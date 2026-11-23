import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from models import DiscreteDiffusion, SeqGAN, TransformerVAE, create_model
from molecular_tokenizer import MolecularTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IonDataset(Dataset):
    def __init__(self, frame, tokenizer, language, max_length):
        self.values = frame[["ion_type", language]].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language = language
        self.max_length = max_length

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        row = self.values.iloc[index]
        return torch.tensor(
            self.tokenizer.encode(row[self.language], row.ion_type, self.max_length), dtype=torch.long
        )


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle, seed=20260919):
        self.lengths = np.asarray(lengths, dtype=int)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        order = np.argsort(self.lengths, kind="stable")
        batches = [
            order[start:start + self.batch_size].copy()
            for start in range(0, len(order), self.batch_size)
        ]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
            self.epoch += 1
        for batch in batches:
            yield batch.tolist()


def trim_padding_collate(batch):
    ids = torch.stack(batch)
    non_padding = ids.ne(0)
    batch_length = int(non_padding.any(dim=0).nonzero(as_tuple=False).max().item()) + 1
    return ids[:, :batch_length].contiguous()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, stage):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "stage": stage,
    }
    if isinstance(model, SeqGAN) and hasattr(model, "_optimizer_d"):
        payload["optimizer_d"] = model._optimizer_d.state_dict()
    torch.save(payload, path)


def pad_generated(ids, max_length, pad_id):
    if ids.size(1) < max_length:
        padding = torch.full((ids.size(0), max_length - ids.size(1)), pad_id, device=ids.device, dtype=ids.dtype)
        ids = torch.cat([ids, padding], dim=1)
    return ids[:, :max_length]


def seqgan_step(model, real, optimizer_g, optimizer_d, tokenizer, max_length):
    batch = real.size(0)
    ion_ids = real[:, 1:2]
    prefix = torch.cat([torch.full_like(ion_ids, tokenizer.bos_id), ion_ids], dim=1)
    with torch.no_grad():
        batch_length = real.size(1)
        fake = pad_generated(
            model.sample(prefix, tokenizer.eos_id, batch_length, 1.0, 32),
            batch_length, tokenizer.pad_id,
        )
    optimizer_d.zero_grad(set_to_none=True)
    real_logits = model.discriminator(real)
    fake_logits = model.discriminator(fake)
    d_loss = nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    d_loss += nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    d_loss.backward()
    nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
    optimizer_d.step()

    optimizer_g.zero_grad(set_to_none=True)
    logits = model.logits(fake[:, :-1])
    log_probs = torch.log_softmax(logits, -1).gather(-1, fake[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = fake[:, 1:].ne(tokenizer.pad_id)
    reward = torch.sigmoid(model.discriminator(fake)).detach()
    policy = -(reward * (log_probs * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    mle = model.loss(real)["total"]
    g_loss = policy + 0.1 * mle
    g_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_g.step()
    return {"total": float(g_loss.detach()), "policy": float(policy.detach()), "discriminator": float(d_loss.detach())}


def run_epoch(model, loader, optimizer, scaler, device, grad_clip, train, epoch, total_epochs, tokenizer, max_length, stage):
    model.train(train)
    totals = {}
    count = 0
    for ids in loader:
        ids = ids.to(device, non_blocking=True)
        use_adversarial = (
            train
            and isinstance(model, SeqGAN)
            and (
                stage == "adversarial"
                or (stage == "finetune" and epoch > max(2, total_epochs - 10))
            )
        )
        if use_adversarial:
            metrics = seqgan_step(model, ids, optimizer, model._optimizer_d, tokenizer, max_length)
        else:
            if train:
                optimizer.zero_grad(set_to_none=True)
            kl_weight = min(1.0, ((epoch - 1) % 10 + 1) / 10)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                losses = model.loss(ids, kl_weight=kl_weight, deterministic=not train)
            if train:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            metrics = {key: float(value.detach()) for key, value in losses.items()}
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * ids.size(0)
        count += ids.size(0)
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_stage(model, train_frame, validation_frame, tokenizer, language, max_length, config, run_dir, stage, epochs, device, smoke):
    if smoke:
        train_frame = train_frame.sample(min(2048, len(train_frame)), random_state=17)
        validation_frame = validation_frame.sample(min(512, len(validation_frame)), random_state=19)
        epochs = 1
    train_data = IonDataset(train_frame, tokenizer, language, max_length)
    validation_data = IonDataset(validation_frame, tokenizer, language, max_length)
    batch_size = config["training"]["batch_size"]
    train_sampler = LengthBucketBatchSampler(
        train_frame[f"{language}_tokens"].to_numpy() + 3, batch_size, shuffle=True
    )
    validation_sampler = LengthBucketBatchSampler(
        validation_frame[f"{language}_tokens"].to_numpy() + 3, batch_size, shuffle=False
    )
    loader_workers = config["training"]["num_workers"] if stage == "pretrain" else 0
    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler,
        num_workers=loader_workers, pin_memory=True,
        persistent_workers=loader_workers > 0, collate_fn=trim_padding_collate,
    )
    validation_loader = DataLoader(
        validation_data, batch_sampler=validation_sampler,
        num_workers=loader_workers, pin_memory=True,
        persistent_workers=loader_workers > 0, collate_fn=trim_padding_collate,
    )
    optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if "discriminator" not in name],
        lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
    )
    if isinstance(model, SeqGAN):
        model._optimizer_d = torch.optim.AdamW(
            model.discriminator.parameters(), lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] and device.type == "cuda")
    best = float("inf")
    stale = 0
    best_path = run_dir / f"{stage}_best.pt"
    latest_path = run_dir / f"{stage}_latest.pt"
    completed_path = run_dir / f"{stage}_COMPLETED.json"
    if completed_path.exists() and best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        return model, float(state["best"])

    start_epoch = 1
    resume_path = latest_path if latest_path.exists() else (best_path if best_path.exists() else None)
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if isinstance(model, SeqGAN) and state.get("optimizer_d") is not None:
            model._optimizer_d.load_state_dict(state["optimizer_d"])
        best = float(state["best"])
        start_epoch = int(state["epoch"]) + 1
        if resume_path == best_path:
            scheduler.step()
        print(json.dumps({"stage": stage, "resumed_from": str(resume_path), "next_epoch": start_epoch}), flush=True)

    log_path = run_dir / f"{stage}_loss.csv"
    loss_names = ["total", "reconstruction", "kl", "denoising", "policy", "discriminator"]
    fields = ["stage", "epoch"]
    fields += [f"train_{name}" for name in loss_names]
    fields += [f"validation_{name}" for name in loss_names]
    fields += ["learning_rate", "seconds"]
    existing_rows = []
    if log_path.exists() and start_epoch > 1:
        with log_path.open("r", newline="", encoding="utf-8") as existing:
            existing_rows = [
                row for row in csv.DictReader(existing)
                if int(row["epoch"]) < start_epoch
            ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        handle.flush()
        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, config["training"]["grad_clip"], True, epoch, epochs, tokenizer, max_length, stage)
            with torch.no_grad():
                validation_metrics = run_epoch(model, validation_loader, optimizer, scaler, device, config["training"]["grad_clip"], False, epoch, epochs, tokenizer, max_length, stage)
            row = {"stage": stage, "epoch": epoch,
                   "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
            for prefix, values in (("train", train_metrics), ("validation", validation_metrics)):
                for name in loss_names:
                    row[f"{prefix}_{name}"] = values.get(name)
            writer.writerow(row)
            handle.flush()
            monitor_name = "reconstruction" if isinstance(model, TransformerVAE) else "total"
            current = validation_metrics.get(monitor_name, float("inf"))
            improved = current < best
            if improved:
                best = current
                stale = 0
            else:
                stale += 1
            scheduler.step()
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best, stage)
            if improved:
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best, stage)
            print(json.dumps(row), flush=True)
            mandatory_seqgan_adversarial = isinstance(model, SeqGAN) and stage == "finetune"
            if stale >= config["training"]["patience"] and not mandatory_seqgan_adversarial:
                break
    selected_path = latest_path if isinstance(model, SeqGAN) and stage == "finetune" else best_path
    state = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    completed_path.write_text(json.dumps({
        "stage": stage, "best": float(best), "best_epoch": int(state["epoch"]),
        "selected_checkpoint": selected_path.name,
        "monitor": "reconstruction" if isinstance(model, TransformerVAE) else "total",
        "requested_epochs": epochs,
    }, indent=2), encoding="utf-8")
    return model, best


@torch.no_grad()
def generate(model, tokenizer, max_length, count, device, length_priors=None, batch_size=256):
    model.eval()
    records = []
    for ion_type, marker in (("cation", "<CAT>"), ("anion", "<ANI>")):
        remaining = count // 2
        while remaining > 0:
            size = min(batch_size, remaining)
            prefix = torch.tensor([[tokenizer.bos_id, tokenizer.vocab[marker]]] * size, device=device)
            sample_kwargs = {}
            if isinstance(model, DiscreteDiffusion):
                prior = np.asarray(length_priors[ion_type], dtype=int)
                sampled = np.random.choice(prior, size=size, replace=True) + 3
                sample_kwargs["lengths"] = torch.as_tensor(sampled, device=device)
            ids = model.sample(prefix, tokenizer.eos_id, max_length, temperature=1.0, top_k=32, **sample_kwargs)
            for sequence in ids.cpu().tolist():
                representation = tokenizer.decode_tokens(sequence)
                try:
                    text = representation if tokenizer.language == "smiles" else __import__("selfies").decoder(representation)
                    error = ""
                except Exception as exc:
                    text = ""
                    error = type(exc).__name__
                records.append({
                    "ion_type": ion_type, "representation": representation,
                    "generated_smiles": text, "decode_error": error,
                })
            remaining -= size
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--language", choices=["smiles", "selfies"], required=True)
    parser.add_argument("--model", choices=["vae", "diffusion", "seqgan", "transformer"], required=True)
    parser.add_argument("--regime", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(args.seed)
    output_root = Path(config["output_root"])
    data_dir = output_root / "data"
    run_root = output_root / ("smoke_runs" if args.smoke else "runs")
    run_dir = run_root / args.language / args.model / args.regime / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "COMPLETED.json"
    if completed.exists():
        print(completed.read_text(encoding="utf-8"))
        return
    manifest = json.loads((data_dir / "PREPARE_COMPLETED.json").read_text(encoding="utf-8"))
    max_length = manifest["maximum_tokens"][args.language]
    tokenizer = MolecularTokenizer.load(data_dir / f"vocab_{args.language}.json")
    target = pd.read_parquet(data_dir / "target_ions.parquet")
    pretrain = pd.read_parquet(data_dir / "pretrain_ions.parquet") if args.regime == "pretrained" else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, len(tokenizer.vocab), max_length, tokenizer, config["model"]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameters = sum(p.numel() for p in model.parameters())
    started = time.time()
    if args.regime == "pretrained":
        model, pretrain_loss = train_stage(
            model, pretrain[pretrain.split == "train"], pretrain[pretrain.split == "validation"],
            tokenizer, args.language, max_length, config, run_dir, "pretrain",
            config["training"]["pretrain_epochs"], device, args.smoke,
        )
    else:
        pretrain_loss = None
    model, finetune_loss = train_stage(
        model, target[target.split == "train"], target[target.split == "validation"],
        tokenizer, args.language, max_length, config, run_dir, "finetune",
        config["training"]["finetune_epochs"], device, args.smoke,
    )
    sample_count = 200 if args.smoke else config["evaluation"]["samples_per_run"]
    training_target = target[target.split == "train"]
    length_column = f"{args.language}_tokens"
    length_priors = {
        ion_type: training_target.loc[training_target.ion_type == ion_type, length_column].astype(int).tolist()
        for ion_type in ("cation", "anion")
    }
    generated = generate(model, tokenizer, max_length, sample_count, device, length_priors)
    generated.to_csv(run_dir / "generated.csv", index=False)
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    payload = {
        "language": args.language, "model": args.model, "regime": args.regime, "seed": args.seed,
        "parameters": parameters, "maximum_length": max_length, "pretrain_best_loss": pretrain_loss,
        "finetune_best_loss": finetune_loss, "generated": len(generated), "seconds": time.time() - started,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda" else 0.0
        ),
    }
    completed.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
