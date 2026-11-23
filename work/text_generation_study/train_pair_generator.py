import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
import selfies as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from models import DiscreteDiffusion, SeqGAN, TransformerVAE, create_model
from molecular_tokenizer import MolecularTokenizer


CONDITION_CENTER = np.asarray([298.15, np.log(100.0), 3.7], dtype=np.float32)
CONDITION_SCALE = np.asarray([30.0, 1.0, 0.5], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pair_bucket(value):
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 20


def representation(smiles, language):
    if language == "smiles":
        return smiles
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise
        kekule = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True, kekuleSmiles=True
        )
        return sf.encoder(kekule)


def extend_pair_tokenizer(base_tokenizer, frame, language, output_path):
    vocab = dict(base_tokenizer.vocab)
    missing = set()
    for column in ("cation_smiles", "anion_smiles"):
        for value in frame[column].drop_duplicates():
            missing.update(
                token for token in base_tokenizer.tokenize(representation(value, language))
                if token not in vocab
            )
    for token in sorted(missing):
        vocab[token] = len(vocab)
    if output_path.exists():
        try:
            existing = MolecularTokenizer.load(output_path)
            if all(token in existing.vocab for token in vocab):
                return existing, sorted(missing)
        except Exception:
            pass
    tokenizer = MolecularTokenizer(language, vocab)
    tokenizer.save(output_path)
    return tokenizer, sorted(missing)


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, language, max_length, conditioned, seed, cache_tokens=False,
                 toxicity_floor=2.5):
        self.frame = frame.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.language = language
        self.max_length = max_length
        self.conditioned = conditioned
        rng = np.random.default_rng(seed)
        if conditioned:
            raw = np.column_stack([
                self.frame.Tm_K.to_numpy(float) + rng.uniform(3.0, 40.0, len(self.frame)),
                self.frame.ln_eta_298.to_numpy(float) + rng.uniform(0.1, 1.2, len(self.frame)),
                self.frame.toxicity.to_numpy(float) - rng.uniform(0.05, 0.8, len(self.frame)),
            ]).astype(np.float32)
            raw[:, 0] = np.clip(raw[:, 0], 270.0, 380.0)
            raw[:, 1] = np.clip(raw[:, 1], np.log(20.0), np.log(500.0))
            raw[:, 2] = np.clip(raw[:, 2], toxicity_floor, 4.5)
            self.conditions = (raw - CONDITION_CENTER) / CONDITION_SCALE
        else:
            self.conditions = None
        self.cached_ids = None
        if cache_tokens:
            ion_smiles = set(self.frame.cation_smiles) | set(self.frame.anion_smiles)
            strings = {smiles: representation(smiles, language) for smiles in ion_smiles}
            self.cached_ids = torch.tensor([
                tokenizer.encode_pair(strings[cation], strings[anion], max_length)
                for cation, anion in zip(self.frame.cation_smiles, self.frame.anion_smiles)
            ], dtype=torch.long)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        if self.cached_ids is None:
            row = self.frame.iloc[index]
            cation = representation(row.cation_smiles, self.language)
            anion = representation(row.anion_smiles, self.language)
            ids = torch.tensor(self.tokenizer.encode_pair(cation, anion, self.max_length), dtype=torch.long)
        else:
            ids = self.cached_ids[index]
        condition = self.conditions[index] if self.conditioned else np.zeros(0, dtype=np.float32)
        return ids, torch.tensor(condition, dtype=torch.float32)


def collate_pairs(batch):
    ids = torch.stack([item[0] for item in batch])
    conditions = torch.stack([item[1] for item in batch])
    final = int(ids.ne(0).any(dim=0).nonzero(as_tuple=False).max().item()) + 1
    return ids[:, :final].contiguous(), conditions


def transfer_pretrained(model, checkpoint):
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target = model.state_dict()
    transferred = {}
    for key, value in source.items():
        if key not in target:
            continue
        if value.shape == target[key].shape:
            transferred[key] = value
        elif key.endswith("position.weight") and value.ndim == 2 and target[key].ndim == 2:
            resized = target[key].clone()
            rows = min(value.size(0), resized.size(0))
            resized[:rows] = value[:rows]
            transferred[key] = resized
        elif (
            key.endswith(("token.weight", "head.weight", "head.bias"))
            and value.ndim == target[key].ndim
            and (value.ndim == 1 or value.shape[1:] == target[key].shape[1:])
        ):
            resized = target[key].clone()
            rows = min(value.size(0), resized.size(0))
            resized[:rows] = value[:rows]
            transferred[key] = resized
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    return {"transferred_tensors": len(transferred), "missing": missing, "unexpected": unexpected}


def model_loss(model, ids, conditions, epoch):
    condition_arg = conditions if conditions.size(1) else None
    kl_weight = min(1.0, ((epoch - 1) % 10 + 1) / 10)
    return model.loss(ids, conditions=condition_arg, kl_weight=kl_weight)


def seqgan_pair_step(model, ids, conditions, optimizer, optimizer_d, tokenizer, grad_clip):
    condition_arg = conditions if conditions.size(1) else None
    batch = ids.size(0)
    prefix = ids[:, :2]
    with torch.no_grad():
        fake = model.sample(
            prefix, tokenizer.eos_id, ids.size(1), temperature=1.0, top_k=32,
            conditions=condition_arg,
        )
        if fake.size(1) < ids.size(1):
            padding = torch.full(
                (batch, ids.size(1) - fake.size(1)), tokenizer.pad_id,
                dtype=fake.dtype, device=fake.device,
            )
            fake = torch.cat([fake, padding], dim=1)
    optimizer_d.zero_grad(set_to_none=True)
    real_logits, fake_logits = model.discriminator(ids), model.discriminator(fake)
    d_loss = nn.functional.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    ) + nn.functional.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    d_loss.backward()
    nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip)
    optimizer_d.step()
    optimizer.zero_grad(set_to_none=True)
    logits = model.logits(fake[:, :-1], condition_arg)
    log_probs = torch.log_softmax(logits, -1).gather(
        -1, fake[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    mask = fake[:, 1:].ne(tokenizer.pad_id)
    reward = torch.sigmoid(model.discriminator(fake)).detach()
    policy = -(reward * (log_probs * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    mle = model.loss(ids, conditions=condition_arg)["total"]
    total = policy + 0.1 * mle
    total.backward()
    nn.utils.clip_grad_norm_(
        [parameter for name, parameter in model.named_parameters()
         if not name.startswith("discriminator.")], grad_clip,
    )
    optimizer.step()
    return {"total": total.detach(), "policy": policy.detach(), "discriminator": d_loss.detach()}


def epoch_pass(model, loader, optimizer, scaler, device, epoch, train, grad_clip,
               tokenizer=None, total_epochs=None, optimizer_d=None):
    model.train(train)
    totals, count = {}, 0
    for ids, conditions in loader:
        ids = ids.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        adversarial = (
            train and isinstance(model, SeqGAN) and total_epochs is not None
            and epoch > max(2, total_epochs - 10)
        )
        if adversarial:
            losses = seqgan_pair_step(
                model, ids, conditions, optimizer, optimizer_d, tokenizer, grad_clip
            )
        else:
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                losses = model_loss(model, ids, conditions, epoch)
            if train:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * ids.size(0)
        count += ids.size(0)
    return {key: value / max(count, 1) for key, value in totals.items()}


def pair_length_statistics(frame, tokenizer, language):
    cation_lengths, anion_lengths, total_lengths = [], [], []
    for row in frame.itertuples(index=False):
        cation_length = len(tokenizer.tokenize(representation(row.cation_smiles, language)))
        anion_length = len(tokenizer.tokenize(representation(row.anion_smiles, language)))
        cation_lengths.append(cation_length)
        anion_lengths.append(anion_length)
        total_lengths.append(cation_length + anion_length + 5)
    maximum = min(256, int(math.ceil(np.quantile(total_lengths, 0.99) / 8) * 8))
    keep = np.asarray(total_lengths) <= maximum
    return maximum, keep, {
        "cation": np.asarray(cation_lengths, dtype=int),
        "anion": np.asarray(anion_lengths, dtype=int),
    }


def canonical_pair(cation, anion):
    cation_mol, anion_mol = Chem.MolFromSmiles(cation), Chem.MolFromSmiles(anion)
    if cation_mol is None or anion_mol is None:
        return None
    cation_charge = sum(atom.GetFormalCharge() for atom in cation_mol.GetAtoms())
    anion_charge = sum(atom.GetFormalCharge() for atom in anion_mol.GetAtoms())
    if cation_charge != 1 or anion_charge != -1 or "." in cation or "." in anion:
        return None
    cation = Chem.MolToSmiles(cation_mol, canonical=True, isomericSmiles=True)
    anion = Chem.MolToSmiles(anion_mol, canonical=True, isomericSmiles=True)
    return cation, anion, f"{cation}.{anion}"


@torch.no_grad()
def generate_pairs(model, tokenizer, language, max_length, count, device, length_priors,
                   conditioned, generation_seed=20260919, condition_vector=None, *,
                   temperature=1.0, top_k=32):
    model.eval()
    records = []
    if conditioned:
        condition_vector = (
            np.zeros(3, dtype=np.float32) if condition_vector is None
            else np.asarray(condition_vector, dtype=np.float32)
        )
        target_condition = torch.as_tensor(
            np.tile(condition_vector, (256, 1)), device=device
        )
    else:
        target_condition = torch.zeros(256, 0, device=device)
    rng = np.random.default_rng(generation_seed)
    while len(records) < count:
        size = min(256, count - len(records))
        prefix = torch.tensor([[tokenizer.bos_id, tokenizer.vocab["<CAT>"]]] * size, device=device)
        conditions = target_condition[:size] if conditioned else None
        kwargs = {}
        if isinstance(model, DiscreteDiffusion):
            cation_lengths = rng.choice(length_priors["cation"], size=size, replace=True)
            anion_lengths = rng.choice(length_priors["anion"], size=size, replace=True)
            pair_positions = cation_lengths + 2
            anion_positions = pair_positions + 1
            total_lengths = cation_lengths + anion_lengths + 5
            kwargs.update({
                "lengths": torch.as_tensor(total_lengths, device=device),
                "fixed_positions": torch.as_tensor(
                    np.column_stack([pair_positions, anion_positions]), device=device
                ),
                "fixed_token_ids": torch.as_tensor(
                    np.tile([tokenizer.vocab["<PAIR>"], tokenizer.vocab["<ANI>"]], (size, 1)),
                    device=device,
                ),
            })
        ids = model.sample(
            prefix, tokenizer.eos_id, max_length, temperature=temperature, top_k=top_k,
            conditions=conditions, **kwargs
        )
        for sequence in ids.cpu().tolist():
            try:
                cation, anion, cation_repr, anion_repr = tokenizer.decode_pair(sequence)
                valid = canonical_pair(cation, anion)
                if valid is None:
                    raise ValueError("invalid charge or component count")
                cation, anion, pair = valid
                error = ""
            except Exception as exc:
                cation = anion = pair = cation_repr = anion_repr = ""
                error = type(exc).__name__
            records.append({
                "cation_smiles": cation, "anion_smiles": anion, "IL_SMILES": pair,
                "cation_representation": cation_repr, "anion_representation": anion_repr,
                "valid_pair": not bool(error), "error": error,
            })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--variant", choices=["unconditional", "hard_conditioned"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    configure_selfies(config)
    set_seed(args.seed)
    root = Path(config["output_root"])
    run_dir = root / "pair_runs" / args.variant / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "COMPLETED.json"
    if completed.exists():
        print(completed.read_text(encoding="utf-8"), flush=True)
        return
    latest_path = run_dir / "latest.pt"
    loss_path = run_dir / "loss.csv"
    epochs = config["training"]["pair_epochs"]
    state = None
    if latest_path.exists():
        print(f"Reading saved checkpoint: {latest_path}", flush=True)
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint verified at epoch {state['epoch']}/{epochs}", flush=True)
    best = json.loads((root / "BEST_MODEL.json").read_text(encoding="utf-8"))
    language, model_name, regime = best["language"], best["model"], best["regime"]
    frame = pd.read_parquet(root / "pair_pool" / "pair_pool_scored.parquet")
    base_tokenizer = MolecularTokenizer.load(root / "data" / f"vocab_{language}.json")
    pair_vocab_path = root / "pair_pool" / f"vocab_{language}_pair.json"
    tokenizer, added_tokens = extend_pair_tokenizer(
        base_tokenizer, frame, language, pair_vocab_path
    )
    maximum, keep, priors_all = pair_length_statistics(frame, tokenizer, language)
    frame = frame.loc[keep].reset_index(drop=True)
    priors = {key: value[keep] for key, value in priors_all.items()}
    buckets = frame.IL_SMILES.map(pair_bucket)
    train_frame, validation_frame = frame[buckets < 18], frame[buckets == 18]
    conditioned = args.variant == "hard_conditioned"
    condition_dim = 3 if conditioned else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name, len(tokenizer.vocab), maximum, tokenizer, config["model"],
        condition_dim=condition_dim,
    ).to(device)
    source_checkpoint = root / "runs" / language / model_name / regime / str(args.seed) / "final_model.pt"
    # Match the regular run's source-weight initialization before selecting the
    # checkpoint. The completed-epoch path needs no data loader or optimizer.
    transfer = transfer_pretrained(model, source_checkpoint)
    if state is not None and int(state["epoch"]) >= epochs:
        best_loss = float(state["best"])
        del state
        print(f"Resuming at sampling: {args.variant}, seed {args.seed}", flush=True)
        finish_pair_run(model, tokenizer, language, maximum, args, device, priors,
                        conditioned, config, run_dir, completed, best_loss,
                        transfer, base_tokenizer, added_tokens, model_name, regime, time.time(),
                        reuse_samples=True)
        return
    cache_tokens = config.get("evaluation", {}).get("allfold_ratio_de", False)
    toxicity_floor = config["evaluation"].get("physical_limits", {}).get("log10_EC50_gt", 2.5)
    train_data = PairDataset(train_frame, tokenizer, language, maximum, conditioned, args.seed, cache_tokens, toxicity_floor)
    validation_data = PairDataset(validation_frame, tokenizer, language, maximum, conditioned, args.seed + 1, cache_tokens, toxicity_floor)
    batch_size = config["training"]["pair_batch_size"]
    # Windows worker subprocesses can exit during long paired runs; GPU training
    # remains unchanged when collation is performed in the parent process.
    loader_workers = 0
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=loader_workers,
        pin_memory=True, persistent_workers=loader_workers > 0, collate_fn=collate_pairs,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=loader_workers,
        pin_memory=True, persistent_workers=loader_workers > 0, collate_fn=collate_pairs,
    )
    generator_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("discriminator.")
    ]
    optimizer = torch.optim.AdamW(
        generator_parameters, lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    optimizer_d = (
        torch.optim.AdamW(
            model.discriminator.parameters(), lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        ) if isinstance(model, SeqGAN) else None
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"])
    fields = [
        "epoch", "train_total", "validation_total", "train_reconstruction",
        "validation_reconstruction", "train_kl", "validation_kl",
        "train_denoising", "validation_denoising", "learning_rate", "seconds",
        "train_policy", "validation_policy", "train_discriminator", "validation_discriminator",
    ]
    best_loss, stale = float("inf"), 0
    start_epoch = 1
    existing_rows = []
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if optimizer_d is not None and state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(state["optimizer_d"])
        best_loss = float(state["best"])
        start_epoch = int(state["epoch"]) + 1
        if loss_path.exists():
            with loss_path.open("r", newline="", encoding="utf-8") as existing:
                existing_rows = [row for row in csv.DictReader(existing) if int(row["epoch"]) < start_epoch]
        stale = int(state.get("stale", 0))
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        del state
        print(json.dumps({"resumed_from": str(latest_path), "next_epoch": start_epoch}), flush=True)
    started_all = time.time()
    with (run_dir / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        handle.flush()
        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            train_metrics = epoch_pass(
                model, train_loader, optimizer, scaler, device, epoch, True,
                config["training"]["grad_clip"], tokenizer, epochs, optimizer_d,
            )
            with torch.no_grad():
                validation_metrics = epoch_pass(
                    model, validation_loader, optimizer, scaler, device, epoch, False,
                    config["training"]["grad_clip"], tokenizer, epochs, optimizer_d,
                )
            row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                   "seconds": time.time() - started}
            for prefix, values in (("train", train_metrics), ("validation", validation_metrics)):
                for key in ("total", "reconstruction", "kl", "denoising", "policy", "discriminator"):
                    row[f"{prefix}_{key}"] = values.get(key)
            writer.writerow(row); handle.flush()
            monitor_name = "reconstruction" if isinstance(model, TransformerVAE) else "total"
            current = validation_metrics[monitor_name]
            if current < best_loss:
                best_loss, stale = current, 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
            else:
                stale += 1
            scheduler.step()
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best_loss,
                "stale": stale, "scaler": scaler.state_dict(),
            }
            if optimizer_d is not None:
                checkpoint["optimizer_d"] = optimizer_d.state_dict()
            pending = latest_path.with_suffix(".partial")
            torch.save(checkpoint, pending)
            pending.replace(latest_path)
            print(json.dumps(row), flush=True)
            if stale >= config["training"]["pair_patience"] and not isinstance(model, SeqGAN):
                break
    finish_pair_run(model, tokenizer, language, maximum, args, device, priors,
                    conditioned, config, run_dir, completed, best_loss,
                    transfer, base_tokenizer, added_tokens, model_name, regime, started_all)


def finish_pair_run(model, tokenizer, language, maximum, args, device, priors,
                    conditioned, config, run_dir, completed, best_loss,
                    transfer, base_tokenizer, added_tokens, model_name, regime, started_all,
                    reuse_samples=False):
    selected_path = run_dir / ("latest.pt" if isinstance(model, SeqGAN) else "best_model.pt")
    selected_state = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(selected_state["model"] if isinstance(selected_state, dict) and "model" in selected_state else selected_state)
    sample_path = run_dir / "generated_pairs.csv"
    if reuse_samples and sample_path.exists():
        generated = pd.read_csv(sample_path)
        if (len(generated) != args.samples or generated.valid_pair.dtype.kind != "b"
                or generated.loc[generated.valid_pair, "IL_SMILES"].isna().any()):
            raise ValueError(f"Saved sample batch did not pass validation: {sample_path}")
        print(f"Reusing {len(generated)} saved draws: {args.variant}, seed {args.seed}", flush=True)
    else:
        print(f"Sampling {args.samples} pairs: {args.variant}, seed {args.seed}", flush=True)
        generated = generate_pairs(
            model, tokenizer, language, maximum, args.samples, device, priors, conditioned,
            generation_seed=args.seed, condition_vector=target_condition(config),
        )
        pending = run_dir / "generated_pairs.partial"
        generated.to_csv(pending, index=False)
        pending.replace(sample_path)
    payload = {
        "variant": args.variant, "seed": args.seed, "language": language, "model": model_name,
        "regime": regime,
        "maximum_length": maximum, "condition_dim": 3 if conditioned else 0,
        "best_validation_loss": best_loss, "generated": len(generated),
        "monitor": "reconstruction" if isinstance(model, TransformerVAE) else "total",
        "selected_checkpoint": selected_path.name,
        "valid_pair_rate": float(generated.valid_pair.mean()), "seconds": time.time() - started_all,
        "pretrained_transfer": transfer,
        "downstream_vocabulary_extension": {
            "base_size": len(base_tokenizer.vocab), "pair_size": len(tokenizer.vocab),
            "added_tokens": added_tokens,
        },
    }
    pending = completed.with_suffix(".partial")
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pending.replace(completed)
    print(json.dumps(payload, indent=2))


def target_condition(config):
    limits = config.get("evaluation", {}).get("physical_limits")
    if limits is None:
        return None
    raw = np.array([limits["Tm_K_lt"], np.log(limits["viscosity_mPas_lt"]),
                    limits["log10_EC50_gt"]], dtype=np.float32)
    return (raw - CONDITION_CENTER) / CONDITION_SCALE


def configure_selfies(config):
    if config.get("evaluation", {}).get("allow_hexacoordinate_phosphorus_anions"):
        constraints = sf.get_semantic_constraints()
        constraints["P-1"] = 6
        sf.set_semantic_constraints(constraints)


if __name__ == "__main__":
    main()


