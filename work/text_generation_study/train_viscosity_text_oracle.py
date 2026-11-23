import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from text_oracles import PHYSICAL_ROOT, ROOT, physical_model_module, physical_tokenizer_module


class ViscosityDataset(Dataset):
    def __init__(self, frame, tokenizer, label_mean, label_std, temperature_mean, temperature_std):
        self.tokens = torch.tensor([
            tokenizer.encode(value, max_length=100, truncation=True, padding="max_length")
            for value in frame["Normalized SMILES"].astype(str)
        ], dtype=torch.long)
        self.temperature = torch.tensor(
            ((frame["T/K"].to_numpy(float) - temperature_mean) / temperature_std),
            dtype=torch.float32,
        )
        self.target = torch.tensor(
            ((frame["ln(n_mPas)"].to_numpy(float) - label_mean) / label_std),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.tokens[index], self.temperature[index], self.target[index]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    predictions, targets = [], []
    for tokens, temperature, target in loader:
        tokens = tokens.to(device, non_blocking=True)
        temperature = temperature.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            prediction = model([tokens, temperature]).reshape(-1)
            loss = F.smooth_l1_loss(prediction, target)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * len(target)
        count += len(target)
        predictions.append(prediction.detach().float().cpu().numpy())
        targets.append(target.detach().float().cpu().numpy())
    return total / max(count, 1), np.concatenate(predictions), np.concatenate(targets)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 11))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    source = ROOT / "viscosity.csv"
    output = ROOT / "outputs" / "IECR_text_generation_revision_20260919" / "predictors" / "viscosity_text_10fold"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    required = ["Normalized SMILES", "T/K", "ln(n_mPas)"]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    groups = frame["Normalized SMILES"].astype(str).to_numpy()
    splits = list(GroupKFold(n_splits=10).split(frame, groups=groups))
    folds = [args.fold] if args.fold else list(range(1, 11))
    config = yaml.safe_load((PHYSICAL_ROOT / "config.yaml").read_text(encoding="utf-8"))
    tokenizer = physical_tokenizer_module.SMILES_Atomwise_Tokenizer(str(PHYSICAL_ROOT / "vocab.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in folds:
        destination = output / f"fold_{fold}_best_model.pt"
        prediction_path = output / f"fold_{fold}_test_predictions.csv"
        metric_path = output / f"fold_{fold}_metrics.json"
        if destination.exists() and prediction_path.exists():
            if not metric_path.exists():
                test_indices = splits[fold - 1][1]
                validation_indices = splits[fold % 10][1]
                held_out = np.zeros(len(frame), dtype=bool)
                held_out[test_indices] = True
                held_out[validation_indices] = True
                train_indices = np.flatnonzero(~held_out)
                train = frame.iloc[train_indices]
                validation = frame.iloc[validation_indices]
                test = frame.iloc[test_indices]
                result = pd.read_csv(prediction_path)
                checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
                history_path = output / f"fold_{fold}_history.csv"
                elapsed = 0.0
                if history_path.exists():
                    elapsed = float(pd.read_csv(history_path)["seconds"].max())
                metrics = {
                    "fold": int(fold), "n_train": int(len(train)),
                    "n_validation": int(len(validation)), "n_test": int(len(test)),
                    "n_train_ils": int(train["Normalized SMILES"].nunique()),
                    "n_validation_ils": int(validation["Normalized SMILES"].nunique()),
                    "n_test_ils": int(test["Normalized SMILES"].nunique()),
                    "mae": float(mean_absolute_error(result["ln(n_mPas)"], result.prediction)),
                    "rmse": float(mean_squared_error(result["ln(n_mPas)"], result.prediction) ** 0.5),
                    "r2": float(r2_score(result["ln(n_mPas)"], result.prediction)),
                    "best_epoch": int(checkpoint["epoch"]), "seconds": elapsed,
                }
                metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(json.dumps({"fold": fold, "status": "already complete"}), flush=True)
            continue
        test_indices = splits[fold - 1][1]
        validation_indices = splits[fold % 10][1]
        held_out = np.zeros(len(frame), dtype=bool)
        held_out[test_indices] = True
        held_out[validation_indices] = True
        train_indices = np.flatnonzero(~held_out)
        train = frame.iloc[train_indices].reset_index(drop=True)
        validation = frame.iloc[validation_indices].reset_index(drop=True)
        test = frame.iloc[test_indices].reset_index(drop=True)
        label_mean = float(train["ln(n_mPas)"].mean())
        label_std = float(train["ln(n_mPas)"].std(ddof=1))
        temperature_mean = float(train["T/K"].mean())
        temperature_std = float(train["T/K"].std(ddof=1))
        datasets = {
            name: ViscosityDataset(part, tokenizer, label_mean, label_std, temperature_mean, temperature_std)
            for name, part in (("train", train), ("validation", validation), ("test", test))
        }
        loaders = {
            "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=device.type == "cuda"),
            "validation": DataLoader(datasets["validation"], batch_size=args.batch_size * 2,
                                     shuffle=False, num_workers=0, pin_memory=device.type == "cuda"),
            "test": DataLoader(datasets["test"], batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=0, pin_memory=device.type == "cuda"),
        }
        seed_everything(20260919 + fold)
        model = physical_model_module.ILBERT_T(**config["transformer"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        best, stale, history = float("inf"), 0, []
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(model, loaders["train"], device, optimizer, scaler)
            validation_loss, _, _ = run_epoch(model, loaders["validation"], device)
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "validation_loss": validation_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "seconds": time.time() - started})
            print(json.dumps({"fold": fold, **history[-1]}), flush=True)
            if validation_loss < best - 1e-5:
                best, stale = validation_loss, 0
                torch.save({
                    "model": model.state_dict(), "fold": fold,
                    "label_mean": label_mean, "label_std": label_std,
                    "temperature_mean": temperature_mean, "temperature_std": temperature_std,
                    "validation_loss": best, "epoch": epoch,
                    "train_groups": sorted(train["Normalized SMILES"].astype(str).unique().tolist()),
                    "validation_groups": sorted(validation["Normalized SMILES"].astype(str).unique().tolist()),
                    "test_groups": sorted(test["Normalized SMILES"].astype(str).unique().tolist()),
                }, destination)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        pd.DataFrame(history).to_csv(output / f"fold_{fold}_history.csv", index=False)
        checkpoint = torch.load(destination, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        _, prediction_scaled, target_scaled = run_epoch(model, loaders["test"], device)
        prediction = prediction_scaled * label_std + label_mean
        target = target_scaled * label_std + label_mean
        result = test[["Normalized SMILES", "T/K", "ln(n_mPas)"]].copy()
        result["prediction"] = prediction
        result["fold"] = fold
        result.to_csv(prediction_path, index=False)
        metrics = {
            "fold": int(fold), "n_train": int(len(train)),
            "n_validation": int(len(validation)), "n_test": int(len(test)),
            "n_train_ils": int(train["Normalized SMILES"].nunique()),
            "n_validation_ils": int(validation["Normalized SMILES"].nunique()),
            "n_test_ils": int(test["Normalized SMILES"].nunique()),
            "mae": float(mean_absolute_error(target, prediction)),
            "rmse": float(mean_squared_error(target, prediction) ** 0.5),
            "r2": float(r2_score(target, prediction)), "best_epoch": int(checkpoint["epoch"]),
            "seconds": float(time.time() - started),
        }
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_files = [output / f"fold_{fold}_metrics.json" for fold in range(1, 11)]
    prediction_files = [output / f"fold_{fold}_test_predictions.csv" for fold in range(1, 11)]
    if all(path.exists() for path in metric_files + prediction_files):
        metrics = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in metric_files])
        metrics.to_csv(output / "fold_metrics.csv", index=False)
        predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
        predictions.to_csv(output / "oof_predictions.csv", index=False)
        summary = {
            "n": int(len(predictions)), "n_unique_ils": int(predictions["Normalized SMILES"].nunique()),
            "mae": float(mean_absolute_error(predictions["ln(n_mPas)"], predictions.prediction)),
            "rmse": float(mean_squared_error(predictions["ln(n_mPas)"], predictions.prediction) ** 0.5),
            "r2": float(r2_score(predictions["ln(n_mPas)"], predictions.prediction)),
            "split": "GroupKFold by canonical ionic liquid with disjoint validation fold",
            "source_sha256": sha256(source), "device": str(device),
        }
        (output / "COMPLETED.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
