"""Run the full-fold, ratio-DE study in its own output directory."""
import argparse
import json
import os
from pathlib import Path
from shutil import copy2
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import yaml

from allfold_oracle import FullFoldOracle, GAS_FOLDS, PHYSICAL_FOLDS, apply_physical_limits
from study_thresholds import PHYSICAL_LIMITS

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
OLD = ROOT / "outputs/IECR_section33_50k_20261023"
PREVIOUS_FULL_FOLD = ROOT / "outputs/IECR_allfold_ratio_DE_20261111"
OUT = Path(os.environ.get("ILGEN_STUDY_OUTPUT",
                          ROOT / "outputs/IECR_allfold_ratio_DE_eta200_EC50gt2_20261111"))
SEEDS = (11, 23, 37)


def save_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def check_predictions(frame):
    for field, folds in (
        ("Tm_K", PHYSICAL_FOLDS), ("ln_eta_298", PHYSICAL_FOLDS),
        ("toxicity", PHYSICAL_FOLDS), ("ln_x_SO2_298", GAS_FOLDS),
        ("ln_x_SO2_313", GAS_FOLDS), ("ln_x_CO2_298", GAS_FOLDS),
    ):
        values = frame[[f"{field}_fold{i:02d}" for i in folds]].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite ensemble predictions for {field}")
        np.testing.assert_allclose(values.mean(axis=1), frame[field], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(values.std(axis=1, ddof=1), frame[f"{field}_sd"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(frame.DE, frame.x_SO2_298 / frame.x_SO2_313)


def prepare():
    config_file = OUT / "config.yaml"
    pool_file = OUT / "pair_pool/pair_pool_scored.parquet"
    protocol_file = OUT / "protocol.json"
    if pool_file.exists() and config_file.exists() and protocol_file.exists():
        return
    for relative in ("BEST_MODEL.json", "data/known_pairs.csv", "data/vocab_selfies.json",
                     "pair_pool/vocab_selfies_pair.json",
                     *(f"runs/selfies/vae/scratch/{seed}/final_model.pt" for seed in SEEDS)):
        dest = OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy2(OLD / relative, dest)
    config = yaml.safe_load((OLD / "config.yaml").read_text())
    config["output_root"] = str(OUT)
    config["data"]["known_pairs"] = str(OUT / "data/known_pairs.csv")
    # The original setup used 80 cosine-scheduler epochs and patience 10.
    config["training"].update(pair_epochs=80, pair_patience=10)
    config["evaluation"]["allfold_ratio_de"] = True
    config["evaluation"]["physical_limits"] = PHYSICAL_LIMITS
    config["evaluation"]["allow_hexacoordinate_phosphorus_anions"] = True
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source = pd.read_parquet(OLD / "pair_pool/pair_pool_scored.parquet",
                             columns=["pair_id", "cation_smiles", "anion_smiles", "IL_SMILES", "source"])
    assert len(source) == source.IL_SMILES.nunique() == 50_000
    assert source.source.value_counts().to_dict() == {
        "known_ion_recombination": 45_353, "known_experimental": 4_647
    }
    saved = PREVIOUS_FULL_FOLD / "pair_pool/pair_pool_scored.parquet"
    if saved.exists():
        all_scores = pd.read_parquet(saved)
        assert all_scores.IL_SMILES.tolist() == source.IL_SMILES.tolist()
        check_predictions(all_scores)
        rescored = apply_physical_limits(all_scores)
        save_parquet(rescored.head(8), OUT / "qa/oracle_probe.parquet")
        save_parquet(rescored, pool_file)
        print("Recomputed the new feasibility labels for 50,000 saved full-fold predictions.", flush=True)
    else:
        score_pool(source, pool_file)
    protocol_file.write_text(json.dumps({
        "physical_folds": PHYSICAL_FOLDS, "gas_folds": GAS_FOLDS,
        "physical_limits": PHYSICAL_LIMITS,
        "DE": "x_SO2(298.15 K, 1 bar) / x_SO2(313.15 K, 1 bar)",
        "gas_aggregation": "exp(mean(ln(x_fold)))", "seeds": SEEDS,
        "selfies_P-1_bond_capacity": 6,
    }, indent=2), encoding="utf-8")


def score_pool(source, pool_file):
    oracle = FullFoldOracle(device="cuda")
    probe = oracle.score(source.IL_SMILES.iloc[:8].tolist())
    check_predictions(probe)
    save_parquet(probe, OUT / "qa/oracle_probe.parquet")
    print("Probe passed: 10 physical folds, 5 gas folds, DE ratio", flush=True)
    chunks = []
    for start in range(0, len(source), 4096):
        stop = min(start + 4096, len(source))
        cache = OUT / "pair_pool/full_fold_chunks" / f"{start:05d}_{stop:05d}.parquet"
        smiles = source.IL_SMILES.iloc[start:stop].tolist()
        if cache.exists():
            scores = pd.read_parquet(cache)
        else:
            scores = oracle.score(smiles)
            check_predictions(scores)
            save_parquet(scores, cache)
        assert scores.IL_SMILES.tolist() == smiles
        chunks.append(scores)
        print(f"Full-fold pool: {stop:,}/{len(source):,}", flush=True)
    scores = pd.concat(chunks, ignore_index=True)
    save_parquet(pd.concat([source, scores.drop(columns="IL_SMILES")], axis=1), pool_file)


def run_script(name, *args):
    print(f"Running {name}: {' '.join(map(str, args))}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full study.")
    prepare()
    if args.prepare_only:
        return
    config = OUT / "config.yaml"
    for seed in SEEDS:
        for arm in ("unconditional", "hard_conditioned"):
            run_script("train_pair_generator.py", "--config", config, "--variant", arm,
                       "--seed", seed, "--samples", 1000)
        run_script("iterate_pair_generator_physical_10k_20261023.py", "--config", config, "--seed", seed)
    run_script("summarize_allfold_ratio_de.py")


if __name__ == "__main__":
    main()
