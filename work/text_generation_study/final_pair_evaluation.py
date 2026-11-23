import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Contrib.SA_Score import sascorer
import torch
import yaml

from models import create_model
from molecular_tokenizer import MolecularTokenizer
from text_oracles import TextOracle
from train_pair_generator import canonical_pair, generate_pairs, pair_length_statistics


def valid_ions(path):
    frame = pd.read_csv(path).fillna("")
    cations, anions = set(), set()
    for row in frame.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.generated_smiles)
        if mol is None or "." in row.generated_smiles:
            continue
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if row.ion_type == "cation" and charge == 1:
            cations.add(canonical)
        elif row.ion_type == "anion" and charge == -1:
            anions.add(canonical)
    return sorted(cations), sorted(anions)


def independent_pairs(path, count, seed):
    cations, anions = valid_ions(path)
    if not cations or not anions:
        raise RuntimeError("Independent-generation baseline lacks valid ions")
    rng, seen, rows = np.random.default_rng(seed), set(), []
    while len(rows) < count:
        cation = cations[int(rng.integers(len(cations)))]
        anion = anions[int(rng.integers(len(anions)))]
        pair = f"{cation}.{anion}"
        if pair in seen:
            if len(seen) >= len(cations) * len(anions):
                raise RuntimeError("Insufficient unique independent ion combinations")
            continue
        seen.add(pair)
        rows.append({"cation_smiles": cation, "anion_smiles": anion,
                     "IL_SMILES": pair, "valid_pair": True, "error": ""})
    return pd.DataFrame(rows)


def generate_trained_variant(root, variant, seed, model_name, tokenizer, maximum,
                             config, device, priors, count):
    cache = root / "final_evaluation" / str(seed) / f"{variant}_generated_100k.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    run_dir = root / "pair_runs" / variant / str(seed)
    metadata = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    conditioned = variant == "hard_conditioned"
    model = create_model(
        model_name, len(tokenizer.vocab), maximum, tokenizer, config["model"],
        condition_dim=3 if conditioned else 0,
    ).to(device)
    checkpoint = torch.load(
        run_dir / metadata.get("selected_checkpoint", "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    generated = generate_pairs(
        model, tokenizer, metadata["language"], maximum, count, device, priors,
        conditioned, generation_seed=seed + (20_000 if conditioned else 15_000),
    )
    generated.to_parquet(cache, index=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return generated


def sa_score(pair):
    scores = []
    for component in pair.split("."):
        mol = Chem.MolFromSmiles(component)
        if mol is None:
            return np.nan
        scores.append(sascorer.calculateScore(mol))
    return float(np.mean(scores))


def cached_pair_sa(frame):
    components = pd.unique(pd.concat(
        [frame.cation_smiles, frame.anion_smiles], ignore_index=True
    )).tolist()
    component_scores = {}
    for index, component in enumerate(components, start=1):
        mol = Chem.MolFromSmiles(component)
        component_scores[component] = (
            float(sascorer.calculateScore(mol)) if mol is not None else np.nan
        )
        if index % 5000 == 0:
            print(f"sa_score unique_components={index}/{len(components)}", flush=True)
    return (
        frame.cation_smiles.map(component_scores)
        + frame.anion_smiles.map(component_scores)
    ) / 2.0


def oracle_columns(frame, oracle, physical_folds, gas_folds, prefix, cache_dir=None,
                   chunk_size=25_000):
    chunks = []
    values = frame.IL_SMILES.tolist()
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        cache = cache_dir / f"{prefix}_{start}_{stop}.parquet" if cache_dir else None
        if cache is not None and cache.exists():
            print(f"{prefix}_oracle rows={start}:{stop} resumed", flush=True)
            chunk = pd.read_parquet(cache)
        else:
            print(f"{prefix}_oracle rows={start}:{stop}", flush=True)
            chunk = oracle.score(
                values[start:stop], physical_folds=physical_folds, gas_folds=gas_folds
            )
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                chunk.to_parquet(cache, index=False)
        chunks.append(chunk)
    scores = pd.concat(chunks, ignore_index=True).drop(columns=["IL_SMILES"])
    scores = scores.rename(columns={column: f"{prefix}_{column}" for column in scores.columns})
    return pd.concat([frame.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)


def add_uq(frame, prefix):
    z = 1.645
    frame[f"{prefix}_Tm_margin_uq_K"] = 298.15 - (
        frame[f"{prefix}_Tm_K"] + z * frame[f"{prefix}_Tm_K_sd"]
    )
    frame[f"{prefix}_ln_eta_margin_uq"] = np.log(100.0) - (
        frame[f"{prefix}_ln_eta_298"] + z * frame[f"{prefix}_ln_eta_298_sd"]
    )
    frame[f"{prefix}_toxicity_margin_uq"] = (
        frame[f"{prefix}_toxicity"] - z * frame[f"{prefix}_toxicity_sd"] - 3.7
    )
    frame[f"{prefix}_hard_pass_uq"] = (
        (frame[f"{prefix}_Tm_margin_uq_K"] > 0)
        & (frame[f"{prefix}_ln_eta_margin_uq"] > 0)
        & (frame[f"{prefix}_toxicity_margin_uq"] > 0)
    )
    return frame


def internal_diversity(values, seed, maximum=2000):
    values = list(dict.fromkeys(values))
    if len(values) > maximum:
        values = pd.Series(values).sample(maximum, random_state=seed).tolist()
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in values]
    similarities = []
    for index in range(1, len(fps)):
        similarities.extend(DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index]))
    return 1.0 - float(np.mean(similarities)) if similarities else np.nan


def evaluate_method(generated, oracle, known_ions, known_pairs, total_count, seed,
                    cache_dir):
    base_cache = cache_dir / "base.parquet"
    if base_cache.exists():
        print(f"evaluate seed={seed} stage=chemical_filter resumed", flush=True)
        valid = pd.read_parquet(base_cache)
    else:
        print(f"evaluate seed={seed} stage=chemical_filter rows={len(generated)}", flush=True)
        valid = generated[generated.valid_pair].drop_duplicates("IL_SMILES").copy()
        valid = valid[valid.IL_SMILES.map(lambda value: canonical_pair(*value.split(".", 1)) is not None)]
        print(f"evaluate seed={seed} stage=sa_score rows={len(valid)}", flush=True)
        valid["SA_score"] = cached_pair_sa(valid)
        valid["pair_novel"] = ~valid.IL_SMILES.isin(known_pairs)
        valid["both_ions_novel"] = valid.apply(
            lambda row: row.cation_smiles not in known_ions and row.anion_smiles not in known_ions,
            axis=1,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        valid.to_parquet(base_cache, index=False)
    print(f"evaluate seed={seed} stage=guidance_oracle rows={len(valid)}", flush=True)
    valid = oracle_columns(
        valid, oracle, [1, 2, 3, 4, 5], [1, 2, 3], "guidance", cache_dir
    )
    print(f"evaluate seed={seed} stage=evaluation_oracle rows={len(valid)}", flush=True)
    valid = oracle_columns(
        valid, oracle, [6, 7, 8, 9, 10], [4, 5], "eval", cache_dir
    )
    valid = add_uq(add_uq(valid, "guidance"), "eval")
    valid["eval_physical_gas"] = (
        valid[["eval_x_SO2_298", "eval_x_SO2_313", "eval_x_CO2_298"]].gt(0).all(axis=1)
        & valid[["eval_x_SO2_298", "eval_x_SO2_313", "eval_x_CO2_298"]].le(1).all(axis=1)
    )
    rank = valid.sort_values(
        ["guidance_hard_pass_uq", "guidance_hard_pass", "guidance_AC",
         "guidance_DE", "guidance_S", "SA_score"],
        ascending=[False, False, False, False, False, True],
    )
    success = {}
    for cutoff in (10, 50, 100, 500, 1000):
        subset = rank.head(cutoff)
        success[f"success_at_{cutoff}"] = float(
            (subset.eval_hard_pass & subset.eval_physical_gas).any()
        )
    metrics = {
        "generated": int(total_count),
        "valid_pair_rate": float(generated.valid_pair.sum() / max(total_count, 1)),
        "unique_valid_pair_rate": float(len(valid) / max(total_count, 1)),
        "pair_novelty": float(valid.pair_novel.mean()) if len(valid) else np.nan,
        "both_ions_novelty": float(valid.both_ions_novel.mean()) if len(valid) else np.nan,
        "internal_diversity": internal_diversity(valid.IL_SMILES.tolist(), seed),
        "Tm_pass_rate": float((valid.eval_Tm_K < 298.15).sum() / max(total_count, 1)),
        "viscosity_pass_rate": float((valid.eval_eta_298_mPas < 100).sum() / max(total_count, 1)),
        "toxicity_pass_rate": float((valid.eval_toxicity > 3.7).sum() / max(total_count, 1)),
        "HCSR": float(valid.eval_hard_pass.sum() / max(total_count, 1)),
        "HCSR_UQ": float(valid.eval_hard_pass_uq.sum() / max(total_count, 1)),
        "physical_gas_rate": float(valid.eval_physical_gas.sum() / max(total_count, 1)),
        **success,
    }
    return valid, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final-samples", type=int, default=100_000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    best = json.loads((root / "BEST_MODEL.json").read_text(encoding="utf-8"))
    language, model_name, regime = best["language"], best["model"], best["regime"]
    tokenizer = MolecularTokenizer.load(root / "pair_pool" / f"vocab_{language}_pair.json")
    pool = pd.read_parquet(root / "pair_pool" / "pair_pool_scored.parquet")
    maximum, keep, priors_all = pair_length_statistics(pool, tokenizer, language)
    priors = {key: values[keep] for key, values in priors_all.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name, len(tokenizer.vocab), maximum, tokenizer, config["model"], condition_dim=3
    ).to(device)
    iteration_dir = root / "iteration_runs" / str(args.seed)
    model.load_state_dict(torch.load(iteration_dir / "final_model.pt", map_location=device, weights_only=True))
    output = root / "final_evaluation" / str(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "COMPLETED.json"
    if completed.exists():
        print(completed.read_text(encoding="utf-8"))
        return

    final_generated = generate_pairs(
        model, tokenizer, language, maximum, args.final_samples, device, priors, True,
        generation_seed=args.seed + 10_000,
    )
    final_generated.to_parquet(output / "iterated_generated_100k.parquet", index=False)
    independent = independent_pairs(
        root / "runs" / language / model_name / regime / str(args.seed) / "generated.csv",
        args.final_samples, args.seed,
    )
    independent.to_parquet(output / "independent_pairing_100k.parquet", index=False)
    methods = {
        "unconditional": generate_trained_variant(
            root, "unconditional", args.seed, model_name, tokenizer, maximum,
            config, device, priors, args.final_samples,
        ),
        "hard_conditioned": generate_trained_variant(
            root, "hard_conditioned", args.seed, model_name, tokenizer, maximum,
            config, device, priors, args.final_samples,
        ),
        "hard_conditioned_iterated": final_generated,
        "independent_pairing": independent,
    }
    known = pd.read_csv(config["data"]["known_pairs"])
    known_pairs = set(known.IL_SMILES)
    known_ions = set(known.cation_smiles) | set(known.anion_smiles)
    oracle = TextOracle(device=device)
    metrics_path = output / "method_metrics.csv"
    if metrics_path.exists():
        metric_rows = pd.read_csv(metrics_path).to_dict("records")
    else:
        metric_rows = []
    completed_methods = {row["method"] for row in metric_rows}
    for method, generated in methods.items():
        if method in completed_methods:
            print(json.dumps({"method": method, "seed": args.seed, "resumed": True}), flush=True)
            continue
        scored, metrics = evaluate_method(
            generated, oracle, known_ions, known_pairs, len(generated), args.seed,
            output / "checkpoints" / method,
        )
        scored.to_parquet(output / f"{method}_scored.parquet", index=False)
        metric_rows.append({"method": method, "seed": args.seed, **metrics})
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        print(json.dumps(metric_rows[-1]), flush=True)
    payload = {
        "seed": args.seed, "methods": list(methods),
        "evaluation_physical_folds": [6, 7, 8, 9, 10],
        "evaluation_gas_folds": [4, 5],
        "guidance_and_evaluation_separated": True,
    }
    completed.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
