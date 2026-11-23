"""Regenerate draws, metrics and the joint front for the full-fold study."""
from functools import lru_cache
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
import torch
import yaml

from final_pair_evaluation import cached_pair_sa, internal_diversity
from models import create_model
from molecular_tokenizer import MolecularTokenizer
from run_allfold_ratio_de import FullFoldOracle, OUT, SEEDS, check_predictions, run_script, save_parquet
from section34_global_local_50k_20261023 import (
    SEED_ANION, SEED_CATION, generate_with_latent,
)
from train_pair_generator import (
    configure_selfies, generate_pairs, pair_bucket, pair_length_statistics, representation, set_seed,
    target_condition,
)

OBJECTIVES = ("AC", "DE", "S", "SA_score")
PASS_COLUMNS = ("hard_pass", "hard_pass_uq", "Tm_pass", "viscosity_pass", "toxicity_pass")
TABLES = OUT / "tables"


@lru_cache(maxsize=100_000)
def charge(smiles):
    if not isinstance(smiles, str) or not smiles or "." in smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.GetFormalCharge(molecule) if molecule else None


def quality(draws, training, seed):
    cations = draws.cation_smiles.map(charge)
    anions = draws.anion_smiles.map(charge)
    structural = cations.notna() & anions.notna()
    valid = structural & cations.eq(1) & anions.eq(-1)
    assert valid.eq(draws.valid_pair).all()
    unique = draws.loc[valid, "IL_SMILES"].drop_duplicates()
    return dict(
        attempts=len(draws),
        structural_validity=structural.mean(),
        charge_validity=valid.sum() / max(1, structural.sum()),
        uniqueness=len(unique) / max(1, valid.sum()),
        training_novelty=(~unique.isin(training)).mean(),
        structural_diversity=internal_diversity(unique.tolist(), seed, maximum=2000),
    )


def pass_rates(draws, scored):
    unique = draws.loc[draws.valid_pair, "IL_SMILES"].drop_duplicates()
    selected = scored.set_index("IL_SMILES").loc[unique]
    return {metric: selected[metric].sum() / len(draws) for metric in PASS_COLUMNS}


def save_summary(rows, stem, keys, metrics):
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / f"{stem}_seed_data.csv", index=False)
    summary = (
        frame.melt(id_vars=[*keys, "seed"], value_vars=metrics, var_name="metric")
        .groupby([*keys, "metric"]).value
        .agg(mean="mean", SD="std", sample_variance="var", n="count")
        .reset_index()
    )
    summary.to_csv(TABLES / f"{stem}_summary.csv", index=False)


def predict_draws(draws, file, oracle, gas=True):
    unique = draws.loc[draws.valid_pair].drop_duplicates("IL_SMILES").reset_index(drop=True)
    if file.exists():
        scores = pd.read_parquet(file)
        assert scores.IL_SMILES.tolist() == unique.IL_SMILES.tolist()
        return scores
    smiles = unique.IL_SMILES.tolist()
    scores = oracle.score(smiles) if gas else oracle.score_physical(smiles)
    if gas:
        check_predictions(scores)
        scores["SA_score"] = cached_pair_sa(unique)
    scores = pd.concat([unique, scores.drop(columns="IL_SMILES")], axis=1)
    save_parquet(scores, file)
    return scores


def generate_final(seed, arm, sigma, model, tokenizer, maximum, priors, latent, oracle, condition):
    folder = OUT / "global_local"
    name = f"{arm}_{seed}" if sigma in (None, 0.25) else f"{arm}_sigma{sigma:.2f}_{seed}"
    filename = folder / f"{name}_generated.parquet"
    if filename.exists():
        draws = pd.read_parquet(filename)
    elif arm == "global":
        set_seed(seed + 20_000)
        draws = generate_pairs(model, tokenizer, "selfies", maximum, 1000, oracle.device, priors,
                               True, generation_seed=seed + 20_000, condition_vector=condition)
    else:
        draw_seed = seed + (30_000 if sigma == 0.25 else int(sigma * 1e6))
        set_seed(draw_seed)
        rng = np.random.default_rng(draw_seed)
        z = latent[None, :] + sigma * rng.normal(size=(1000, latent.size))
        draws = generate_with_latent(model, tokenizer, "selfies", maximum, z, oracle.device,
                                     generation_seed=draw_seed, condition_vector=condition)
    if not filename.exists():
        save_parquet(draws, filename)
    assert len(draws) == 1000
    scored = predict_draws(draws, folder / f"{name}_scored.parquet", oracle)
    return draws, scored.assign(arm=arm, seed=seed)


def joint_front(scored):
    feasible = scored.loc[scored.hard_pass_uq & scored.physical_gas].drop_duplicates("IL_SMILES").copy()
    finite = np.isfinite(feasible[list(OBJECTIVES)]).all(axis=1)
    feasible = feasible.loc[finite].copy()
    scores = feasible[list(OBJECTIVES)].to_numpy() * [1, 1, 1, -1]
    front = [not ((scores >= v).all(axis=1) & (scores > v).any(axis=1)).any() for v in scores]
    feasible["point_estimate_front"] = np.asarray(front, dtype=bool)
    feasible.to_csv(TABLES / "joint_feasible_candidates.csv", index=False)
    feasible[list(OBJECTIVES)].corr().to_csv(TABLES / "objective_pearson.csv")
    selected = feasible.loc[feasible.point_estimate_front].copy()
    rank_columns = []
    for objective in OBJECTIVES:
        column = f"{objective}_rank"
        selected[column] = selected[objective].rank(ascending=objective == "SA_score")
        rank_columns.append(column)
    selected["mean_rank"] = selected[rank_columns].mean(axis=1)
    selected = selected.sort_values(["mean_rank", "IL_SMILES"])
    selected.insert(0, "front_ID", [f"AF{i:03d}" for i in range(1, len(selected) + 1)])
    selected.to_csv(TABLES / "joint_pareto_front_ranked.csv", index=False)
    selected.head(3).to_csv(TABLES / "top_three.csv", index=False)
    return feasible, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-training", type=Path, help="Read trained models and feedback from a completed study.")
    args = parser.parse_args()
    source = args.reuse_training or OUT
    RDLogger.DisableLog("rdApp.*")
    TABLES.mkdir(exist_ok=True)
    config = yaml.safe_load((OUT / "config.yaml").read_text(encoding="utf-8"))
    sampling_seeds = config["evaluation"].get("sampling_seeds", list(SEEDS))
    if len(sampling_seeds) != len(SEEDS) or len(set(sampling_seeds)) != len(SEEDS):
        raise ValueError("Supply one distinct sampling seed for each trained checkpoint.")
    if sampling_seeds != list(SEEDS) and not args.reuse_training:
        raise ValueError("Use --reuse-training when supplying different sampling seeds.")
    configure_selfies(config)
    condition = target_condition(config)
    anchor = config["evaluation"].get("local_sampling_reference", {})
    anchor_cation = anchor.get("cation_smiles", SEED_CATION)
    anchor_anion = anchor.get("anion_smiles", SEED_ANION)
    pool = pd.read_parquet(source / "pair_pool/pair_pool_scored.parquet")
    tokenizer = MolecularTokenizer.load(source / "pair_pool/vocab_selfies_pair.json")
    maximum, keep, priors_all = pair_length_statistics(pool, tokenizer, "selfies")
    priors = {k: v[keep] for k, v in priors_all.items()}
    train = set(pool.loc[keep].loc[lambda d: d.IL_SMILES.map(pair_bucket) < 18, "IL_SMILES"])
    assert maximum == 144 and len(train) == 44_698
    oracle = FullFoldOracle(device="cuda")
    if anchor:
        smiles = f"{anchor_cation}.{anchor_anion}"
        reference = oracle.score([smiles])
        check_predictions(reference)
        reference["SA_score"] = cached_pair_sa(pd.DataFrame({
            "cation_smiles": [anchor_cation], "anion_smiles": [anchor_anion],
        }))
        reference.to_csv(TABLES / "local_reference_full_fold_predictions.csv", index=False)
    baseline, feedback, trajectory, sampling, sweep = [], [], [], [], []
    attempts, final_scores = [], []
    for seed, sampling_seed in zip(SEEDS, sampling_seeds, strict=True):
        for arm in (() if args.reuse_training else ("unconditional", "hard_conditioned")):
            run = OUT / f"pair_runs/{arm}/{seed}"
            draws = pd.read_csv(run / "generated_pairs.csv")
            assert len(draws) == 1000
            scored = predict_draws(draws, run / "allfold_scored.parquet", oracle)
            baseline.append(dict(seed=seed, arm=arm, **quality(draws, train, seed), **pass_rates(draws, scored)))
        training = train.copy()
        path = source / f"iteration_physical_10r_runs/{seed}"
        for round_index in range(10):
            if not args.reuse_training:
                draws = pd.read_parquet(path / f"round_{round_index}_generated.parquet")
                scored = pd.read_parquet(path / f"round_{round_index}_scored.parquet")
                assert len(draws) == 10_000
                metrics = quality(draws, training, seed + round_index)
                feedback.append(dict(seed=seed, round=round_index, **metrics))
                for budget in (1000, 10_000):
                    trajectory.append(dict(seed=seed, round=round_index, budget=budget,
                                           **pass_rates(draws.head(budget), scored)))
            selected = pd.read_csv(path / f"round_{round_index}_selected.csv")
            assert not (set(selected.IL_SMILES) & training)
            training.update(selected.IL_SMILES)
        model = create_model("vae", len(tokenizer.vocab), maximum, tokenizer, config["model"],
                             condition_dim=3).to(oracle.device)
        model.load_state_dict(torch.load(path / "final_model.pt", map_location=oracle.device, weights_only=True))
        model.eval()
        encoded_anchor = tokenizer.encode_pair(representation(anchor_cation, "selfies"),
                                               representation(anchor_anion, "selfies"), maximum)
        if anchor and tokenizer.unk_id in encoded_anchor:
            raise ValueError("The local sampling reference contains tokens absent from the trained vocabulary.")
        seed_ids = torch.tensor([encoded_anchor], device=oracle.device)
        with torch.inference_mode():
            target = torch.as_tensor(condition, device=oracle.device).unsqueeze(0)
            latent = model.encode(seed_ids, target)[0][0].cpu().numpy()
        (OUT / "global_local").mkdir(exist_ok=True)
        np.save(OUT / f"global_local/z_seed_{sampling_seed}.npy", latent)
        for arm, sigma in (("global", None), ("local", 0.25), ("local", 0.10), ("local", 0.50)):
            print(f"Sampling seed={sampling_seed}, checkpoint={seed}, arm={arm}, sigma={sigma}", flush=True)
            draws, scored = generate_final(sampling_seed, arm, sigma, model, tokenizer, maximum, priors, latent, oracle, condition)
            scored["checkpoint_seed"] = seed
            properties = {k: scored.loc[scored.physical_gas, k].mean() for k in OBJECTIVES}
            row = dict(seed=sampling_seed, checkpoint_seed=seed, arm=arm, **quality(draws, training, sampling_seed),
                       **pass_rates(draws, scored), **properties)
            if sigma is not None:
                sweep.append(dict(sigma=sigma, **row))
            if sigma in (None, 0.25):
                sampling.append(row)
                attempts.append(draws.assign(arm=arm, seed=sampling_seed, checkpoint_seed=seed,
                                             draw=np.arange(1, len(draws)+1)).merge(
                    scored[["IL_SMILES", "arm", "seed", *OBJECTIVES, *PASS_COLUMNS, "physical_gas",
                            "Tm_K", "eta_298_mPas", "toxicity", "x_SO2_298", "x_SO2_313", "x_CO2_298"]],
                    on=["IL_SMILES", "arm", "seed"], how="left", validate="many_to_one"))
                final_scores.append(scored)
        del model
    all_attempts = pd.concat(attempts, ignore_index=True)
    assert len(all_attempts) == 6000
    all_attempts.to_csv(TABLES / "all6000_attempts_and_predictions.csv", index=False)
    scored = pd.concat(final_scores, ignore_index=True)
    save_parquet(scored, OUT / "global_local/all_scored.parquet")
    feasible, front = joint_front(scored)
    summary_metrics = ["structural_validity", "charge_validity", "uniqueness",
                       "training_novelty", "structural_diversity"]
    if not args.reuse_training:
        save_summary(baseline, "section32_pair_generation", ["arm"], [*summary_metrics, *PASS_COLUMNS])
        save_summary(feedback, "section33_feedback_quality", ["round"], summary_metrics)
        save_summary(trajectory, "section33_feedback_constraints", ["round", "budget"], list(PASS_COLUMNS))
    save_summary(sampling, "section34_global_local", ["arm"], [*summary_metrics, *PASS_COLUMNS, *OBJECTIVES])
    save_summary(sweep, "section34_sigma_sweep", ["sigma"], [*summary_metrics, *PASS_COLUMNS, *OBJECTIVES])
    (TABLES / "screening_counts.json").write_text(json.dumps(
        {"attempts": len(all_attempts), "valid_attempts": int(all_attempts.valid_pair.sum()),
         "scored_within_arm": len(scored), "gas_nonphysical_within_arm": int((~scored.physical_gas).sum()),
         "feasible_unique": len(feasible), "front": len(front)}, indent=2), encoding="utf-8")
    run_script("plot_allfold_ratio_de.py", *(["--screening-only"] if args.reuse_training else []))


if __name__ == "__main__":
    main()
