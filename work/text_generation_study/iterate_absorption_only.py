"""Ten-round pilot using only joint absorption improvements for replay."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, rdBase
from rdkit.Contrib.SA_Score import sascorer
from torch.utils.data import DataLoader, TensorDataset

from allfold_oracle import FullFoldOracle, GAS_FOLDS, store_folds
from models import create_model
from molecular_tokenizer import MolecularTokenizer
from run_allfold_ratio_de import save_parquet
from text_oracles import TextOracle
from train_pair_generator import (
    collate_pairs, configure_selfies, epoch_pass, generate_pairs, representation, set_seed,
    target_condition,
)

REFERENCE = "CCCCn1cc[n+](C)c1.CC(=O)[O-]"
METRICS = ["AC", "DE", "S"]
ROUNDS = 10


@lru_cache(maxsize=100_000)
def ion_sa(smiles):
    return sascorer.calculateScore(Chem.MolFromSmiles(smiles))


def score(oracle, smiles):
    columns = {"IL_SMILES": smiles}
    tokens = oracle.encode(smiles, oracle.gas_tokenizer)
    for label, gas, kelvin in (("SO2_298", "SO2", 298.15), ("SO2_313", "SO2", 313.15),
                               ("CO2_298", "CO2", 298.15)):
        values = oracle.gas(smiles, gas, kelvin, 1.0, GAS_FOLDS, tokens, return_folds=True)
        store_folds(columns, f"ln_x_{label}", values, GAS_FOLDS)
    result = pd.DataFrame(columns)
    ac, des, co2 = (np.exp(result[f"ln_x_{label}"]) for label in ("SO2_298", "SO2_313", "CO2_298"))
    result["AC"], result["DE"], result["S"] = ac, ac / des, ac / co2
    result["physical_gas"] = np.isfinite(result[METRICS]).all(axis=1) & (
        ac.between(0, 1, inclusive="right") & des.between(0, 1, inclusive="right")
        & co2.between(0, 1, inclusive="right"))
    return result


def inchi(smiles):
    with rdBase.BlockLogs():
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def top_half_feedback(scored):
    """Select the intersection of the four favorable halves of the feasible pool."""
    objectives = [*METRICS, "SA_max"]
    eligible = scored.loc[scored.hard_pass & scored.library_novel & scored.physical_gas]
    eligible = eligible.loc[np.isfinite(eligible[objectives]).all(axis=1)].drop_duplicates("InChIKey")
    cutoffs = eligible[objectives].median()
    keep = eligible[METRICS].ge(cutoffs[METRICS]).all(axis=1) & eligible.SA_max.le(cutoffs.SA_max)
    return eligible.loc[keep].assign(identity=lambda rows: rows.InChIKey), len(eligible), cutoffs


def train(model, replay, tokenizer, maximum, condition, device):
    ids = []
    fallback_count = 0
    for row in replay.itertuples():
        try:
            tokens = tokenizer.encode_pair(representation(row.cation_smiles, "selfies"),
                                           representation(row.anion_smiles, "selfies"), maximum)
        except ValueError:
            tokens = []
        if not tokens or tokenizer.unk_id in tokens:
            # Canonicalization may produce SELFIES tokens outside the frozen vocabulary.
            # The original generated representation was decoded into this same pair.
            tokens = tokenizer.encode_pair(row.cation_representation, row.anion_representation, maximum)
            fallback_count += 1
        if tokenizer.unk_id in tokens:
            raise ValueError(f"Generated pair cannot be encoded with the frozen vocabulary: {row.IL_SMILES}")
        ids.append(tokens)
    dataset = TensorDataset(torch.tensor(ids), torch.tensor(np.tile(condition, (len(ids), 1))))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_pairs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    losses = [epoch_pass(model, loader, optimizer, scaler, device, e, True, 1.0)
              for e in (1, 2)]
    return losses, fallback_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-seed", type=int, default=137)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--reference-smiles", default=REFERENCE)
    parser.add_argument("--reuse-initial-samples", type=Path)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--joint-point-replay", action="store_true")
    parser.add_argument("--max-sa-better-than-reference", action="store_true")
    parser.add_argument("--local-count", type=int, default=0)
    parser.add_argument("--local-sigma", type=float, default=0.25)
    parser.add_argument("--evaluation-checkpoint", type=Path)
    parser.add_argument("--top-half-feedback", action="store_true")
    parser.add_argument("--product-feedback", action="store_true")
    args = parser.parse_args()
    if args.max_sa_better_than_reference and not args.joint_point_replay:
        parser.error("--max-sa-better-than-reference requires --joint-point-replay")
    if not 0 <= args.local_count < args.count or args.local_sigma <= 0:
        parser.error("Local count must be smaller than total count and local sigma must be positive")
    if args.local_count and args.reuse_initial_samples:
        parser.error("Mixed sampling cannot reuse global-only samples")
    if args.evaluation_checkpoint and args.rounds != 0:
        parser.error("Evaluation of a saved checkpoint requires --rounds 0")
    if args.top_half_feedback and args.product_feedback:
        parser.error("Choose one feedback selection rule")
    if (args.top_half_feedback or args.product_feedback) and not (args.joint_point_replay and args.max_sa_better_than_reference):
        parser.error("Alternative feedback requires the full-fold and maximum-ion-SA scoring flags")
    torch.set_num_threads(2)
    src, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((src / "config.yaml").read_text(encoding="utf-8"))
    configure_selfies(config)
    condition = target_condition(config)
    tokenizer = MolecularTokenizer.load(src / "pair_pool/vocab_selfies_pair.json")
    checkpoint_dir = src / "pair_runs/hard_conditioned/11"
    metadata = json.loads((checkpoint_dir / "COMPLETED.json").read_text())
    maximum = metadata["maximum_length"]
    device = torch.device("cuda")
    model = create_model("vae", len(tokenizer.vocab), maximum, tokenizer, config["model"],
                         condition_dim=3).to(device)
    protocol = dict(reference=args.reference_smiles, rounds=args.rounds, count=args.count,
                    checkpoint=str(checkpoint_dir / metadata["selected_checkpoint"]),
                    seed=args.sample_seed, gas_folds=list(GAS_FOLDS), temperature_K=298.15,
                    desorption_temperature_K=313.15, pressure_bar=1, epochs_per_update=2,
                    learning_rate=3e-5, physical_property_filter=args.joint_point_replay, sa_filter=False)
    if args.joint_point_replay:
        protocol.update(library_exclusion="InChIKey", physical_rule="point prediction")
    if args.max_sa_better_than_reference:
        reference_sa = [ion_sa(ion) for ion in args.reference_smiles.split(".")]
        protocol.update(sa_filter=True, sa_rule="maximum ion SA strictly below reference",
                        reference_sa_max=max(reference_sa))
    if args.local_count:
        anchor = tokenizer.encode_pair(*(representation(ion, "selfies") for ion in args.reference_smiles.split(".")), maximum)
        if tokenizer.unk_id in anchor:
            raise ValueError("Local reference contains tokens outside the trained vocabulary")
        protocol.update(local_count=args.local_count, local_sigma=args.local_sigma,
                        local_anchor=args.reference_smiles, temperature=1.0, top_k=32)
    if args.evaluation_checkpoint:
        protocol["checkpoint"] = str(args.evaluation_checkpoint)
    if args.top_half_feedback:
        protocol["feedback_selection"] = "joint intersection of four median thresholds on novel feasible pairs"
    if args.product_feedback:
        protocol["feedback_selection"] = "AC * DE * S / maximum ion SA strictly above reference"
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output already belongs to a different run.")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    checkpoint = torch.load(protocol["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    oracle = FullFoldOracle(device=device) if args.joint_point_replay else TextOracle(device=device)
    known = set()
    if args.joint_point_replay:
        known = set(pd.read_csv(src / "data/known_pairs.csv").IL_SMILES.map(inchi)) - {""}
    ref_path = out / "reference.csv"
    reference = pd.read_csv(ref_path) if ref_path.exists() else score(oracle, [args.reference_smiles])
    if args.max_sa_better_than_reference:
        reference["SA_cation"], reference["SA_anion"] = reference_sa
        reference["SA_max"] = protocol["reference_sa_max"]
    reference.to_csv(ref_path, index=False)
    target = reference.iloc[0][METRICS].astype(float)
    reusable = args.reuse_initial_samples
    if reusable is not None:
        prior_protocol = json.loads((reusable / "protocol.json").read_text())
        for field in ("checkpoint", "seed", "count", "gas_folds", "physical_rule"):
            if prior_protocol[field] != protocol[field]:
                raise ValueError(f"Cached samples use a different {field}.")
        if not (prior_protocol["sa_filter"] and protocol["sa_filter"]):
            raise ValueError("Cached samples must include full-fold predictions and SA.")
    replay_frames, metrics = [], []
    for step in range(args.rounds + 1):
        path = out / f"round_{step:02d}"
        path.mkdir(exist_ok=True)
        marker = path / "metrics.json"
        if marker.exists():
            row = json.loads(marker.read_text())
            metrics.append(row)
            if step < args.rounds:
                replay_frames.append(pd.read_csv(path / "selected.csv"))
                model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
            continue
        seed = args.sample_seed + step * 1000
        set_seed(seed)
        print(f"Round {step}: generating {args.count} draws", flush=True)
        draws_file = path / "draws.parquet"
        previous = set().union(*(set(frame.identity) for frame in replay_frames))
        cached = reusable / path.name if reusable is not None and not previous else None
        if cached is not None:
            prior_round = cached / "metrics.json"
            prior_update = cached.parent / f"round_{step-1:02d}" / "metrics.json"
            if not prior_round.exists() or (step > 0 and (
                not prior_update.exists() or json.loads(prior_update.read_text())["replay_size"]
            )):
                cached = None
        if draws_file.exists():
            draws = pd.read_parquet(draws_file)
        elif cached is not None:
            draws = pd.read_parquet(cached / "draws.parquet")
            save_parquet(draws, draws_file)
        else:
            draws = generate_pairs(model, tokenizer, "selfies", maximum, args.count - args.local_count, device, {},
                                   True, generation_seed=seed, condition_vector=condition)
            if args.local_count:
                from section34_global_local_50k_20261023 import generate_with_latent

                local_seed = seed + 500_000
                set_seed(local_seed)
                model.eval()
                with torch.inference_mode():
                    ids = torch.tensor([anchor], device=device)
                    target_condition_tensor = torch.as_tensor(condition, device=device).unsqueeze(0)
                    center = model.encode(ids, target_condition_tensor)[0][0].cpu().numpy()
                noise = np.random.default_rng(local_seed).normal(size=(args.local_count, center.size))
                local = generate_with_latent(model, tokenizer, "selfies", maximum,
                                             center + args.local_sigma * noise, device,
                                             generation_seed=local_seed, condition_vector=condition)
                draws = pd.concat([draws.assign(arm="global"), local.assign(arm="local")], ignore_index=True)
            save_parquet(draws, draws_file)
        unique = draws.loc[draws.valid_pair].drop_duplicates("IL_SMILES").reset_index(drop=True)
        if unique.empty:
            raise RuntimeError(f"No valid pairs at round {step}.")
        score_file = path / "scores.parquet"
        if score_file.exists():
            scored = pd.read_parquet(score_file)
        elif cached is not None:
            scored = pd.read_parquet(cached / "scores.parquet")
            if scored.IL_SMILES.tolist() != unique.IL_SMILES.tolist():
                raise ValueError("Cached predictions do not match the saved draw batch.")
            scored["all_three"] = scored.physical_gas & scored[METRICS].gt(target).all(axis=1)
            scored["sa_pass"] = scored.SA_max.lt(protocol["reference_sa_max"])
            scored["joint_hit"] = scored.all_three & scored.hard_pass & scored.library_novel & scored.sa_pass
            save_parquet(scored, score_file)
            print(f"Round {step}: re-screened {len(scored)} saved scores from the initial checkpoint", flush=True)
        else:
            print(f"Round {step}: scoring {len(unique)} distinct SMILES", flush=True)
            scores = score(oracle, unique.IL_SMILES.tolist())
            scored = unique.merge(scores, on="IL_SMILES", validate="one_to_one")
            scored["InChIKey"] = scored.IL_SMILES.map(inchi)
            scored["all_three"] = scored.physical_gas & scored[METRICS].gt(target).all(axis=1)
            if args.joint_point_replay:
                physical = oracle.score_physical(unique.IL_SMILES.tolist())
                fields = ["IL_SMILES", "Tm_K", "eta_298_mPas", "toxicity", "Tm_pass",
                          "viscosity_pass", "toxicity_pass", "hard_pass"]
                scored = scored.merge(physical[fields], on="IL_SMILES", validate="one_to_one")
                scored["library_novel"] = scored.InChIKey.ne("") & ~scored.InChIKey.isin(known)
                scored["joint_hit"] = scored.all_three & scored.hard_pass & scored.library_novel
                if args.max_sa_better_than_reference:
                    scored["SA_cation"] = scored.cation_smiles.map(ion_sa)
                    scored["SA_anion"] = scored.anion_smiles.map(ion_sa)
                    scored["SA_max"] = scored[["SA_cation", "SA_anion"]].max(axis=1)
                    scored["sa_pass"] = scored.SA_max.lt(protocol["reference_sa_max"])
                    scored["joint_hit"] &= scored.sa_pass
            save_parquet(scored, score_file)
        valid = scored.loc[scored.physical_gas]
        hits = valid.loc[valid.all_three].copy()
        hits["identity"] = hits.InChIKey.mask(hits.InChIKey.eq(""), hits.IL_SMILES)
        hits = hits.drop_duplicates("identity")
        row = dict(round=step, attempts=len(draws), valid_draws=int(draws.valid_pair.sum()),
                   valid_unique=len(unique), physical_gas_unique=len(valid),
                   hits_unique=len(hits), hits_per_1000=1000 * len(hits) / len(draws),
                   hit_draws=int(draws.IL_SMILES.isin(scored.loc[scored.all_three, "IL_SMILES"]).sum()))
        if args.joint_point_replay:
            filters = [("melting", "Tm_pass"), ("viscosity", "viscosity_pass"),
                                  ("toxicity", "toxicity_pass"), ("physical", "hard_pass"),
                                  ("joint", "joint_hit")]
            if args.max_sa_better_than_reference:
                filters.append(("sa", "sa_pass"))
            for label, column in filters:
                passing = scored.loc[scored[column]]
                n = int(draws.IL_SMILES.isin(passing.IL_SMILES).sum())
                row[f"{label}_passing_draws"] = n
                row[f"{label}_pass_rate"] = n / len(draws)
                row[f"{label}_passing_identities"] = passing.InChIKey.mask(
                    passing.InChIKey.eq(""), passing.IL_SMILES).nunique()
            hits = scored.loc[scored.joint_hit].copy()
            hits["identity"] = hits.InChIKey
            hits = hits.drop_duplicates("identity")
        for metric in METRICS:
            row[f"{metric}_above"] = float(valid[metric].gt(target[metric]).mean())
            for name, quantile in (("p10", .1), ("median", .5), ("p90", .9)):
                row[f"{metric}_{name}"] = float(valid[metric].quantile(quantile))
        if args.product_feedback:
            product_reference = float(target.prod() / reference.iloc[0].SA_max)
            product_scores = scored[METRICS].prod(axis=1, skipna=False) / scored.SA_max
            eligible = scored.hard_pass & scored.library_novel & scored.physical_gas
            product_hits = scored.loc[eligible & np.isfinite(product_scores) & product_scores.gt(product_reference)]
            product_hits = product_hits.assign(R=product_scores, identity=product_hits.InChIKey).drop_duplicates("identity")
            row.update(R_reference=product_reference, R_passing_ILs=len(product_hits),
                       R_passing_cations=product_hits.cation_smiles.nunique(),
                       R_passing_draws=int(draws.IL_SMILES.isin(scored.loc[
                           eligible & np.isfinite(product_scores) & product_scores.gt(product_reference),
                           "IL_SMILES"]).sum()))
            product_hits.to_csv(path / "product_hits.csv", index=False)
        if step < args.rounds:
            feedback = hits
            if args.product_feedback:
                feedback = product_hits
            if args.top_half_feedback:
                feedback, pool_size, cutoffs = top_half_feedback(scored)
                row.update(feedback_pool_size=pool_size, feedback_passing_identities=len(feedback),
                           feedback_passing_cations=feedback.cation_smiles.nunique(),
                           **{f"feedback_{metric}_cutoff": float(value) if pd.notna(value) else None
                              for metric, value in cutoffs.items()})
            selected = feedback.loc[~feedback.identity.isin(previous)]
            selected.to_csv(path / "selected.csv", index=False)
            replay_frames.append(selected)
            replay = pd.concat(replay_frames, ignore_index=True)
            row.update(new_elites=len(selected), replay_size=len(replay))
            if len(replay):
                set_seed(seed + 1)
                row["training_losses"], fallback_count = train(
                    model, replay, tokenizer, maximum, condition, device)
                if fallback_count:
                    row["replay_original_representation_count"] = fallback_count
            torch.save(model.state_dict(), path / "model.pt")
        marker.write_text(json.dumps(row, indent=2), encoding="utf-8")
        metrics.append(row)
        pd.DataFrame(metrics).to_csv(out / "iteration_metrics.csv", index=False)
        print(json.dumps({k: v for k, v in row.items() if k != "training_losses"}), flush=True)
    pd.DataFrame(metrics).to_csv(out / "iteration_metrics.csv", index=False)


if __name__ == "__main__":
    main()
