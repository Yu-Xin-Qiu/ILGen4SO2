"""Select a feedback round and screen fresh samples from its checkpoints."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from iterate_absorption_only import top_half_feedback
from report_joint_feedback_20r import OBJECTIVES, QUALITY, reconstruct_training, quality


SEEDS = (137, 251, 367)
FLAGS = ("Tm_pass", "viscosity_pass", "toxicity_pass", "hard_pass", "sa_pass", "joint_hit")
METRICS = (*OBJECTIVES, "SA_max")
REFERENCE = "CCCCn1cc[n+](C)c1.COS(=O)(=O)[O-]"


def summarize_batch(folder, seed, round_index, training, replay, reference, product=False):
    draws = pd.read_parquet(folder / "draws.parquet")
    scores = pd.read_parquet(folder / "scores.parquet").set_index("IL_SMILES")
    assert draws.arm.value_counts().to_dict() == {"global": 5000, "local": 5000}
    rows, hits = [], []
    for arm, batch in (("pooled", draws), *draws.groupby("arm")):
        valid = batch.loc[batch.valid_pair]
        evaluated = scores.loc[valid.IL_SMILES].reset_index()
        distinct = evaluated.drop_duplicates("InChIKey")
        passing = evaluated.loc[evaluated.joint_hit].drop_duplicates("InChIKey").copy()
        fresh = passing.loc[~passing.InChIKey.isin(replay)]
        row = dict(seed=seed, round=round_index, arm=arm,
                   **quality(batch, training, seed + round_index),
                   distinct_cations=valid.cation_smiles.nunique(),
                   dominant_cation_fraction=valid.cation_smiles.value_counts(normalize=True).max(),
                   distinct_joint_ILs=len(passing), distinct_joint_cations=passing.cation_smiles.nunique(),
                   new_joint_ILs=len(fresh), new_joint_cations=fresh.cation_smiles.nunique(),
                   prior_replay_size=len(replay))
        if arm == "pooled":
            marker = json.loads((folder / "metrics.json").read_text())
            assert len(passing) == marker["joint_passing_identities"]
            assert np.isclose(evaluated.joint_hit.sum() / len(batch), marker["joint_pass_rate"])
        for flag in FLAGS:
            row[f"{flag}_rate"] = evaluated[flag].sum() / len(batch)
        for objective in METRICS:
            for population, frame in (("draws", evaluated), ("unique", distinct)):
                if objective != "SA_max":
                    frame = frame.loc[frame.physical_gas]
                values = frame[objective]
                better = values.lt(reference[objective]) if objective == "SA_max" else values.gt(reference[objective])
                row[f"{population}_{objective}_better"] = better.mean()
                for label, q in (("p10", .10), ("median", .50), ("p90", .90)):
                    row[f"{population}_{objective}_{label}"] = values.quantile(q)
        if product:
            eligible = evaluated.assign(R=evaluated[list(OBJECTIVES)].prod(axis=1) / evaluated.SA_max)
            r_ref = reference[list(OBJECTIVES)].astype(float).prod() / reference.SA_max
            r_hits = eligible.loc[eligible.hard_pass & eligible.library_novel & eligible.physical_gas
                                  & np.isfinite(eligible.R) & eligible.R.gt(r_ref)]
            r_unique = r_hits.drop_duplicates("InChIKey")
            row.update(R_pass_rate=len(r_hits) / len(batch), R_distinct_ILs=len(r_unique),
                       R_distinct_cations=r_unique.cation_smiles.nunique(),
                       R_new_ILs=(~r_unique.InChIKey.isin(replay)).sum())
            for population, values in (("draws", eligible), ("unique", eligible.drop_duplicates("InChIKey"))):
                r_values = values.loc[values.physical_gas & np.isfinite(values.R), "R"]
                for label, q in (("p10", .10), ("median", .50), ("p90", .90)):
                    row[f"{population}_R_{label}"] = r_values.quantile(q)
        rows.append(row)
        if arm != "pooled":
            hits.append(passing.assign(seed=seed, arm=arm))
    return rows, hits


def write_statistics(data, group, path):
    stats = (data.melt(id_vars=["seed", *group]).groupby([*group, "variable"]).value
             .agg(mean="mean", sample_variance="var", SD="std", n="count").reset_index())
    stats.to_csv(path, index=False)
    return stats


def plot_feedback(stats, out):
    figures = {
        "quality": [*QUALITY, "dominant_cation_fraction"],
        "constraints": [f"{flag}_rate" for flag in FLAGS],
        "candidate_yield": ["new_joint_ILs", "new_joint_cations", "distinct_joint_ILs", "distinct_joint_cations"],
        "absorption": [f"unique_{objective}_median" for objective in METRICS],
    }
    if "R_pass_rate" in stats.variable.values:
        figures["R"] = ["R_pass_rate", "R_distinct_ILs", "R_distinct_cations", "unique_R_median"]
    for name, columns in figures.items():
        fig, axes = plt.subplots(2, 3 if len(columns) == 6 else 2, figsize=(10, 6.2), layout="constrained")
        for ax, column in zip(axes.flat, columns):
            for arm, color in (("global", "#346BA7"), ("local", "#C9714B"), ("pooled", "#2A887C")):
                values = stats.loc[stats.variable.eq(column) & stats.arm.eq(arm)]
                ax.errorbar(values["round"], values["mean"], yerr=values.SD, color=color,
                            label=arm.capitalize(), marker="o", ms=2, lw=1, capsize=2)
            final_round = int(stats["round"].max())
            ax.set(xlabel="Feedback round", ylabel=column.replace("_", " "), xticks=range(0, final_round + 1, 5))
        axes.flat[0].legend(frameon=False, fontsize=8)
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"feedback_{name}.{suffix}", dpi=450)
        plt.close(fig)


def screen(hits, out, rank_by_product=False):
    pooled = pd.concat(hits, ignore_index=True)
    if pooled.empty:
        pooled.to_csv(out / "final_joint_candidates.csv", index=False)
        return 0, 0
    provenance = pooled.assign(origin=pooled.seed.astype(str) + ":" + pooled.arm).groupby("InChIKey").origin.agg(
        lambda origins: "; ".join(sorted(set(origins))))
    candidates = pooled.drop_duplicates("InChIKey").copy().reset_index(drop=True)
    candidates.insert(0, "ID", [f"IL{i:04d}" for i in range(1, len(candidates) + 1)])
    candidates["origins"] = candidates.InChIKey.map(provenance)
    objective = candidates[list(METRICS)].to_numpy() * [1, 1, 1, -1]
    on_front = []
    for block in np.array_split(objective, max(1, (len(objective) + 255) // 256)):
        dominates = (objective[:, None] >= block).all(axis=2) & (objective[:, None] > block).any(axis=2)
        on_front.extend(~dominates.any(axis=0))
    candidates["pareto"] = on_front
    front = candidates.loc[candidates.pareto].copy()
    if rank_by_product:
        candidates["R"] = candidates[list(OBJECTIVES)].prod(axis=1) / candidates.SA_max
        front["R"] = candidates.loc[front.index, "R"]
        front = front.sort_values(["R", "AC", "ID"], ascending=[False, False, True])
    else:
        front["mean_rank"] = front[list(OBJECTIVES)].rank(ascending=False, method="average").mean(axis=1)
        front = front.sort_values(["mean_rank", "AC", "ID"], ascending=[True, False, True])
    front.insert(1, "rank", range(1, len(front) + 1))
    top = front.head(3)
    candidates.to_csv(out / "final_joint_candidates.csv", index=False)
    front.to_csv(out / "final_pareto_front.csv", index=False)
    top.to_csv(out / "final_top3.csv", index=False)
    for label, frame, columns in (("top3", top, 3), ("pareto", front, 4)):
        opts = Draw.MolDrawOptions()
        opts.legendFontSize = 23
        opts.legendFraction = .29
        for page, start in enumerate(range(0, len(frame), 32), 1):
            part = frame.iloc[start:start + 32]
            mols = [Chem.MolFromSmiles(s) for s in part.IL_SMILES]
            assert all(mol is not None for mol in mols)
            legends = [f"#{r.rank} | {r.ID}" + (f" | R {r.R:.4f}" if rank_by_product else "")
                       + f"\nAC {r.AC:.4f}  DE {r.DE:.5f}\nS {r.S:.2f}  SA-max {r.SA_max:.3f}"
                       for r in part.itertuples()]
            file = out / f"{label}_structures_{page:02d}"
            for svg in (True, False):
                picture = Draw.MolsToGridImage(mols, molsPerRow=columns, subImgSize=(460, 295),
                                               legends=legends, drawOptions=opts, useSVG=svg)
                if svg:
                    file.with_suffix(".svg").write_text(picture, encoding="utf-8")
                else:
                    picture.save(file.with_suffix(".png"))
    return len(candidates), len(front)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--rank-by-product", action="store_true")
    args = parser.parse_args()
    source, out = args.source, args.output
    initial = reconstruct_training(source)
    rows, states = [], {}
    for seed in SEEDS:
        run = out / f"seed_{seed}"
        training, replay = initial.copy(), set()
        reference = pd.read_csv(run / "reference.csv").iloc[0]
        assert reference.IL_SMILES == REFERENCE
        for round_index in range(args.rounds + 1):
            folder = run / f"round_{round_index:02d}"
            states[seed, round_index] = replay.copy()
            batch, _ = summarize_batch(folder, seed, round_index, training, replay, reference, args.rank_by_product)
            rows.extend(batch)
            if round_index < args.rounds:
                selected = pd.read_csv(folder / "selected.csv")
                marker = json.loads((folder / "metrics.json").read_text())
                if "feedback_pool_size" in marker:
                    scored = pd.read_parquet(folder / "scores.parquet")
                    feedback, size, cutoffs = top_half_feedback(scored)
                    assert size == marker["feedback_pool_size"]
                    assert set(selected.identity) == set(feedback.identity) - replay
                    for row in batch:
                        row["feedback_new_ILs_pooled"] = len(selected)
                        row["feedback_new_cations_pooled"] = selected.cation_smiles.nunique()
                        row["feedback_pool_size_pooled"] = size
                        row["feedback_passing_ILs_pooled"] = len(feedback)
                        row["feedback_passing_cations_pooled"] = feedback.cation_smiles.nunique()
                        row.update({f"feedback_{metric}_cutoff_pooled": value for metric, value in cutoffs.items()})
                training.update(selected.IL_SMILES)
                replay.update(selected.identity)
        print(f"Summarized feedback seed {seed}", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(out / "feedback_metrics_per_seed_round_arm.csv", index=False)
    stats = write_statistics(data, ["round", "arm"], out / "feedback_mean_variance_sd.csv")
    plot_feedback(stats, out)
    criteria = (["distinct_joint_cations", "distinct_joint_ILs", "R_pass_rate"] if args.rank_by_product
                else ["new_joint_ILs", "new_joint_cations", "joint_hit_rate"])
    selection = data.loc[data.arm.eq("pooled")].groupby("round")[criteria].mean().reset_index()
    selection = selection.sort_values([*criteria, "round"], ascending=[False, False, False, True])
    selection.to_csv(out / "checkpoint_selection.csv", index=False)
    round_index = int(selection.iloc[0]["round"])
    print(f"Selected common round {round_index} across three seeds", flush=True)
    final_rows, hits = [], []
    iterator = Path(__file__).with_name("iterate_absorption_only.py")
    for seed in SEEDS:
        run = out / f"seed_{seed}"
        final = out / "final" / f"seed_{seed}"
        protocol = json.loads((run / "protocol.json").read_text())
        checkpoint = (run / f"round_{round_index - 1:02d}" / "model.pt" if round_index
                      else Path(protocol["checkpoint"]))
        subprocess.run([sys.executable, "-u", str(iterator), str(source), str(final),
                        "--sample-seed", str(seed + 1_000_000), "--count", "10000", "--local-count", "5000",
                        "--local-sigma", "0.25", "--reference-smiles", REFERENCE, "--rounds", "0",
                        "--joint-point-replay", "--max-sa-better-than-reference",
                        "--evaluation-checkpoint", str(checkpoint)], check=True)
        training = initial.copy()
        for prior in range(round_index):
            training.update(pd.read_csv(run / f"round_{prior:02d}" / "selected.csv").IL_SMILES)
        reference = pd.read_csv(run / "reference.csv").iloc[0]
        batch, qualified = summarize_batch(final / "round_00", seed, round_index, training,
                                           states[seed, round_index], reference, args.rank_by_product)
        final_rows.extend(batch)
        hits.extend(qualified)
    final_data = pd.DataFrame(final_rows)
    final_data.to_csv(out / "final" / "metrics_per_seed_arm.csv", index=False)
    write_statistics(final_data, ["round", "arm"], out / "final" / "mean_variance_sd.csv")
    n_hits, n_front = screen(hits, out / "final", args.rank_by_product)
    print(f"Final screening: {n_hits} distinct passing ILs, {n_front} on the Pareto front", flush=True)
    (out / "COMPLETED.json").write_text(json.dumps(dict(round=round_index, candidates=n_hits, pareto=n_front)))


if __name__ == "__main__":
    main()
