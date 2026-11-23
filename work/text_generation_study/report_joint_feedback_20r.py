"""Verify and summarize independent joint-feedback runs from saved draws."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from molecular_tokenizer import MolecularTokenizer
from iterate_absorption_only import inchi, ion_sa
from summarize_allfold_ratio_de import quality
from train_pair_generator import configure_selfies, pair_bucket, pair_length_statistics

OBJECTIVES = ("AC", "DE", "S")
RATES = ("melting_pass_rate", "viscosity_pass_rate", "toxicity_pass_rate",
         "physical_pass_rate", "joint_pass_rate")
QUALITY = ("structural_validity", "charge_validity", "uniqueness",
           "training_novelty", "structural_diversity")


def reconstruct_training(source):
    config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    configure_selfies(config)
    tokenizer = MolecularTokenizer.load(source / "pair_pool/vocab_selfies_pair.json")
    pool = pd.read_parquet(source / "pair_pool/pair_pool_scored.parquet")
    maximum, keep, _ = pair_length_statistics(pool, tokenizer, "selfies")
    training = set(pool.loc[keep].loc[lambda p: p.IL_SMILES.map(pair_bucket) < 18, "IL_SMILES"])
    assert maximum == 144 and len(training) == 44_698
    return training


def collect(run, seed, initial_training, library_keys, count=1000):
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["rounds"] == 20 and protocol["count"] == count
    assert protocol["seed"] == seed and protocol["physical_rule"] == "point prediction"
    reference_row = pd.read_csv(run / "reference.csv").iloc[0]
    assert reference_row.IL_SMILES == protocol["reference"]
    reference = reference_row[list(OBJECTIVES)].astype(float)
    sa_limit = None
    objectives = list(OBJECTIVES)
    if protocol["sa_filter"]:
        sa_limit = max(map(ion_sa, reference_row.IL_SMILES.split(".")))
        assert np.isclose(sa_limit, reference_row.SA_max)
        assert np.isclose(sa_limit, protocol["reference_sa_max"])
        objectives.append("SA_max")
    training = initial_training.copy()
    identities = set()
    updates = 0
    metrics, distributions, hits = [], [], []
    for index in range(21):
        folder = run / f"round_{index:02d}"
        marker = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert marker["round"] == index
        draws = pd.read_parquet(folder / "draws.parquet")
        scores = pd.read_parquet(folder / "scores.parquet")
        assert len(draws) == count
        assert draws.loc[draws.valid_pair, "IL_SMILES"].nunique() == len(scores)
        physical = scores.Tm_K.lt(298.15) & scores.eta_298_mPas.lt(200) & scores.toxicity.gt(2)
        np.testing.assert_array_equal(physical, scores.hard_pass)
        above = scores[list(OBJECTIVES)].gt(reference).all(axis=1) & scores.physical_gas
        novel = scores.InChIKey.notna() & scores.InChIKey.ne("") & ~scores.InChIKey.isin(library_keys)
        np.testing.assert_array_equal(scores.library_novel, novel)
        joint_filter = physical & above & scores.library_novel
        if sa_limit is not None:
            sa_cation = scores.cation_smiles.map(ion_sa)
            sa_anion = scores.anion_smiles.map(ion_sa)
            np.testing.assert_allclose(scores.SA_cation, sa_cation, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores.SA_anion, sa_anion, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores.SA_max, np.maximum(sa_cation, sa_anion), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(scores.sa_pass, scores.SA_max.lt(sa_limit))
            joint_filter &= scores.sa_pass
        np.testing.assert_array_equal(scores.joint_hit, joint_filter)
        joint = scores.loc[scores.joint_hit].drop_duplicates("InChIKey").copy()
        joint["seen_in_replay"] = joint.InChIKey.isin(identities)
        if index == 20:
            hits.append(joint.assign(seed=seed))
        row = dict(seed=seed, round=index, **quality(draws, training, seed + index))
        row["completed_model_updates"] = updates
        for metric, flag in zip(RATES, ("Tm_pass", "viscosity_pass", "toxicity_pass", "hard_pass", "joint_hit")):
            row[metric] = draws.IL_SMILES.isin(scores.loc[scores[flag], "IL_SMILES"]).mean()
            assert np.isclose(row[metric], marker[metric])
        row["distinct_joint_hits"] = len(joint)
        if sa_limit is not None:
            row["sa_pass_rate"] = draws.IL_SMILES.isin(scores.loc[scores.sa_pass, "IL_SMILES"]).mean()
            assert np.isclose(row["sa_pass_rate"], marker["sa_pass_rate"])
        row["distinct_joint_hits_outside_replay"] = (~joint.seen_in_replay).sum()
        assert len(joint) == marker["joint_passing_identities"]
        cols = ["IL_SMILES", *objectives, "physical_gas", "hard_pass", "joint_hit", "InChIKey"]
        evaluated = draws.loc[draws.valid_pair, ["IL_SMILES"]].merge(
            scores[cols], on="IL_SMILES", validate="many_to_one")
        evaluated = evaluated.loc[evaluated.physical_gas].assign(seed=seed, round=index)
        for population, values in (
            ("valid_draws", evaluated),
            ("distinct_valid_SMILES", evaluated.drop_duplicates("IL_SMILES")),
        ):
            for objective in objectives:
                for label, q in (("p10", .1), ("median", .5), ("p90", .9)):
                    row[f"{population}_{objective}_{label}"] = values[objective].quantile(q)
        distributions.append(evaluated)
        if index < 20:
            selected = pd.read_csv(folder / "selected.csv")
            assert selected.joint_hit.all() and selected.hard_pass.all() and selected.library_novel.all()
            assert selected[list(OBJECTIVES)].gt(reference).all().all()
            if sa_limit is not None:
                assert selected.SA_max.lt(sa_limit).all()
            assert set(selected.identity) == set(joint.InChIKey) - identities
            training.update(selected.IL_SMILES)
            identities.update(selected.identity)
            assert len(identities) == marker["replay_size"]
            updates += int(bool(identities))
        row["replay_size"] = len(identities)
        metrics.append(row)
    return pd.DataFrame(metrics), pd.concat(distributions), pd.concat(hits), reference


def plot_curves(summary, columns, labels, out, name, scale=1):
    fig, axes = plt.subplots(1, len(columns), figsize=(3.15 * len(columns), 3.45),
                             layout="constrained", squeeze=False)
    for ax, metric, label in zip(axes[0], columns, labels):
        values = summary.loc[summary.metric.eq(metric)]
        ax.errorbar(values["round"], scale * values["mean"], yerr=scale * values.SD,
                    marker="o", ms=2.5, lw=1.1, capsize=2, color="#285984")
        ax.set(xlabel="Feedback round", ylabel=label, xticks=range(0, 21, 5))
    fig.savefig(out / f"{name}.png", dpi=450)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def plot_distribution(data, reference, out, reference_label):
    names = ("SO₂ absorption capacity (AC)", "SO₂ desorption ratio (DE)",
             "SO₂/CO₂ solubility ratio (S)", "Maximum ion SA")
    objectives = [metric for metric in (*OBJECTIVES, "SA_max") if metric in data]
    fig, axes = plt.subplots(1, len(objectives), figsize=(3.4 * len(objectives), 3.55), layout="constrained")
    colors = plt.cm.Blues(np.linspace(.32, .96, 5))
    bins_table = []
    for ax, key, label in zip(axes, objectives, names):
        limits = np.quantile(data[key], [.002, .998])
        edges = np.linspace(min(limits[0], reference[key]), max(limits[1], reference[key]), 61)
        centers = (edges[:-1] + edges[1:]) / 2
        for index, chunk in data.groupby("round"):
            curves = np.array([100 * np.histogram(v[key], bins=edges)[0] / len(v)
                               for _, v in chunk.groupby("seed")])
            mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            bins_table.extend((key, index, x, a, b) for x, a, b in zip(centers, mean, sd))
            if index % 5 == 0:
                ax.plot(centers, mean, color=colors[index // 5], label=str(index))
        ax.axvline(reference[key], ls="--", lw=1, color="#ac4a35", label=reference_label)
        ax.set(xlabel=label, ylabel="Mean relative frequency per bin (%)")
    fig.legend(*axes[-1].get_legend_handles_labels(), loc="outside lower center", ncol=6,
               frameon=False, title="Feedback round")
    fig.savefig(out / "absorption_distributions.png", dpi=450)
    fig.savefig(out / "absorption_distributions.pdf")
    plt.close(fig)
    pd.DataFrame(bins_table, columns=["metric", "round", "bin_center", "mean_pct", "SD_pct"]).to_csv(
        out / "absorption_distribution_bins_all_rounds.csv", index=False)

    fig, axes = plt.subplots(1, len(objectives), figsize=(3.4 * len(objectives), 3.5), layout="constrained")
    for ax, key, label in zip(axes, objectives, names):
        ranges = data.groupby("round")[key].quantile([.1, .5, .9]).unstack()
        ax.fill_between(ranges.index, ranges[.1], ranges[.9], color="#c7dcec", label="P10–P90, pooled")
        ax.plot(ranges.index, ranges[.5], color="#285984", label="Median, pooled")
        ax.axhline(reference[key], ls="--", lw=1, color="#ac4a35", label=reference_label)
        ax.set(xlabel="Feedback round", ylabel=label, xticks=range(0, 21, 5))
    fig.legend(*axes[-1].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "absorption_quantile_trajectory.png", dpi=450)
    fig.savefig(out / "absorption_quantile_trajectory.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--reference-label", default="[BMIM][OAc]")
    args = parser.parse_args()
    out = args.output
    training = reconstruct_training(args.source)
    library = pd.read_csv(args.source / "data/known_pairs.csv")
    library_keys = set(library.IL_SMILES.map(inchi)) - {""}
    runs = [collect(out / f"seed_{seed}", seed, training, library_keys, args.count)
            for seed in (137, 251, 367)]
    for _, _, _, ref in runs[1:]:
        np.testing.assert_allclose(ref, runs[0][3], rtol=1e-6)
    metrics = pd.concat([run[0] for run in runs], ignore_index=True)
    data = pd.concat([run[1] for run in runs], ignore_index=True)
    hits = pd.concat([run[2] for run in runs], ignore_index=True)
    metrics.to_csv(out / "metrics_per_seed_and_round.csv", index=False)
    data.to_csv(out / "absorption_predictions_per_valid_draw.csv", index=False)
    hits.to_csv(out / "round20_joint_hits_by_seed.csv", index=False)
    summary = (metrics.melt(id_vars=["seed", "round"], var_name="metric")
               .groupby(["round", "metric"]).value
               .agg(mean="mean", variance="var", SD="std", n="count").reset_index())
    assert summary.n.eq(3).all()
    summary.to_csv(out / "metrics_by_round_mean_variance_sd.csv", index=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "pdf.fonttype": 42})
    plot_curves(summary, RATES, ("Melting point pass (%)", "Viscosity pass (%)", "IPC-81 pass (%)",
                                "Three-property pass (%)", "Joint pass (%)"), out, "constraint_trajectories", 100)
    plot_curves(summary, QUALITY, ("Valid-pair yield", "Retained-pair charge check", "Uniqueness",
                                  "Training-set novelty", "Structural diversity"), out, "generation_quality")
    plot_curves(summary, ["distinct_joint_hits", "distinct_joint_hits_outside_replay"],
                (f"Distinct joint hits per {args.count:,}", f"Outside replay per {args.count:,}"),
                out, "distinct_joint_hits")
    reference = pd.read_csv(out / "seed_137/reference.csv").iloc[0]
    if "SA_max" in data:
        plot_curves(summary, ["sa_pass_rate"], ["Maximum ion SA pass (%)"], out, "sa_pass_trajectory", 100)
    plot_distribution(data, reference, out, args.reference_label)
    print(summary.loc[summary["round"].isin([0, 5, 10, 15, 20]) & summary.metric.isin(
        [*RATES, "distinct_joint_hits", "uniqueness", "structural_diversity"]
    )].to_string(index=False))
    print(f"Round 20: {hits.InChIKey.nunique()} distinct joint hits across the three seeds.")


if __name__ == "__main__":
    main()
