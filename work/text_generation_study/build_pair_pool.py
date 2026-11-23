import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
import yaml

from text_oracles import TextOracle


RDLogger.DisableLog("rdApp.*")
SEEDS = [11, 23, 37, 41, 59]


def canonical_ion(value, expected_charge):
    if not isinstance(value, str) or not value.strip() or "." in value:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if charge != expected_charge:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_pair(cation, anion):
    cation = canonical_ion(cation, 1)
    anion = canonical_ion(anion, -1)
    if cation is None or anion is None:
        return None
    return cation, anion, f"{cation}.{anion}"


def sample_unique_pairs(cations, anions, count, excluded, rng, source):
    records = []
    seen = set(excluded)
    attempts = 0
    maximum_attempts = count * 100
    while len(records) < count and attempts < maximum_attempts:
        attempts += 1
        cation = cations[int(rng.integers(len(cations)))]
        anion = anions[int(rng.integers(len(anions)))]
        key = f"{cation}.{anion}"
        if key in seen:
            continue
        seen.add(key)
        records.append({"cation_smiles": cation, "anion_smiles": anion,
                        "IL_SMILES": key, "source": source})
    if len(records) != count:
        raise RuntimeError(f"Could create only {len(records):,} of {count:,} unique {source} pairs")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--skip-scoring", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    output = root / "pair_pool"
    output.mkdir(parents=True, exist_ok=True)
    best = json.loads((root / "BEST_MODEL.json").read_text(encoding="utf-8"))
    language, model, regime = best["language"], best["model"], best["regime"]
    rng = np.random.default_rng(args.seed)

    known_raw = pd.read_csv(config["data"]["known_pairs"])
    known = []
    for row in known_raw.itertuples(index=False):
        result = canonical_pair(row.cation_smiles, row.anion_smiles)
        if result is None:
            raise ValueError(f"Invalid known pair: {row.IL_SMILES}")
        cation, anion, pair = result
        known.append({"cation_smiles": cation, "anion_smiles": anion,
                      "IL_SMILES": pair, "source": "known_experimental"})
    if len(known) != 4647:
        raise ValueError(f"Expected 4,647 known pairs, found {len(known):,}")
    known_frame = pd.DataFrame(known)
    known_cations = sorted(known_frame.cation_smiles.unique())
    known_anions = sorted(known_frame.anion_smiles.unique())
    excluded = set(known_frame.IL_SMILES)

    recombined = sample_unique_pairs(
        known_cations, known_anions, 45_353, excluded, rng, "known_ion_recombination"
    )
    excluded.update(row["IL_SMILES"] for row in recombined)

    generated_frames = []
    for seed in SEEDS:
        path = root / "runs" / language / model / regime / str(seed) / "generated.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing best-model output for seed {seed}: {path}")
        generated_frames.append(pd.read_csv(path))
    generated = pd.concat(generated_frames, ignore_index=True)
    novel_cations = sorted({
        value for value in (
            canonical_ion(row.generated_smiles, 1)
            for row in generated[generated.ion_type == "cation"].itertuples(index=False)
        ) if value is not None
    })
    novel_anions = sorted({
        value for value in (
            canonical_ion(row.generated_smiles, -1)
            for row in generated[generated.ion_type == "anion"].itertuples(index=False)
        ) if value is not None
    })
    if not novel_cations or not novel_anions:
        raise RuntimeError("The selected model did not yield both valid cations and valid anions")
    proposed = sample_unique_pairs(
        novel_cations, novel_anions, 50_000, excluded, rng, "pretrained_model_proposal"
    )

    pool = pd.concat([known_frame, pd.DataFrame(recombined), pd.DataFrame(proposed)], ignore_index=True)
    if len(pool) != 100_000:
        raise AssertionError(f"Pair pool has {len(pool):,} rows instead of 100,000")
    pool.insert(0, "pair_id", np.arange(len(pool), dtype=int))
    pool.to_parquet(output / "pair_pool_unscored.parquet", index=False)

    manifest = {
        "selected_language": language, "selected_model": model, "selected_regime": regime,
        "seed": args.seed, "counts": pool.source.value_counts().to_dict(),
        "unique_cations": int(pool.cation_smiles.nunique()),
        "unique_anions": int(pool.anion_smiles.nunique()),
        "unique_pairs": int(pool.IL_SMILES.nunique()),
        "guidance_physical_folds": [1, 2, 3, 4, 5],
        "guidance_gas_folds": [1, 2, 3],
        "evaluation_oracle_used": False,
    }
    if not args.skip_scoring:
        oracle = TextOracle()
        scores = oracle.score(
            pool.IL_SMILES.tolist(), physical_folds=[1, 2, 3, 4, 5], gas_folds=[1, 2, 3]
        )
        scores = scores.drop(columns=["IL_SMILES"])
        scored = pd.concat([pool.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        z = 1.645
        scored["Tm_margin_uq_K"] = 298.15 - (scored.Tm_K + z * scored.Tm_K_sd)
        scored["ln_eta_margin_uq"] = np.log(100.0) - (
            scored.ln_eta_298 + z * scored.ln_eta_298_sd
        )
        scored["toxicity_margin_uq"] = (
            scored.toxicity - z * scored.toxicity_sd - 3.7
        )
        scored["hard_pass_uq"] = (
            (scored.Tm_margin_uq_K > 0)
            & (scored.ln_eta_margin_uq > 0)
            & (scored.toxicity_margin_uq > 0)
        )
        scored.to_parquet(output / "pair_pool_scored.parquet", index=False)
        scored.to_csv(output / "pair_pool_scored.csv.gz", index=False, compression="gzip")
        manifest["ordinary_hcsr"] = float(scored.hard_pass.mean())
        manifest["uq_hcsr"] = float(scored.hard_pass_uq.mean())
    (output / "PAIR_POOL_COMPLETED.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
