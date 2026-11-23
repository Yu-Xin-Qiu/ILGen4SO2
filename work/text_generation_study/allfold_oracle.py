"""Full-ensemble inference for the ratio-DE study."""
import numpy as np
import pandas as pd

from text_oracles import TextOracle
from study_thresholds import TM_LIMIT_K, TOXICITY_MIN, VISCOSITY_LIMIT_MPAS

PHYSICAL_FOLDS = tuple(range(1, 11))
GAS_FOLDS = tuple(range(1, 6))
Z = 1.645


def store_folds(columns, name, values, folds):
    if values.shape != (len(folds), len(columns["IL_SMILES"])):
        raise ValueError(f"Unexpected ensemble shape for {name}: {values.shape}")
    columns[name] = values.mean(axis=0)
    columns[f"{name}_sd"] = values.std(axis=0, ddof=1)
    for fold, predictions in zip(folds, values):
        columns[f"{name}_fold{fold:02d}"] = predictions


def apply_physical_limits(frame):
    ln_eta_limit = np.log(VISCOSITY_LIMIT_MPAS)
    frame["eta_298_mPas"] = np.exp(frame.ln_eta_298)
    frame["Tm_pass"] = frame.Tm_K < TM_LIMIT_K
    frame["viscosity_pass"] = frame.ln_eta_298 < ln_eta_limit
    frame["toxicity_pass"] = frame.toxicity > TOXICITY_MIN
    frame["hard_pass"] = frame[["Tm_pass", "viscosity_pass", "toxicity_pass"]].all(axis=1)
    frame["Tm_margin_uq_K"] = TM_LIMIT_K - (frame.Tm_K + Z * frame.Tm_K_sd)
    frame["ln_eta_margin_uq"] = ln_eta_limit - (frame.ln_eta_298 + Z * frame.ln_eta_298_sd)
    frame["toxicity_margin_uq"] = (frame.toxicity - Z * frame.toxicity_sd) - TOXICITY_MIN
    frame["hard_pass_uq"] = frame[["Tm_margin_uq_K", "ln_eta_margin_uq", "toxicity_margin_uq"]].gt(0).all(axis=1)
    return frame


class FullFoldOracle(TextOracle):
    def score_physical(self, smiles):
        columns = {"IL_SMILES": smiles}
        tokens = self.encode(smiles, self.physical_tokenizer)
        for name, task, temperature in (
            ("Tm_K", "melting_point", None),
            ("ln_eta_298", "viscosity", 298.15),
            ("toxicity", "toxicity", None),
        ):
            values = self.physical(smiles, task, temperature, PHYSICAL_FOLDS, tokens, return_folds=True)
            store_folds(columns, name, values, PHYSICAL_FOLDS)
        return apply_physical_limits(pd.DataFrame(columns))

    def score(self, smiles):
        physical = self.score_physical(smiles)
        columns = {"IL_SMILES": smiles}
        tokens = self.encode(smiles, self.gas_tokenizer)
        for suffix, gas, temperature in (
            ("SO2_298", "SO2", 298.15), ("SO2_313", "SO2", 313.15), ("CO2_298", "CO2", 298.15)
        ):
            values = self.gas(smiles, gas, temperature, 1.0, GAS_FOLDS, tokens, return_folds=True)
            store_folds(columns, f"ln_x_{suffix}", values, GAS_FOLDS)
            columns[f"x_{suffix}"] = np.exp(columns[f"ln_x_{suffix}"])
        frame = pd.concat([physical, pd.DataFrame(columns).drop(columns="IL_SMILES")], axis=1)
        gas_values = frame[["x_SO2_298", "x_SO2_313", "x_CO2_298"]]
        frame["physical_gas"] = np.isfinite(gas_values).all(axis=1) & gas_values.gt(0).all(axis=1) & gas_values.le(1).all(axis=1)
        frame["AC"] = frame.x_SO2_298
        frame["DE"] = frame.x_SO2_298 / frame.x_SO2_313.where(frame.x_SO2_313 > 0)
        frame["S"] = frame.x_SO2_298 / frame.x_CO2_298.where(frame.x_CO2_298 > 0)
        return frame
