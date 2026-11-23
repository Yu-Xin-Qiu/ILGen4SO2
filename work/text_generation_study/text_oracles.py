import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_ROOT = ROOT / "previous" / "ILBERT"
GAS_ROOT = ROOT / "previous" / "IL4SO2"
OUTPUT_ROOT = ROOT / "outputs" / "IECR_text_generation_revision_20260919"
UPDATED_VISCOSITY_ROOT = OUTPUT_ROOT / "predictors" / "viscosity_text_10fold"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


physical_model_module = load_module("physical_ilbert_model", PHYSICAL_ROOT / "model.py")
physical_tokenizer_module = load_module("physical_ilbert_tokenizer", PHYSICAL_ROOT / "ILtokenizer.py")
gas_model_module = load_module("gas_ilbert_model", GAS_ROOT / "model.py")
gas_tokenizer_module = load_module("gas_ilbert_tokenizer", GAS_ROOT / "ILtokenizer.py")


PHYSICAL_TASKS = {
    "melting_point": {"folder": "Melting point", "class": "ILBERT", "mean": 363.0857, "std": 75.0207},
    "viscosity": {"folder": "Viscosity", "class": "ILBERT_T", "mean": 4.1952, "std": 1.6220},
    "toxicity": {"folder": "IPC-81", "class": "ILBERT", "mean": 3.1156, "std": 1.0641},
}


class TextOracle:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.physical_config = yaml.safe_load((PHYSICAL_ROOT / "config.yaml").read_text(encoding="utf-8"))
        self.gas_config = yaml.safe_load((GAS_ROOT / "config-cv.yaml").read_text(encoding="utf-8"))
        self.physical_tokenizer = physical_tokenizer_module.SMILES_Atomwise_Tokenizer(str(PHYSICAL_ROOT / "vocab.txt"))
        self.gas_tokenizer = gas_tokenizer_module.SMILES_Atomwise_Tokenizer(str(GAS_ROOT / "vocab_IL.txt"))

    def encode(self, smiles, tokenizer):
        return torch.tensor([tokenizer.encode(value, max_length=100, truncation=True, padding="max_length")
                             for value in smiles], dtype=torch.long)

    def _run_model(self, model, token_ids, temperatures=None, pressures=None, batch_size=512):
        predictions = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(token_ids), batch_size):
                stop = min(len(token_ids), start + batch_size)
                inputs = [token_ids[start:stop].to(self.device)]
                if temperatures is not None:
                    inputs.append(torch.as_tensor(temperatures[start:stop], dtype=torch.float32, device=self.device))
                if pressures is not None:
                    inputs.append(torch.as_tensor(pressures[start:stop], dtype=torch.float32, device=self.device))
                predictions.extend(model(inputs).reshape(-1).cpu().numpy())
        return np.asarray(predictions)

    def physical(self, smiles, task, temperature=None, folds=None, token_ids=None, return_folds=False):
        spec = PHYSICAL_TASKS[task]
        folds = list(folds or range(1, 11))
        token_ids = token_ids if token_ids is not None else self.encode(smiles, self.physical_tokenizer)
        outputs = []
        model_class = getattr(physical_model_module, spec["class"])
        use_updated_viscosity = (
            task == "viscosity" and (UPDATED_VISCOSITY_ROOT / "COMPLETED.json").exists()
        )
        for fold in folds:
            model = model_class(**self.physical_config["transformer"]).to(self.device)
            updated_path = UPDATED_VISCOSITY_ROOT / f"fold_{fold}_best_model.pt"
            if use_updated_viscosity:
                if not updated_path.exists():
                    raise FileNotFoundError(
                        f"Updated viscosity ensemble is marked complete but fold {fold} is missing"
                    )
                checkpoint = torch.load(updated_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model"], strict=True)
                temperatures = (
                    np.full(len(smiles), temperature, dtype=np.float32)
                    - float(checkpoint["temperature_mean"])
                ) / float(checkpoint["temperature_std"])
                scaled = self._run_model(model, token_ids, temperatures)
                outputs.append(
                    scaled * float(checkpoint["label_std"]) + float(checkpoint["label_mean"])
                )
            else:
                path = PHYSICAL_ROOT / "model_weight" / "final" / spec["folder"] / f"fold_{fold}_best_model.pth"
                model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False), strict=True)
                temperatures = (
                    np.full(len(smiles), temperature, dtype=np.float32)
                    if task == "viscosity" else np.zeros(len(smiles), dtype=np.float32)
                )
                scaled = self._run_model(model, token_ids, temperatures)
                outputs.append(scaled * spec["std"] + spec["mean"])
            del model
            if self.device.type == "cuda": torch.cuda.empty_cache()
        values = np.asarray(outputs)
        if return_folds:
            return values
        return values.mean(0), values.std(0, ddof=1) if len(values) > 1 else np.zeros(len(smiles))

    def gas(self, smiles, gas, temperature, pressure, folds=None, token_ids=None, return_folds=False):
        token_ids = token_ids if token_ids is not None else self.encode(smiles, self.gas_tokenizer)
        temperatures = np.full(len(smiles), temperature, dtype=np.float32)
        pressures = np.full(len(smiles), pressure, dtype=np.float32)
        outputs = []
        if gas == "CO2":
            folds = list(folds or range(1, 6))
            labels = pd.read_csv(ROOT / "CO2 solubility.csv")["ln(x_CO2)"].dropna()
            global_mean, global_std = float(labels.mean()), float(labels.std(ddof=1))
            paths = [GAS_ROOT / "model_weight" / "SCREEN" / "CO2" / "TF-IL" / f"fold_{fold}_best_model.pth" for fold in folds]
            checkpoints = [(path, global_mean, global_std) for path in paths]
        elif gas == "SO2":
            folds = list(folds or range(1, 6))
            base = OUTPUT_ROOT / "predictors" / "SO2_text_5fold"
            checkpoints = []
            for fold in folds:
                checkpoint = torch.load(base / f"fold_{fold}_best_model.pt", map_location="cpu", weights_only=False)
                checkpoints.append((base / f"fold_{fold}_best_model.pt", float(checkpoint["mean"]), float(checkpoint["std"])))
        else:
            raise ValueError(gas)
        for path, mean, std in checkpoints:
            model = gas_model_module.ILBERT_T_P(**self.gas_config["transformer"]).to(self.device)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state, strict=True)
            scaled = self._run_model(model, token_ids, temperatures, pressures)
            outputs.append(scaled * std + mean)
            del model
            if self.device.type == "cuda": torch.cuda.empty_cache()
        values = np.asarray(outputs)
        if return_folds:
            return values
        return values.mean(0), values.std(0, ddof=1) if len(values) > 1 else np.zeros(len(smiles))

    def score(self, smiles, physical_folds=None, gas_folds=None):
        physical_tokens = self.encode(smiles, self.physical_tokenizer)
        gas_tokens = self.encode(smiles, self.gas_tokenizer)
        tm, tm_sd = self.physical(smiles, "melting_point", folds=physical_folds, token_ids=physical_tokens)
        ln_eta, ln_eta_sd = self.physical(
            smiles, "viscosity", temperature=298.15, folds=physical_folds, token_ids=physical_tokens
        )
        toxicity, toxicity_sd = self.physical(smiles, "toxicity", folds=physical_folds, token_ids=physical_tokens)
        so2_298, so2_298_sd = self.gas(
            smiles, "SO2", 298.15, 1.0, folds=gas_folds, token_ids=gas_tokens
        )
        so2_313, so2_313_sd = self.gas(
            smiles, "SO2", 313.15, 1.0, folds=gas_folds, token_ids=gas_tokens
        )
        co2_298, co2_298_sd = self.gas(
            smiles, "CO2", 298.15, 1.0, folds=gas_folds, token_ids=gas_tokens
        )
        result = pd.DataFrame({"IL_SMILES": smiles, "Tm_K": tm, "Tm_K_sd": tm_sd,
                               "ln_eta_298": ln_eta, "ln_eta_298_sd": ln_eta_sd,
                               "eta_298_mPas": np.exp(ln_eta), "toxicity": toxicity, "toxicity_sd": toxicity_sd,
                               "ln_x_SO2_298": so2_298, "ln_x_SO2_298_sd": so2_298_sd,
                               "ln_x_SO2_313": so2_313, "ln_x_SO2_313_sd": so2_313_sd,
                               "ln_x_CO2_298": co2_298, "ln_x_CO2_298_sd": co2_298_sd})
        result["x_SO2_298"] = np.exp(result.ln_x_SO2_298)
        result["x_SO2_313"] = np.exp(result.ln_x_SO2_313)
        result["x_CO2_298"] = np.exp(result.ln_x_CO2_298)
        result["AC"] = result.x_SO2_298
        result["DE"] = result.x_SO2_298 - result.x_SO2_313
        result["S"] = result.x_SO2_298 / (result.x_CO2_298 + 1e-12)
        result["hard_pass"] = (result.Tm_K < 298.15) & (result.eta_298_mPas < 100) & (result.toxicity > 3.7)
        result["physical_gas"] = result[["x_SO2_298", "x_SO2_313", "x_CO2_298"]].gt(0).all(axis=1) & result[["x_SO2_298", "x_SO2_313", "x_CO2_298"]].le(1).all(axis=1)
        return result


if __name__ == "__main__":
    oracle = TextOracle()
    example = ["Cn1cc[nH+]c1.[Cl-]"]
    print(oracle.score(example).to_json(orient="records", indent=2))
