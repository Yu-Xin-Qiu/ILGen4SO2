import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "previous" / "IL4SO2"
sys.path.insert(0, str(SOURCE))
from dataset import SMILES_dataset
from ILtokenizer import SMILES_Atomwise_Tokenizer
from model import ILBERT_T_P


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def canonical_groups(smiles):
    from rdkit import Chem
    result = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        result.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else value)
    return result


def loader(frame, tokenizer, batch, shuffle):
    return DataLoader(SMILES_dataset(frame.reset_index(drop=True), tokenizer, "ln(x_SO2)"),
                      batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)


def evaluate(model, data_loader, mean, std, device):
    model.eval(); observed=[]; predicted=[]; structures=[]; temperatures=[]; pressures=[]
    with torch.no_grad():
        for inputs, labels, smiles, t, p in data_loader:
            inputs = [value.to(device) for value in inputs]
            output = model(inputs).reshape(-1) * std + mean
            observed.extend(labels.reshape(-1).numpy()); predicted.extend(output.cpu().numpy())
            structures.extend(smiles); temperatures.extend(t.reshape(-1).numpy()); pressures.extend(p.reshape(-1).numpy())
    observed=np.asarray(observed); predicted=np.asarray(predicted)
    metrics={"mae":mean_absolute_error(observed,predicted),"rmse":math.sqrt(mean_squared_error(observed,predicted)),
             "r2":r2_score(observed,predicted)}
    rows=pd.DataFrame({"experimental":observed,"predicted":predicted,"IL_SMILES":structures,
                       "T_K":temperatures,"P_bar":pressures})
    return metrics, rows


def main():
    config = yaml.safe_load((SOURCE / "config-cv.yaml").read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / "SO2 solubility.csv").dropna().reset_index(drop=True)
    frame["group"] = canonical_groups(frame["IL_SMILES"].astype(str))
    output_dir = ROOT / "outputs" / "IECR_text_generation_revision_20260919" / "predictors" / "SO2_text_5fold"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "COMPLETED.json").exists():
        print((output_dir / "COMPLETED.json").read_text(encoding="utf-8")); return
    existing_oof = output_dir / "oof_predictions.csv"
    if existing_oof.exists() and all((output_dir / f"fold_{fold}_best_model.pt").exists() for fold in range(1, 6)):
        oof = pd.read_csv(existing_oof)
        overall = {"n": int(len(oof)), "n_unique_ils": int(frame.group.nunique()),
                   "mae": float(mean_absolute_error(oof.experimental, oof.predicted)),
                   "rmse": float(math.sqrt(mean_squared_error(oof.experimental, oof.predicted))),
                   "r2": float(r2_score(oof.experimental, oof.predicted)),
                   "split": "GroupKFold by canonical ionic liquid",
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (output_dir / "COMPLETED.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
        print(json.dumps(overall, indent=2)); return
    tokenizer = SMILES_Atomwise_Tokenizer(str(SOURCE / "vocab_IL.txt"))
    splitter = GroupKFold(n_splits=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows=[]; fold_metrics=[]
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, groups=frame.group), 1):
        set_seed(9100 + fold)
        train_frame=frame.iloc[train_index].copy(); test_frame=frame.iloc[test_index].copy()
        groups=train_frame.group.drop_duplicates().sample(frac=0.15, random_state=fold).tolist()
        validation_frame=train_frame[train_frame.group.isin(groups)].copy()
        fit_frame=train_frame[~train_frame.group.isin(groups)].copy()
        mean=float(fit_frame["ln(x_SO2)"].mean()); std=float(fit_frame["ln(x_SO2)"].std(ddof=0))
        model=ILBERT_T_P(**config["transformer"]).to(device)
        pretrained=torch.load(SOURCE / "model_weight" / "pretrain" / "TF-SO2" / "test_best_model.pth",
                              map_location=device, weights_only=False)
        model.load_state_dict(pretrained, strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-5)
        fit_loader=loader(fit_frame,tokenizer,16,True); validation_loader=loader(validation_frame,tokenizer,64,False)
        best=float("inf"); stale=0; history=[]; started=time.time()
        for epoch in range(1,101):
            model.train(); losses=[]
            for inputs, labels, *_ in fit_loader:
                inputs=[value.to(device) for value in inputs]; labels=labels.to(device).reshape(-1)
                optimizer.zero_grad(set_to_none=True)
                prediction=model(inputs).reshape(-1)
                loss=F.mse_loss(prediction,(labels-mean)/std); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
            validation_metrics,_=evaluate(model,validation_loader,mean,std,device)
            history.append({"epoch":epoch,"train_mse_scaled":float(np.mean(losses)),**{f"validation_{k}":v for k,v in validation_metrics.items()},"lr":optimizer.param_groups[0]["lr"]})
            if validation_metrics["rmse"] < best:
                best=validation_metrics["rmse"]; stale=0
                torch.save({"model":model.state_dict(),"mean":mean,"std":std,"fold":fold},output_dir/f"fold_{fold}_best_model.pt")
            else: stale+=1
            scheduler.step()
            if stale>=15: break
        pd.DataFrame(history).to_csv(output_dir/f"fold_{fold}_history.csv",index=False)
        checkpoint=torch.load(output_dir/f"fold_{fold}_best_model.pt",map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model"])
        metrics,rows=evaluate(model,loader(test_frame,tokenizer,64,False),mean,std,device)
        rows["fold"]=fold; all_rows.append(rows); fold_metrics.append({"fold":fold,"n_train":len(fit_frame),"n_validation":len(validation_frame),"n_test":len(test_frame),"seconds":time.time()-started,**metrics})
        print(f"fold={fold} {metrics}",flush=True)
    oof=pd.concat(all_rows,ignore_index=True); oof.to_csv(output_dir/"oof_predictions.csv",index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir/"fold_metrics.csv",index=False)
    overall={"n":int(len(oof)),"n_unique_ils":int(frame.group.nunique()),"mae":float(mean_absolute_error(oof.experimental,oof.predicted)),
             "rmse":float(math.sqrt(mean_squared_error(oof.experimental,oof.predicted))),"r2":float(r2_score(oof.experimental,oof.predicted)),
             "split":"GroupKFold by canonical ionic liquid","device":torch.cuda.get_device_name(0) if device.type=="cuda" else "cpu"}
    (output_dir/"COMPLETED.json").write_text(json.dumps(overall,indent=2),encoding="utf-8"); print(json.dumps(overall,indent=2))


if __name__ == "__main__":
    main()
