"""Reproduce the current screen from the latest saved desktop point predictions."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json
import numpy as np
import pandas as pd
from rdkit import Chem,rdBase
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=50000)
def identity(s):
    with rdBase.BlockLogs():
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError(s)
        return Chem.MolToInchiKey(mol)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'reproduced');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(ROOT/'data/generated/final_draws.parquet')
    assert len(f)==30000
    assert f.groupby(['seed','arm']).size().eq(5000).all()
    f=f.loc[f.valid_pair].copy()
    f['InChIKey']=f.IL_SMILES.map(identity)
    gas=pd.DataFrame({'abs':f.AC,'des':f.AC/f.DE,'co2':f.AC/f.S})
    physical=np.isfinite(gas).all(axis=1)&gas.gt(0).all(axis=1)&gas.le(1).all(axis=1)
    hard=(f.Tm_K<298.15)&(f.eta_298_mPas<200)&(f.toxicity>2)
    assert (hard==f.hard_pass).all()
    ref=pd.read_csv(ROOT/'data/generated/reference.csv').iloc[0]
    strict=hard&physical&f.library_novel&f.AC.gt(ref.AC)&f.DE.gt(ref.DE)&f.S.gt(ref.S)&f.SA_max.lt(ref.SA_max)
    assert (strict==f.joint_hit).all()
    np.testing.assert_allclose(f.R,f.AC*f.DE*f.S/f.SA_max,rtol=2e-6)
    feasible=f.loc[hard&physical].drop_duplicates('InChIKey')
    hits=f.loc[strict].drop_duplicates('InChIKey').copy()
    assert len(feasible)==10285,len(feasible)
    assert len(hits)==15,len(hits)
    obj=hits[['AC','DE','S','SA_max']].to_numpy()*[1,1,1,-1]
    keep=[not ((obj>=row).all(axis=1)&(obj>row).any(axis=1)).any() for row in obj]
    front=hits.loc[keep].sort_values('R',ascending=False)
    published=pd.read_csv(ROOT/'data/generated/pareto_front.csv')
    assert len(front)==11 and set(front.InChIKey)==set(published.InChIKey)
    front=front.merge(published[['InChIKey','ID','cation_smiles','anion_smiles']],on='InChIKey',validate='one_to_one').sort_values('R',ascending=False)
    assert front.ID.head(4).tolist()==['IL0006','IL0014','IL0003','IL0008']
    np.testing.assert_allclose(front.R.head(4),pd.read_csv(ROOT/'data/processed/top4.csv').R,rtol=2e-6)
    for name,df in [('feasible_candidates',feasible),('strict_candidates',hits),('pareto_front',front),('top4',front.head(4))]:
        df.to_csv(a.out/(name+'.csv'),index=False)
    checked=0
    if (ROOT/'checksums.csv').exists():
        for row in pd.read_csv(ROOT/'checksums.csv').itertuples():
            assert hashlib.sha256((ROOT/row.path).read_bytes()).hexdigest()==row.sha256,row.path
            checked+=1
    report=dict(attempts=30000,feasible=len(feasible),strict=len(hits),pareto=len(front),top4=front.ID.head(4).tolist(),files_verified=checked)
    (a.out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
