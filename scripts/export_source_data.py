"""Export losslessly bundled source tables/QC inputs for Origin or Gaussian."""
from pathlib import Path
import argparse,gzip,json
ROOT=Path(__file__).resolve().parents[1]
def load_tables():
    with gzip.open(ROOT/'data/figure_tables.json.gz','rt',encoding='utf-8') as f:b=json.load(f)
    tables=b['tables']
    return {**tables,**{k:tables[v] for k,v in b['aliases'].items()}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'reproduced/source_data');a=p.parse_args()
    for key,text in load_tables().items():
        q=a.out/'figure_source'/key;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
    for key,text in json.loads((ROOT/'data/quantum_chemistry.json').read_text(encoding='utf-8')).items():
        q=a.out/'quantum_chemistry'/key;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
    print(a.out)
if __name__=='__main__':main()
