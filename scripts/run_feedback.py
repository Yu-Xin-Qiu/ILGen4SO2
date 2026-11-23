"""Run current R-based feedback with explicit current defaults (requires CUDA/weights)."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=9);p.add_argument('--seed',type=int,choices=[137,251,367]);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
for seed in ([a.seed] if a.seed else [137,251,367]):
    subprocess.run([sys.executable,'work/text_generation_study/iterate_absorption_only.py',
      'outputs/IECR_allfold_ratio_DE_eta200_EC50gt2_20261111',f'reproduced/feedback/seed_{seed}',
      '--sample-seed',str(seed),'--count','10000','--local-count','5000','--local-sigma','0.25',
      '--reference-smiles','CCCCn1cc[n+](C)c1.COS(=O)(=O)[O-]',
      '--rounds',str(a.rounds),'--joint-point-replay','--max-sa-better-than-reference',
      '--product-feedback'],cwd=root,check=True)
