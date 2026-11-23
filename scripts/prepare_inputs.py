"""Restore compressed original ion lists for optional full corpus rebuilding."""
from pathlib import Path
import gzip, shutil
root=Path(__file__).resolve().parents[1]
for name in ('cations.txt','anions.txt'):
    target=root/name
    if not target.exists():
        with gzip.open(root/(name+'.gz'),'rb') as src,target.open('wb') as dst:
            shutil.copyfileobj(src,dst)
print('Original ion lists restored.')
