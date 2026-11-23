"""Copy locally held external weights; never downloads or deserializes weights."""
import argparse,csv,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--from-workspace',required=True,type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
for row in csv.DictReader((root/'model_weights_manifest.csv').open()):
    src=a.from_workspace/row['path'];dst=root/row['path']
    if hashlib.sha256(src.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(src)
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
print('All verified weights staged. Use GitHub Releases for weights; .gitignore excludes them.')
