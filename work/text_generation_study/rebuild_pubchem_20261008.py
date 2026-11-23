"""Rebuild charged molecular-text corpora from the complete PubChem CID dump."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from itertools import islice
import math
import multiprocessing as mp
from pathlib import Path
import re
import queue
import sqlite3
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import selfies as sf

from molecular_tokenizer import SPECIAL_TOKENS, tokenize_smiles

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'PubChem_dataset_revision_20261008'
CHARGED_ATOM = re.compile(r'\[[^\]]*[+-][^\]]*\]')
ATOM_TOKEN = re.compile(r'\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|b|c|n|o|p|s|\*')
HYDROGEN_ATOM = re.compile(r'^\[(?:\d+)?H(?:[+\-\d:]|\])')
PROPERTIES = ['melting point', 'viscosity', 'toxicity', 'CO2 solubility', 'SO2 solubility']
SCHEMA = pa.schema([
    ('cid', pa.int64()), ('ion_type', pa.string()), ('smiles', pa.string()),
    ('selfies', pa.string()), ('formal_charge', pa.int64()),
    ('smiles_tokens', pa.int64()), ('selfies_tokens', pa.int64()),
    ('scaffold', pa.string()), ('split', pa.string()),
])


def init_worker():
    RDLogger.DisableLog('rdApp.*')


def isolated_worker(function, input_queue, output_queue, current):
    init_worker()
    while True:
        batch = input_queue.get()
        if batch is None:
            return
        result = []
        for item in batch:
            current.value = item[0]
            result.append(function(item))
        output_queue.put(result)
        current.value = 0


def safe_iter(function, items, workers, out, stage):
    """Keep native failures local to one CID and bound queued molecular work."""
    context = mp.get_context('spawn')
    iterator = iter(items)
    slots = []
    exhausted = False

    def start_slot(slot):
        iq, oq = context.Queue(1), context.Queue(1)
        current = context.Value('q', 0)
        proc = context.Process(target=isolated_worker, args=(function, iq, oq, current))
        proc.start()
        slot.update(proc=proc, iq=iq, oq=oq, current=current, observed=0, changed=time.monotonic())

    def stop_slot(slot):
        if slot['proc'].is_alive():
            slot['proc'].terminate()
        slot['proc'].join(timeout=3)
        for key in ('iq','oq'):
            slot[key].cancel_join_thread()
            slot[key].close()

    try:
        for _ in range(workers):
            slot={'job':None, 'restart_count':0}
            start_slot(slot)
            slots.append(slot)
        while True:
            active = False
            for slot in slots:
                if slot['job'] is None and not exhausted:
                    job=list(islice(iterator,128))
                    if job:
                        slot['job']=job
                        slot['current'].value=0
                        slot['observed']=0
                        slot['changed']=time.monotonic()
                        slot['iq'].put(job)
                    else:
                        exhausted=True
                if slot['job'] is None:
                    continue
                active=True
                try:
                    results=slot['oq'].get_nowait()
                except queue.Empty:
                    results=None
                if results is not None:
                    slot['job']=None
                    slot['restart_count']=0
                    yield from results
                    continue
                cid=slot['current'].value
                if cid != slot['observed']:
                    slot['observed']=cid
                    slot['changed']=time.monotonic()
                alive=slot['proc'].is_alive()
                try:
                    rss=psutil.Process(slot['proc'].pid).memory_info().rss if alive else 0
                except psutil.Error:
                    rss=0
                timed_out=time.monotonic()-slot['changed'] > (30 if cid else 120)
                if not alive or rss > 1024**3 or timed_out:
                    job=slot['job']
                    stop_slot(slot)
                    offending=next((item for item in job if item[0]==cid),None)
                    if offending is None:
                        slot['restart_count']+=1
                        if slot['restart_count'] > 3:
                            raise RuntimeError(f'Worker startup repeatedly failed during {stage}')
                    else:
                        cause='native_exit' if not alive else 'memory_limit' if rss > 1024**3 else 'timeout'
                        failure={'stage':stage,'cid':cid,'cause':cause,'record':offending}
                        with (out/'resource_exceptions.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(failure,ensure_ascii=False)+'\n')
                        print(json.dumps({'stage':stage,'resource_exception_cid':cid,'cause':cause}),flush=True)
                        job=[item for item in job if item[0]!=cid]
                        yield None, 'resource_exception_'+cause
                    start_slot(slot)
                    slot['job']=job if job else None
                    if job:
                        slot['iq'].put(job)
            if exhausted and not active:
                break
            time.sleep(.005)
    finally:
        for slot in slots:
            stop_slot(slot)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def fraction(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') / 2**64


def split_record(kind, smiles, scaffold=None):
    p = fraction(f'{kind}:{scaffold if scaffold is not None else smiles}')
    cuts = (0.8, 0.9) if scaffold is not None else (0.98, 0.99)
    return 'train' if p < cuts[0] else 'validation' if p < cuts[1] else 'test'


def structure(item):
    cid, raw = item
    if '.' in raw:
        return None, 'multicomponent'
    if len(raw) > 253:
        atom_count = 0
        for match in ATOM_TOKEN.finditer(raw):
            if not HYDROGEN_ATOM.match(match.group()):
                atom_count += 1
                if atom_count > 253:
                    return None, 'guaranteed_above_256_token_cap'
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None, 'rdkit_invalid'
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, 'multicomponent'
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if charge == 0:
        return None, 'net_neutral'
    kind = 'cation' if charge > 0 else 'anion'
    if charge > 0 and not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return None, 'carbon_free_cation'
    # Canonical sanitization retains ionic charge, stereochemistry, and metal bonds.
    return (cid, kind, canonical(mol), charge), 'accepted'


def representation(item):
    cid, kind, smiles, charge = item
    try:
        text = sf.encoder(smiles)
        mol = Chem.MolFromSmiles(sf.decoder(text))
        if mol is None or canonical(mol) != smiles:
            return None, 'selfies_roundtrip_changed'
        st = tokenize_smiles(smiles)
        ft = list(sf.split_selfies(text))
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True) or smiles
        return dict(cid=cid, ion_type=kind, smiles=smiles, selfies=text,
                    formal_charge=charge, smiles_tokens=len(st), selfies_tokens=len(ft),
                    scaffold=scaffold, split=''), 'accepted'
    except Exception as e:
        return None, 'selfies_failed' if not isinstance(e, ValueError) or 'tokenizer' not in str(e) else 'tokenization_failed'


def property_ions(out):
    out.mkdir(parents=True, exist_ok=True)
    sources = defaultdict(set)
    observations = Counter()
    audit = {}
    pairs = {}
    rejection_rows = []
    for name in PROPERTIES:
        path = ROOT / f'{name}.csv'
        frame = pd.read_csv(path)
        column = 'IL_SMILES' if 'IL_SMILES' in frame else 'IL SMILES'
        tally = Counter()
        for row_idx, raw in enumerate(frame[column].fillna('').astype(str)):
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                tally['rdkit_invalid'] += 1
                rejection_rows.append([name, row_idx, raw, 'rdkit_invalid'])
                continue
            cat, ani = [], []
            for part in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
                q = sum(a.GetFormalCharge() for a in part.GetAtoms())
                if q == 0:
                    tally['neutral_components'] += 1
                    continue
                smi = canonical(part)
                kind = 'cation' if q > 0 else 'anion'
                sources[(kind, smi, q)].add(name)
                observations[(kind, smi, q)] += 1
                (cat if q > 0 else ani).append(smi)
            if cat and ani:
                tally['rows_with_both_ion_signs'] += 1
                key = canonical(mol)
                pairs[key] = (json.dumps(sorted(cat)), json.dumps(sorted(ani)))
                if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
                    tally['non_neutral_il_rows'] += 1
                if len(cat) != 1 or len(ani) != 1:
                    tally['non_1_to_1_pair_rows'] += 1
            else:
                tally['rows_without_both_ion_signs'] += 1
        audit[name] = {'rows': len(frame), 'sha256': sha256(path), 'counts': dict(tally)}
    raw_rows = []
    for (kind, smi, q), names in sorted(sources.items()):
        raw_rows.append(dict(ion_type=kind, smiles=smi, formal_charge=q,
                             property_sources=';'.join(sorted(names)), observations=observations[(kind, smi, q)]))
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out / 'target_ions_all_standardized.csv', index=False)
    pd.DataFrame([[k, *v] for k, v in sorted(pairs.items())],
                 columns=['IL_SMILES', 'cations_json', 'anions_json']).to_csv(out / 'known_pairs_all_properties.csv', index=False)
    pd.DataFrame(rejection_rows, columns=['property', 'row', 'raw_smiles', 'reason']).to_csv(out / 'property_invalid_rows.csv', index=False)
    accepted, failures = [], Counter()
    for row in raw.itertuples():
        rec, reason = representation((0, row.ion_type, row.smiles, row.formal_charge))
        if rec is None:
            failures[f'{row.ion_type}:{reason}'] += 1
            continue
        rec['split'] = split_record(rec['ion_type'], rec['smiles'], rec['scaffold'])
        accepted.append(rec)
    target = pd.DataFrame(accepted)
    target.to_parquet(out / 'target_ions_before_length.parquet', index=False)
    save_json(out / 'property_audit.json', {'properties': audit, 'known_pairs': len(pairs),
             'raw_unique_ions': raw.groupby('ion_type').size().to_dict(),
             'target_before_length': target.groupby('ion_type').size().to_dict(),
             'target_representation_failures': dict(failures)})
    return set(raw.smiles), target


def scan(source, out):
    marker = out / 'SCAN_COMPLETED.json'
    if marker.exists():
        return json.loads(marker.read_text())
    counters = Counter()
    started = time.monotonic()
    with gzip.open(source, 'rt', encoding='utf-8', errors='strict') as f, gzip.open(out / 'charged_candidates.tsv.gz', 'wt', encoding='utf-8') as g:
        for line in f:
            counters['total_records'] += 1
            fields = line.rstrip('\n\r').split('\t', 1)
            if len(fields) != 2 or not fields[0].isdigit():
                counters['malformed_records'] += 1
                continue
            raw = fields[1]
            # A bond '-' is not an atomic negative charge. Inspect bracket atoms.
            if '+' in raw or ('-' in raw and '[' in raw):
                if CHARGED_ATOM.search(raw):
                    counters['charge_token_records'] += 1
                    g.write(line)
                else:
                    counters['no_atomic_charge_token'] += 1
            else:
                counters['no_atomic_charge_token'] += 1
            if counters['total_records'] % 5_000_000 == 0:
                print(json.dumps({'stage': 'scan', **counters, 'seconds': round(time.monotonic()-started)}), flush=True)
    result = {'source': str(source), 'source_bytes': source.stat().st_size,
              'source_sha256': sha256(source), 'counts': dict(counters)}
    save_json(marker, result)
    return result


def read_candidates(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            cid, raw = line.rstrip('\n\r').split('\t', 1)
            yield int(cid), raw


def clean(out, workers):
    marker = out / 'STANDARDIZE_COMPLETED.json'
    if marker.exists():
        return json.loads(marker.read_text())
    db = sqlite3.connect(out / 'canonical_ions.sqlite')
    db.execute('DROP TABLE IF EXISTS ions')
    db.execute('CREATE TABLE ions (smiles TEXT PRIMARY KEY, cid INTEGER, ion_type TEXT, charge INTEGER)')
    counts, types = Counter(), Counter()
    started = time.monotonic()
    if True:
        for rec, reason in safe_iter(structure, read_candidates(out / 'charged_candidates.tsv.gz'), workers, out, 'standardize'):
            counts['processed'] += 1
            if rec is None:
                counts[reason] += 1
            else:
                cid, kind, smiles, charge = rec
                types[f'{kind}:before_deduplication'] += 1
                changed = db.execute('INSERT OR IGNORE INTO ions VALUES (?,?,?,?)', (smiles, cid, kind, charge)).rowcount
                counts['unique_standardized' if changed else 'canonical_duplicate'] += 1
                if changed:
                    types[f'{kind}:unique_standardized'] += 1
            if counts['processed'] % 100_000 == 0:
                db.commit()
                print(json.dumps({'stage': 'standardize', **counts, 'seconds': round(time.monotonic()-started)}), flush=True)
    db.commit()
    db.close()
    result = {'counts': dict(counts), 'ion_counts': dict(types)}
    save_json(marker, result)
    return result


def verified_corpus(out, workers, targets):
    marker = out / 'REPRESENTATION_COMPLETED.json'
    if marker.exists():
        return json.loads(marker.read_text())
    db = sqlite3.connect(out / 'canonical_ions.sqlite', check_same_thread=False)
    cursor = db.execute('SELECT cid, ion_type, smiles, charge FROM ions ORDER BY cid')
    counters, types = Counter(), Counter()
    with pq.ParquetWriter(out / 'pretrain_ions_before_length.parquet', SCHEMA, compression='zstd') as writer:
        batch = []
        for rec, reason in safe_iter(representation, cursor, workers, out, 'representations'):
            counters['processed'] += 1
            if rec is None:
                counters[reason] += 1
            elif rec['smiles'] in targets:
                counters['downstream_overlap_removed'] += 1
                types[f"{rec['ion_type']}:overlap_removed"] += 1
            else:
                rec['split'] = split_record(rec['ion_type'], rec['smiles'])
                types[f"{rec['ion_type']}:verified_nonoverlapping"] += 1
                batch.append(rec)
                counters['verified_nonoverlapping'] += 1
            if len(batch) >= 4000:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                batch.clear()
            if counters['processed'] % 100_000 == 0:
                print(json.dumps({'stage': 'representations', **counters}), flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    db.close()
    result = {'counts': dict(counters), 'ion_counts': dict(types)}
    save_json(marker, result)
    return result


def export_training(out, target, targets):
    print(json.dumps({'stage':'export','step':'length_percentiles'}),flush=True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    lengths = pd.read_parquet(out / 'pretrain_ions_before_length.parquet', columns=['smiles_tokens', 'selfies_tokens', 'split'], use_threads=False)
    train = lengths[lengths.split == 'train']
    # Reserve BOS, type, EOS; percentiles use only the pretraining train partition.
    maximum = {name: min(256, int(math.ceil((train[name+'_tokens'].quantile(.99)+3)/8)*8)) for name in ('smiles', 'selfies')}
    print(json.dumps({'stage':'export','maximum_tokens':maximum}),flush=True)
    types, removed, charges, token_counts = Counter(), Counter(), Counter(), {'smiles': Counter(), 'selfies': Counter()}
    overlap = 0
    with pq.ParquetWriter(out / 'pretrain_ions.parquet', SCHEMA, compression='zstd') as writer, \
         gzip.open(out / 'pretrain_cations.smi.gz', 'wt') as cat, gzip.open(out / 'pretrain_anions.smi.gz', 'wt') as ani:
        for batch in pq.ParquetFile(out / 'pretrain_ions_before_length.parquet').iter_batches(batch_size=5000, use_threads=False):
            records=[]
            for rec in batch.to_pylist():
                if rec['smiles_tokens']+3 > maximum['smiles'] or rec['selfies_tokens']+3 > maximum['selfies']:
                    removed[rec['ion_type']] += 1
                    continue
                overlap += rec['smiles'] in targets
                types[f"{rec['ion_type']}:{rec['split']}"] += 1
                charges[f"{rec['ion_type']}:{rec['formal_charge']}"] += 1
                records.append(rec)
                (cat if rec['ion_type']=='cation' else ani).write(rec['smiles']+'\n')
                if rec['split']=='train':
                    token_counts['smiles'].update(tokenize_smiles(rec['smiles']))
                    token_counts['selfies'].update(sf.split_selfies(rec['selfies']))
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
            if sum(types.values()) % 100_000 < 5000:
                print(json.dumps({'stage':'export','retained':sum(types.values())}),flush=True)
    keep = (target.smiles_tokens+3 <= maximum['smiles']) & (target.selfies_tokens+3 <= maximum['selfies'])
    target_final=target[keep].copy()
    target_final.to_parquet(out / 'target_ions.parquet', index=False)
    target_final.to_csv(out / 'target_ions.csv', index=False)
    for kind in ('cation','anion'):
        (out / f'target_{kind}s.smi').write_text('\n'.join(target_final.loc[target_final.ion_type==kind,'smiles'])+'\n', encoding='utf-8')
    oov = {}
    for language, counts in token_counts.items():
        vocab={t:i for i,t in enumerate(SPECIAL_TOKENS+sorted(counts))}
        save_json(out / f'vocab_{language}.json', {'language': language, 'vocab': vocab})
        pd.DataFrame(counts.most_common(),columns=['token','count']).to_csv(out / f'token_frequencies_{language}.csv',index=False)
        oov_count=0; total=0
        tok = tokenize_smiles if language=='smiles' else sf.split_selfies
        for value in target_final[language]:
            tokens=list(tok(value)); total += len(tokens); oov_count += sum(t not in vocab for t in tokens)
        oov[language]={'target_oov_tokens':oov_count,'target_tokens':total,'target_oov_rate':oov_count/max(total,1),'vocabulary_size':len(vocab)}
    assert overlap == 0
    for kind in ('cation','anion'):
        sets=[set(target_final.loc[(target_final.ion_type==kind)&(target_final.split==s),'scaffold']) for s in ('train','validation','test')]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return {'maximum_tokens':maximum,'pretrain_counts':dict(types),'pretrain_charge_counts':dict(charges),
            'pretrain_length_removed':dict(removed),'target_counts':{f'{k}:{s}':int(n) for (k,s),n in target_final.groupby(['ion_type','split']).size().items()},
            'target_length_removed':target[~keep].groupby('ion_type').size().to_dict(),
            'pretraining_target_overlap':int(overlap),'vocabulary_statistics':oov}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    init_worker()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'DATASET_COMPLETED.json').exists():
        print((OUT/'DATASET_COMPLETED.json').read_text(encoding='utf-8')); return
    targets,target=property_ions(OUT)
    scan_info=scan(ROOT/'CID-SMILES.gz',OUT)
    clean_info=clean(OUT,args.workers)
    representation_info=verified_corpus(OUT,args.workers,targets)
    training_info=export_training(OUT,target,targets)
    manifest={'date':'2026-10-08','rdkit_version':Chem.rdBase.rdkitVersion,'selfies_version':sf.__version__,
              'rules':{'positive_ions':'net formal charge > 0; contains carbon; organometallics allowed',
                       'negative_ions':'net formal charge < 0; no carbon restriction',
                       'mixed_charge_signs':'allowed','multicomponent_pubchem':'excluded',
                       'safe_early_length_filter':'more than 253 non-hydrogen atom tokens cannot satisfy the 256-token cap including 3 control tokens',
                       'standardization':'RDKit sanitized canonical isomeric SMILES; no neutralization, tautomer replacement, or metal disconnection',
                       'length':'both languages synchronized; 99th-percentile pretrain training lengths including 3 controls rounded up to multiples of 8; cap 256',
                       'pretrain_split':'SHA256 ion-type/canonical-SMILES 98/1/1',
                       'target_split':'SHA256 ion-type/scaffold 80/10/10; acyclic scaffold falls back to full canonical SMILES'},
              'scan':scan_info,'standardize':clean_info,'representations':representation_info,
              'properties':json.loads((OUT/'property_audit.json').read_text(encoding='utf-8')),**training_info}
    manifest['output_sha256']={name:sha256(OUT/name) for name in ['pretrain_ions.parquet','target_ions.parquet','target_ions_all_standardized.csv','vocab_smiles.json','vocab_selfies.json']}
    save_json(OUT/'DATASET_COMPLETED.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    mp.freeze_support()
    main()
