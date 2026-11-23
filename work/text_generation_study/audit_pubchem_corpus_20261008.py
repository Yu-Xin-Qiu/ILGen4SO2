import csv
import json
from pathlib import Path
import sqlite3

import pandas as pd
import fastparquet as fp

from rebuild_pubchem_20261008 import OUT, init_worker, representation, save_json


def main():
    init_worker()
    m=json.loads((OUT/'DATASET_COMPLETED.json').read_text(encoding='utf-8'))
    source_counts=m['scan']['counts']
    assert source_counts['total_records']==source_counts['charge_token_records']+source_counts['no_atomic_charge_token']+source_counts.get('malformed_records',0)
    c=m['standardize']['counts']
    assert c['processed']==source_counts['charge_token_records']
    assert c['processed']==sum(v for k,v in c.items() if k!='processed')
    r=m['representations']['counts']
    assert r['processed']==c['unique_standardized']
    assert r['processed']==sum(v for k,v in r.items() if k!='processed')
    db=sqlite3.connect(OUT/'canonical_ions.sqlite')
    db.execute('DROP TABLE IF EXISTS verified_pretrain')
    db.execute('CREATE TABLE verified_pretrain (smiles TEXT PRIMARY KEY)')
    verified=OUT/'pretrain_ions_before_length.parquet'
    for batch in fp.ParquetFile(verified).iter_row_groups(columns=['smiles']):
        db.executemany('INSERT INTO verified_pretrain VALUES (?)',[(s,) for s in batch.smiles])
    db.commit()
    excluded=[]
    q='SELECT i.cid,i.ion_type,i.smiles,i.charge FROM ions i LEFT JOIN verified_pretrain v ON i.smiles=v.smiles WHERE v.smiles IS NULL'
    raw_target=pd.read_csv(OUT/'target_ions_all_standardized.csv')
    target_set=set(raw_target.smiles)
    resource_ids=set()
    if (OUT/'resource_exceptions.jsonl').exists():
        for line in (OUT/'resource_exceptions.jsonl').read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e['stage']=='representations':resource_ids.add(e['cid'])
    for cid,kind,smiles,charge in db.execute(q):
        if cid in resource_ids:
            reason='resource_exception'
        else:
            rec,reason=representation((cid,kind,smiles,charge))
            if rec is not None:
                assert smiles in target_set
                reason='downstream_overlap_removed'
        excluded.append(dict(cid=cid,ion_type=kind,smiles=smiles,formal_charge=charge,reason=reason))
    db.close()
    exclusion=pd.DataFrame(excluded)
    exclusion.to_csv(OUT/'representation_exclusions.csv',index=False)
    exclusion[exclusion.reason!='downstream_overlap_removed'].to_csv(OUT/'selfies_failure_records.csv',index=False)
    target_failed=[]
    for row in raw_target.itertuples():
        rec,reason=representation((0,row.ion_type,row.smiles,row.formal_charge))
        if rec is None:
            target_failed.append(dict(ion_type=row.ion_type,smiles=row.smiles,formal_charge=row.formal_charge,reason=reason))
    pd.DataFrame(target_failed).to_csv(OUT/'target_selfies_failure_records.csv',index=False)
    overlap=0;length_failed=[];counts={}
    for f in fp.ParquetFile(OUT/'pretrain_ions.parquet').iter_row_groups():
        overlap+=int(f.smiles.isin(target_set).sum())
        assert f.smiles.is_unique
        assert (((f.ion_type=='cation')&(f.formal_charge>0))|((f.ion_type=='anion')&(f.formal_charge<0))).all()
        assert (f.smiles_tokens+3<=m['maximum_tokens']['smiles']).all()
        assert (f.selfies_tokens+3<=m['maximum_tokens']['selfies']).all()
        for (kind,split),n in f.groupby(['ion_type','split']).size().items():
            key=f'{kind}:{split}';counts[key]=counts.get(key,0)+int(n)
    assert counts==m['pretrain_counts']
    assert overlap==0
    for f in fp.ParquetFile(verified).iter_row_groups():
        failed=(f.smiles_tokens+3>m['maximum_tokens']['smiles'])|(f.selfies_tokens+3>m['maximum_tokens']['selfies'])
        length_failed.extend(f.loc[failed,['cid','ion_type','smiles','smiles_tokens','selfies_tokens']].to_dict('records'))
    pd.DataFrame(length_failed).to_csv(OUT/'pretrain_length_exclusions.csv',index=False)
    t=pd.read_parquet(OUT/'target_ions.parquet',engine='fastparquet')
    assert t.smiles.is_unique
    assert t.smiles_tokens.add(3).le(m['maximum_tokens']['smiles']).all()
    assert t.selfies_tokens.add(3).le(m['maximum_tokens']['selfies']).all()
    charge_rows=[]
    for (kind,charge),n in t.groupby(['ion_type','formal_charge']).size().items():
        charge_rows.append({'corpus':'fine_tuning','ion_type':kind,'formal_charge':int(charge),'count':int(n)})
    for key,n in m['pretrain_charge_counts'].items():
        kind,charge=key.split(':');charge_rows.append({'corpus':'pretraining','ion_type':kind,'formal_charge':int(charge),'count':n})
    pd.DataFrame(charge_rows).to_csv(OUT/'formal_charge_distributions.csv',index=False)
    qa={'source_and_stage_count_reconciliation':'passed','all_pretrain_rows_charge_sign_checked':True,
        'all_pretrain_and_target_rows_within_both_length_limits':True,
        'exact_overlap_with_all_standardized_property_ions':overlap,
        'representation_failure_records':int((exclusion.reason!='downstream_overlap_removed').sum()),
        'target_representation_failure_records':len(target_failed),
        'pretrain_length_exclusion_records':len(length_failed),'pretrain_partition_counts':counts}
    save_json(OUT/'CORPUS_QA_COMPLETED.json',qa)
    print(json.dumps(qa,indent=2),flush=True)


if __name__=='__main__':main()
