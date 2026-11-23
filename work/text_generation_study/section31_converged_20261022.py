"""Convergence-oriented from-scratch benchmark on the October ion corpus.

Differences from section31_20261022.py:
- supervised models train up to 200 epochs with early stopping (patience 20);
- SeqGAN first trains its MLE phase with early stopping, then performs
  10 adversarial epochs from the best MLE checkpoint;
- outputs are written to a separate directory so the previous benchmark
  remains available for comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from molecular_tokenizer import MolecularTokenizer, SPECIAL_TOKENS
from models import create_model, AutoregressiveTransformer, TransformerVAE, SeqGAN
from train_generator import IonDataset, LengthBucketBatchSampler, trim_padding_collate, run_epoch, generate, set_seed
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/PubChem_dataset_revision_20261008'
OUT=ROOT/'outputs/IECR_section31_converged_20261022'
SEEDS=[11,23,37,41,59]
METRICS=['structural_validity','charge_validity','uniqueness','novelty','structural_diversity']
MAX_EPOCHS=200
PATIENCE=20
ADVERSARIAL_EPOCHS=10
ADVERSARIAL_LR=1e-4


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=True),encoding='utf-8')
    tmp.replace(path)


def prepare(refresh=False):
    data=OUT/'data';data.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SOURCE/'DATASET_COMPLETED.json').read_text())
    target=pd.read_parquet(SOURCE/'target_ions.parquet',engine='fastparquet')
    if not refresh and (data/'protocol.json').exists():
        protocol=json.loads((data/'protocol.json').read_text())
        return target,protocol['config'],manifest
    audit={}
    for language in ['smiles','selfies']:
        tok=MolecularTokenizer.load(SOURCE/f'vocab_{language}.json')
        training=target[target.split=='train']
        missing=sorted({t for text in training[language] for t in tok.tokenize(text) if t not in tok.vocab})
        vocab=dict(tok.vocab)
        for token in missing:vocab[token]=len(vocab)
        new=MolecularTokenizer(language,vocab);new.save(data/f'vocab_{language}.json')
        audit[language]={'base_size':len(tok.vocab),'training_only_additions':missing,'size':len(vocab),'splits':{}}
        for split,frame in target.groupby('split'):
            tokens=[t for text in frame[language] for t in new.tokenize(text)]
            audit[language]['splits'][split]={'rows':len(frame),'tokens':len(tokens),'oov':sum(t not in vocab for t in tokens)}
        assert audit[language]['splits']['train']['oov']==0
    config=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    config['output_root']=str(OUT)
    config['data']={'source_manifest':str(SOURCE/'DATASET_COMPLETED.json'),
                    'target_ions':str(SOURCE/'target_ions.parquet'),
                    'pretraining_weights_loaded':False,'maximum_length':168}
    config['training']['num_workers']=0
    config['training']['finetune_epochs']=MAX_EPOCHS
    config['training']['patience']=PATIENCE
    config['training']['seqgan_adversarial_epochs']=ADVERSARIAL_EPOCHS
    config['training']['seqgan_adversarial_lr']=ADVERSARIAL_LR
    dump(data/'protocol.json',{'dataset':manifest,'tokenizers':audit,'config':config,
        'charge_rule':'positive for cation request; negative for anion request',
        'reference':'all 2659 retained target ions; training/validation/test combined',
        'structural_validity':'RDKit sanitizable nonempty molecule / all attempts',
        'charge_validity':'correct net charge sign / structurally valid attempts',
        'uniqueness':'canonical unique structurally valid molecules / structurally valid attempts',
        'novelty':'unique valid molecules absent from all target ions / unique valid molecules',
        'diversity':'mean pairwise Morgan radius2 2048-bit Tanimoto distance on at most 1000 unique structurally valid molecules; seed17',
        'sampling':'5000 cation plus 5000 anion requests; temperature1; topk32; final sampling seed=2000000+training seed',
        'single_component':'reported separately, not part of primary structural validity',
        'early_stopping':{'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
                          'monitor':'validation reconstruction for VAE; validation total for diffusion and autoregressive Transformer',
                          'seqgan':'MLE phase uses early stopping; 10 adversarial epochs follow from the best MLE checkpoint at a fixed learning rate'},
        'selection':'checkpoints use held-out validation reconstruction loss for VAE, denoising loss for diffusion, CE for AR; SeqGAN uses the terminal adversarial checkpoint; no selection from final generated metrics'})
    target.to_csv(data/'target_ions.csv',index=False)
    return target,config,manifest


@torch.no_grad()
def sample_autoregressive(self,prefix,eos_id,max_length,temperature=1.,top_k=32,conditions=None):
    ids=prefix
    finished=torch.zeros(len(ids),dtype=torch.bool,device=ids.device)
    latent=torch.randn(len(ids),self.latent_dim,device=ids.device) if isinstance(self,TransformerVAE) else None
    while ids.size(1)<max_length and not finished.all():
        active=(~finished).nonzero(as_tuple=False).squeeze(1)
        sub_ids=ids[active]
        sub_latent=latent[active] if latent is not None else None
        sub_cond=conditions[active] if conditions is not None else None
        logits=(self.decode_logits(sub_ids,sub_latent,sub_cond) if sub_latent is not None else self.logits(sub_ids,sub_cond))[:,-1]/temperature
        forbidden=[i for i in range(len(SPECIAL_TOKENS)) if i!=eos_id]
        logits[:,forbidden]=torch.finfo(logits.dtype).min
        values,indices=logits.topk(min(top_k,logits.size(-1)-len(forbidden)),dim=-1)
        nxt=indices.gather(1,torch.multinomial(torch.softmax(values,-1),1)).squeeze(1)
        new_col=torch.full((len(ids),1),self.pad_id,dtype=ids.dtype,device=ids.device)
        new_col[active,0]=nxt
        ids=torch.cat([ids,new_col],dim=1)
        finished[active]|=nxt.eq(eos_id)
    return ids


def build_loaders(target,tok,language,config,seed):
    tc=config['training']
    frames=[target[target.split==s] for s in ['train','validation']]
    loaders=[]
    for i,frame in enumerate(frames):
        sampler=LengthBucketBatchSampler(frame[f'{language}_tokens']+3,tc['batch_size'],i==0,seed)
        loaders.append(DataLoader(IonDataset(frame,tok,language,168),batch_sampler=sampler,
                                  collate_fn=trim_padding_collate,num_workers=0,pin_memory=True))
    return loaders


def save_training_state(path,model,opt,sched,scaler,epoch,best,stale,logs,extra=None):
    state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
           'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'stale':stale,'logs':logs,
           'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    if isinstance(model,SeqGAN) and hasattr(model,'_optimizer_d'):
        state['optimizer_d']=model._optimizer_d.state_dict()
    if extra:state.update(extra)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(state,temp);temp.replace(path)


def training_seqgan(model,loaders,opt,sched,scaler,device,tc,epochs,patience,tok,max_length,directory,seed):
    # Stage 1: maximum-likelihood warm-up with early stopping.
    mle_done=directory/'MLE_TRAINED.json'
    if mle_done.exists():
        state=torch.load(directory/'mle_best.pt',weights_only=False,map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        mle_meta=json.loads(mle_done.read_text())
        mle_logs=pd.read_csv(directory/'mle_loss.csv').to_dict('records')
    else:
        latest=directory/'mle_latest.pt';best_path=directory/'mle_best.pt'
        start=1;best=float('inf');stale=0;mle_logs=[]
        if latest.exists():
            state=torch.load(latest,weights_only=False,map_location='cpu')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            start=state['epoch']+1;best=state['best'];stale=state['stale'];mle_logs=state['logs']
            loaders[0].batch_sampler.epoch=start-1
        for epoch in range(start,epochs+1):
            begin=time.time()
            train=run_epoch(model,loaders[0],opt,scaler,device,tc['grad_clip'],True,epoch,epochs,tok,max_length,'mle')
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                torch.manual_seed(7919);torch.cuda.manual_seed_all(7919)
                val=run_epoch(model,loaders[1],opt,scaler,device,tc['grad_clip'],False,epoch,epochs,tok,max_length,'mle')
            current=val['total']
            if not np.isfinite(current):raise RuntimeError('Non-finite validation loss')
            improved=current<best
            best=min(best,current);stale=0 if improved else stale+1
            row={'epoch':epoch,'stage':'mle','seconds':time.time()-begin,'lr':opt.param_groups[0]['lr'],
                 **{'train_'+k:v for k,v in train.items()},**{'validation_'+k:v for k,v in val.items()}}
            mle_logs.append(row);sched.step()
            save_training_state(latest,model,opt,sched,scaler,epoch,best,stale,mle_logs)
            if improved:shutil.copy2(latest,best_path)
            pd.DataFrame(mle_logs).to_csv(directory/'loss.csv',index=False)
            if epoch%10==0:print(json.dumps({'run':str(directory.relative_to(OUT)),'stage':'mle','epoch':epoch,'val':current}),flush=True)
            if stale>=patience:break
        state=torch.load(best_path,weights_only=False,map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        mle_meta={'selected':'mle_best.pt','selected_epoch':state['epoch'],'last_epoch':mle_logs[-1]['epoch'],
                  'monitor':'total','validation_best':best,'early_stopped':mle_logs[-1]['epoch']<epochs,
                  'patience':patience,'max_epochs':epochs}
        dump(mle_done,mle_meta)
    # Stage 2: fixed adversarial phase from the best MLE checkpoint.
    adv_done=directory/'ADV_TRAINED.json'
    latest=directory/'latest.pt'
    if adv_done.exists():
        state=torch.load(latest,weights_only=False,map_location=device)
        model.load_state_dict(state['model'])
        adv_meta=json.loads(adv_done.read_text())
        adv_logs=pd.read_csv(directory/'adv_loss.csv').to_dict('records')
    else:
        for group in opt.param_groups:group['lr']=ADVERSARIAL_LR
        model._optimizer_d=torch.optim.AdamW(model.discriminator.parameters(),lr=ADVERSARIAL_LR,weight_decay=tc['weight_decay'])
        adv_logs=[]
        for epoch in range(1,ADVERSARIAL_EPOCHS+1):
            begin=time.time()
            train=run_epoch(model,loaders[0],opt,scaler,device,tc['grad_clip'],True,epoch,ADVERSARIAL_EPOCHS,tok,max_length,'adversarial')
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                torch.manual_seed(7919);torch.cuda.manual_seed_all(7919)
                val=run_epoch(model,loaders[1],opt,scaler,device,tc['grad_clip'],False,epoch,ADVERSARIAL_EPOCHS,tok,max_length,'adversarial')
            row={'epoch':mle_meta['last_epoch']+epoch,'stage':'adversarial','seconds':time.time()-begin,'lr':ADVERSARIAL_LR,
                 **{'train_'+k:v for k,v in train.items()},**{'validation_'+k:v for k,v in val.items()}}
            adv_logs.append(row)
            save_training_state(latest,model,opt,sched,scaler,row['epoch'],mle_meta['validation_best'],0,adv_logs)
            pd.DataFrame(adv_logs).to_csv(directory/'adv_loss.csv',index=False)
            if epoch%5==0:print(json.dumps({'run':str(directory.relative_to(OUT)),'stage':'adversarial','epoch':epoch}),flush=True)
        adv_meta={'adversarial_epochs':ADVERSARIAL_EPOCHS,'adversarial_lr':ADVERSARIAL_LR}
        dump(adv_done,adv_meta)
    combined=list(mle_logs)+list(adv_logs)
    pd.DataFrame(combined).to_csv(directory/'loss.csv',index=False)
    result={'selected':'latest.pt','selected_epoch':combined[-1]['epoch'],'last_epoch':combined[-1]['epoch'],
            'monitor':'total','validation_best':mle_meta['validation_best'],
            'early_stopped':mle_meta['early_stopped'],'patience':patience,'max_epochs':epochs,
            'mle_epochs':mle_meta['last_epoch'],'adversarial_epochs':ADVERSARIAL_EPOCHS,
            'adversarial_lr':ADVERSARIAL_LR}
    dump(directory/'TRAINED.json',result)
    return result


def training(model,target,tok,language,config,directory,seed):
    device=torch.device('cuda');max_length=168
    tc=config['training'];epochs=tc['finetune_epochs'];patience=tc['patience']
    loaders=build_loaders(target,tok,language,config,seed)
    opt=torch.optim.AdamW([p for n,p in model.named_parameters() if 'discriminator' not in n],lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    if isinstance(model,SeqGAN):
        model._optimizer_d=torch.optim.AdamW(model.discriminator.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    done=directory/'TRAINED.json'
    if done.exists():
        meta=json.loads(done.read_text());model.load_state_dict(torch.load(directory/meta['selected'],weights_only=False,map_location=device)['model']);return meta
    if isinstance(model,SeqGAN):
        return training_seqgan(model,loaders,opt,sched,scaler,device,tc,epochs,patience,tok,max_length,directory,seed)
    latest=directory/'latest.pt';best_path=directory/'best.pt'
    start=1;best=float('inf');stale=0;logs=[]
    if latest.exists():
        state=torch.load(latest,weights_only=False,map_location='cpu')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['epoch']+1;best=state['best'];stale=state['stale'];logs=state['logs']
        loaders[0].batch_sampler.epoch=start-1
    for epoch in range(start,epochs+1):
        begin=time.time()
        train=run_epoch(model,loaders[0],opt,scaler,device,tc['grad_clip'],True,epoch,epochs,tok,max_length,'finetune')
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(7919);torch.cuda.manual_seed_all(7919)
            val=run_epoch(model,loaders[1],opt,scaler,device,tc['grad_clip'],False,epoch,epochs,tok,max_length,'finetune')
        monitor='reconstruction' if isinstance(model,TransformerVAE) else 'total'
        current=val[monitor]
        if not np.isfinite(current):raise RuntimeError('Non-finite validation loss')
        improved=current<best
        best=min(best,current);stale=0 if improved else stale+1
        row={'epoch':epoch,'stage':'finetune','seconds':time.time()-begin,'lr':opt.param_groups[0]['lr'],
             **{'train_'+k:v for k,v in train.items()},**{'validation_'+k:v for k,v in val.items()}}
        logs.append(row);sched.step()
        save_training_state(latest,model,opt,sched,scaler,epoch,best,stale,logs)
        if improved:shutil.copy2(latest,best_path)
        pd.DataFrame(logs).to_csv(directory/'loss.csv',index=False)
        if epoch%10==0:print(json.dumps({'run':str(directory.relative_to(OUT)),'epoch':epoch,'val':current}),flush=True)
        if stale>=patience:break
    selected='best.pt'
    state=torch.load(directory/selected,weights_only=False,map_location=device);model.load_state_dict(state['model'])
    result={'selected':selected,'selected_epoch':state['epoch'],'last_epoch':logs[-1]['epoch'],
            'monitor':monitor,'validation_best':best,'early_stopped':logs[-1]['epoch']<epochs,
            'patience':patience,'max_epochs':epochs}
    dump(done,result);return result


def diversity(values, fallback_sources=None):
    if len(values)<2:return float('nan')
    chosen=sorted(values)
    if len(chosen)>1000:chosen=np.random.default_rng(17).choice(chosen,1000,replace=False).tolist()
    generator=AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[]
    for s in chosen:
        mol=Chem.MolFromSmiles(s)
        if mol is None and fallback_sources is not None:
            mol=Chem.MolFromSmiles(fallback_sources[s])
        if mol is None:raise ValueError('Cannot recover a structurally validated molecule')
        fps.append(generator.GetFingerprint(mol))
    total=0.;n=0
    for i in range(1,len(fps)):
        total+=sum(DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]));n+=i
    return 1-total/n


def evaluate(generated,target,language,model,seed,directory):
    RDLogger.DisableLog('rdApp.*');ref=set(target.smiles)
    cache={};records=[];fallback_sources={}
    for r in generated.fillna('').itertuples():
        text=r.generated_smiles
        if text not in cache:
            mol=Chem.MolFromSmiles(text)
            valid=mol is not None and mol.GetNumAtoms()>0
            cache[text]=(valid,Chem.MolToSmiles(mol,isomericSmiles=True) if valid else '',Chem.GetFormalCharge(mol) if valid else 0,len(Chem.GetMolFrags(mol)) if valid else 0)
        valid,can,q,components=cache[text]
        if valid:fallback_sources.setdefault(can,text)
        correct=valid and (q>0 if r.ion_type=='cation' else q<0)
        records.append({'ion_type':r.ion_type,'canonical':can,'structural_valid':valid,'charge_valid':correct,'charge':q,'components':components})
    scored=pd.DataFrame(records);scored.to_csv(directory/'generated_scored.csv',index=False)
    rows=[]
    for kind in ['cation','anion','pooled']:
        group=scored if kind=='pooled' else scored[scored.ion_type==kind]
        good=group[group.structural_valid];unique=set(good.canonical);novel=unique-ref
        divide=lambda x,y:x/y if y else float('nan')
        rows.append({'language':language,'model':model,'seed':seed,'ion_type':kind,'n_generated':len(group),
                     'n_structural_valid':len(good),'n_charge_valid':int(group.charge_valid.sum()),'n_unique':len(unique),'n_novel':len(novel),
                     'structural_validity':len(good)/len(group),'charge_validity':divide(group.charge_valid.sum(),len(good)),
                     'uniqueness':divide(len(unique),len(good)),'novelty':divide(len(novel),len(unique)),
                     'structural_diversity':diversity(unique,fallback_sources),'single_component_rate':divide((good.components==1).sum(),len(good)),
                     'joint_ion_validity':((group.components==1)&group.charge_valid).mean()})
    dump(directory/'metrics.json',rows);return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);parser.add_argument('--model');parser.add_argument('--language');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(),'QYX CUDA is required'
    target,config,manifest=prepare()
    AutoregressiveTransformer.sample=sample_autoregressive;TransformerVAE.sample=sample_autoregressive
    for seed in ([args.seed] if args.seed else SEEDS):
        for language in ([args.language] if args.language else ['smiles','selfies']):
            for name in ([args.model] if args.model else ['vae','diffusion','seqgan','transformer']):
                directory=OUT/'runs'/language/name/str(seed);directory.mkdir(parents=True,exist_ok=True)
                if (directory/'COMPLETED.json').exists():continue
                resumed=(directory/'latest.pt').exists() or (directory/'mle_latest.pt').exists()
                set_seed(seed);torch.cuda.reset_peak_memory_stats();begin=time.time()
                tok=MolecularTokenizer.load(OUT/'data'/f'vocab_{language}.json')
                model=create_model(name,len(tok.vocab),168,tok,config['model']).cuda()
                if name=='diffusion':model.protected_ids=tuple(model.protected_ids)+(tok.unk_id,)
                meta=training(model,target,tok,language,config,directory,seed)
                torch.save(model.state_dict(),directory/'final_model.pt')
                set_seed(2000000+seed)
                train=target[target.split=='train']
                lengths={k:train.loc[train.ion_type==k,f'{language}_tokens'].tolist() for k in ['cation','anion']}
                if (directory/'generated.csv').exists():
                    generated=pd.read_csv(directory/'generated.csv',keep_default_na=False)
                else:
                    torch.cuda.empty_cache()
                    generated=generate(model,tok,168,10000,torch.device('cuda'),lengths,batch_size=512)
                    generated.to_csv(directory/'generated.csv',index=False)
                evaluate(generated,target,language,name,seed,directory)
                meta.update(language=language,model=name,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                            seconds=time.time()-begin,peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                            device=torch.cuda.get_device_name(),vocab_size=len(tok.vocab),maximum_length=168,
                            runtime_scope='resumed_segment' if resumed else 'full_run',
                            dataset_sha256=manifest['output_sha256']['target_ions.parquet'])
                dump(directory/'COMPLETED.json',meta)
                print(json.dumps({'COMPLETED':meta}),flush=True)
                del model;torch.cuda.empty_cache()
    if args.seed is None:
        paths=list((OUT/'runs').rglob('metrics.json'))
        data=pd.DataFrame([row for p in paths for row in json.loads(p.read_text())])
        (OUT/'tables').mkdir(exist_ok=True);data.to_csv(OUT/'tables/metrics_raw.csv',index=False)
        if len(paths)==40:dump(OUT/'ALL_RUNS_COMPLETED.json',{'runs':40,'seeds':SEEDS,'dataset':str(SOURCE)})


if __name__=='__main__':main()
