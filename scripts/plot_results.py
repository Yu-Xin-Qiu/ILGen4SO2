"""Plot saved point-prediction summaries and original density curves at 600 dpi."""
from pathlib import Path
import argparse,io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from export_source_data import load_tables
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'reproduced/figures');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    tables=load_tables()
    def read(key):return pd.read_csv(io.StringIO(tables[key]))
    def save(fig,name):fig.savefig(a.out/name,dpi=600,bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':12,'axes.labelweight':'bold','axes.spines.top':False,'axes.spines.right':False})
    f=pd.read_csv(ROOT/'data/generated/feedback_metrics.csv');f=f.loc[f.arm.eq('pooled')&f['round'].between(0,9)]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(key,label) in zip(axes.flat,[('structural_validity','Structural validity'),('uniqueness','Uniqueness'),('training_novelty','Novelty'),('structural_diversity','Diversity'),('hard_pass_rate','CSR'),('R_pass_rate','R feedback eligibility')]):
        s=f.groupby('round')[key].agg(['mean','std'])
        ax.errorbar(s.index,s['mean'],yerr=s['std'],fmt='o-',capsize=3);ax.set(xlabel='Round',ylabel=label,xticks=range(10))
    save(fig,'feedback_metrics.png')
    for key,label in [('c_AC','AC'),('d_DE','DE'),('e_S','S'),('f_SAmax',r'SA$_{\max}$'),('extra_R','R')]:
        df=read(f'analysis_export/Figure06__{key}_KDE.csv')
        fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
        xcol=df.columns[0]
        for i,col in enumerate(df.columns[1:]):ax.plot(df[xcol],df[col],color=plt.cm.Blues(.3+.65*i/max(1,len(df.columns)-2)),label=col)
        ax.set(xlabel=label,ylabel='Probability density');ax.legend(fontsize=7,ncol=2)
        save(fig,f'rounds_{key}.png')
    specs=[('c_Tm_curve',r'$T_m$ (K)',298.15),('d_eta_curve',r'$\eta$ (mPa s)',200),('e_toxicity_curve',r'$\log_{10}EC_{50}$',2),('f_SO2_curve',r'SO$_2$ solubility',None),('g_CO2_curve',r'CO$_2$ solubility',None),('h_SAmax_curve',r'SA$_{\max}$',None)]
    for name,label,cut in specs:
        df=read(f'feasible_population/Figure08__{name}.csv');fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
        for pop,c in [('Experimental','#637788'),('Generated','#008A98')]:
            y=df[pop+'_Relative_frequency_pct'];ax.plot(df.X,y,color=c,label=pop);ax.fill_between(df.X,0,y,color=c,alpha=.1)
        if cut is not None:ax.axvline(cut,color='gray',ls='--')
        ax.set(xlabel=label,ylabel='Relative frequency (%)',ylim=(0,None));ax.legend(fontsize=9)
        save(fig,f'population_{name}.png')
    f=pd.read_csv(ROOT/'data/processed/top4.csv');mols=[];legends=[]
    for r in f.itertuples():
        for ion in ['cation','anion']:
            mols.append(Chem.MolFromSmiles(getattr(r,ion+'_smiles')));legends.append(f'{r.ID} {ion}  R={r.R:.3f}')
    (a.out/'top4.svg').write_text(Draw.MolsToGridImage(mols,molsPerRow=2,subImgSize=(500,250),legends=legends,useSVG=True),encoding='utf-8')
    Draw.MolsToGridImage(mols,molsPerRow=2,subImgSize=(1500,750),legends=legends).save(a.out/'top4.png',dpi=(600,600))
    print(a.out)
if __name__=='__main__':main()
