"""Export a single long table with source row IDs, units and unchanged observations."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    parts=[];summary=[]
    specs=[('melting point.csv','melting_point','MP_K','K','IL_SMILES',False),
           ('viscosity.csv','viscosity','n_mPas','mPa s','IL SMILES',False),
           ('toxicity.csv','toxicity_log10_EC50','logEC50','log10(EC50)','IL_SMILES',False),
           ('SO2 solubility.csv','SO2_solubility','ln(x_SO2)','mole fraction','IL_SMILES',True),
           ('CO2 solubility.csv','CO2_solubility','ln(x_CO2)','mole fraction','IL_SMILES',True)]
    for filename,prop,col,unit,smiles,log in specs:
        f=pd.read_csv(ROOT/filename)
        raw=pd.to_numeric(f[col],errors='coerce')
        out=pd.DataFrame({'source_file':filename,'source_row_1based':np.arange(len(f))+2,
            'IL_SMILES':f[smiles],'property':prop,'value':np.exp(raw) if log else raw,'unit':unit,
            'original_value':raw,'original_column':col})
        out['temperature_K']=pd.to_numeric(f['T/K'],errors='coerce') if 'T/K' in f else np.nan
        out['pressure_bar']=pd.to_numeric(f['P/bar'],errors='coerce') if 'P/bar' in f else np.nan
        parts.append(out)
        summary.append(dict(file=filename,observations=len(f),distinct_IL_SMILES=f[smiles].nunique(),columns=' | '.join(f.columns)))
    target=ROOT/'data/processed'
    target.mkdir(exist_ok=True)
    pd.concat(parts,ignore_index=True).to_csv(target/'experimental_properties_long.csv',index=False)
    pd.DataFrame(summary).to_csv(ROOT/'data/database_summary.csv',index=False)
    print('Experimental observations exported without deduplication or condition filtering.')

if __name__=='__main__':main()
