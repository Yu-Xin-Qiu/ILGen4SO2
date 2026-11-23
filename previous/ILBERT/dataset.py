from __future__ import print_function, division
import functools
import  numpy  as  np
import torch
from torch.utils.data import Dataset



class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, tokenizer, target):
        self.target = target
        self.smiles = df['Normalized SMILES']
        self.tokens = np.array(
            [tokenizer.encode(i, max_length=100, truncation=True, padding='max_length') for i in self.smiles])
        # print(self.tokens)
        self.label = df[target]
        self.tokenizer = tokenizer

        if self.target == 'lnhc' or self.target == 'lnEC' or self.target == 'ln(n_mPas)' or self.target == 's_mNm' or self.target == 'R':
            self.t = df['T/K']


        elif self.target == 'x_CO2' or self.target == 'TC/W m-1 K-1' or self.target == 'd_kg m-3':
            self.t = df['T/K']
            self.p = df['P/bar']


    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):

        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        smiles=self.smiles[index]
        t,p=0,0

        if self.target == 'lnhc' or self.target == 'lnEC' or self.target == 'ln(n_mPas)' or self.target == 's_mNm' or self.target == 'R':

            t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)

            return (X,t), y.float(),smiles,t,p
        

        elif self.target == 'x_CO2' or self.target == 'TC/W m-1 K-1' or self.target == 'd_kg m-3':

            t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
            p = torch.from_numpy(np.asarray(self.p[index])).view(-1, 1)

            return (X,t,p), y.float(),smiles,t,p
        
        else:
            return (X,y), y.float(),smiles,t,p


