from __future__ import print_function, division
import functools
import  numpy  as  np
import torch
from torch.utils.data import Dataset



from tqdm import tqdm
import numpy as np
import torch
import functools
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer, target):
        self.target = target
        self.smiles = df['IL_SMILES']

        # Use joblib for parallel tokenization with tqdm for progress bar
        self.tokens = np.array(
            list(Parallel(n_jobs=3)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles, desc='Tokenizing SMILES', total=len(self.smiles))
            ))
        )

        self.label = df[target]
        self.tokenizer = tokenizer
        self.t = df['T/K']
        self.p = df['P/bar']

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        smiles = self.smiles[index]
        t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
        p = torch.from_numpy(np.asarray(self.p[index])).view(-1, 1)
        return (X, t, p), y.float(), smiles, t, p
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import functools
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

class SMILES_dataset_MTL(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer, target):
        self.target = target
        self.smiles = df['IL_SMILES']
        self.gas_index = df['gas_index']

        # One-hot encoding for gas_index
        self.encoder = OneHotEncoder(categories=[range(1, 11)])  # sparse=True for older versions
        self.gas_index_one_hot = self.encoder.fit_transform(self.gas_index.values.reshape(-1, 1)).toarray()  # Convert to dense array

        # Use joblib for parallel tokenization with tqdm for progress bar
        self.tokens = np.array(
            list(Parallel(n_jobs=10)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles, desc='Tokenizing SMILES', total=len(self.smiles))
            ))
        )

        self.label = df[target]
        self.tokenizer = tokenizer
        self.t = df['T/K']
        self.p = df['P/bar']

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Convert gas_index to one-hot encoded vector
        gas_index = torch.from_numpy(np.asarray(self.gas_index_one_hot[index])).float()

        # Tokenize and prepare other inputs
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        smiles = self.smiles[index]
        t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
        p = torch.from_numpy(np.asarray(self.p[index])).view(-1, 1)

        return (X, t, p), y.float(), smiles, t, p, gas_index


# class SMILES_dataset_MTL(Dataset):
#     'Characterizes a dataset for PyTorch'

#     def __init__(self, df, tokenizer, target):
#         self.target = target
#         self.smiles = df['IL_SMILES']
#         self.gas_index= df['gas_index']

#         # Use joblib for parallel tokenization with tqdm for progress bar
#         self.tokens = np.array(
#             list(Parallel(n_jobs=10)(
#                 delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
#                 for i in tqdm(self.smiles, desc='Tokenizing SMILES', total=len(self.smiles))
#             ))
#         )

#         self.label = df[target]
#         self.tokenizer = tokenizer
#         self.t = df['T/K']
#         self.p = df['P/bar']

#     def __len__(self):
#         return len(self.label)

#     @functools.lru_cache(maxsize=None)
#     def __getitem__(self, index):
#         X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
#         y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
#         smiles = self.smiles[index]
#         gas_index = torch.from_numpy(np.asarray(self.gas_index[index])).view(-1, 1)
#         t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
#         p = torch.from_numpy(np.asarray(self.p[index])).view(-1, 1)
        # return (X, t, p), y.float(), smiles, t, p, gas_index



from torch_geometric.data import Batch,Data
from descriptor import atom_features,bond_features,etype_features
import numpy as np
import torch
from rdkit import Chem


def combine_Graph(Graph_list):
    """
    merge a Graph with multiple subgraph
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object

    """
    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr

    combined_Graph = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return combined_Graph



def add_global(graph):
    """
    add a global point, all the attribute are set to zero
    :param graph: pyg.data
    :return: pyg.data
    """
    node = torch.zeros(1, 40)
    # node.shape
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1
    start = []
    end = []
    attr = []

    for i in range(num_node):
        # print(i)
        start.append(new_node)
        end.append(i)
        attr.append([0] * 10)

    start = torch.tensor(start).reshape(1, -1)
    end = torch.tensor(end).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)
    g = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return g




class Graph_dataset(torch.utils.data.Dataset):
    """
    torch dataset
    """
    def __init__(self, df,target):
        super(Graph_dataset, self).__init__()
        self.smiles = df['IL_SMILES']
        self.label = df[target]
        # self.index = df['index']
        self.length = len(df)
        self.target=target
        self.df=df
        self.T = df['T/K']
        self.P = df['P/bar']

        # show basic information
        print("----info----")
        print("data_length", self.length)
        print("------------")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        t = torch.from_numpy(np.asarray(self.T[idx])).view(-1, 1)
        p = torch.from_numpy(np.asarray(self.P[idx])).view(-1, 1)
        smiles=row['IL_SMILES']
        cation_smiles = row['IL_SMILES'].split('.')[0]
        anion_smiles = row['IL_SMILES'].split('.')[1]

        cation = self.smiles2graph(cation_smiles)
        anion = self.smiles2graph(anion_smiles)
        combine_graph = combine_Graph([cation, anion])
        # num_bond = combine_graph.edge_index.shape[1]
        
        combine_graph = add_global(combine_graph)
        label = torch.tensor(row[self.target], dtype=torch.float).view(1, -1)

        data = Data(x=combine_graph.x, edge_index=combine_graph.edge_index, edge_attr=combine_graph.edge_attr, y=label,smiles=smiles,T=t,P=p)

        return data,label,smiles,t,p


    def smiles2graph(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        nodes = []
        edges = []
        edge_attrs = []

        for atom in mol.GetAtoms():
            node_feat = atom_features(atom)
            nodes.append(node_feat)

        for bond in mol.GetBonds():
            bond_feat = bond_features(bond)
            etype_feat = etype_features(bond)
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_attrs.append(etype_feat + bond_feat)

        x = np.array(nodes, dtype=np.float32)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)

        x = torch.from_numpy(x)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def collate_fn(self,batch):
        graphs = [data[0] for data in batch]
        batch = torch.cat([data.batch for data in graphs], dim=0)

        return batch


class Graph_dataset_MTL(torch.utils.data.Dataset):
    """
    torch dataset
    """

    def __init__(self, df, target):
        super(Graph_dataset_MTL, self).__init__()
        self.smiles = df['IL_SMILES']
        self.label = df[target]
        # self.index = df['index']
        self.length = len(df)
        self.target = target
        self.df = df
        self.T = df['T/K']
        self.P = df['P/bar']
        self.gas_index = df['gas_index']
        self.encoder = OneHotEncoder(categories=[range(1, 11)])  # sparse=True for older versions
        self.gas_index_one_hot = self.encoder.fit_transform(self.gas_index.values.reshape(-1, 1)).toarray()  # Convert to dense array
        # show basic information
        # print("----info----")
        # print("data_length", self.length)
        # print("------------")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        t = torch.from_numpy(np.asarray(self.T[idx])).view(-1, 1)
        p = torch.from_numpy(np.asarray(self.P[idx])).view(-1, 1)
        smiles = row['IL_SMILES']
        cation_smiles = row['IL_SMILES'].split('.')[0]
        anion_smiles = row['IL_SMILES'].split('.')[1]

        cation = self.smiles2graph(cation_smiles)
        anion = self.smiles2graph(anion_smiles)
        combine_graph = combine_Graph([cation, anion])
        # num_bond = combine_graph.edge_index.shape[1]

        combine_graph = add_global(combine_graph)
        label = torch.tensor(row[self.target], dtype=torch.float).view(1, -1)
        gas_index = torch.from_numpy(np.asarray(self.gas_index_one_hot[idx])).float()
        data = Data(x=combine_graph.x, edge_index=combine_graph.edge_index, edge_attr=combine_graph.edge_attr, y=label,
                    smiles=smiles, T=t, P=p)

        return data, label, smiles, t, p,gas_index

    def smiles2graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        nodes = []
        edges = []
        edge_attrs = []

        for atom in mol.GetAtoms():
            node_feat = atom_features(atom)
            nodes.append(node_feat)

        for bond in mol.GetBonds():
            bond_feat = bond_features(bond)
            etype_feat = etype_features(bond)
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_attrs.append(etype_feat + bond_feat)

        x = np.array(nodes, dtype=np.float32)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)

        x = torch.from_numpy(x)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def collate_fn(self, batch):
        graphs = [data[0] for data in batch]
        batch = torch.cat([data.batch for data in graphs], dim=0)

        return batch


class SMILES_dataset_MTL(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer, target):
        self.target = target
        self.smiles = df['IL_SMILES']
        self.gas_index = df['gas_index']

        # One-hot encoding for gas_index
        self.encoder = OneHotEncoder(categories=[range(1, 11)])  # sparse=True for older versions
        self.gas_index_one_hot = self.encoder.fit_transform(self.gas_index.values.reshape(-1, 1)).toarray()  # Convert to dense array

        # Use joblib for parallel tokenization with tqdm for progress bar
        self.tokens = np.array(
            list(Parallel(n_jobs=10)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles, desc='Tokenizing SMILES', total=len(self.smiles))
            ))
        )

        self.label = df[target]
        self.tokenizer = tokenizer
        self.t = df['T/K']
        self.p = df['P/bar']

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Convert gas_index to one-hot encoded vector
        gas_index = torch.from_numpy(np.asarray(self.gas_index_one_hot[index])).float()

        # Tokenize and prepare other inputs
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        smiles = self.smiles[index]
        t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
        p = torch.from_numpy(np.asarray(self.p[index])).view(-1, 1)

        return (X, t, p), y.float(), smiles, t, p, gas_index