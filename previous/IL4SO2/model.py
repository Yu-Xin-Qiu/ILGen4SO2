import torch
import torch.nn.functional as F
from torch import nn
from torch import nn
from transformers import RobertaConfig, RobertaModel


import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter,LSTMCell
from torch import nn
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, GINConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch import nn, Tensor
from transformers import RobertaConfig, RobertaModel
import numpy as np

class nd_conv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        return self.lin2(x_j)


class GNN(torch.nn.Module):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input = None
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.nd_conv = nd_conv(hidden_channels, hidden_channels, edge_dim,
                               dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GINConv(nn.Linear(hidden_channels,hidden_channels))
        self.mol_lstm = LSTMCell(hidden_channels, hidden_channels)


        self.pred_head = nn.Sequential(
            nn.Linear(hidden_channels+2,hidden_channels//2),
            nn.Softplus(),
            nn.Linear(hidden_channels//2, 1)
        )


        self.reset_parameters()
        self.final_conv_acts = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.nd_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_lstm.reset_parameters()
        # self.lin2.reset_parameters()

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data):
        """"""
        '''for explainablity'''


        x,edge_index,edge_attr,batch,t,p=data.x,data.edge_index,data.edge_attr,data.batch,data.T,data.P
        T,P=t.view(-1, 1),p.view(-1, 1)

        h0, edge_index, edge_weight = x, edge_index, sum(edge_attr)
        h0.float().requires_grad = True
        self.input = h0

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.nd_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            with torch.enable_grad():
                self.final_conv_acts = gru(h, x).relu_()
            self.final_conv_acts.register_hook(self.activations_hook)

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(self.final_conv_acts, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((self.final_conv_acts, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out,h = self.mol_lstm(out,(h,h))




        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.pred_head(torch.cat((out,T.float(),P.float()), dim=1))
    




    def forward(self, data):
        """"""
        '''for explainablity'''

        x,edge_index,edge_attr,batch,t,p=data.x,data.edge_index,data.edge_attr,data.batch,data.T,data.P
        T,P=t.view(-1, 1),p.view(-1, 1)

        h0, edge_index, edge_weight = x, edge_index, sum(edge_attr)
        h0.requires_grad = True
        self.input = h0

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.nd_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            with torch.enable_grad():
                self.final_conv_acts = gru(h, x).relu_()
            self.final_conv_acts.register_hook(self.activations_hook)

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(self.final_conv_acts, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((self.final_conv_acts, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out,h = self.mol_lstm(out,(h,h))

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)

        return F.elu_(self.lin2(torch.cat((out,T.float(),P.float()), dim=1)))




class CNN(nn.Module):
    """Network for fine-tuning on IL properties datasets"""
    def __init__(self,
                 dropout,
                 embed_size,
                 output_size=1,
                 num_filters=(100, 200, 200, 200, 200, 100, 100),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7),
                 IL_num_filters=(100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160),
                 IL_ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)):
        super(CNN, self).__init__()
        
        self.num_filters = num_filters
        self.IL_num_filters = IL_num_filters
        
        self.IL_textcnn = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=nf, kernel_size=ks)
                                        for nf, ks in zip(IL_num_filters, IL_ngram_filter_sizes)])
        self.output = nn.Linear(sum(IL_num_filters), embed_size)


    def forward(self, IL_src_nd):
        IL_encoded = IL_src_nd.permute(1,2,0)

        IL_textcnn_out = [F.relu(conv(IL_encoded)) for conv in self.IL_textcnn]
        IL_textcnn_out = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in IL_textcnn_out]  # Max pooling
        IL_textcnn_out = torch.cat(IL_textcnn_out, 1)  # Concatenate all the pooled features
        # input_vecs = torch.cat((IL_textcnn_out, T.view(-1, 1), P.view(-1, 1)), dim=1)
        input_vecs = IL_textcnn_out
        # input_vecs = torch.cat((self.lin(IL_textcnn_out), T.view(-1, 1)), dim=1)

        out = self.output(input_vecs.float())

        # out = self.output(IL_encoded[:,0,:].float())

        return out


class ILBERT(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, _ = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)
        output = self.pred_head(output)

        return output






class ILBERT1(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )




    def forward(self, input):
        if not isinstance(input, torch.Tensor):
            x = torch.from_numpy(input).long()
        else:
            x = input.long()


        # 生成attention_mask，类型为布尔张量
        attention_mask = (x != 0)

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x, attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)
        output = self.pred_head(output)

        return outputs


class ILBERT_NOCNN(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)

    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, _ = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求


        return last_hidden_states


class ILBERT_WITHCNN(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
        )



    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, _ = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)


        output = self.pred_head(output)


        return output




class ILBERT_T(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model+1, d_model//2),
            nn.Softplus(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, T = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)
        T=T.view(-1, 1)

        output = self.pred_head(torch.cat((output, T.float()), dim=1))

        return output






class ILBERT_T_P(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model+2, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, T,P = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)

        T,P=T.view(-1, 1),P.view(-1, 1)

        output = self.pred_head(torch.cat((output, T.float(),P.float()), dim=1))

        return output





class MTL_ILBERT_T_P(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model+2, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 10)
        )

    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, T,P = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)

        T,P=T.view(-1, 1),P.view(-1, 1)

        output = self.pred_head(torch.cat((output, T.float(),P.float()), dim=1))

        return output


import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter,LSTMCell
from torch import nn
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, GINConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor


class nd_conv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        return self.lin2(x_j)

class ILGNN(torch.nn.Module):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input = None
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.nd_conv = nd_conv(hidden_channels, hidden_channels, edge_dim,
                               dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GINConv(nn.Linear(hidden_channels,hidden_channels))
        self.mol_lstm = LSTMCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels+2, out_channels)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_channels + 2, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1))
        self.reset_parameters()
        self.final_conv_acts = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.nd_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_lstm.reset_parameters()
        self.lin2.reset_parameters()

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data):
        """"""
        '''for explainablity'''

        x,edge_index,edge_attr,batch,t,p=data.x,data.edge_index,data.edge_attr,data.batch,data.T,data.P
        T,P=t.view(-1, 1),p.view(-1, 1)

        h0, edge_index, edge_weight = x, edge_index, sum(edge_attr)
        h0.requires_grad = True
        self.input = h0

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.nd_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            with torch.enable_grad():
                self.final_conv_acts = gru(h, x).relu_()
            self.final_conv_acts.register_hook(self.activations_hook)

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(self.final_conv_acts, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((self.final_conv_acts, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out,h = self.mol_lstm(out,(h,h))

        # Predictor:
        # out = F.dropout(out, p=self.dropout, training=self.training)

        # return self.lin2(torch.cat((out,T.float(),P.float()), dim=1))
        # return self.lin2(torch.cat((out,T.float(),P.float()), dim=1))

        # # print(out.shape,cond.shape)
        out = torch.cat([out, T.float(), P.float()], dim=1)

        return self.pred_head(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')



from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax



class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)




        
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        gru = GRUCell(hidden_channels, hidden_channels)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_channels+2, hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels//2, 1))


        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

        
    def forward(self, data):
        # x, edge_index, edge_attr, batch=data.x,data.edge_index, data.edge_attr, data.batch
        x,edge_index,edge_attr,batch,t,p=data.x,data.edge_index,data.edge_attr,data.batch,data.T,data.P
        T,P=t.view(-1, 1),p.view(-1, 1)
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x.float()))

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()



        # # print(out.shape,cond.shape)
        out = torch.cat([out,T.float(),P.float()], dim=1)

        return self.pred_head(out)


class MTL_AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        gru = GRUCell(hidden_channels, hidden_channels)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_channels + 2, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 10))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # x, edge_index, edge_attr, batch=data.x,data.edge_index, data.edge_attr, data.batch
        x, edge_index, edge_attr, batch, t, p = data.x, data.edge_index, data.edge_attr, data.batch, data.T, data.P
        T, P = t.view(-1, 1), p.view(-1, 1)
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x.float()))

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # # print(out.shape,cond.shape)
        out = torch.cat([out, T.float(), P.float()], dim=1)

        return self.pred_head(out)
