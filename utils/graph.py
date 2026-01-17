import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['GraphConvolution', 'GCN', 'GraphAttentionLayer',
           'GAT', 'GatedGNN', 'SimpleGatedGNN']


class SimpleGatedGNN(nn.Module):
    '''
    pre defined affinity matrix A
    need gated operation to fuse representations
    '''

    def __init__(self, input_dim, output_dim, alpha=0.2, dropout=0.1):
        super(SimpleGatedGNN, self).__init__()
        self.dropout = dropout
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = alpha

        self.gru = nn.GRUCell(input_size=input_dim, hidden_size=output_dim)
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W.data)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, affinity_matrix, x):
        batch_size = x.size(0)
        Wh = torch.matmul(x, self.W)  # (B, N, d)
        # we pre divide total num in Mask
        graph_out = torch.matmul(affinity_matrix, Wh)
        graph_out = graph_out.reshape(-1, self.out_features)
        x = x.reshape(-1, self.out_features)

        output = self.gru(graph_out, x)
        output = output.reshape(batch_size, -1, self.out_features)  # (B, N, d)

        return output


class GatedGNN(nn.Module):
    '''
    need self-attention and adj_mask to generate adj,
    need gated operation to fuse old and new representations
    '''

    def __init__(self, input_dim, output_dim, alpha=0.2, dropout=0.1):
        super(GatedGNN, self).__init__()
        self.dropout = dropout
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = alpha

        self.gru = nn.GRUCell(input_size=input_dim, hidden_size=output_dim)
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.Tensor(2*output_dim, 1))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, adj_mask, x, x_topic):
        batch_size = x.size(0)
        Wh = torch.matmul(x, self.W)  # (B, N, out_features)
        e = F.cosine_similarity(x_topic.unsqueeze(1), x_topic.unsqueeze(2), dim=-1)
        
        padding = (-2 ** 31) * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, padding)  # (B, N, N)
        attention = F.softmax(attention, dim=-1)
        
        attention = F.dropout(attention, self.dropout)
        
        attention = attention.float()
        Wh = Wh.float()
        
        graph_out = torch.matmul(attention, Wh)

        graph_out = graph_out.reshape(-1, self.out_features)
        x = x.reshape(-1, self.out_features)

        output = self.gru(graph_out, x)

        output = output.reshape(batch_size, -1, self.out_features)

        return output


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # Xavier initialization, which is Glorot initialization from the paper
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(self.dropout(inputs), self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_layers, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        if n_layers == 1:
            self.first_layer = GraphConvolution(n_features, n_classes, dropout)
        else:
            self.first_layer = GraphConvolution(
                n_features, hidden_dim, dropout)
            self.last_layer = GraphConvolution(hidden_dim, n_classes, dropout)
            if n_layers > 2:
                self.gc_layers = nn.ModuleList([
                    GraphConvolution(hidden_dim, hidden_dim, 0) for _ in range(n_layers - 2)
                ])

        self.n_layers = n_layers
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        if self.n_layers == 1:
            x = self.first_layer(inputs, adj)
        else:
            x = self.relu(self.first_layer(inputs, adj))
            if self.n_layers > 2:
                for i, layer in enumerate(self.gc_layers):
                    x = self.relu(layer(x, adj))
            x = self.last_layer(x, adj)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        h: (N, in_features)
        adj: sparse matrix with shape (N, N)
        '''
        Wh = torch.mm(h, self.W)  # (N, out_features)
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])  # (N, 1)

        # Wh1 + Wh2.T is an N*N matrix, where element [i,j] is Wh1[i]+Wh2[j]
        # The [i,j] element of Wh1 + Wh2.T is exactly a^T*[Whi||Whj] from the paper
        # This represents the attention of node i to node j
        e = self.leakyrelu(Wh1 + Wh2.T)  # (N, N)
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)
        attention = torch.where(adj > 0, e, padding)  # (N, N)
        attention = F.softmax(attention, dim=1)  # (N, N)
        # The [i,j] element of the attention matrix represents the attention of node_i to node_j
        # Apply dropout to attention weights (if the attention matrix is highly sparse after masking, is this still necessary?)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nfeat)
        x = torch.cat([head(x, adj)
                      for head in self.MH], dim=1)  # (N, nheads*nhid)
        # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
