import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import init
import logging
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import wrappers
'''
1. use topic model to split document representation to k representation
2. use multi-round q-a pair
3. based on topic profet, add GNN

The next version TextGNN will not use profet anymore
'''


def visualize_adj(tensor_data, name):
    # Convert PyTorch tensor to NumPy array
    numpy_data = tensor_data.cpu().numpy()

    # Visualize using Seaborn heatmap
    sns.set()
    plt.figure(figsize=(12, 2))
    sns.heatmap(numpy_data, annot=False, cmap="Reds", linewidths=.5)
    plt.savefig(f'sa_{name}.png')


def load_topic_rep(path):
    data = pickle.load(open(path, 'rb'))
    topic_rep = []
    for i in range(len(data.keys())):
        topic_rep.append(data[i])
    return np.array(topic_rep)


def print_topics_by_indices(indices_list):
    model = wrappers.LdaMallet.load("./result/checkpoint/LDA/sent_lda.model")
    if indices_list == []:
        for topic_id in range(50):
            words = model.show_topic(topic_id)
            mod = f'topic {topic_id}: '
            for word, prob in words:
                mod = mod + word + ', '
            print(mod)
    for topic_id in indices_list:
        words = model.show_topic(topic_id)
        mod = f'topic {topic_id}: '
        for word, prob in words:
            mod = mod + word + ', '
        print(mod)


def load_clustering_rep(path):
    data = pickle.load(open(path, 'rb'))
    return data


class TopicProfetGNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 embedding_size=None,
                 hidden_layers=None,
                 dropout=None,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 make_ff=True,
                 ):
        super().__init__()
        logging.info(f"initialize {self.__class__.__name__}")
        self.embedding_size = embedding_size
        
        # Load pretrained topic representation
        data = load_topic_rep("dataset/processed/sent_lda_rep_v2_0707.pkl")
        self.topic_rep = torch.from_numpy(data).to(torch.float).cuda()
        self.topic_rep.requires_grad = False

        if make_ff:
            self.ff = self._make_ff(dropout,
                                    self.embedding_size*2,
                                    hidden_size,
                                    hidden_layers,
                                    in_bn=in_bn,
                                    hid_bn=hid_bn,
                                    out_bn=out_bn)

        self.sa = ScaledDotProductAttention(
            d_model=self.embedding_size, d_k=self.embedding_size, d_v=self.embedding_size, h=1, dropout=dropout)
        
        self.gru_pre = nn.GRU(input_size=self.embedding_size,
                              hidden_size=self.embedding_size, batch_first=True, bidirectional=True)

        self.gru_qa = nn.GRU(input_size=self.embedding_size,
                             hidden_size=self.embedding_size, batch_first=True, bidirectional=True)
        
        
        self.ggan = GatedGNN(input_dim=self.embedding_size,
                             output_dim=self.embedding_size, dropout=dropout)
        
        self.linear = nn.Linear(384, self.embedding_size)

    def _make_ff(self, dropout, in_size, hidden_size, hidden_layers, in_bn=True, hid_bn=True, out_bn=True, out=True):
        def get_block(in_size, hidden_size, bn, act=True, drop=True):
            result = nn.Sequential(
                nn.BatchNorm1d(in_size) if bn else None,
                nn.Dropout(p=dropout) if drop else None,
                nn.Linear(in_size, hidden_size),
                nn.ReLU() if act else None,
            )
            return result

        ff_seq = list()
        ff_seq.extend(get_block(in_size, hidden_size[0], bn=in_bn))
        for i in range(1, hidden_layers):
            ff_seq.extend(
                get_block(hidden_size[i - 1], hidden_size[i], bn=hid_bn))
        if out:
            ff_seq.extend(
                get_block(hidden_size[-1], 1, bn=out_bn, act=False, drop=False))

        return Sequential(
            *ff_seq
        )

    def ablation_GNN_straightforward(self, q_rep, a_rep, pre_rep, pre_mask, padded_len):
        qa_rep = (q_rep + a_rep) / 2

        Q = torch.cat([pre_rep, qa_rep], dim=1)
        topic_indices = torch.matmul(
            Q, self.cluster_centriods.t()).sum(dim=1).topk(25).indices
        anchor = self.topic_emb(topic_indices)
        pre_mask = torch.unsqueeze(
            pre_mask, dim=1).expand(-1, anchor.size(1), -1)

        output_sa, att = self.sa(anchor, pre_rep, pre_rep, pre_mask)
        out = []
        cat_res = torch.cat([output_sa, qa_rep], dim=1)
        for l, sample in zip(padded_len, cat_res):
            out.append(sample[:l].mean(dim=0))
        out = torch.stack(out, dim=0)
        return self.ff(out)

    def origin_method(self, q_rep, a_rep, pre_rep, gnn_mask, pre_mask, qa_mask, padded_len, pre_topic, qa_topic):
        topic_rep = self.linear(self.topic_rep)
        
        pre_rep = pre_rep.float()
        q_rep = q_rep.float()
        a_rep = a_rep.float()
        
        score = pre_rep @ topic_rep.t()  # [b, l, 50]
        score = F.softmax(score, dim=-1)
        score = score.permute(0, 2, 1)  # [b, 50, l]
        batch_topic_pre = score @ pre_rep  # [b, 50, d]
        
        qa_rep = (q_rep + a_rep) / 2
        
        gnn_topic = torch.cat([pre_topic, qa_topic], dim=1)
        
        gnn_in = torch.cat([batch_topic_pre, qa_rep], dim=1)
        
        gnn_out = self.ggan(gnn_mask, gnn_in, gnn_topic)
        
        h_pre_pack = gnn_out[:, :50, :]
        h_qa_pack = gnn_out[:, 50:, :] 
        qa_mask = qa_mask.unsqueeze(-1)
        
        h_qa_pack = h_qa_pack * qa_mask    
        
        topic_weights = torch.softmax(torch.sum(qa_topic, dim=1), dim=-1)  # [b, 50]
        
        h_pre_weighted = h_pre_pack * topic_weights.unsqueeze(-1) # [b, 50, d]
        h_pre = h_pre_weighted.sum(dim=1)  # [b, d]
        
        qa_weights = qa_topic * topic_weights.unsqueeze(1)  # [b, l, 50]
        
        qa_weights = torch.softmax(torch.sum(qa_weights, dim=-1), dim=-1)  # [b, l]
        
        h_qa_weighted = h_qa_pack * qa_weights.unsqueeze(-1)  # [b, l, d]
        h_qa = h_qa_weighted.sum(dim=1)  # [b, d]
        
        h = torch.cat([h_pre, h_qa], dim=-1)
        h = h.float()
        
        return self.ff(h)


    def forward(self,
                pre_rep,
                q_rep,
                a_rep,
                pre_mask,
                qa_mask,
                gnn_mask=None,            
                padded_len=None,
                pre_topic=None,
                qa_topic=None,
                ):

        B, L_pre, D = pre_rep.shape
        L_qa = q_rep.size(1)

        # ① padded_len: total sequence length for each sample
        if padded_len is None:
            padded_len = torch.full((B,), L_pre + L_qa,
                                    dtype=torch.long, device=pre_rep.device)

        # ② pre_topic: batch of 50 × 50 one-hot matrices
        if pre_topic is None:
            pre_topic = torch.eye(50, device=pre_rep.device)            \
                           .unsqueeze(0).repeat(B, 1, 1)                # [B,50,50]

        if qa_topic is None:
            topic_rep = self.linear(self.topic_rep)                     # [50,D]
            qa_topic = torch.softmax(q_rep @ topic_rep.t(), dim=-1)     # [B,L_qa,50]

        if gnn_mask is None:
            N = 50 + L_qa
            gnn_mask = torch.ones(B, N, N, device=pre_rep.device)

        out = self.origin_method(q_rep, a_rep, pre_rep, gnn_mask, pre_mask, qa_mask, padded_len, pre_topic, qa_topic)
        return out
