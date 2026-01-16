import torch
import torch.nn as nn
from utils.sequential import Sequential


class SentenceSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SentenceSelector, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.gru(x)  # Get outputs for all time steps
        logits = self.fc(output)  # [B, N, 1]
        probs = torch.sigmoid(logits)
        selected = (probs > 0.5).float().squeeze(-1)  # [B, N], convert probabilities to 0 or 1
        return selected, probs.squeeze(-1)


class QA_Attention(nn.Module):
    def __init__(self):
        super(QA_Attention, self).__init__()

    def forward(self, q, a, q_mask, a_mask):
        attention_scores = torch.bmm(q, a.transpose(1, 2))  # [B, N, N], compute attention score matrix
        mask = torch.bmm(q_mask.unsqueeze(-1), a_mask.unsqueeze(1))  # [B, N, N], create mask matrix
        masked_attention_scores = attention_scores * mask  # Apply mask matrix

        q_s = masked_attention_scores.sum(dim=-1).softmax(dim=-1).unsqueeze(dim=1)
        rep_q = torch.bmm(q_s, q).squeeze()

        a_s = masked_attention_scores.sum(dim=1).softmax(dim=-1).unsqueeze(dim=1)
        rep_a = torch.bmm(a_s, a).squeeze()

        return torch.cat([rep_q, rep_a], dim=-1)


class MRQA_simple(nn.Module):
    def __init__(self,
                 input_dim=384,  # e.g., D can be the word embedding dimension
                 hidden_dim=192  # hidden layer dimension can be adjusted as needed
                 ):
        super(MRQA_simple, self).__init__()

        # Instantiate models
        self.selector = SentenceSelector(input_dim, hidden_dim)
        self.attention = QA_Attention()

        hidden_size = [500, 250, 100]
        hidden_layers = 3
        in_bn = False
        hid_bn = False
        out_bn = True
        num_features = 384
        self.mlp = self.make_ff(0.5,
                                num_features * 2,
                                hidden_size,
                                hidden_layers,
                                in_bn=in_bn,
                                hid_bn=hid_bn,
                                out_bn=out_bn)

    def make_ff(self, dropout, in_size, hidden_size, hidden_layers, in_bn=True, hid_bn=True, out_bn=True, out=True):
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

    def forward(self, Q, A):
        # Pass through selector model
        Q_selected, _ = self.selector(Q)
        A_selected, _ = self.selector(A)
        return Q_selected.squeeze(), A_selected.squeeze()
