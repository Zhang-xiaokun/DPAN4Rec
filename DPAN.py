
"""
Created on 11 Nov, 2019

@author: zhangxiaokun
Reference: https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch/blob/master/NARM.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import pandas as pd
from scipy import stats


class DPAN(nn.Module):
    """Dual Part-pooling attentive networks for session-based recommendation

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of SA\CA item embedding
        alpha_pool(float): the degree of SA (alpha) pooling
        beta_pool(float): the degree of CA (beta) pooling
        n_layers(int): the number of gru layers
    """

    def __init__(self, n_items, hidden_size, embedding_dim, batch_size,alpha_pool,beta_pool, n_layers=1):
        super(DPAN, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.alpha_pool =alpha_pool
        self.beta_pool = beta_pool
        # sequential embedding
        self.emb_s = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        # collectove embedding
        self.emb_c = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)
        # Sequential Acquisition
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_3 = nn.Linear(self.embedding_dim, self.hidden_size, bias=False)
        self.va_t = nn.Linear(self.hidden_size, 1, bias=True)

        # Collective Acquisition
        self.b_1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.b_2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.b_3 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.vb_t = nn.Linear(self.embedding_dim, 1, bias=True)

        self.ct_dropout = nn.Dropout(0.25)

        self.item_fuse = nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2, bias=True)
        self.mlp_active = nn.Tanh()

        self.b = nn.Linear(self.embedding_dim*2, self.hidden_size + self.embedding_dim, bias=False)
        self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        hidden = self.init_hidden(seq.size(1))
        # sequential embedding
        emb_sa = self.emb_dropout(self.emb_s(seq))
        # collective embedding
        emb_ca = self.emb_dropout(self.emb_c(seq))
        last_itemid = torch.stack([seq[lengths[i] - 1, i] for i in range(len(lengths))])
        q2_c = self.emb_dropout(self.emb_c(last_itemid))
        q3_c = emb_ca[0]

        emb_ca = emb_ca.permute(1, 0, 2)

        embs_padded = pack_padded_sequence(emb_sa, lengths)
        gru_out, hidden = self.gru(embs_padded, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        ht = hidden[-1]
        hf = emb_sa[0]
        gru_out = gru_out.permute(1, 0, 2)
        qs1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        qs2 = self.a_2(ht) #last
        qs3 = self.a_3(hf) #first
        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))
        qs2_expand = qs2.unsqueeze(1).expand_as(qs1)
        qs2_masked = mask.unsqueeze(2).expand_as(qs1) * qs2_expand

        qs3_expand = qs3.unsqueeze(1).expand_as(qs1)
        qs3_masked = mask.unsqueeze(2).expand_as(qs1) * qs3_expand

        alpha = self.va_t(torch.sigmoid(qs1 + qs2_masked + qs3_masked).view(-1, self.hidden_size)).view(mask.size())
        # part-pooling
        alpha = self.sf(alpha)
        alpha_max = torch.max(alpha, 1, True)[0].expand_as(alpha)
        alpha_mask = alpha - self.alpha_pool * alpha_max

        alpha = torch.where(alpha_mask > 0, alpha,
                            torch.tensor([0.], device=self.device))

        # session embedding containing sequential dependencies
        sess_s = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        qc1 = self.b_1(emb_ca.contiguous().view(-1, self.embedding_dim)).view(emb_ca.size())
        qc2 = self.b_2(q2_c)  # last
        qc3 = self.b_3(q3_c)  # first

        qc2_expand = qc2.unsqueeze(1).expand_as(qc1)
        qc2_masked = mask.unsqueeze(2).expand_as(qc1) * qc2_expand

        qc3_expand = qc3.unsqueeze(1).expand_as(qc1)
        qc3_masked = mask.unsqueeze(2).expand_as(qc1) * qc3_expand

        beta = self.vb_t(torch.sigmoid(qc1 + qc2_masked +qc3_masked).view(-1, self.embedding_dim)).view(mask.size())
        # part-pooling
        beta = self.sf(beta)
        beta_v = torch.max(beta, 1, True)[0].expand_as(beta)
        beta_mask = beta - self.beta_pool * beta_v

        beta = torch.where(beta_mask > 0, beta,
                           torch.tensor([0.], device=self.device))

        # session embedding containing collective dependencies
        sess_c = torch.sum(beta.unsqueeze(2).expand_as(emb_ca) * emb_ca, 1)

        # MLP to merge session embeddings from SA&CA
        sess_final = torch.cat([sess_c, sess_s], 1)
        sess_final = self.ct_dropout(sess_final)
        # MLP to merge item embeddings from SA&CA
        item_emb_sa = self.emb_s(torch.arange(self.n_items).to(self.device))
        item_emb_ca = self.emb_c(torch.arange(self.n_items).to(self.device))
        item_final = torch.cat([item_emb_sa, item_emb_ca], 1)
        item_final = self.ct_dropout(item_final)
        item_final = self.item_fuse(item_final)

        scores = torch.matmul(sess_final, self.b(item_final).permute(1, 0))

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

