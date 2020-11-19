import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(2*self.dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.shape[0]
        assert self.dim == h.shape[1]
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # attention = F.softmax(e, dim=1)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(attention, h)

class SelfAttentionLayer_batch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer_batch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(2*self.dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, mask):
        N = h.shape[0]
        assert self.dim == h.shape[2]
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # attention = F.softmax(e, dim=1)
        mask=1e-30*mask.float()

        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        #print(e.size())
        #print(mask.size())
        attention = F.softmax(e+mask.unsqueeze(-1),dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(torch.transpose(attention,1,2), h).squeeze(1),attention

class SelfAttentionLayer2(nn.Module):
    def __init__(self, dim, da):
        super(SelfAttentionLayer2, self).__init__()
        self.dim = dim
        self.Wq = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.Wk = nn.Parameter(torch.zeros(self.dim, self.dim))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.shape[0]
        assert self.dim == h.shape[1]
        q = torch.matmul(h, self.Wq)
        k = torch.matmul(h, self.Wk)
        e = torch.matmul(q, k.t()) / math.sqrt(self.dim)
        attention = F.softmax(e, dim=1)
        attention = attention.mean(dim=0)
        x = torch.matmul(attention, h)
        return x


# http://dbpedia.org/ontology/director


