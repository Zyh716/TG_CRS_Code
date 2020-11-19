# coding: UTF-8
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json
from os.path import join
import time 
import pandas as pd 
from tqdm import tqdm
from torch.utils.data import *
import ipdb
import copy
import math
from copy import deepcopy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU4REC(nn.Module):
    def __init__(self, args):
        super(GRU4REC, self).__init__()
        self.args = args

        self.input_size = args.item_size
        self.hidden_size = args.gru_hidden_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.dropout_hidden = args.dropout_hidden
        self.dropout_input = args.dropout_input
        self.embedding_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.device = args.device
        self.h2o = nn.Linear(args.hidden_size, args.output_size)
        self.create_final_activation(args.final_act)
        self.item_embeddings = nn.Embedding(args.item_size, self.embedding_dim)
        self.item_embeddings.load_state_dict(torch.load(args.sasrec_emb_path, map_location=self.args.device))
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first = True)
        # if self.embedding_dim != -1:
        #     self.look_up = nn.Embedding(input_size, self.embedding_dim)
        #     self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        # else:
        #     self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, len_input, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        # if self.embedding_dim == -1:
        #     # bs, item_size
        #     embedded = self.onehot_encode(input)
        #     if self.training and self.dropout_input > 0: embedded = self.embedding_dropout(embedded)
        #     # 1, bs, item_size?
        #     embedded = embedded.unsqueeze(0)
        # else:
        #     embedded = input.unsqueeze(0)
        #     embedded = self.look_up(embedded)
        # print(input.shape)
        # input: bs， seq_len,  
        embedded = self.item_embeddings(input)
        # print(embedded.shape)
        # (batch, seq_len, hidden_size):
        # (num_layers , batch, hidden_size)
        embedded = pack_padded_sequence(embedded, len_input, enforce_sorted=False, batch_first=True)
        # print(embedded.data.shape)

        output, hidden = self.gru(embedded, hidden) 
        output, output_len = pad_packed_sequence(output,batch_first=True)
        # print(output.shape) # 以最长的序列的长度为最长
        # print(output_len)
        # output = output.view(-1, output.size(-1))  #(B,H)
        
        batch, seq_len,  hidden_size = output.size()
        # output = output.view(-1, hidden_size)
        # output = self.final_activation(self.h2o(output))
        logit = output.view(batch, seq_len,  hidden_size)

        # (bs, output_size) #(num_layer, B, H)
        return logit, hidden[-1], max([len_ for len_ in output_len])

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # try:
        #     h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # except:
        #     self.device = 'cpu'
        #     h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids, input_mask):
        
        # [batch seq_len hidden_size]
        # print(type(neg_ids))
        # print(neg_ids.shape)
        # ipdb.set_trace()
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)        

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        # [batch*seq_len hidden_size]
        # print(seq_out.shape)
        # print(self.args.gru_hidden_size)
        # ipdb.set_trace()
        seq_emb = seq_out.contiguous().view(-1, self.args.gru_hidden_size)

        # [batch*seq_len]
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)

        # [batch*seq_len]
        istarget = (pos_ids > 0).view(pos_ids.size(0) * pos_ids.size(1)).float()
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)


        return loss

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)

        self.to(self.args.device)
    
    # def load_model(self, path):
    #     self.load_state_dict(torch.load(path))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
