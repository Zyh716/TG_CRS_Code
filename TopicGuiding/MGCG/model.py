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
# from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from tqdm import tqdm
from torch.utils.data import *
import ipdb
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class IntentionClassifier(nn.Module):
    def __init__(self, args):
        super(IntentionClassifier, self).__init__()

        self.state2topic_id = nn.Linear(args.hidden_size * 3,
                                        args.topic_class_num)

    def forward(self, context_rep, tp_rep, profile_pooled):
        # [bs, hidden_size*3]
        state_rep = torch.cat((context_rep, tp_rep, profile_pooled), 1)

        out_topic_id = self.state2topic_id(state_rep)

        return out_topic_id

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        embed_matrix = np.load(args.bpe2vec)
        vocab_size = embed_matrix.shape[0]
        self.embeddings = nn.Embedding(vocab_size, self.args.embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_matrix))

        self.context_lstm = nn.LSTM(self.args.embedding_dim,
                                    self.args.hidden_size,
                                    self.args.num_layers,
                                    dropout=self.args.dropout_hidden,
                                    batch_first=True)

        self.topic_lstm = nn.LSTM(self.args.embedding_dim,
                                  self.args.hidden_size,
                                  self.args.num_layers,
                                  dropout=self.args.dropout_hidden,
                                  batch_first=True)

        self.profile_lstm = nn.LSTM(self.args.embedding_dim,
                                    self.args.hidden_size,
                                    self.args.num_layers,
                                    dropout=self.args.dropout_hidden,
                                    batch_first=True)

        self.intention_classifier = IntentionClassifier(self.args)

        if args.init_from_fineturn:
            self.load_state_dict(torch.load(join(self.args.lstm_path,
                                                 'model')))

        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)

    def forward(self, x):
        context, len_context, topic_path_kw, topic_path_attitude, len_tp, user_profile, len_profile = x
        bs, sent_num, word_num = user_profile.shape

        user_profile = user_profile.view(-1,
                                         word_num)  # (bs*sent_num, word_num)
        len_profile = len_profile.view(-1)  #(bs*sent_num)

        context = self.embeddings(context)
        topic_path_kw = self.embeddings(topic_path_kw)
        user_profile = self.embeddings(user_profile)

        context = pack_padded_sequence(context,
                                       len_context,
                                       enforce_sorted=False,
                                       batch_first=True)
        topic_path_kw = pack_padded_sequence(topic_path_kw,
                                             len_tp,
                                             enforce_sorted=False,
                                             batch_first=True)
        user_profile = pack_padded_sequence(user_profile,
                                            len_profile,
                                            enforce_sorted=False,
                                            batch_first=True)

        init_h0 = (torch.zeros(self.args.num_layers, bs,
                               self.args.hidden_size).to(self.args.device),
                   torch.zeros(self.args.num_layers, bs,
                               self.args.hidden_size).to(self.args.device))

        # batch, seq_len, num_directions * hidden_size        # num_layers * num_directions, batch, hidden_size
        context_output, (context_h, _) = self.context_lstm(context, init_h0)
        topic_output, (topic_h, _) = self.topic_lstm(topic_path_kw, init_h0)
        # batch*sent_num, seq_len, num_directions * hidden_size
        init_h0 = (torch.zeros(self.args.num_layers, bs * sent_num,
                               self.args.hidden_size).to(self.args.device),
                   torch.zeros(self.args.num_layers, bs * sent_num,
                               self.args.hidden_size).to(self.args.device))
        profile_output, (profile_h,
                         _) = self.profile_lstm(user_profile, init_h0)

        # batch, hidden_size
        context_rep = context_h[-1]
        topic_rep = topic_h[-1]

        profile_rep = profile_h[-1]
        profile_rep = profile_rep.view(bs, sent_num, -1)
        # batch, hidden_size
        profile_rep = torch.mean(profile_rep, dim=1)

        out_topic_id = self.intention_classifier(context_rep, topic_rep,
                                                 profile_rep)

        return out_topic_id

    def compute_loss(self, output, y_type, y_topic_id, subset='test'):
        out_topic_id = output

        loss_topic_id = F.cross_entropy(out_topic_id, y_topic_id)
        return loss_topic_id

    def save_model(self, save_path):
        # 存储时, 传入一个存储位置
        torch.save(self.state_dict(), self.args.model_save_path + '/model')

    def load_addition_params(self, path):
        self.intention_classifier.load_model(path)
