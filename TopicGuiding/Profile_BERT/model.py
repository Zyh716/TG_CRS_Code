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


class IntentionClassifier(nn.Module):
    def __init__(self, args, bert_embed_size=768):
        super(IntentionClassifier, self).__init__()
        self.state2topic_id = nn.Linear(bert_embed_size, args.topic_class_num)

    def forward(self, profile_pooled, bs, sent_num, word_num):
        profile_pooled = profile_pooled.view(bs, sent_num, -1)
        # (bs, hidden)
        profile_pooled = torch.mean(profile_pooled, dim=1)

        out_topic_id = self.state2topic_id(profile_pooled)

        return out_topic_id

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))


class Model(nn.Module):
    def __init__(self, args, bert_embed_size=768):
        super(Model, self).__init__()
        self.args = args

        # define and init bert
        bert_path1, bert_path2, bert_path3 = [args.bert_path] * 3
        if args.init_bert_from_pretrain:
            pass
        elif args.init_from_fineturn:
            # 如果是加载已经微调好的
            bert_path3 = bert_path3 + '/3'

        self.profile_bert = BertModel.from_pretrained(
            bert_path3)  # /bert_pretrain/
        # to check
        for model in [self.profile_bert]:
            for param in model.parameters():
                param.requires_grad = True  # 每个参数都要 求梯度

        # define IntentionClassifier
        self.intention_classifier = IntentionClassifier(
            self.args, bert_embed_size)
        # init if need, save and load both in bert1_path
        self.addition_save_name = 'addition_model.pth'
        if args.init_add:
            self.load_addition_params(join(bert_path3,
                                           self.addition_save_name))

        # 记录save path
        self.save_path3 = args.model_save_path + '/3'

        if not os.path.exists(self.save_path3):
            os.mkdir(self.save_path3)

    def forward(self, x):
        context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask = x

        bs, sent_num, word_num = user_profile.shape
        user_profile = user_profile.view(-1, user_profile.shape[-1])
        profile_mask = profile_mask.view(-1, profile_mask.shape[-1])
        # (bs, word_num, hidden)
        profile_last_hidden_state, profile_pooled = self.profile_bert(
            user_profile, profile_mask)

        out_topic_id = self.intention_classifier(profile_pooled, bs, sent_num,
                                                 word_num)

        return out_topic_id

    def compute_loss(self, output, y_topic_id):
        out_topic_id = output

        loss_topic_id = F.cross_entropy(out_topic_id, y_topic_id)
        return loss_topic_id

    def save_model(self, save_path):
        self.profile_bert.save_pretrained(self.save_path3)

        self.intention_classifier.save_model(
            join(self.save_path3, self.addition_save_name))

    def load_addition_params(self, path):
        self.intention_classifier.load_model(path)
