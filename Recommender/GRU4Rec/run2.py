#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from tqdm import tqdm
from math import exp
import os
import signal
import json
import argparse
from dataset import CRSdataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn as nn
from torch import optim
import torch
from nltk.translate.bleu_score import sentence_bleu
import nltk
import re
import pickle
import logging
import time 
import torch.nn.functional as F 
# from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from torch.utils.data import *
import ipdb
import math
import random
from os.path import join
from torch.optim import Adam
from model import GRU4REC

def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
 
def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-exp_name","--exp_name",type=str,default='modelv1')

    # about train setting
    train.add_argument("-batch_size","--batch_size",type=int,default=64)  # todo
    train.add_argument("-lr_GRU","--lr_GRU",type=float,default=1e-3)
    train.add_argument("-epoch","--epoch",type=int,default=500)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True) 
    train.add_argument("-gpu","--gpu",type=str,default='2')

    # about model setting
    train.add_argument("-load_dict","--load_dict",type=str,\
        default="../../pretrain_model/wwm_ext/", help='要加载的模型的位置') 
    train.add_argument("-init_add", "--init_add", action="store_true", default=False)
    train.add_argument("-model_save_path","--model_save_path",type=str,default='saved_model/{}') # todo
    
    # about dataset and data setting
    #  下面两个只有一个不为None, load_builded_data有则直接加载，没有就重新处理，但是可能不保存
    train.add_argument("-raw","--raw", action="store_true", default=False)

    train.add_argument("-train_data_file","--train_data_file",type=str,\
        default="../../data/train_data.pkl", help='要处理的数据的位置')  
    train.add_argument("-valid_data_file","--valid_data_file",type=str,\
        default="../../data/valid_data.pkl", help='要处理的数据的位置')  
    train.add_argument("-test_data_file","--test_data_file",type=str,\
        default="../../data/test_data.pkl", help='要处理的数据的位置')  
    
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)  # pad_size，与其他模型不统一
    train.add_argument("-use_size","--use_size",type=int,default=-1)  # pad_size，与其他模型不统一
    train.add_argument("-vocab_path","--vocab_path",type=str,\
        default="../../pretrain_model/wwm_ext/vocab.txt", help='用于初始化分词器的字典') 

    # other
    train.add_argument('--log_path', default='log/{}.log', type=str, required=False, help='训练日志存放位置') #todo
    
    # SASRec
    train.add_argument('--max_seq_length', default=100, type=int)
    train.add_argument('--item_size', default=33834, type=int) #

    train.add_argument("-load_model_path","--load_model_path",type=str,\
        default="0526.pth", help='要加载的模型的位置') 
    train.add_argument("-load_model", "--load_model", action="store_true", default=False)
    train.add_argument("-GRU_save_path","--GRU_save_path",type=str,default='gru_{}.pth') # todo

    train.add_argument('--gru_hidden_size', default=50, type=int) #
    train.add_argument('--output_size', default=50, type=int) #
    train.add_argument('--num_layers', default=3, type=int) #
    train.add_argument('--embedding_dim', default=50, type=int) #
    train.add_argument('--dropout_input', default=0, type=int) #
    train.add_argument('--dropout_hidden', default=0.0, type=float) #
    train.add_argument("--final_act",type=str,default='tanh') # todo

    train.add_argument('--do_eval', action='store_true')

    train.add_argument("-seed","--seed",type=int,default=42)  # todo
    train.add_argument("--weight_decay", type=float, default=0.0000, help="weight_decay of adam")
    train.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    train.add_argument("--adam_beta2", type=float, default=0.99, help="adam second beta value")

    train.add_argument('--sasrec_emb_path', default='data/sasrec_embed.pth', type=str)
    train.add_argument("--hidden_size",
                        type=int,
                        default=50,
                        help="hidden size of transformer model")
        
    return train

class TrainLoop_GRU():
    def __init__(self, opt, args):
        self.opt=opt
        self.args=args

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']

        self.device = "cuda:{}".format(self.args.gpu) if self.use_cuda else 'cpu'
        self.args.device = self.device
        # equal to torch.device(f"cuda:{self.opt['gpu']}")
        self.cuda_condition = self.use_cuda
        self.args.use_cuda = self.use_cuda

        self.build_data()

        # bs, item_num+1: [gt, all_item_id]
        self.default_neg_sampled = torch.tensor(
            [0] + [i for i in range(1, self.args.item_size)], dtype=torch.long).repeat(
                self.args.batch_size, 1).to(self.device)         

        self.build_model()

        self.init_optim()

    def build_model(self):
        self.model = GRU4REC(args=self.args)
        if self.args.load_model:
            self.model.load_model(self.args.load_model_path)
        if self.use_cuda:
            self.model.to(self.device)  # todo

    def build_data(self):
        # 初始化分词器
        self.tokenizer = BertTokenizer(vocab_file=self.opt['vocab_path'])  # 初始化分词器
        # build and save self.dataset
        self.dataset = {'train': None, 'valid': None, 'test': None}
        self.dataset_loader = {'train': None, 'valid': None, 'test': None}
        for subset in self.dataset:
            self.dataset[subset] = CRSdataset(subset, self.opt[f'{subset}_data_file'], self.opt, self.args, self.tokenizer, self.opt['use_size'])
            self.dataset_loader[subset] =  torch.utils.data.DataLoader(dataset=self.dataset[subset],
                                                            batch_size=self.batch_size,
                                                            shuffle=True) # todo
        # self.args.item_size += 1
        self.movie_num = self.dataset['train'].movie_num + 1
        self.args.item_size = self.dataset['train'].movie_num + 1
        self.item_size = self.dataset['train'].movie_num + 1
        
    def train(self):
        losses=[]  # 预报一次清零一IC
        best_val_NDCG=0.0
        gen_stop=False
        patience = 0
        max_patience = 5

        for i in range(self.epoch):
            train_loss = []
            # for batch_idx, batch_data in tqdm(enumerate(self.rec_train_dataloader)):
            for batch_idx, batch_data in enumerate(self.dataset_loader['train']):
                self.model.train()
                self.zero_grad()
                batch_data  = [data.to(self.device) for data in batch_data]

                gt, input_ids, target_pos, input_mask, sample_negs, len_input = batch_data[-6:]
                # print(input_ids) # bs, seq_len
                # print(target_pos) # bs, seq_len
                # print(gt)
                # print(len_input)
                # len_ = []
                # for input in input_ids:
                #     len_.append(0)
                #     for id in input:
                #         if id != 0:
                #             len_[-1] += 1

                sequence_output, _, max_out_len = self.model(input_ids, len_input, None)

                loss = self.model.cross_entropy(sequence_output, target_pos[:,:max_out_len], sample_negs[:,:max_out_len], input_mask)

                train_loss.append(loss.item())

                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if (batch_idx+1) % 1000000000000000==0:
                    # 从上次预报到现在为止的loss均值，每50个batch预报一次
                    loss = sum(losses) / len(losses)
                    logger.info('loss is %.4f' % (loss))
                    losses=[]
            ########## a batch success
            logger.info(f'Epoch {i}, train loss = {sum(train_loss)/len(train_loss)}')

            # metrics_test = self.val('train')
            metrics_test = self.val('valid')
            _ = self.val('test')
            # False是什么鬼
            if best_val_NDCG > metrics_test["NDCG50"]:
                patience += 1
                logger.info(f"[Patience = {patience}]")
                if patience >= max_patience:
                    gen_stop=True
            else:
                patience = 0
                best_val_NDCG = metrics_test["NDCG50"]
                self.model.save_model(self.opt['GRU_save_path'])
                logger.info(f"[Model saved in {self.opt['GRU_save_path']}]")

            if gen_stop==True:
                break
        metrics_test = self.val('test')

    def val(self, subset):
        assert subset in ['train', 'test', 'valid']
        self.model.eval()
        val_dataset_loader = self.dataset_loader[subset]

        metrics_test = {"Loss":0, "NDCG1":0,"NDCG10":0,"NDCG50":0,"MRR1":0,"MRR10":0,"MRR50":0,"count":0}  
        losses=[]
        # for batch_idx, batch_data in tqdm(enumerate(val_dataset_loader)):           
        for batch_idx, batch_data in enumerate(val_dataset_loader):           
            with torch.no_grad():
                batch_data  = [data.to(self.device) for data in batch_data]
                predict_ids, input_ids, target_pos, input_mask, sample_negs, len_input = batch_data[-6:]
             
                sequence_output, hidden, max_out_len = self.model(input_ids, len_input, None)

                loss = self.model.cross_entropy(sequence_output, target_pos[:,:max_out_len], sample_negs[:,:max_out_len], input_mask)

                # loss = self.model.cross_entropy(sequence_output, target_pos, sample_negs, input_mask)
                
                # bs, item_num
                for i in range(predict_ids.shape[0]):
                    self.default_neg_sampled[i][0] = predict_ids[i]
                # 推荐的结果
                test_logits = self.predict(hidden, None, self.default_neg_sampled[:predict_ids.shape[0]], self.cuda_condition)                
                # print(hidden.shape) #()
                # print(test_logits.shape) #()
                self.compute_metircs(test_logits, metrics_test)

                losses.append(loss.item())
        # test 结束
        metrics_test['Loss'] = sum(losses) / len(losses)

        for key in metrics_test:
            if 'NDCG' in key or 'MRR' in key:
                # ipdb.set_trace()
                metrics_test[key] = round(metrics_test[key] / metrics_test['count'], 4)

        logger.info(f"{subset} set's metrics = {metrics_test}")

        return metrics_test

    def predict(self, hidden, attr_ids, test_neg_sample, cuda_condition=True):
        # shorten: 只要每个batch最后一个item的representation与所有candidate rep的点击
        # hidden=(bs, hidden)

        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch 1 hidden]
        # seq_out = seq_out[:, -1, :].unsqueeze(1)
        seq_out = hidden.unsqueeze(1)

        # [batch 1 item_num]
        test_logits = torch.matmul(seq_out, test_item_emb.transpose(1, 2))
        # print(test_logits.shape) #p
        # [batch item_num]
        test_logits = test_logits[:, -1, :]

        return test_logits

    def compute_metircs(self, logit, metrics):
        MRR1, NDCG1 = self.get_metric(logit, topk=1)     
        # ipdb.set_trace()           
        metrics['MRR1'] += MRR1
        metrics['NDCG1'] += NDCG1

        MRR10, NDCG10 = self.get_metric(logit, topk=10)                
        metrics['MRR10'] += MRR10
        metrics['NDCG10'] += NDCG10

        MRR50, NDCG50 = self.get_metric(logit, topk=50)                
        metrics['MRR50'] += MRR50
        metrics['NDCG50'] += NDCG50

        # recall= self.get_recall(logit, topk=self.args.item_size)                
        # print('recallall = {}'.format(recall))

        metrics['count'] += 1

    def get_metric(self, test_logits, topk=10):
        NDCG = 0.0
        MRR = 0.0
        # [batch] 最终每个 example 中 正确答案的排位
        ranks = test_logits.argsort(descending=True).argsort()[:, 0].cpu()
        ranks_size = int(ranks.size(0))
        for rank in ranks:
            if rank < topk:
                NDCG += float(1.0 / np.log2(rank + 2.0))
            # ipdb.set_trace()
                MRR += float(1.0 / np.array(rank + 1.0))
            
        # return MRR / ranks.size(0), NDCG / ranks.size(0)
        return MRR / ranks_size, NDCG / ranks_size
    
    def get_recall(self, test_logits, topk=10):
        recall = 0
        # [batch] 最终每个 example 中 正确答案的排位
        ranks = test_logits.argsort(descending=True).argsort()[:, 0].cpu()
        ranks_size = int(ranks.size(0))
        for rank in ranks:
            if rank < topk:
                recall +=1
            
        # return MRR / ranks.size(0), NDCG / ranks.size(0)
        return recall / ranks_size

    def vector2sentence(self,batch_sen, compat=True):
        # 一个batch的sentence 从id换成token
        sentences=[]
        # for sen in batch_sen.numpy():
        #     sentences.append(self.tokenizer.convert_ids_to_tokens(sen))
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:              
                if word != 0:                    
                    sentence.append(self.tokenizer.convert_ids_to_tokens(word))
                # elif word==3:
                #     sentence.append('_UNK_')
            if compat:
                sentence = ''.join(sentence)
            sentences.append(sentence)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass
        logger.info(optims)
        return optims

    def init_optim(self):
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr_GRU, betas=betas, weight_decay=self.args.weight_decay)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

if __name__ == '__main__':    
    args=setup_args().parse_args()
    args.log_path = args.log_path.format(args.exp_name)
    args.model_save_path = args.model_save_path.format(args.exp_name)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    args.GRU_save_path = join(args.model_save_path, args.GRU_save_path.format(args.exp_name))

    global logger
    logger = create_logger(args)

    logger.info(vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loop=TrainLoop_GRU(vars(args), args)
    if args.do_eval:
        loop.val('test')
    else:
        loop.train()


