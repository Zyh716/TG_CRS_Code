import numpy as np
from tqdm import tqdm
from math import exp
import os
import signal
import json
import argparse
from dataset_p import CRSdataset
from model import BERTModel, SASRecModel, SASBERT
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

import inspect
import re
from torch.optim import Adam


def var_name(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=args.log_path)
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
    train.add_argument("-model_type", "--model_type", type=str, default='Ours')
    train.add_argument("-exp_name", "--exp_name", type=str, default='modelv1')

    # about train setting
    train.add_argument("-batch_size", "--batch_size", type=int,
                       default=8)  # todo
    train.add_argument("-lr_bert", "--lr_bert", type=float, default=1e-5)
    train.add_argument("-lr_sasrec", "--lr_sasrec", type=float, default=1e-3)
    train.add_argument("-epoch", "--epoch", type=int, default=500)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-gpu", "--gpu", type=str, default='1')
    train.add_argument('--do_eval', action='store_true')
    train.add_argument("-use_size", "--use_size", type=int,
                       default=-1)  # pad_size，与其他模型不统一
    train.add_argument("-seed", "--seed", type=int, default=43)  # todo

    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)  # pad_size，与其他模型不统一

    # about model setting
    train.add_argument("-init_add",
                       "--init_add",
                       action="store_true",
                       default=False)

    train.add_argument("-bert_path","--bert_path",type=str,\
        default="../../pretrain_model/wwm_ext/", help='要加载的模型的位置')

    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/{}')  # todo
    train.add_argument("-sasrec_save_path","--sasrec_save_path",type=str, \
        default='sasrec_{}.pth') # todo
    train.add_argument("-fusion_save_path","--fusion_save_path",type=str, \
        default='fusion_save_path_{}.pth') # todo

    train.add_argument("-load_exp_name","--load_exp_name",type=str, \
        default='v1')
    train.add_argument("-model_load_path",
                       "--model_load_path",
                       type=str,
                       default='saved_model/{}')  # todo
    train.add_argument("-load_model",
                       "--load_model",
                       action="store_true",
                       default=False)
    train.add_argument("-sasrec_load_path","--sasrec_load_path",type=str,\
        default="sasrec_{}.pth", help='要加载的模型的位置')
    train.add_argument("-fusion_load_path","--fusion_load_path",type=str, \
        default='fusion_save_path_{}.pth') # todo

    # about dataset and data setting
    train.add_argument("--raw", action="store_true", default=False)
    train.add_argument("-train_data_file","--train_data_file",type=str,\
        default="../../data/train_data.pkl", help='要处理的数据的位置')
    train.add_argument("-valid_data_file","--valid_data_file",type=str,\
        default="../../data/valid_data.pkl", help='要处理的数据的位置')
    train.add_argument("-test_data_file","--test_data_file",type=str,\
        default="../../data/test_data.pkl", help='要处理的数据的位置')
    train.add_argument("-vocab_path","--vocab_path",type=str,\
        default="../../pretrain_model/wwm_ext/vocab.txt", help='用于初始化分词器的字典')

    # other
    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='训练日志存放位置')  #todo

    # SASRec
    train.add_argument("--hidden_size", type=int, default=50, \
        help="hidden size of transformer model")
    train.add_argument("--num_hidden_layers", type=int, default=2, \
        help="number of layers")
    train.add_argument('--num_attention_heads', default=1, type=int)
    train.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    train.add_argument("--attention_probs_dropout_prob", type=float, \
        default=0.2, help="attention dropout p")
    train.add_argument("--hidden_dropout_prob", type=float, default=0.2, \
        help="hidden dropout p")
    train.add_argument("--initializer_range", type=float, default=0.02)
    train.add_argument('--max_seq_length', default=100, type=int)
    train.add_argument('--item_size', default=33834, type=int)  #

    train.add_argument("--weight_decay",
                       type=float,
                       default=0.0000,
                       help="weight_decay of adam")
    train.add_argument("--adam_beta1",
                       type=float,
                       default=0.9,
                       help="adam first beta value")
    train.add_argument("--adam_beta2",
                       type=float,
                       default=0.99,
                       help="adam second beta value")
    train.add_argument("--sasrec_emb_save_path",
                       type=str,
                       default='saved_model/sasrec_embed.pth')
    train.add_argument("--is_save_sasrec_embed",
                       default=False,
                       action='store_true')

    return train


class TrainLoop_Ours():
    def __init__(self, opt, args):
        self.opt = opt
        self.args = args

        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']
        self.use_cuda = opt['use_cuda']

        self.device = "cuda:{}".format(
            self.args.gpu) if self.use_cuda else 'cpu'
        self.args.device = self.device

        self.build_data()
        self.build_model()
        self.optimizer = self.model.get_optimizer()

    def build_data(self):
        # 初始化分词器
        self.tokenizer = BertTokenizer(
            vocab_file=self.opt['vocab_path'])  # 初始化分词器
        # build and save self.dataset
        self.dataset = {'train': None, 'valid': None, 'test': None}
        self.dataset_loader = {'train': None, 'valid': None, 'test': None}
        for subset in self.dataset:
            self.dataset[subset] = CRSdataset(logger, subset,
                                              self.opt[f'{subset}_data_file'],
                                              self.args, self.tokenizer)
            self.dataset_loader[subset] = torch.utils.data.DataLoader(
                dataset=self.dataset[subset],
                batch_size=self.batch_size,
                shuffle=True)
        # self.args.item_size += 1
        self.movie_num = self.dataset['train'].movie_num
        self.args.item_size = self.dataset['train'].movie_num

    def build_model(self):
        self.model = SASBERT(self.opt, self.args, self.movie_num)
        if self.use_cuda:
            self.model.to(self.device)  # todo

    def train(self):
        losses = []  # 预报一次清零一IC
        best_val_NDCG = 0.0
        gen_stop = False
        patience = 0
        max_patience = 3

        for i in range(self.epoch):
            train_loss = []
            for batch_idx, batch_data in tqdm(
                    enumerate(self.dataset_loader['train'])):
                ####################################### 检验输入输出ok
                # print("[Context] ", batch_data[0])
                # print("[Context] ", '\n'.join(self.vector2sentence(batch_data[0])))
                # print("[Movie]", batch_data[3])
                # ipdb.set_trace()

                self.model.train()
                self.zero_grad()
                batch_data = [data.to(self.device) for data in batch_data]

                logit = self.model(batch_data)

                y = batch_data[3]
                loss = self.model.compute_loss(logit, y, 'train')

                train_loss.append(loss.item())
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 50 == 0:
                    # 从上次预报到现在为止的loss均值，每50个batch预报一次
                    loss = sum(losses) / len(losses)
                    logger.info('loss is %.4f' % (loss))
                    losses = []

            logger.info(
                f'Epoch {i}, train loss = {sum(train_loss)/len(train_loss)}')

            # metrics_test = self.val('train')
            metrics_test = self.val('valid')
            _ = self.val('test')
            if best_val_NDCG > metrics_test["NDCG50"]:
                patience += 1
                logger.info(f"[Patience = {patience}]")
                if patience >= max_patience:
                    gen_stop = True
            else:
                patience = 0
                best_val_NDCG = metrics_test["NDCG50"]
                self.model.save_model('BERT SASRec Fusion')
                logger.info(f"[Model saved]")

            if gen_stop == True:
                break

    def val(self, subset):
        assert subset in ['train', 'test', 'valid']
        self.model.eval()
        val_dataset_loader = self.dataset_loader[subset]

        self.identity2movieId = {}

        metrics_test = {
            "Loss": 0,
            "NDCG1": 0,
            "NDCG10": 0,
            "NDCG50": 0,
            "MRR1": 0,
            "MRR10": 0,
            "MRR50": 0,
            "count": 0
        }
        losses = []
        # for batch_idx, batch_data in tqdm(enumerate(val_dataset_loader)):
        for batch_idx, batch_data in enumerate(val_dataset_loader):
            # print("[Context] ", batch_data[0])
            # print("[Context] ", '\n'.join(self.vector2sentence(batch_data[0])))
            # print("[MovieHistory]\n", batch_data[4])
            # print("[MovieMask]\n", batch_data[6])
            # print("[Movie]", batch_data[3])
            # ipdb.set_trace()

            with torch.no_grad():
                identity = batch_data[0]
                batch_data = [data.to(self.device) for data in batch_data[1:]]
                logit = self.model(batch_data)

                y = batch_data[3]
                loss = self.model.compute_loss(logit, y)

                self.compute_metircs(logit, y, metrics_test, identity)

                losses.append(loss.item())
        # test 结束
        metrics_test['Loss'] = sum(losses) / len(losses)

        for key in metrics_test:
            if 'NDCG' in key or 'MRR' in key:
                # metrics_test[key] = round(metrics_test[key] / metrics_test['count'] * 3, 4)
                metrics_test[key] = round(
                    metrics_test[key] / metrics_test['count'], 4)

        logger.info(f"{subset} set's metrics = {metrics_test}")

        # save predicted topic_id
        json.dump(self.identity2movieId, open('data/data_p_Ours/identity2movieId.json',
                                              'w'))
        logger.info('共预测了{}identity的movie id'.format(len(
            self.identity2movieId)))

        return metrics_test

    def compute_metircs(self, logit, y, metrics, identities):
        for K in [1, 10, 50]:
            # pred = logit.max(-1, keepdim=True)[1]
            # acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
            pred, pred_id = torch.topk(logit, K, dim=1)  # id=[bs, K]
            for i, gt in enumerate(y):
                gt = gt.item()
                cand_ids = pred_id[i].tolist()
                if gt in cand_ids:
                    rank = cand_ids.index(gt)
                    metrics['NDCG' + str(K)] += 1.0 / math.log(rank + 2.0, 2)
                    metrics['MRR' + str(K)] += 1.0 / (rank + 1.0)

            # record
            if K == 1:
                for i, gt in enumerate(y):
                    cand_id = pred_id[i].tolist()[0]
                    self.identity2movieId[identities[i]] = cand_id
                    if identities[i] in self.identity2movieId:
                        assert cand_id == self.identity2movieId[identities[i]]

        assert len(y.shape) == 1
        metrics['count'] += y.shape[0]
        # metrics['count'] = int(metrics['count']/3)

    def vector2sentence(self, batch_sen, compat=True):
        # 一个batch的sentence 从id换成token
        sentences = []
        # for sen in batch_sen.numpy():
        #     sentences.append(self.tokenizer.convert_ids_to_tokens(sen))
        for sen in batch_sen.numpy().tolist():
            sentence = []
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
        optims = {
            k.lower(): v
            for k, v in optim.__dict__.items()
            if not k.startswith('__') and k[0].isupper()
        }
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
        bert_param_optimizer = list(
            self.model.BERT.named_parameters())  # 模型参数名字列表
        other_param_optimizer = list(self.model.SASRec.named_parameters()) + \
            list(self.model.fusion.named_parameters())

        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt['lr_bert'])
        self.optimizer = transformers.AdamW([{
            'params': bert_param_optimizer,
            'lr': self.opt['lr_bert']
        }, {
            'params': other_param_optimizer
        }],
                                            lr=self.opt['lr_sasrec'])
        # self.scheduler = transformers.WarmupLinearSchedule(\
        #     self.optimizer, warmup_steps=self.opt['warmup_steps'], t_total=len(self.dataset_loader['train']) * self.epoch)

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()


class TrainLoop_SASRec():
    pass


class TrainLoop_BERT():
    pass


def main():
    args = setup_args().parse_args()
    args.log_path = args.log_path.format(args.exp_name)

    global logger
    logger = create_logger(args)
    logger.info(vars(args))

    if not args.do_eval:
        args.model_save_path = args.model_save_path.format(args.exp_name)
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        args.fusion_save_path = join(
            args.model_save_path, args.fusion_save_path.format(args.exp_name))
        args.sasrec_save_path = join(
            args.model_save_path, args.sasrec_save_path.format(args.exp_name))

    if args.load_model:
        args.model_load_path = args.model_load_path.format(args.load_exp_name)
        if not os.path.exists(args.model_load_path):
            logger.info('!No existing load exp dictionary')
            exit(0)
        args.fusion_load_path = join(
            args.model_load_path,
            args.fusion_load_path.format(args.load_exp_name))
        args.sasrec_load_path = join(
            args.model_load_path,
            args.sasrec_load_path.format(args.load_exp_name))

    set_seed(args)

    if args.model_type == 'Ours':
        loop = TrainLoop_Ours(vars(args), args)
    elif args.model_type == 'BERT':
        loop = TrainLoop_BERT(vars(args), args)
    elif args.model_type == 'SASRec':
        loop = TrainLoop_SASRec(vars(args), args)

    if args.do_eval:
        loop.val('test')
        if args.is_save_sasrec_embed:
            loop.save_embed()
    else:
        loop.train()


if __name__ == '__main__':
    main()