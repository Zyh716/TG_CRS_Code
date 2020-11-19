import numpy as np
from tqdm import tqdm
from math import exp
import os
import signal
import json
import argparse
from dataset import dataset, CRSdataset
from model import Model
import torch.nn as nn
from torch import optim
import torch
import nltk
import re
import pickle
import logging
import time
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import *
import ipdb
import math


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
    train.add_argument("-exp_name", "--exp_name", type=str, default='modelv1')

    # about train setting
    train.add_argument("-batch_size", "--batch_size", type=int,
                       default=64)  # todo
    train.add_argument("-lr", "--lr", type=float, default=1e-3)
    train.add_argument("-warmup", "--warmup", type=float, default=0.05)
    train.add_argument("-warmup_steps",
                       "--warmup_steps",
                       type=int,
                       default=2000)
    train.add_argument("-epoch", "--epoch", type=int, default=500)
    train.add_argument("-base_epoch", "--base_epoch", type=int, default=0)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-gpu", "--gpu", type=str, default='3')

    # about model setting
    train.add_argument("-load_dict",
                       "--load_dict",
                       action="store_true",
                       default=False)
    train.add_argument("-model_load_path","--model_load_path",type=str,\
        default="saved_model/v1.pth", help='要加载的模型的位置')
    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/{}.pth')  # todo

    # about dataset and data setting
    #  下面两个只有一个不为None, load_builded_data有则直接加载，没有就重新处理，但是可能不保存
    train.add_argument("-save_build_data",
                       "--save_build_data",
                       action="store_true",
                       default=False)
    train.add_argument("-load_builded_data",
                       "--load_builded_data",
                       action="store_true",
                       default=False)

    train.add_argument("-train_data_file","--train_data_file",type=str,\
        default="../../data/train_data.pkl", help='要处理的数据的位置')
    train.add_argument("-valid_data_file","--valid_data_file",type=str,\
        default="../../data/valid_data.pkl", help='要处理的数据的位置')
    train.add_argument("-test_data_file","--test_data_file",type=str,\
        default="../../data/test_data.pkl", help='要处理的数据的位置')

    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)  # pad_size，与其他模型不统一
    train.add_argument("-use_size", "--use_size", type=int,
                       default=-1)  # pad_size，与其他模型不统一
    train.add_argument("-vocab_path","--vocab_path",type=str,\
        default="data/vocab.pkl", help='用于初始化分词器的字典')

    # other
    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='训练日志存放位置')  #todo

    # train.add_argument('--dataset', default='../Chinese-Text-Classification-Pytorch2/THUCNews', type=str, help='原项目数据存放位置') #todo
    train.add_argument('--embedding',
                       default='data/embedding_SougouNews.npz',
                       type=str)

    train.add_argument("-dropout", "--dropout", type=float, default=0.5)
    train.add_argument("-num_filters", "--num_filters", type=int, default=256)
    train.add_argument(
        "-filter_sizes", "--filter_sizes", type=str,
        default='(2, 3, 4)')  # filter_sizes = eval(filter_sizes)
    train.add_argument("-embed",
                       "--embed",
                       type=int,
                       default=300,
                       help='embedding长度')
    train.add_argument("-n_vocab",
                       "--n_vocab",
                       type=int,
                       default=10000,
                       help='词表数量')
    train.add_argument("-embedding_pretrained",
                       "--embedding_pretrained",
                       type=bool,
                       default=True)
    train.add_argument('--do_eval', action='store_true')

    return train


class Tokenizer():
    def __init__(self, vocab):
        '''
        params: vocab: {token: id}
        '''
        pass

    def convert_tokens_to_ids(tokens):
        pass

    def convert_ids_to_tokens(ids):
        pass

    def tokenizer(word_string):
        pass


class TrainLoop():
    def __init__(self, opt, args):
        self.opt = opt
        self.args = args
        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']

        self.use_cuda = opt['use_cuda']
        self.device = "cuda:{}".format(self.args.gpu) if self.use_cuda else 'cpu'
        self.args.device = "cuda:{}".format(self.args.gpu) if self.use_cuda else 'cpu'
        # equal to torch.device(f"cuda:{self.opt['gpu']}")

        self.build_data()

        self.build_model()

        self.init_optim()

    def build_model(self):
        self.model = Model(self.args, self.movie_num)
        if self.use_cuda:
            self.model.to(self.device)  # todo

        if self.args.load_dict:
            self.args.base_epoch = self.model.load_model(args.model_load_path)

    def build_data(self):
        # 初始化分词器
        self.tokenizer = lambda x: [y for y in x]
        # vocab
        self.vocab = pickle.load(open(args.vocab_path, 'rb'))
        self.id2token = {id: token for token, id in self.vocab.items()}
        self.id2token['<SENT>'] = len(self.id2token)

        # build and save self.dataset
        self.dataset = {'train': None, 'valid': None, 'test': None}
        self.dataset_loader = {'train': None, 'valid': None, 'test': None}
        for subset in self.dataset:
            self.dataset[subset] = CRSdataset(subset, self.opt[f'{subset}_data_file'], self.opt, \
                                                self.args, self.tokenizer, self.vocab, \
                                                self.opt['save_build_data'], self.opt['load_builded_data'], self.opt['use_size'])
            self.dataset_loader[subset] = torch.utils.data.DataLoader(
                dataset=self.dataset[subset],
                batch_size=self.batch_size,
                shuffle=True)  # todo
            self.movie_num = self.dataset[subset].movie_num

    def train(self):
        losses = []
        best_val_NDCG = 0.0
        gen_stop = False
        patience = 0
        max_patience = 5

        for epoch in range(self.args.epoch):
            epoch = self.args.base_epoch + epoch
            train_loss = []
            # print('Epoch [{}/{}]'.format(epoch + 1, self.args.epoch))
            for batch_idx, batch_data in enumerate(
                    self.dataset_loader['train']):
                ####################################### 检验输入输出ok
                # print("[Context] ", batch_data[0])
                # print("[Context] ", '\n'.join(self.vector2sentence(batch_data[0])))
                # ipdb.set_trace()
                self.model.train()
                self.zero_grad()

                contexts, length, y = (data.to(self.device)
                                       for data in batch_data)
                logit = self.model([contexts, length])
                # scalar
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
                f'Epoch {epoch}, train loss = {sum(train_loss)/len(train_loss):.4f}'
            )
            # metrics_test = self.val('train')
            metrics_test = self.val('valid')
            _ = self.val('test')
            if best_val_NDCG > metrics_test["NDCG50"]:
                patience += 1
                logger.info(f"[Patience = {patience}]")
                if patience >= 5:
                    gen_stop = True
            else:
                patience = 0
                best_val_NDCG = metrics_test["NDCG50"]
                self.model.save_model(self.opt['model_save_path'],
                                      self.optimizer, epoch)
                logger.info(f"[Model saved in {self.opt['model_save_path']}]")
            if gen_stop == True:
                break

    def val(self, subset):
        assert subset in ['train', 'test', 'valid']
        self.model.eval()
        val_dataset_loader = self.dataset_loader[subset]

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
            with torch.no_grad():
                contexts, length, y = (data.to(self.device)
                                       for data in batch_data)
                logit = self.model([contexts, length])
                loss = self.model.compute_loss(logit, y)

                self.compute_metircs(logit, y, metrics_test)
                losses.append(loss.item())
        # test 结束
        metrics_test['Loss'] = round(sum(losses) / len(losses), 4)

        for key in metrics_test:
            if 'NDCG' in key or 'MRR' in key:
                metrics_test[key] = round(
                    metrics_test[key] / metrics_test['count'] * 3, 4)

        logger.info(f"{subset} set's metrics = {metrics_test}")

        return metrics_test

    def compute_metircs(self, logit, y, metrics):
        for K in [1, 10, 50]:
            pred = logit.max(-1, keepdim=True)[1]
            # acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
            pred, pred_id = torch.topk(logit, K, dim=1)  # id=[bs, K]
            for i, gt in enumerate(y.squeeze()):
                gt = gt.item()
                cand_ids = pred_id[i].tolist()
                if gt in cand_ids:
                    rank = cand_ids.index(gt)
                    metrics['NDCG' + str(K)] += 1.0 / math.log(rank + 2.0, 2)
                    metrics['MRR' + str(K)] += 1.0 / (rank + 1.0)
                metrics['count'] += 1
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
                    token = self.id2token[word]
                    if token == '<PAD>':
                        break
                    sentence.append(token)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.lr)
        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler.step() # 学习率衰减

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()


if __name__ == '__main__':
    args = setup_args().parse_args()
    args.log_path = args.log_path.format(args.exp_name)
    args.model_save_path = args.model_save_path.format(args.exp_name)

    args.filter_sizes = eval(args.filter_sizes)

    global logger
    logger = create_logger(args)

    logger.info(vars(args))

    loop = TrainLoop(vars(args), args)
    if args.do_eval:
        loop.val('test')
    else:
        loop.train()
