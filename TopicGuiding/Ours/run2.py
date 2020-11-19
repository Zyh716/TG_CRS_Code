import numpy as np
from tqdm import tqdm
from math import exp
import os
import signal
import json
import argparse
from dataset import CRSdataset, collate_fn
from model import Model
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


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler,用于写入日志文件
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler,用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-exp_name", "--exp_name", type=str, default='modelv1')

    # about train setting
    train.add_argument("-batch_size", "--batch_size", type=int, default=8)
    train.add_argument("-lr", "--lr", type=float, default=1e-5)
    train.add_argument("-epoch", "--epoch", type=int, default=500)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-gpu", "--gpu", type=str, default='1')

    train.add_argument("-bert_path",
                       "--bert_path",
                       type=str,
                       default="../../pretrain_model/wwm_ext/",
                       help='要加载的模型的位置')
    train.add_argument("-vocab_path",
                       "--vocab_path",
                       type=str,
                       default="../../pretrain_model/wwm_ext/vocab.txt",
                       help='用于初始化分词器的字典')

    train.add_argument("-init_add",
                       "--init_add",
                       action="store_true",
                       default=False)
    train.add_argument("-init_bert_from_pretrain",
                       "--init_bert_from_pretrain",
                       action="store_true",
                       default=False,
                       help='加载预训练的BERT')
    train.add_argument("-init_from_fineturn",
                       "--init_from_fineturn",
                       action="store_true",
                       default=False,
                       help='加载已经本项目存储的BERT')
    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/{}')

    # about dataset and data setting
    train.add_argument("-raw", "--raw", action="store_true", default=False)

    train.add_argument("-train_data_file",
                       "--train_data_file",
                       type=str,
                       default="../../data/train_data.pkl",
                       help='要处理的数据的位置')
    train.add_argument("-valid_data_file",
                       "--valid_data_file",
                       type=str,
                       default="../../data/valid_data.pkl",
                       help='要处理的数据的位置')
    train.add_argument("-test_data_file",
                       "--test_data_file",
                       type=str,
                       default="../../data/test_data.pkl",
                       help='要处理的数据的位置')
    train.add_argument("-topic_to_id",
                       "--topic_to_id",
                       type=str,
                       default="../../data/topic_to_id_2k5.json",
                       help='要处理的数据的位置')

    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)  # pad_size,与其他模型不统一
    train.add_argument("-use_size", "--use_size", type=int,
                       default=-1)  # pad_size,与其他模型不统一

    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='训练日志存放位置')  #todo
    train.add_argument('--do_eval', action='store_true')

    return train


class TrainLoop():
    def __init__(self, opt, args):
        self.opt = opt
        self.args = args

        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']

        self.use_cuda = opt['use_cuda']
        self.device = "cuda:{}".format(args.gpu) if self.use_cuda else 'cpu'
        self.args.device = self.device

        self.build_data()
        self.build_model()
        self.init_optim()

        self.mean = lambda x_list: sum(x_list) / len(x_list)

    def build_data(self):
        # 初始化分词器
        self.tokenizer = BertTokenizer(
            vocab_file=self.opt['vocab_path'])  # 初始化分词器
        # build and save self.dataset
        self.dataset = {'train': None, 'valid': None, 'test': None}
        self.dataset_loader = {'train': None, 'valid': None, 'test': None}
        for subset in self.dataset:
            self.dataset[subset] = CRSdataset(
                subset,
                self.opt[f'{subset}_data_file'],
                self.opt,
                self.args,
                self.tokenizer,
            )
            self.dataset_loader[subset] = torch.utils.data.DataLoader(
                dataset=self.dataset[subset],
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn)  # todo
        args.topic_class_num = self.dataset['train'].topic_class_num

    def build_model(self):
        self.model = Model(self.args)
        if self.use_cuda:
            self.model.to(self.device)  # todo

    def train(self):
        losses = []
        best_val_Hits = 0.0
        gen_stop = False
        patience = 0
        max_patience = 3

        for i in range(self.epoch):
            train_loss = []
            for batch_idx, batch_data in tqdm(
                    enumerate(self.dataset_loader['train'])):
                # for batch_idx, batch_data in enumerate(self.dataset_loader['train']):
                # context, context_mask,
                #     topic_path_kw, topic_path_attitude, topic_mask,
                #     user_profile, profile_mask,
                #     target_type, target_topic= batch_data
                # print(batch_data[0])
                # print("[Context] ", '\n'.join(self.vector2sentence(batch_data[0])), '\n')
                # print("[topic_path_kw] ", '\n'.join(self.vector2sentence(batch_data[2])), '\n')
                # print("[target_topic] ", '\n'.join([self.dataset['train'].id_to_topic[id] for id in batch_data[-1].tolist()]), '\n')
                # # for one_user_profile in user_profile:
                # #     print("[one_user_profile] ", '\n'.join(self.vector2sentence(one_user_profile)), '\n')
                # print("[user_profile] ", '\n'.join(self.vector2sentence(batch_data[6])), '\n')

                # ipdb.set_trace()

                self.model.train()
                self.zero_grad()

                batch_data = [data.to(self.device) for data in batch_data[1:]]

                # print("[Context] ", '\n'.join(self.vector2sentence(batch_data[0].cpu())), '\n')
                # print("[topic_path_kw] ", '\n'.join(self.vector2sentence(batch_data[2].cpu())), '\n')
                # print("[target_topic] ", '\n'.join([self.dataset['train'].id_to_topic[id] for id in batch_data[-1].cpu().tolist()]), '\n')
                # for one_user_profile in batch_data[5].cpu():
                #     print("[one_user_profile] ", '\n'.join(self.vector2sentence(one_user_profile)), '\n')

                input = batch_data[:-1]
                y_topic_id = batch_data[6]
                output = self.model(input)

                loss = self.model.compute_loss(output, y_topic_id)

                train_loss.append(loss.item())
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 50 == 0:
                    logger.info('loss is %.4f' % (self.mean(losses)))
                    losses = []

            logger.info('Epoch {}, train loss = {:.4f} '.format(
                i, self.mean(train_loss)))

            # metrics_test = self.val('train')
            metrics_test = self.val('valid')
            _ = self.val('test')

            if best_val_Hits > metrics_test["TopicId_Hits@3"]:
                patience += 1
                logger.info(f"[Patience = {patience}]")
                if patience >= max_patience:
                    gen_stop = True
            else:
                patience = 0
                best_val_Hits = metrics_test["TopicId_Hits@3"]
                self.model.save_model(self.opt['model_save_path'])
                logger.info(f"[Model saved in {self.opt['model_save_path']}]")

            if gen_stop:
                break

    def val(self, subset):
        assert subset in ['train', 'test', 'valid']
        self.model.eval()
        val_dataset_loader = self.dataset_loader[subset]

        self.identity2topicId = {}

        metrics_test = {
            "Loss": 0,
            "TopicId_Hits@1": 0,
            "TopicId_Hits@3": 0,
            "TopicId_Hits@5": 0,
            "count": 0
        }
        losses = []

        for batch_idx, batch_data in enumerate(val_dataset_loader):
            with torch.no_grad():
                identity = batch_data[0]
                batch_data = [data.to(self.device) for data in batch_data[1:]]
                
                input = batch_data[:-1]
                y_topic_id = batch_data[6]
                output = self.model(input)

                loss = self.model.compute_loss(output, y_topic_id)
                self.compute_metircs(output, y_topic_id, metrics_test, identity)

                losses.append(loss.item())

        metrics_test['Loss'] = round(sum(losses) / len(losses), 4)
        metrics_test['TopicId_Hits@1'] = round(
            metrics_test['TopicId_Hits@1'] / metrics_test['count'], 4)
        metrics_test['TopicId_Hits@3'] = round(
            metrics_test['TopicId_Hits@3'] / metrics_test['count'], 4)
        metrics_test['TopicId_Hits@5'] = round(
            metrics_test['TopicId_Hits@5'] / metrics_test['count'], 4)

        logger.info(f"{subset} set's metrics = {metrics_test}")

        # save predicted topic_id
        json.dump(self.identity2topicId, open('data/identity2topicId.json',
                                              'w'))
        logger.info('共预测了{}identity的topic id'.format(len(
            self.identity2topicId)))

        return metrics_test

    def compute_metircs(self, output, y_topic_id, metrics, identities=None):
        logit = output
        y = y_topic_id
        pred, pred_id = torch.topk(logit, 1, dim=1)  # id=[bs, K]
        # ipdb.set_trace()
        for i, gt in enumerate(y):
            gt = gt.item()
            cand_ids = pred_id[i].tolist()
            if gt in cand_ids:
                metrics['TopicId_Hits@1'] += 1
            metrics['count'] += 1

        # record
        for i, gt in enumerate(y):
            cand_id = pred_id[i].tolist()[0]
            self.identity2topicId[identities[i]] = cand_id
            if identities[i] in self.identity2topicId:
                assert cand_id == self.identity2topicId[identities[i]]

        pred, pred_id = torch.topk(logit, 3, dim=1)  # id=[bs, K]
        for i, gt in enumerate(y):
            gt = gt.item()
            cand_ids = pred_id[i].tolist()
            if gt in cand_ids:
                metrics['TopicId_Hits@3'] += 1

        pred, pred_id = torch.topk(logit, 5, dim=1)  # id=[bs, K]
        for i, gt in enumerate(y):
            gt = gt.item()
            cand_ids = pred_id[i].tolist()
            if gt in cand_ids:
                metrics['TopicId_Hits@5'] += 1

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

    def init_optim(self):
        param_optimizer = list(self.model.named_parameters())  # 模型参数名字列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]

        self.optimizer = transformers.AdamW(optimizer_grouped_parameters,
                                            lr=self.opt['lr'])

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
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    global logger
    logger = create_logger(args)

    logger.info(vars(args))

    loop = TrainLoop(vars(args), args)
    if args.do_eval:
        loop.val('test')
    else:
        loop.train()
