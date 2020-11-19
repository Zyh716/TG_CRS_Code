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
# from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
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
    train.add_argument("-lr","--lr",type=float,default=1e-3)
    # train.add_argument("-warmup","--warmup",type=float,default=0.05)
    # train.add_argument("-warmup_steps","--warmup_steps",type=int,default=2000)
    train.add_argument("-epoch","--epoch",type=int,default=500)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True) 
    train.add_argument("-gpu","--gpu",type=str,default='1')

    # about model setting
    train.add_argument("-lstm_path","--lstm_path",type=str,\
        default="saved_model/{}", help='要加载的模型的位置') 
    train.add_argument("-init_add", "--init_add", action="store_true", default=False)
    train.add_argument("-init_bert_from_pretrain", "--init_bert_from_pretrain", action="store_true", default=False)
    train.add_argument("-init_from_fineturn", "--init_from_fineturn", action="store_true", default=False)
    train.add_argument("-model_save_path","--model_save_path",type=str,default='saved_model/{}') # todo
    
    # about dataset and data setting
    #  下面两个只有一个不为None, load_builded_data有则直接加载，没有就重新处理，但是可能不保存
    train.add_argument("-save_build_data","--save_build_data", action="store_true", default=False)
    train.add_argument("-load_builded_data","--load_builded_data", action="store_true", default=False)
 
    train.add_argument("-train_data_file","--train_data_file",type=str,\
        default="../../data/data1030/output/train_cut.pkl", help='要处理的数据的位置')  
    train.add_argument("-valid_data_file","--valid_data_file",type=str,\
        default="../../data/data1030/output/valid_cut.pkl", help='要处理的数据的位置')  
    train.add_argument("-test_data_file","--test_data_file",type=str,\
        default="../../data/data1030/output/test_cut.pkl", help='要处理的数据的位置')  
    train.add_argument("-topic_to_id","--topic_to_id",type=str,\
        default="../../data/topic_to_id_2k5.json", help='要处理的数据的位置')  

    train.add_argument("-bpe2index","--bpe2index",type=str,\
        default="../../data/data1030/output/bpe2index.json", help='要处理的数据的位置')  
    train.add_argument("-bpe2vec","--bpe2vec",type=str,\
        default="../../data/data1030/bpe2vec.npy", help='要处理的数据的位置')  
    train.add_argument("-jieba_dict","--jieba_dict",type=str,\
        default="../../data/data1030/output/dict.txt", help='要处理的数据的位置')

    train.add_argument("-max_c_length","--max_c_length",type=int,default=128)  # pad_size，与其他模型不统一
    train.add_argument("-use_size","--use_size",type=int,default=-1)  # pad_size，与其他模型不统一
    train.add_argument("-vocab_path","--vocab_path",type=str,\
        default="../../pretrain_model/wwm_ext/vocab.txt", help='用于初始化分词器的字典') 
    train.add_argument("-embedding_dim","--embedding_dim",type=int,default=300)  # pad_size，与其他模型不统一
    train.add_argument("-hidden_size","--hidden_size",type=int,default=300)  # pad_size，与其他模型不统一
    train.add_argument("-num_layers","--num_layers",type=int,default=1)  # pad_size，与其他模型不统一
    train.add_argument("-dropout_hidden","--dropout_hidden",type=float,default=0)  # pad_size，与其他模型不统一
    # train.add_argument("-num_layers","--num_layers",type=int,default=1)  # pad_size，与其他模型不统一
    

    # other
    train.add_argument('--log_path', default='log/{}.log', type=str, required=False, help='训练日志存放位置') #todo
    train.add_argument('--do_eval', action='store_true')
    
    return train

class TrainLoop():
    def __init__(self, opt, args):
        self.opt=opt
        self.args=args

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        self.device = "cuda:{}".format(args.gpu) if self.use_cuda else 'cpu'
        self.args.device = self.device
        self.build_data()

        self.build_model()

        self.init_optim()

        self.mean = lambda x_list: sum(x_list) / len(x_list)

    def build_data(self):
        # 初始化分词器
        # self.tokenizer = BertTokenizer(vocab_file=self.opt['vocab_path'])  # 初始化分词器
        # build and save self.dataset
        self.dataset = {'train': None, 'valid': None, 'test': None}
        self.dataset_loader = {'train': None, 'valid': None, 'test': None}
        for subset in self.dataset:
            self.dataset[subset] = CRSdataset(subset, self.opt[f'{subset}_data_file'], self.opt, self.args, \
                self.opt['save_build_data'], self.opt['load_builded_data'])
            self.dataset_loader[subset] =  torch.utils.data.DataLoader(dataset=self.dataset[subset],
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn = collate_fn) # todo
            args.topic_class_num = self.dataset[subset].topic_class_num

    def build_model(self):
        self.model = Model(self.args)
        if self.use_cuda:
            self.model.to(self.device)  # todo

    def train(self):
        #self.model.load_model()
        losses=[]
        # topic_losses, req_losses, rec_losses, topic_id_loss = [], [], [], []
        best_val_Hits=0.0
        gen_stop=False
        patience = 0
        max_patience = 3

        for i in range(self.epoch):
            train_loss = []
            for batch_idx, batch_data in tqdm(enumerate(self.dataset_loader['train'])):
            # for batch_idx, batch_data in enumerate(self.dataset_loader['train']):
                # context, context_mask, \
                #     topic_path_kw, topic_path_attitude, topic_mask, \
                #     user_profile, profile_mask, \
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

                batch_data = [data.to(self.device) for data in batch_data]

                # print("[Context] ", '\n'.join(self.vector2sentence(batch_data[0].cpu())), '\n')
                # print("[topic_path_kw] ", '\n'.join(self.vector2sentence(batch_data[2].cpu())), '\n')
                # print("[target_topic] ", '\n'.join([self.dataset['train'].id_to_topic[id] for id in batch_data[-1].cpu().tolist()]), '\n')
                # for one_user_profile in batch_data[5].cpu():
                #     print("[one_user_profile] ", '\n'.join(self.vector2sentence(one_user_profile)), '\n')

                input = batch_data[:7]
                y_type = batch_data[7]
                y_topic_id = batch_data[8]
                output = self.model(input)

                # loss, loss_topic, loss_req, loss_rec,loss_t_id = self.model.compute_loss(output, y_type, y_topic_id, 'train')
                # print('output.shape = ', output.shape)
                # print('y_topic_id.shape = ', y_topic_id.shape)
                # ipdb.set_trace()
                loss = self.model.compute_loss(output, y_type, y_topic_id, 'train')

                train_loss.append(loss.item())
                # train_topic_losses.append(loss_topic.item())
                # train_req_losses.append(loss_req.item())
                # train_rec_losses.append(loss_rec.item())
                # train_topic_id_loss.append(loss_t_id.item())
                
                losses.append(loss.item())
                # topic_losses.append(loss_topic.item())
                # req_losses.append(loss_req.item())
                # rec_losses.append(loss_rec.item())
                # topic_id_loss.append(loss_t_id.item())

                loss.backward()

                self.optimizer.step()

                if (batch_idx+1) % 50==0:
                    # logger.info('loss is %.4f (%.4f, %.4f, %.4f, %.4f)' % (
                    #     self.mean(losses), self.mean(topic_losses), self.mean(req_losses), self.mean(rec_losses), self.mean(topic_id_loss)))
                    # losses=[]
                    # topic_losses, req_losses, rec_losses, topic_id_loss= [], [], [], []

                    logger.info('loss is %.4f' % (
                        self.mean(losses)))
                    losses=[]

            ########## a batch success
            # logger.info('Epoch {}, train loss = {:.4f} ({:.4f} {:.4f} {:.4f} {:.4f})'.format(
            #     i, self.mean(train_loss), self.mean(train_topic_losses), self.mean(train_req_losses), self.mean(train_rec_losses), self.mean(train_topic_id_loss)))
            logger.info('Epoch {}, train loss = {:.4f} '.format(
                i, self.mean(train_loss)))
            
            # metrics_test = self.val('train')
            metrics_test = self.val('valid')
            _ = self.val('test')

            if best_val_Hits > metrics_test["TopicId_Hits@3"]:
                patience += 1
                logger.info(f"[Patience = {patience}]")
                if patience >= max_patience:
                    gen_stop=True
            else:
                patience = 0
                best_val_Hits = metrics_test["TopicId_Hits@3"]
                self.model.save_model(self.opt['model_save_path'])
                logger.info(f"[Model saved in {self.opt['model_save_path']}]")

            if gen_stop==True:
                break

    def val(self, subset):
        assert subset in ['train', 'test', 'valid']
        self.model.eval()
        val_dataset_loader = self.dataset_loader[subset]

        # metrics_test = {"Loss":0, "topic_loss":0, "req_loss":0, "rec_loss":0, \
        #     "Hits1":0, "Hits10":0, "Hits50":0, "Recall1":0, "Recall10":0, "Recall50":0,\
        #     "Req":0, "Rec":0, "count":0}  
        # metrics_test = {"Loss":0, "topic_loss":0, "req_loss":0, "rec_loss":0, "topic_id_loss":0\
        #     "Recall1":0, "Recall10":0, "Recall50":0,\
        #     "Topic":0, "Req":0, "Rec":0, "TopicId":0, "count":0}  # todo 改名
        # metrics_test = {"Loss":0, "topic_loss":0, "req_loss":0, "rec_loss":0, "topic_id_loss":0, \
        #     "Topic":0, "Req":0, "Rec":0, "TopicId":0, "count":0}  # todo 改名
        # metrics_test = {"Loss":0, "topic_loss":0, "req_loss":0, "rec_loss":0, "topic_id_loss":0, \
        #     "Topic_F1":0, "Req_F1":0, "Rec_F1":0, "TopicId_P":0, "count":0,
        #     "Topic_TP":0, "Topic_TN":0, "Topic_FP":0, "Topic_FN":0, "Topic_R":0, "Topic_P":0,
        #     "Req_TP":0, "Req_TN":0, "Req_FP":0, "Req_FN":0, "Req_R":0, "Req_P":0,
        #     "Rec_TP":0, "Rec_TN":0, "Rec_FP":0, "Rec_FN":0, "Rec_R":0, "Rec_P":0,}
        metrics_test = {"Loss":0, \
            "TopicId_Hits@1":0, "TopicId_Hits@3":0, "TopicId_Hits@5":0,"count":0}
        losses=[]
        # topic_losses, req_losses, rec_losses, topic_id_loss = [], [], [], []
        
        # for batch_idx, batch_data in tqdm(enumerate(val_dataset_loader)):           
        for batch_idx, batch_data in enumerate(val_dataset_loader):           
            with torch.no_grad():
                batch_data = [data.to(self.device) for data in batch_data]
                input = batch_data[:7]
                y_type = batch_data[7]
                y_topic_id = batch_data[8]
                output = self.model(input)
                
                # loss, loss_topic, loss_req, loss_rec, loss_t_id = self.model.compute_loss(output, y_type, y_topic_id)
                loss = self.model.compute_loss(output, y_type, y_topic_id, subset)

                # print('output.shape = ', output.shape)
                # print('y_topic_id.shape = ', y_topic_id.shape)
                # ipdb.set_trace()
                self.compute_metircs(output, y_type, y_topic_id, metrics_test)

                losses.append(loss.item())
                # topic_losses.append(loss_topic.item())
                # req_losses.append(loss_req.item())
                # rec_losses.append(loss_rec.item())
                # topic_id_loss.append(loss_t_id.item())
        # test 结束
        metrics_test['Loss'] = round(sum(losses) / len(losses), 4)
        # metrics_test['topic_loss'] = round(sum(topic_losses) / len(topic_losses), 4)
        # metrics_test['req_loss'] = round(sum(req_losses) / len(req_losses), 4)
        # metrics_test['rec_loss'] = round(sum(rec_losses) / len(rec_losses), 4)
        # metrics_test['topic_id_loss'] = round(sum(topic_id_loss) / len(topic_id_loss), 4)

        metrics_test['TopicId_Hits@1'] = round(metrics_test['TopicId_Hits@1'] / metrics_test['count'], 4)
        metrics_test['TopicId_Hits@3'] = round(metrics_test['TopicId_Hits@3'] / metrics_test['count'], 4)
        metrics_test['TopicId_Hits@5'] = round(metrics_test['TopicId_Hits@5'] / metrics_test['count'], 4)
        # for key in metrics_test:
        #     # if 'Recall' in key:
        #     #     metrics_test[key] = round(metrics_test[key] / metrics_test['count'], 4)
        #     if '_F1' in key:
        #         pre = key[:-3]
        #         R = metrics_test[pre+'_TP'] / (metrics_test[pre+'_TP']  +  metrics_test[pre+'_FN'])
        #         P = metrics_test[pre+'_TP'] / (metrics_test[pre+'_TP']  +  metrics_test[pre+'_FP'])
        #         metrics_test[pre+'_P'] = P
        #         metrics_test[pre+'_R'] = R
        #         metrics_test[key] = 2*P*R/(P+R)

        logger.info(f"{subset} set's metrics = {metrics_test}")

        return metrics_test

    def compute_metircs(self, output, y_type, y_topic_id, metrics):
        # out_topic, out_request, out_rec, out_topic_id = output
        out_topic_id = output
        y_topic, y_request, y_rec = y_type[:, 0], y_type[:, 1], y_type[:, 2]
        
            
        logit = out_topic_id
        y = y_topic_id
        pred, pred_id = torch.topk(logit, 1, dim=1) # id=[bs, K]
        # ipdb.set_trace()
        for i, gt in enumerate(y):
            gt = gt.item()
            cand_ids = pred_id[i].tolist()
            if gt in cand_ids:
                metrics['TopicId_Hits@1'] += 1
            metrics['count'] += 1
        
        pred, pred_id = torch.topk(logit, 3, dim=1) # id=[bs, K]
        for i, gt in enumerate(y):
            gt = gt.item()
            cand_ids = pred_id[i].tolist()
            if gt in cand_ids:
                metrics['TopicId_Hits@3'] += 1

        pred, pred_id = torch.topk(logit, 5, dim=1) # id=[bs, K]
        for i, gt in enumerate(y):
            gt = gt.item()
            cand_ids = pred_id[i].tolist()
            if gt in cand_ids:
                metrics['TopicId_Hits@5'] += 1
        
    def vector2sentence(self,batch_sen, compat=True):
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
        param_optimizer = list(self.model.named_parameters())  # 模型参数名字列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # self.optimizer = BertAdam(optimizer_grouped_parameters,
        #                     lr=self.opt['lr'],
        #                     warmup=self.opt['warmup'],
        #                     t_total=len(self.dataset_loader['train']) * self.epoch
        #                     )

        # self.optimizer = transformers.AdamW(self.model.parameters(), lr=self.opt['lr'])
        self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt['lr'])
        # self.scheduler = transformers.WarmupLinearSchedule(\
        #     self.optimizer, warmup_steps=self.opt['warmup_steps'], t_total=len(self.dataset_loader['train']) * self.epoch)

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
    args.lstm_path = args.lstm_path.format(args.exp_name)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    # os.environ['CUDA_VISIBLE_DEVICES']=args.gpu    
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    global logger
    logger = create_logger(args)

    logger.info(vars(args))

    loop=TrainLoop(vars(args), args)
    if args.do_eval:
        loop.val('test')
    else:
        loop.train()
