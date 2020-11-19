import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import nltk
import jieba
import pickle
import os
import ipdb
import random


class CRSdataset(Dataset):
    def __init__(self, subset, filename, opt, args, tokenizer, use_size=-1):
        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.opt = opt
        self.args = args
        
        # load data
        only_first_movie = False
        if self.args.use_size == -1:
            f = pickle.load(open(filename, 'rb'))[:]
        else:
            f = pickle.load(open(filename, 'rb'))[:self.args.use_size]

        self.load_movie()
        self.unk_movie_id = len(self.db2id)
        self.movie_num = len(self.db2id) + 1
        print("[Load {} movies(+1)]".format(self.movie_num))

        self.tokenizer = tokenizer
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.unk_id = self.tokenizer.convert_tokens_to_ids('[UNK]')
        self.sent_split_id = self.tokenizer.convert_tokens_to_ids('[unused1]') #1
        self.word_split_id = self.tokenizer.convert_tokens_to_ids('[unused2]') #2
        self.word_split = '[unused2]' #2

        # define data structs
        self.data=[] # 
        self.both_data = {} # identity: sample, 能够根据索引抽取，后面会加入user history
        save_file = 'data/{}processed_data.pkl' 
        empty_conv_ids_file = open('empty_conv_ids.txt', 'a')

        if not self.args.raw:
            self.data = pickle.load(open(save_file.format(subset), 'rb'))
            if use_size != -1:
                self.data = self.data[:use_size]
            print(f"[Load {len(self.data)} cases, from {save_file.format(subset)}]")

        else:   
            empty_ids_set = set()
            for conv in tqdm(f):
                # contexts_token = ["[CLS]"]  # list of token, ['[CLS]' UTTER1  "[SEP]"  UTTER2  "[SEP]" ]
                conv_id = conv['conv_id']
                contexts_index = [] # list of token, ['[CLS]' UTTER1  UTTER2  "[SEP]" Target ]

                for message in conv['messages']:
                    message_id, content, role = message['local_id'], message['content'], message['role']
                    # 如果这一步要推荐电影，就产生一个sample：上文context + 当前要推荐的电影 这个pair
                    if role == 'Recommender' and message_id in conv['mentionMovies']:
                        # 将douban-id映射为global-id
                        movie_id = int(conv['mentionMovies'][message_id][0])
                        movie_id = self.db2id[movie_id]
                        # 获得context对应的两种mask: seg_id, att_mask # 短则补齐，长则切断
                        cur_contexts_index = contexts_index[:-1] + [self.sep_id]
                        self.max_len_inside = self.max_c_length - 1 # cls

                        if len(cur_contexts_index) < self.max_len_inside:
                            cur_contexts_index = [self.cls_id] + cur_contexts_index

                            cur_contexts_index = cur_contexts_index + [0] * (self.max_c_length - len(cur_contexts_index))

                            types = [0] * len(cur_contexts_index) + [1] * (self.max_c_length - len(cur_contexts_index))  # mask部分 segment置为1
                            masks = [1] * len(cur_contexts_index) + [0] * (self.max_c_length - len(cur_contexts_index))
                        else:
                            cur_contexts_index = [self.cls_id] + cur_contexts_index[-self.max_len_inside:]
                            types = [0] * len(cur_contexts_index)
                            masks = [1] * len(cur_contexts_index)

                        assert len(cur_contexts_index) == self.max_c_length
                        assert cur_contexts_index[0] == self.cls_id
                        assert cur_contexts_index[-1] == self.sep_id or cur_contexts_index[-1] == self.pad_id

                        case = [cur_contexts_index, types, masks, movie_id]
                        self.data.append(case)
                        identity = str(conv_id) + '/' + str(message_id)
                        self.both_data[identity] = case

                        if only_first_movie:
                            break

                    content_token = tokenizer.tokenize(content)
                    content_index = tokenizer.convert_tokens_to_ids(content_token) + [self.sent_split_id]
                    contexts_index.extend(content_index)

            print(f"[Load {len(f)} convs, Extract {len(self.data)} (bert_used) cases, from {filename}]")

            # 以上是bert用的，以下是sasrec用的

            conv2user = pickle.load(open('../../data/0619conv2user.pkl', 'rb'))  # 检验无误
            id2history = pickle.load(open('../../data/{}_identity2history.pkl'.format(subset), 'rb'))
            
            num_history = 0
            history_total_num = 0
            self.max_len = self.args.max_seq_length
            # print(empty_ids_set)
            for conv in f[:]:
                conv_id = conv['conv_id'] # int
                # if int(conv_id) in empty_ids_set:
                #     continue
                # ipdb.set_trace()
                user = conv2user[str(conv_id)] # str
                conv_movie_list = []
                for message_id, (movieId, m_name) in conv['mentionMovies'].items():
                    # 每个要推荐的位置生成一个history，与context的data结合在一起
                    identity = str(conv_id) + '/' + str(message_id)
                    if identity not in self.both_data:
                        print(identity, file=empty_conv_ids_file)
                        continue
                    seq = id2history[identity]

                    movie_list = seq + conv_movie_list # list of int, long-history + short history
                    movie_list = [id + 1 for id in movie_list]
                    history_total_num += len(movie_list)
                    input_ids = movie_list
                    if input_ids == []:
                        input_ids = [self.unk_movie_id]
                    self.both_data[identity][-1] += 1
                    target_pos = movie_list[1:] + [self.both_data[identity][-1]] # no use
                    # print('self.both_data[identity][-1] = ', self.both_data[identity][-1])

                    input_mask = [1]*len(input_ids)

                    sample_negs = []# no use
                    seq_set = set(input_ids)
                    for _ in input_ids:
                        sample_negs.append(self.neg_sample(seq_set)) #用于训练的
                    len_input = len(input_ids)
                    len_input = self.max_len if len_input>self.max_len else len_input

                    if len(input_ids) < self.max_len:
                        pad_len = self.max_len - len(input_ids)
                        input_ids = input_ids + [0] * pad_len 
                        target_pos = target_pos + [0] * pad_len 
                        input_mask = input_mask + [0] * pad_len 
                        sample_negs = sample_negs + [0] * pad_len 
                    else:
                        input_ids = input_ids[-self.max_len:]
                        target_pos = target_pos[-self.max_len:]
                        input_mask = input_mask[-self.max_len:]
                        sample_negs = sample_negs[-self.max_len:]
                    
                    # if len(target_pos) < self.max_len:
                    #     pad_len = self.max_len - len(target_pos)                    
                    #     target_pos = [0] * pad_len + target_pos
                    # else:                      
                    #     target_pos = target_pos[-self.max_len:]

                    assert len(input_ids) == self.max_len
                    assert len(target_pos) == self.max_len, "{}/{}".format(len(target_pos), self.max_len)
                    assert len(input_mask) == self.max_len
                    assert len(sample_negs) == self.max_len
                    if identity in self.both_data:
                        assert self.db2id[int(movieId)] + 1 == self.both_data[identity][-1]
                        self.both_data[identity].extend(
                            [input_ids, target_pos, input_mask, sample_negs, len_input])
                    else:
                        print(identity, file=empty_conv_ids_file)
                        # empty_ids_set.add(conv_id)
                    conv_movie_list.append(self.db2id[int(movieId)])
                    
            # convert both_data to data
            self.data = [sample for identity, sample in self.both_data.items()]      
            pickle.dump(self.data, open(save_file.format(subset), 'wb'))
            if use_size != -1:
                self.data = self.data[:use_size]
            print(f"[Save processed data to {save_file.format(subset)}]")
            # for id in empty_ids_set:
            #     print(id, file=empty_conv_ids_file)
            empty_conv_ids_file.close()

    def load_movie(self, path='../../data/movies_with_mentions.csv'):
        # 获得
        import csv
        self.name2id = {}
        self.db2id = {}
        self.movie_num = 0
        reader = csv.reader(open(path, 'r', encoding='utf-8-sig'))
        next(reader)
        for line in reader:
            global_id, name_time, db_id, _ = line
            name = name_time.split('(')[0]
            self.name2id[name] = int(global_id)
            self.db2id[int(db_id)] = int(global_id)        
        ###############################
        # self.movie_num = 2

    def neg_sample(self, item_set):
        # item = random.randint(1, self.movie_num)
        # while item in item_set:
        #     item = random.randint(1, self.movie_num)
        # return item

        item = random.randint(1, self.movie_num)
        while item in item_set:
            item = random.randint(1, self.movie_num)
        return item

    def __getitem__(self, index):
        contexts_index, types, masks, movie_id, input_ids, target_pos, input_mask, sample_negs, len_input = self.data[index]
        # 如果是第一种模式：
        #   训练集、测试集：movie_list后面不用拼接gt
        # 第二种模式：
        #   训练集:拼接，然后序列所有内容计算sasrec-loss，倒数第二个位置的rep计算fusion loss
        #   测试集：不拼接
    
        # todo
        return np.array(contexts_index), np.array(types), np.array(masks), movie_id, \
            np.array(input_ids), np.array(target_pos), np.array(input_mask), np.array(sample_negs), len_input
        #####################################
        # return np.array(contexts_index), np.array(types), np.array(masks), 0

    def __len__(self):
        return len(self.data)

