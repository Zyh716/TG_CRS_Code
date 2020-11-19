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
    def __init__(self, logger, subset, filename, args, tokenizer):
        super(CRSdataset, self).__init__()
        if args.model_type == 'Ours':
            self._init_Ours(logger, subset, filename, args, tokenizer)

    def _init_Ours(self,
                   logger,
                   subset,
                   filename,
                   args,
                   tokenizer,
                   use_size=-1):
        raw = args.raw
        self.batch_size = args.batch_size
        self.max_c_length = args.max_c_length
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

        logger.info("[Load {} movies(+1)]".format(self.movie_num))

        self.tokenizer = tokenizer
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.unk_id = self.tokenizer.convert_tokens_to_ids('[UNK]')
        self.sent_split_id = self.tokenizer.convert_tokens_to_ids(
            '[unused1]')  #1
        self.word_split_id = self.tokenizer.convert_tokens_to_ids(
            '[unused2]')  #2
        self.word_split = '[unused2]'  #2

        self.both_data = {}  # identity: sample, 能够根据索引抽取，后面会加入user history
        save_file = 'data/data_p_{}/{}processed_data.pkl'.format(
            args.model_type, subset)
        empty_conv_ids_file = open('empty_conv_ids.txt', 'a')

        if not raw:
            self.data = pickle.load(open(save_file, 'rb'))
            if use_size != -1:
                self.data = self.data[:use_size]
            logger.info(f"[Load {len(self.data)} cases, from {save_file}]")

        else:
            for conv in tqdm(f):
                # contexts_token = ["[CLS]"]  # list of token, ['[CLS]' UTTER1  "[SEP]"  UTTER2  "[SEP]" ]
                conv_id = conv['conv_id']
                contexts_index = [
                ]  # list of token, ['[CLS]' UTTER1  UTTER2  "[SEP]" Target ]

                for message in conv['messages']:
                    message_id, content, role = message['local_id'], message[
                        'content'], message['role']
                    # 如果这一步要推荐电影，就产生一个sample：上文context + 当前要推荐的电影 这个pair
                    if role == 'Recommender' and message_id in conv[
                            'mentionMovies']:
                        # 将douban-id映射为global-id
                        movie_id = int(conv['mentionMovies'][message_id][0])
                        movie_id = self.db2id[movie_id]
                        # 获得context对应的两种mask: seg_id, att_mask # 短则补齐，长则切断
                        cur_contexts_index = contexts_index[:-1] + [
                            self.sep_id
                        ]  #？是不是加重复了
                        self.max_len_inside = self.max_c_length - 1  # cls

                        if len(cur_contexts_index) < self.max_len_inside:
                            cur_contexts_index = [self.cls_id
                                                  ] + cur_contexts_index

                            cur_contexts_index = cur_contexts_index + [0] * (
                                self.max_c_length - len(cur_contexts_index))

                            types = [0] * len(cur_contexts_index) + [1] * (
                                self.max_c_length - len(cur_contexts_index)
                            )  # mask部分 segment置为1
                            masks = [1] * len(cur_contexts_index) + [0] * (
                                self.max_c_length - len(cur_contexts_index))
                        else:
                            cur_contexts_index = [
                                self.cls_id
                            ] + cur_contexts_index[-self.max_len_inside:]
                            types = [0] * len(cur_contexts_index)
                            masks = [1] * len(cur_contexts_index)

                        assert len(cur_contexts_index) == self.max_c_length
                        assert cur_contexts_index[0] == self.cls_id
                        assert cur_contexts_index[
                            -1] == self.sep_id or cur_contexts_index[
                                -1] == self.pad_id

                        case = [cur_contexts_index, types, masks, movie_id]
                        identity = str(conv_id) + '/' + str(message_id)
                        self.both_data[identity] = case

                        if only_first_movie:
                            break

                    content_token = tokenizer.tokenize(content)
                    content_index = tokenizer.convert_tokens_to_ids(
                        content_token) + [self.sent_split_id]
                    contexts_index.extend(content_index)

            # 以上是bert用的，以下是sasrec用的

            data_base = '../../data/'
            user_long_history = os.path.join(data_base,
                                             'user_long_history.json')
            conv_long_history = os.path.join(data_base,
                                             'conv_long_history.json')
            user_long_history = json.load(open(
                user_long_history, 'r'))  # {user: [(id, rate, time)]}
            conv_long_history = json.load(
                open(conv_long_history,
                     'r'))  # {user: {conv_id:(start_index, length)}

            conv2user = pickle.load(open('../../data/0619conv2user.pkl',
                                         'rb'))  # 检验无误

            num_history = 0
            history_total_num = 0
            self.max_len = self.args.max_seq_length

            for conv in f[:]:
                conv_id = conv['conv_id']  # int
                # ipdb.set_trace()
                user = conv2user[str(conv_id)]  # str
                conv_movie_list = []
                for message_id, (movieId,
                                 m_name) in conv['mentionMovies'].items():
                    # 每个要推荐的位置生成一个history，与context的data结合在一起
                    identity = str(conv_id) + '/' + str(message_id)
                    # ipdb.set_trace()
                    if user not in conv_long_history or str(
                            conv_id) not in conv_long_history[user]:
                        seq = []
                    else:
                        num_history += 1
                        start_index, length = conv_long_history[user][str(
                            conv_id)]
                        seq = user_long_history[user][start_index:start_index +
                                                      length]
                        # seq = [self.db2id.get(int(movie_id), len(self.db2id)+1) for (movie_id, rate, time_) in seq] # todo
                        seq = [
                            self.db2id.get(int(movie_id), self.unk_movie_id)
                            for (movie_id, rate, time_) in seq
                        ]  # todo

                    movie_list = seq + conv_movie_list  # list of int, long-history + short history
                    history_total_num += len(movie_list)
                    input_ids = movie_list
                    target_pos = movie_list  # no use

                    input_mask = [1] * len(input_ids)

                    sample_negs = []  # no use
                    seq_set = set(input_ids)
                    for _ in input_ids:
                        sample_negs.append(self.neg_sample(seq_set))  #用于训练的

                    if len(input_ids) < self.max_len:
                        pad_len = self.max_len - len(input_ids)
                        input_ids = [0] * pad_len + input_ids
                        target_pos = [0] * pad_len + target_pos
                        input_mask = [0] * pad_len + input_mask
                        sample_negs = [0] * pad_len + sample_negs
                    else:
                        input_ids = input_ids[-self.max_len:]
                        target_pos = target_pos[-self.max_len:]
                        input_mask = input_mask[-self.max_len:]
                        sample_negs = sample_negs[-self.max_len:]

                    assert len(input_ids) == self.max_len
                    assert len(target_pos) == self.max_len
                    assert len(input_mask) == self.max_len
                    assert len(sample_negs) == self.max_len
                    if identity in self.both_data:
                        assert self.db2id[int(
                            movieId)] == self.both_data[identity][-1]
                        self.both_data[identity].extend(
                            [input_ids, target_pos, input_mask, sample_negs])
                    else:
                        logger.info(identity, file=empty_conv_ids_file)
                    conv_movie_list.append(self.db2id[int(movieId)])

            logger.info("Load {} user-history, including {} movies".format(
                num_history, history_total_num))

            # convert both_data to data
            self.data = [[identity] + sample for identity, sample in self.both_data.items()]
            pickle.dump(self.data, open(save_file, 'wb'))
            if use_size != -1:
                self.data = self.data[:use_size]
            logger.info(f"[Save processed data to {save_file}]")
            empty_conv_ids_file.close()

    def load_movie(self, path='../../data/movies_with_mentions.csv'):
        import csv
        self.name2id = {}
        self.db2id = {}
        reader = csv.reader(open(path, 'r', encoding='utf-8-sig'))
        next(reader)
        for line in reader:
            global_id, name_time, db_id, _ = line
            name = name_time.split('(')[0]
            self.name2id[name] = int(global_id)
            self.db2id[int(db_id)] = int(global_id)

    def neg_sample(self, item_set):
        item = random.randint(1, self.movie_num)
        while item in item_set:
            item = random.randint(1, self.movie_num)
        return item

    def __getitem__(self, index):
        if len(self.data[index]) == 8+1:
            identity, contexts_index, types, masks, movie_id, input_ids, target_pos, input_mask, sample_negs = self.data[
                index]

            return identity, np.array(contexts_index), np.array(types), \
                np.array(masks), movie_id, \
                np.array(input_ids), np.array(target_pos), \
                np.array(input_mask), np.array(sample_negs)

        elif len(self.data[index]) == 4+1:
            identity, contexts_index, types, masks, movie_id = self.data[index]
            return np.array(contexts_index), np.array(types), np.array(
                masks), movie_id, movie_id, movie_id, movie_id, movie_id

    def __len__(self):
        return len(self.data)
