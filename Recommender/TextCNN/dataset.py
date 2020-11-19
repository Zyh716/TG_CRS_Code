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


class dataset(object):
    # opt = var(args)
    def __init__(self, filename, opt, args, tokenizer, vocab):
        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_count=opt['max_count']  # 一个context最多使用多少sentence
        
        # load data
        f = pickle.load(open(filename, 'rb'))[:]
        self.load_movie()

        # define data structs
        self.data=[] # 
        self.corpus=[]  # list of all compat-token-sentence

        UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
        UNK_ID, PAD_ID, SENT_ID = vocab.get(UNK), vocab.get(PAD), vocab.get('SENT')

        for conv in tqdm(f):
            # contexts_token = ["[CLS]"]  # list of token, ['[CLS]' UTTER1  "[SEP]"  UTTER2  "[SEP]" ]
            contexts_token = []  # list of token, ['[CLS]' UTTER1  "[SEP]"  UTTER2  "[SEP]" ]
            for message in conv['messages']:
                message_id, content, role = message['local_id'], message['content'], message['role']
                # 如果这一步要推荐电影，就产生一个sample：上文context + 当前要推荐的电影 这个pair
                if role == 'Recommender' and message_id in conv['mentionMovies']:
                    contexts_index = []
                    for token in contexts_token:
                        contexts_index.append(vocab.get(word, vocab.get(UNK)))

                    # 将douban-id映射为global-id
                    movie_id = int(conv['mentionMovies'][message_id][0])
                    movie_id = self.db2id(movie_id)
                    # 短则补齐，长则切断
                    if len(contexts_index) < self.max_c_length:
                        context_len = len(contexts_index)
                        contexts_index = contexts_index + [PAD_ID] * (self.max_c_length - len(contexts_index))
                    else:
                        context_len = self.max_c_length
                        contexts_index = contexts_index[-self.max_c_length:]
                    
                    case = [contexts_index, context_len, movie_id]
                    self.data.append(case)

                content_token = tokenizer(content) + ["[SEP]"]
                contexts_token.extend(content_token)

        print(f"[Load {len(f)} convs, Extract {len(self.data)} cases, from {filename}]")


class CRSdataset(Dataset):
    def __init__(self, subset, filename, opt, args, tokenizer, vocab, save_build_data=None, load_builded_data=None, use_size=-1):
        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        
        # load data
        only_first_movie = False
        f = pickle.load(open(filename, 'rb'))[:]
        self.load_movie()

        # define data structs
        self.data=[] # 
        save_file = 'data/{}_processed_data.pkl' 
        save_file = save_file.format(subset)

        # define special token
        UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
        UNK_ID, PAD_ID = vocab.get(UNK), vocab.get(PAD)

        if load_builded_data:
            self.data = pickle.load(open(save_file, 'rb'))
            if use_size != -1:
                self.data = self.data[:use_size]
            print(f"[Load {len(self.data)} cases, from {save_file}]")

        else:   
            for conv in tqdm(f):
                # contexts_token = ["[CLS]"]  # list of token, ['[CLS]' UTTER1  "[SEP]"  UTTER2  "[SEP]" ]
                contexts_token = []  # list of token, ['[CLS]' UTTER1  "[SEP]"  UTTER2  "[SEP]" ]
                for message in conv['messages']:
                    message_id, content, role = message['local_id'], message['content'], message['role']
                    # 如果这一步要推荐电影，就产生一个sample：上文context + 当前要推荐的电影 这个pair
                    if role == 'Recommender' and message_id in conv['mentionMovies']:
                        # convet token to id
                        contexts_index = []
                        for token in contexts_token:
                            contexts_index.append(vocab.get(token, vocab.get(UNK)))
                        # 将douban-id映射为global-id
                        movie_id = int(conv['mentionMovies'][message_id][0])
                        movie_id = self.db2id[movie_id]
                        # 短则补齐，长则切断
                        if len(contexts_index) < self.max_c_length:
                            context_len = len(contexts_index)
                            contexts_index = contexts_index + [PAD_ID] * (self.max_c_length - len(contexts_index))
                        else:
                            context_len = self.max_c_length
                            contexts_index = contexts_index[-self.max_c_length:]
                        
                        case = [contexts_index, context_len, movie_id]
                        self.data.append(case)
                        if only_first_movie:
                            break

                    # content_token = tokenizer.tokenize(content) + ["[SEP]"]
                    content_token = tokenizer(content)
                    contexts_token.extend(content_token)

            if save_build_data:
                pickle.dump(self.data, open(save_file, 'wb'))
            if use_size != -1:
                self.data = self.data[:use_size]
            print(f"[Load {len(f)} convs, Extract {len(self.data)} cases, from {filename}]")
            print(f"[Save processed data to {save_file}]")

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
        self.movie_num = len(self.db2id)
        ###############################
        # self.movie_num = 2

    def __getitem__(self, index):
        contexts_index, length, movie_id= self.data[index]

        return np.array(contexts_index), length, movie_id
        #####################################
        # return np.array(contexts_index), np.array(types), np.array(masks), 0

    def __len__(self):
        return len(self.data)

