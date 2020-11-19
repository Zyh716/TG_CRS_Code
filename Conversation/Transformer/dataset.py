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
    def __init__(self, filename, opt, subset):
        self.entity2entityId = pkl.load(open('data/entity2entityId.pkl', 'rb'))
        self.entity_max = len(self.entity2entityId)

        self.id2entity = pkl.load(open('data/id2entity.pkl', 'rb'))
        # self.subkg=pkl.load(open('data/subkg.pkl','rb'))    #need not back process
        # self.text_dict=pkl.load(open('data/text_dict.pkl','rb'))

        self.batch_size = opt['batch_size']
        self.max_c_length = opt['max_c_length']
        self.max_r_length = opt['max_r_length']
        self.max_count = opt['max_count']  # 一个context最多使用多少sentence
        # self.entity_num=opt['n_entity']

        if self.args.use_size == -1:
            f = pickle.load(open(filename, 'rb'))[:]
        else:
            f = pickle.load(open(filename, 'rb'))[:self.args.use_size]

        self.data = []  # list response-info， response-info is dict
        self.corpus = []  # list of all compat-token-sentence

        total_reponse_num = 0
        num = 0
        for conv in tqdm(f):
            contexts = []  # [{}]
            for message in conv['messages']:
                contexts.append({
                    'text': message['content'],
                    'senderWorkerId': message['role']
                })
                assert not isinstance(message['content'], str), conv  #8407

            seekerid = None
            recommenderid = 'Recommender'
            contexts = contexts
            movies = {}
            altitude = None
            initial_altitude = None
            cases, reponse_num = self._context_reformulate(
                contexts, movies, altitude, initial_altitude, seekerid,
                recommenderid)
            self.data.extend(cases)
            total_reponse_num += reponse_num
        print(
            f"[Extract {len(self.data)} cases and {total_reponse_num} non-start-response from {filename}]"
        )
        # self.prepare_word2vec()

        # bpe2index = {'<pad>': 0, '<unk>': 1, '<go>': 2, '<end>': 3, '<movie>': 4}

        self.word2index = json.load(
            open('../../data/data1030/output/bpe2index.json',
                 encoding='utf-8'))

    def prepare_word2vec(self):
        '''
        使用HERD里面定义的 bpe2vec  bpe2index, context里的句子分隔符不一样，要改成 _split_
        old: 前面添加4个，最后添加1个token，都是啥呢： [pad=0,end=2,unk=3] [...] [_split_]
        output:
            word2vec_redial.npy <- list of list of int
            word2index_redial.json
        '''
        pass

    def padding_w2v(self,
                    sentence,
                    max_length,
                    transformer=True,
                    pad=0,
                    end=3,
                    unk=1):
        '''
        将一个token sentence 转换为 id sentence，需要padding或者截断
        params: transformer: 在sentence达到最大长度以后，如果该参数为True，则使用倒数max-length个token，否则使用前max-length个
        '''
        vector = []
        concept_mask = []
        dbpedia_mask = []
        for word in sentence:
            # vector.append(self.word2index.get(word,unk))  # 电影 都变成unk了？
            vector.append(word)  # 电影 都变成unk了？
            # concept_mask.append(self.key2index.get(word.lower(),0))
            # if '@' in word:
            #     try:
            #         entity = self.id2entity[int(word[1:])]
            #         id=self.entity2entityId[entity]
            #     except:
            #         id=self.entity_max
            #     dbpedia_mask.append(id)
            # else:
            #     dbpedia_mask.append(self.entity_max)
        vector.append(end)
        # concept_mask.append(0)
        # dbpedia_mask.append(self.entity_max)

        if len(vector) > max_length:
            if transformer:
                return vector[-max_length:], max_length, concept_mask[
                    -max_length:], dbpedia_mask[-max_length:]
            else:
                return vector[:
                              max_length], max_length, concept_mask[:
                                                                    max_length], dbpedia_mask[:
                                                                                              max_length]
        else:
            length = len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]
        # if len(vector)>max_length:
        #     if transformer:
        #         return vector[-max_length:],max_length, None, None
        #     else:
        #         return vector[:max_length],max_length, None, None
        # else:
        #     length=len(vector)
        #     return vector+(max_length-len(vector))*[pad],length, None, None

    def padding_context(self, contexts, pad=0, transformer=True):
        # contexts是list of token sentence
        vectors = []
        vec_lengths = []
        if transformer == False:
            # 如果不适用transformer，则使用全部的context，
            # 长的截断（使用最后的max-count个sent），短的补全（补若干 max_c_length个pad组成的句子）
            if len(contexts) > self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec, v_l = self.padding_w2v(sen, self.max_r_length,
                                                transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors, vec_lengths, self.max_count
            else:
                length = len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen, self.max_r_length,
                                                transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors + (self.max_count - length) * [
                    [pad] * self.max_c_length
                ], vec_lengths + [0] * (self.max_count - length), length
        else:
            # 如果用transformer, 就是用最后max-count个sentence，紧凑得拼在一起，使用_split_分割句子
            # 然后获得整个contexts的sentences
            contexts_com = []
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                # 已经有_split_分隔符了
                # contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec, v_l, concept_mask, dbpedia_mask = self.padding_w2v(
                contexts_com, self.max_c_length, transformer)
            return vec, v_l, concept_mask, dbpedia_mask, 0

    def data_process(self, is_finetune=False):
        '''
        将list of reponse-info-token 转换成list of response-info-id-np版
        '''
        data_set = []
        context_before = []
        for line in self.data:
            # fineturn时，如果上一个response的context与这个response的context一样，那么这个response不被返回
            # generation的时候，一个context只要生成一个回复就好
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']
            context, c_lengths, concept_mask, dbpedia_mask, _ = self.padding_context(
                line['contexts'])
            # context,c_lengths=self.padding_context(line['contexts'])
            response, r_length, _, _ = self.padding_w2v(
                line['response'], self.max_r_length)
            # 啥也不动就对了
            if False:
                mask_response, mask_r_length, _, _ = self.padding_w2v(
                    self.response_delibration(line['response']),
                    self.max_r_length)
            else:
                mask_response, mask_r_length = response, r_length
            assert len(context) == self.max_c_length
            # assert len(concept_mask)==self.max_c_length   # empty
            # assert len(dbpedia_mask)==self.max_c_length

            # data_set.append([np.array(context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,line['entity'],
            #                  line['movie'],concept_mask,dbpedia_mask,line['rec']])
            data_set.append([
                np.array(context), c_lengths,
                np.array(response), r_length,
                np.array(mask_response), mask_r_length, [], line['movie'], [],
                [], line['rec']
            ])
        return data_set

    def entities2ids(self, entities):
        #v2: didnot use
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self, sentence, movies):
        # return list of compat-token, list of movie entityId (of each movie)

        # 获得list of token， 输入string　（对于 @123 是怎么切分的， 看样子是 @ 123）
        ########################################################
        # token_text = word_tokenize(sentence)
        num = 0
        token_text_com = []
        # 将 @ 123 拼接回 @123，其他的原样子copy到token_text_com
        # while num<len(sentence):
        #     if token_text[num]=='@' and num+1<len(token_text):
        #         token_text_com.append(token_text[num]+token_text[num+1])
        #         num+=2
        #     else:
        #         token_text_com.append(token_text[num])
        #         num+=1
        # 将对话中出现的电影id(str)存放到movie_rec中
        movie_rec = []
        # for word in token_text_com:
        #     if word[1:] in movies:
        #         movie_rec.append(word[1:])
        # 将电影id 转换成 entityId
        movie_rec_trans = []
        # for movie in movie_rec:
        #     entity = self.id2entity[int(movie)]
        #     try:
        #         movie_rec_trans.append(self.entity2entityId[entity])
        #     except:
        #         pass
        # return token_text_com,movie_rec_trans
        return sentence, movie_rec_trans

    def _context_reformulate(self, context, movies, altitude, ini_altitude,
                             s_id, re_id):
        '''
        # 纯transformer，entity一点用也没了， 这里还没转换成id
        context,movies,altitude,ini_altitude,s_id,re_id
        X       X      None      None        None    X
        params: context: 若干sentence组成的list
        params: movies: movie-mentions 

        return： 这个对话中每个response相关信息记录
        '''
        reponse_num = 0
        last_id = None
        # perserve the list of dialogue
        context_list = []
        # 将所有message分条加入到context-list中，相邻message相同send-id合并为一个message
        
        for message in context:
            # 获得这段sentence对应的entity
            # v2: didnot use entities
            entities = []
            # try:
            #     for entity in self.text_dict[message['text']]:
            #         try:
            #             entities.append(self.entity2entityId[entity])
            #         except:
            #             pass
            # except:
            #     pass

            # return list of compat-token, list of movie entityId
            token_text, movie_rec = self.detect_movie(message['text'], movies)
            # 如果message是第一次utterance，记录有关信息后continue
            # eneitis只保留电影名
            if len(context_list) == 0:
                context_dict = {
                    'text': token_text,
                    'entity': entities + movie_rec,
                    'user': message['senderWorkerId'],
                    'movie': movie_rec
                }
                # context_dict={'text':token_text,'entity':[] + movie_rec,'user':message['senderWorkerId'],'movie':movie_rec}
                context_list.append(context_dict)
                last_id = message['senderWorkerId']
                continue
            # old implenment 如果不是第一次发言，分情况
            if message['senderWorkerId'] == last_id:
                # 如果是同一个role连续发言
                context_list[-1]['text'] += token_text
                context_list[-1]['entity'] += entities + movie_rec
                # context_list[-1]['entity']+=movie_rec
                context_list[-1]['movie'] += movie_rec
            else:
                # 如果是一个新的role发言（之前已经发过言
                context_dict = {
                    'text': token_text,
                    'entity': entities + movie_rec,
                    'user': message['senderWorkerId'],
                    'movie': movie_rec
                }
                # context_dict = {'text': token_text, 'entity': [] + movie_rec,
                #            'user': message['senderWorkerId'], 'movie':movie_rec}
                context_list.append(context_dict)
                last_id = message['senderWorkerId']

            # new implement:不合并message
            # context_dict = {'text': token_text, 'entity': entities+movie_rec,
            #                'user': message['senderWorkerId'], 'movie':movie_rec}
            # # context_dict = {'text': token_text, 'entity': [] + movie_rec,
            # #            'user': message['senderWorkerId'], 'movie':movie_rec}
            # context_list.append(context_dict)
            # last_id = message['senderWorkerId']

        cases = []
        contexts = []
        entities_set = set()
        entities = []
        # 更新corpus
        # 记录每个response-movie的contexts、之前出现的所有entitis、推荐的电影、是否推荐了电影
        # 更新contexts、entities
        # 每个response-movie相关信息记录到cases中： reponse含有多个movie,则每个movie一个记录，没有movie，则只有一个记录
        for context_dict in context_list:
            self.corpus.append(context_dict['text'])
            if context_dict['user'] == re_id and len(contexts) > 0:
                reponse_num += 1
                response = context_dict['text']

                # if len(context_dict['movie'])!=0:
                #     for movie in context_dict['movie']:
                #         if movie not in entities_set:
                #             cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1})
                # else:
                #     cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': 0, 'rec':0})
                cases.append({
                    'contexts': deepcopy(contexts),
                    'response': response,
                    'entity': deepcopy(entities),
                    'movie': 0,
                    'rec': 0
                })
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
            else:
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
        return cases, reponse_num


class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        '''
        dataset: list of response-info-id-np
        '''
        self.data = dataset
        self.entity_num = entity_num
        self.concept_num = concept_num + 1

    def __getitem__(self, index):
        # 目的：将entity、concept、concept搞成np-id的形式
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        # entity concept_mask dbpedia_mask: []
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[
            index]
        # entity_vec[index]=1，if id为index的entity出现过
        # entity_vector，按照出现次序存放entityId， note：这个entity包括电影

        entity_vec = np.zeros(self.entity_num)
        entity_vector = np.zeros(50, dtype=np.int)
        point = 0
        for en in entity:
            entity_vec[en] = 1
            entity_vector[point] = en
            point += 1

        # ？ conceptId=0意味着什么
        concept_vec = np.zeros(self.concept_num)
        for con in concept_mask:
            if con != 0:
                concept_vec[con] = 1

        db_vec = np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db != 0:
                db_vec[db] = 1
        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(
            concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = dataset('data/test_data.jsonl')

# test: [Extract 7087 cases and 7087 non-start-response from data/test_data.jsonl]
'''
原版是# 每个(合并后的)response-movie相关信息记录到cases中： reponse含有多个movie,则每个movie一个记录，没有movie，则只有一个记录
kbrd是context和test只要有一个不为空就行

我的版本：每个response都只记录一次，但是context不能为空。testset里会有7087个sample

ours数据处理流程：
todo： 改进pair：考虑电影名
# 使用HERD那里的程序，获得若干context-response pair， 这里已经是word-id了。 获得bpe2index bpe2vec
# 使用pair，稍微转变一下格式，data_process处理：padding
'''