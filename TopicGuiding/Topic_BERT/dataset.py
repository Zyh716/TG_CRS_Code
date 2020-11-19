import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
from copy import deepcopy
import nltk
import jieba
import pickle
import random
import ipdb
import torch


def collate_fn(batch):
    def convert_tuple2list(x_tuple):
        return [list(element) for element in x_tuple]

    # int, list of list of int
    identity, context, context_mask, topic_path_kw, user_profile, target = \
        convert_tuple2list(zip(*batch))

    # padding topic_path
    tp_max_len = 0
    for kw_list in topic_path_kw:
        tp_max_len = max(tp_max_len, len(kw_list))

    topic_mask = []
    for i, kw_list in enumerate(topic_path_kw):
        cur_len = len(kw_list)
        pad_len = tp_max_len - cur_len
        topic_path_kw[i] = kw_list + [0] * pad_len
        topic_mask.append([1] * cur_len + [0] * pad_len)

    profile_max_len = 0
    for sent4oneuser in user_profile:
        for sent in sent4oneuser:
            profile_max_len = max(profile_max_len, len(sent))

    profile_mask = []  #
    for i, sent4oneuser in enumerate(user_profile):
        mask4oneuser = []
        for j, sent in enumerate(sent4oneuser):
            cur_len = len(sent)
            pad_len = profile_max_len - cur_len
            user_profile[i][j] = sent + [0] * pad_len
            mask4oneuser.append([1] * cur_len + [0] * pad_len)
        profile_mask.append(mask4oneuser)

    target_topic = []
    for sample in target:
        target_topic.append(sample[-1])

    batch = (context, context_mask, \
        topic_path_kw, topic_mask, \
        user_profile, profile_mask, \
        target_topic)

    # tocheck
    batch = (torch.tensor(d, dtype=torch.long) for d in batch)

    return batch


class CRSdataset(Dataset):
    def __init__(
        self,
        subset,
        filename,
        opt,
        args,
        tokenizer,
    ):
        self.args = args
        self.opt = opt
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

        self.max_c_length = opt['max_c_length']
        save_file = 'data/{}processed_data.pkl'

        self.topic_to_id = json.load(open(args.topic_to_id, 'r'))  # todo
        self.topic_class_num = len(self.topic_to_id)
        self.id_to_topic = {
            id: topic
            for topic, id in self.topic_to_id.items()
        }
        self.conv2user = pickle.load(open("../../data/0619conv2user.pkl",
                                          'rb'))

        if not self.args.raw:
            self.data = pickle.load(open(save_file.format(subset), 'rb'))
            if args.use_size != -1:
                self.data = self.data[:args.use_size]
            print(
                f"[Load {len(self.data)} cases, from {save_file.format(subset)}]"
            )

        else:
            ## Load conversations, extract context
            self.id_to_context = {}
            conv_id_no_messages = []
            f = pickle.load(open(filename, 'rb'))[:]
            args.use_size = len(f) if args.use_size == -1 else args.use_size
            for conv in tqdm(f[:args.use_size]):
                # for conv in tqdm(f[:]):
                conv_id = int(conv['conv_id'])
                if len(conv['messages']) == 0:
                    conv_id_no_messages.append(conv_id)
                    continue
                contexts_index = [
                ]  # list of token, ['[CLS]' UTTER1  UTTER2  "[SEP]" Target ]
                for message in conv['messages']:
                    message_id, content, role = int(
                        message['local_id']
                    ), message['content'], message['role']
                    identity = str(conv_id) + '/' + str(message_id)
                    # 记录每个message-id对应的context-index
                    if message_id >= 3 and role == 'Recommender':  # 第一句和第二句不需要预测
                        self.id_to_context[identity] = deepcopy(
                            contexts_index[:-1])  #没有cls和sep # 消除最后的split

                    content_token = tokenizer.tokenize(content)
                    content_index = tokenizer.convert_tokens_to_ids(
                        content_token) + [self.sent_split_id]
                    contexts_index.extend(content_index)

            ## extract topic guiding
            self.id_to_topic_path = {}
            self.id_to_target = {}
            for conv in tqdm(f[:args.use_size]):
                # for conv in tqdm(f[:]):
                conv_id = int(conv['conv_id'])
                if conv_id in conv_id_no_messages:
                    continue
                topic_path = []  # element = (kw, attitude) 代表每个kw以及态度
                # 态度粗略的分：接收/拒绝 1/0
                last_mess_id = 0
                for message_id, goal_info in conv['goal_path'].items():
                    # 每一步要做两件事: 1.将当前topic以及性质记录到本对话的topic path中
                    #                  2.满足条件时，生成以当前topic为目标的sample
                    identity = str(conv_id) + '/' + str(message_id)
                    assert last_mess_id < message_id
                    last_mess_id = message_id
                    role = goal_info[0]
                    if message_id % 2 == 1:
                        assert role == 'Rec'  # pass

                    actions = goal_info[1::2]
                    kws = goal_info[2::2]
                    # 做第2事
                    if role == 'Rec':
                        target = [
                            0, 0, 0
                        ] 
                        have_talk_topic = False
                        for action, kw in zip(actions, kws):
                            if '谈论' in action:
                                # topic_path.append(kw)
                                target[0] = [kw]
                                assert isinstance(kw, str), str(kw)
                                have_talk_topic = True
                            elif '请求' in action:
                                target[0] = [kw] if isinstance(kw, str) else kw
                                have_talk_topic = True
                            elif '推荐电影' in action:
                                target[2] = 1
                            else:
                                print('Other action by rec = ', action)
                        if have_talk_topic:
                            self.id_to_topic_path[identity] = deepcopy(
                                topic_path)
                            self.id_to_target[identity] = deepcopy(target)
                    # 做第1件事
                    for action, kw_unk_type in zip(actions, kws):
                        kw_list = [kw_unk_type] if isinstance(
                            kw_unk_type, str) else kw_unk_type
                        if '谈论' in action:
                            for kw in kw_list:
                                topic_path.append([kw, 1])
                        elif '请求推荐' in action or '允许推荐' in action:
                            for kw in kw_list:
                                topic_path.append([kw, 1])
                        elif '拒绝' in action:
                            for i, (kw,
                                    attitude) in enumerate(topic_path[::-1]):
                                if kw in kw_list:
                                    topic_path[-i - 1][1] = 0
                                    break
                        elif '反馈' in action:
                            continue
                        elif role == 'Seeker':
                            # seeker的额外的action
                            print('Other action by seeker = ', action)
            self.id_to_final_topic = {}  # :list of kw
            for conv in tqdm(f[:args.use_size]):
                # for conv in tqdm(f[:]):
                conv_id = int(conv['conv_id'])
                if conv_id in conv_id_no_messages:
                    continue
                last_mess_id = 0
                # 获得每次推荐的整个过程的开始messageid和结束messageid
                segment_end_ids = list(conv['mentionMovies'].keys())
                for i, segment_end_id in enumerate(segment_end_ids):
                    segment_start_id = 3 if i == 0 else segment_end_ids[i -
                                                                        1] + 1
                    message_range = range(segment_start_id, segment_end_id + 1)

                    assert last_mess_id < segment_end_id
                    last_mess_id = segment_end_id

                    get_final_topic = False
                    for message_id in message_range:
                        goal_info = conv['goal_path'][message_id]
                        actions = goal_info[1::2]
                        kws = goal_info[2::2]
                        for action, kw_unk_type in zip(actions, kws):
                            if '请求推荐' in action:
                                # assert isinstance(kw, str), 'List了' + str(kw)
                                kw_list = [kw_unk_type] if isinstance(
                                    kw_unk_type, str) else kw_unk_type
                                for included_message_id in message_range:
                                    identity = str(conv_id) + '/' + str(
                                        included_message_id)
                                    self.id_to_final_topic[identity] = kw_list

                                get_final_topic = True
                                break
                        if get_final_topic:
                            break

            ## load user_profile
            self.id_to_profile = {}  #id: list of topic
            user_to_topic_sents = pickle.load(
                open("../../data/user2TopicSent.pkl", 'rb'))

            self.id_to_final_topic = {
                key: value
                for key, value in self.id_to_final_topic.items()
                if key in self.id_to_target
            }
            assert set(self.id_to_target) & set(self.id_to_context) == set(
                self.id_to_target)

            self.data = []
            for identity in self.id_to_target:
                conv_id = int(identity.split('/')[0])

                context_index = self.id_to_context[identity]
                topic_path = self.id_to_topic_path[identity]
                user_profile = self.id_to_profile[conv_id]
                final_topic = self.id_to_final_topic[identity]
                target_info_list = self.id_to_target[
                    identity]  # [list of kw, ..., ...]
                for target_topic in target_info_list[0]:
                    target = [
                        target_topic, target_info_list[1], target_info_list[2]
                    ]
                    self.data.append([
                        identity, context_index, topic_path, user_profile,
                        final_topic, target
                    ])

            pickle.dump(self.data, open(save_file.format(subset), 'wb'))
            print(
                f"[Load {len(f)} convs, Extract {len(self.data)} cases, from {filename}]"
            )
            print(f"[Save processed data to {save_file.format(subset)}]")

    def __getitem__(self, index):
        identity, context_index, topic_path, user_profile, final_topic, target = deepcopy(
            self.data[index])
        # context_index = list of idx
        # topic_path = list of (kw, 1/0)
        # user_profile = set of str(sentence)
        # final topic = list of topic
        # target = [target1, target2, target3]

        # final topic -> Topic1 . Topic2 [SEP]  # note .
        final_topic_list = []
        for topic in final_topic:
            final_topic_list.extend(
                self.tokenizer.tokenize(topic) + [self.word_split])
        final_topic = final_topic_list[:-1]
        final_topic_ids = self.tokenizer.convert_tokens_to_ids(final_topic) + [
            self.sep_id
        ]

        # padding context_index
        context_index.append(self.sep_id)
        context_index.extend(final_topic_ids)
        max_len_inside = self.max_c_length - 1  # cls
        if len(context_index) >= max_len_inside:
            context_index = context_index[-max_len_inside:]
            context_index = [self.cls_id] + context_index
            cotnext_mask = [1] * len(context_index)
            # context_length = self.max_c_length
        else:
            # context_length = len(context_index)
            context_index = [self.cls_id] + context_index
            cotnext_mask = [1] * len(context_index) + [0] * (
                self.max_c_length - len(context_index))
            context_index = context_index + [0] * (self.max_c_length -
                                                   len(context_index))

        # convert all kw to kw_id
        topic_path_kw = [kw for kw, attitude in topic_path]
        topic_path_kw_token_list = ['[CLS]']
        for kw in topic_path_kw:
            token_list = self.tokenizer.tokenize(kw) + [self.word_split]
            topic_path_kw_token_list.extend(token_list)
        topic_path_kw_token_list = topic_path_kw_token_list[:-1] + ['[SEP]']
        topic_path_kw = self.tokenizer.convert_tokens_to_ids(
            topic_path_kw_token_list)
        topic_path_kw.extend(final_topic_ids)

        # topic_path_attitude = [0]
        # for kw, attitude in topic_path:
        #     token_list = self.tokenizer.tokenize(kw)
        #     for token in token_list:
        #         topic_path_attitude.append(attitude)
        # topic_path_attitude += [0] * (
        #     len(topic_path_kw) - len(topic_path_attitude)
        # )  # final topic 也设置为0了

        user_profile_list = []  # list of list of int
        for sent in user_profile:
            token_list = self.tokenizer.tokenize(sent)
            id_list = [self.cls_id] + self.tokenizer.convert_tokens_to_ids(
                token_list) + [self.sep_id] + final_topic_ids
            user_profile_list.append(id_list)

        topic_id = int(self.topic_to_id[target[0]])  # [0, topic_id_num)
        target = [1, target[1], target[2], topic_id]

        # string, list of int, list of int
        # list of int, list of int
        # list of list of int
        # list of int
        return identity, context_index, cotnext_mask, \
            topic_path_kw, \
            user_profile_list, \
            target

    def __len__(self):
        return len(self.data)
