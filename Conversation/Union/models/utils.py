import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import GPT2Dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from math import exp
import ipdb
import pickle
from torch.nn import functional as F


def transformer_setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-exp_name", "--exp_name", type=str, default='v1')


    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)
    train.add_argument("-max_r_length", "--max_r_length", type=int, default=50)
    train.add_argument("-batch_size", "--batch_size", type=int, default=64)
    train.add_argument("-max_count", "--max_count", type=int, default=20)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-process_data",
                       "--process_data",
                       type=bool,
                       default=False)
    train.add_argument("-load_dict", "--load_dict", type=str, default=None)
    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/net_parameter1.pkl')
    train.add_argument("-learningrate",
                       "--learningrate",
                       type=float,
                       default=1)
    train.add_argument("-optimizer", "--optimizer", type=str, default='sgd')
    train.add_argument("-momentum", "--momentum", type=float, default=0)
    train.add_argument("-embedding_type",
                       "--embedding_type",
                       type=str,
                       default='random')
    train.add_argument("-epoch", "--epoch", type=int, default=1000)
    train.add_argument("-gpu", "--gpu", type=str, default='1')
    train.add_argument("-gradient_clip",
                       "--gradient_clip",
                       type=float,
                       default=0.1)
    train.add_argument("-embedding_size",
                       "--embedding_size",
                       type=int,
                       default=300)

    train.add_argument("-n_heads", "--n_heads", type=int, default=2)
    train.add_argument("-n_layers", "--n_layers", type=int, default=2)
    train.add_argument("-ffn_size", "--ffn_size", type=int, default=300)

    train.add_argument("-dropout", "--dropout", type=float, default=0.1)
    train.add_argument("-attention_dropout",
                       "--attention_dropout",
                       type=float,
                       default=0.0)
    train.add_argument("-relu_dropout",
                       "--relu_dropout",
                       type=float,
                       default=0.1)

    train.add_argument("-learn_positional_embeddings",
                       "--learn_positional_embeddings",
                       type=bool,
                       default=False)
    train.add_argument("-embeddings_scale",
                       "--embeddings_scale",
                       type=bool,
                       default=True)

    train.add_argument("-n_entity", "--n_entity", type=int, default=64368)
    train.add_argument("-n_relation", "--n_relation", type=int, default=214)
    train.add_argument("-n_concept", "--n_concept", type=int, default=29308)
    train.add_argument("-n_con_relation",
                       "--n_con_relation",
                       type=int,
                       default=48)
    train.add_argument("-dim", "--dim", type=int, default=128)
    train.add_argument("-n_hop", "--n_hop", type=int, default=2)
    train.add_argument("-kge_weight", "--kge_weight", type=float, default=1)
    train.add_argument("-l2_weight", "--l2_weight", type=float, default=2.5e-6)
    train.add_argument("-n_memory", "--n_memory", type=float, default=32)
    train.add_argument("-item_update_mode",
                       "--item_update_mode",
                       type=str,
                       default='0,1')
    train.add_argument("-using_all_hops",
                       "--using_all_hops",
                       type=bool,
                       default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    train.add_argument("-same_data", "--same_data", type=bool, default=False)
    # train.add_argument("--bpe2vec",type=str,default='../../data/bpe2vec.npy')
    train.add_argument("--bpe2index",
                       type=str,
                       default='../../data/data1030/output/bpe2index.json')
    train.add_argument('--do_eval', action='store_true', default=False)

    train.add_argument("-train_data_file",
                       "--train_data_file",
                       type=str,
                       default="../../data/data1030/output/train_cut.pkl",
                       help='要处理的数据的位置')
    train.add_argument("-valid_data_file",
                       "--valid_data_file",
                       type=str,
                       default="../../data/data1030/output/valid_cut.pkl",
                       help='要处理的数据的位置')
    train.add_argument("-test_data_file",
                       "--test_data_file",
                       type=str,
                       default="../../data/data1030/output/test_cut.pkl",
                       help='要处理的数据的位置')

    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='训练日志存放位置')  #todo
    train.add_argument("-use_size", "--use_size", type=int,
                       default=-1)  # pad_size，与其他模型不统一
    return train.parse_args()


def gpt2_setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        default='v0',
        type=str,
        required=False,
    )
    parser.add_argument('--model_config',
                        default='config/model_config_dialogue_small.json',
                        type=str,
                        required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path',
                        default='vocabulary/vocab_small.txt',
                        type=str,
                        required=False,
                        help='选择词库')
    parser.add_argument('--train_raw_path',
                        default='data/data_{}/train.txt',
                        type=str,
                        required=False,
                        help='原始训练语料')
    parser.add_argument('--valid_raw_path',
                        default='data/data_{}/valid.txt',
                        type=str,
                        required=False,
                        help='原始训练语料')
    parser.add_argument('--test_raw_path',
                        default='data/data_{}/test.txt',
                        type=str,
                        required=False,
                        help='原始训练语料')
    parser.add_argument('--train_tokenized_path',
                        default='data/data_{}/train_tokenized.txt',
                        type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--valid_tokenized_path',
                        default='data/data_{}/valid_tokenized.txt',
                        type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--test_tokenized_path',
                        default='data/data_{}/test_tokenized.txt',
                        type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path',
                        default='log/{}.log',
                        type=str,
                        required=False,
                        help='训练日志存放位置')
    parser.add_argument('--raw',
                        action='store_true',
                        help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs',
                        default=50,
                        type=int,
                        required=False,
                        help='训练的轮次')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        required=False,
                        help='训练batch size')
    parser.add_argument('--lr',
                        default=1.5e-4,
                        type=float,
                        required=False,
                        help='学习率')
    parser.add_argument('--warmup_steps',
                        default=2000,
                        type=int,
                        required=False,
                        help='warm up步数')
    parser.add_argument('--log_step',
                        default=100,
                        type=int,
                        required=False,
                        help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation',
                        default=1,
                        type=int,
                        required=False,
                        help='梯度积累')
    parser.add_argument('--max_grad_norm',
                        default=1.0,
                        type=float,
                        required=False)
    parser.add_argument('--is_model_output',
                        default=False,
                        action='store_true',
                        help='是否输出对话模型参数')
    parser.add_argument('--dialogue_model_output_path',
                        default='saved_model/{}',
                        type=str,
                        help='对话模型参数输出路径')
    parser.add_argument('--pretrained_model',
                        default='',
                        type=str,
                        required=False,
                        help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir',
                        default='tensorboard_summary/',
                        type=str,
                        required=False,
                        help='Tensorboard路径')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--mmi_model_output_path',
                        default='mmi_model',
                        type=str,
                        required=False,
                        help='MMI模型保存路径')
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--is_gen', action='store_true', default=False)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--identity2movieId',
                        type=str,
                        default='data/data_{}/identity2movieId.json')
    parser.add_argument('--identity2topicId',
                        type=str,
                        default='data/data_{}/identity2topicId.json')
    parser.add_argument('--topic_to_id',
                        type=str,
                        default='data/data_{}/topic_to_id.json')
    parser.add_argument('--movieid2name',
                        type=str,
                        default='data/data_{}/movies_with_mentions.csv')
    parser.add_argument('--test_path',
                        type=str,
                        default="../../data/test_data.pkl")
    parser.add_argument('--temperature',
                        default=1,
                        type=float,
                        required=False,
                        help='生成的temperature')
    parser.add_argument('--topk',
                        default=8,
                        type=int,
                        required=False,
                        help='最高k选1')
    parser.add_argument('--topp',
                        default=0,
                        type=float,
                        required=False,
                        help='最高积累概率')
    parser.add_argument('--save_samples_path',
                        default="generation/{}_output.txt",
                        type=str,
                        required=False,
                        help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty',
                        default=1.0,
                        type=float,
                        required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--max_len',
                        type=int,
                        default=50,
                        help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len',
                        type=int,
                        default=5,
                        help="dialogue history的最大长度")
    parser.add_argument('--max_context_len',
                        type=int,
                        default=250,
                        help="dialogue history的最大长度")
    parser.add_argument('--model_type',
                        type=str,
                        default='Ours',
                        help="Ours/GPT2/Transformer",
                        choices=['Ours', 'GPT2', 'Transformer'])

    return parser.parse_args()


def set_random_seed(seed, use_cuda):
    """
    设置训练的随机种子
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if use_cuda:
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


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim(
    ) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def load_movie(path):
    # 获得
    import csv
    name2id = {}
    db2id = {}
    id2name = {}
    movie_num = 0
    reader = csv.reader(open(path, 'r', encoding='utf-8-sig'))
    next(reader)
    for line in reader:
        global_id, name_time, db_id, _ = line
        name = '('.join(name_time.split('(')[:-1])
        name2id[name] = int(global_id)
        id2name[int(global_id)] = name
        db2id[int(db_id)] = int(global_id)

    return id2name
