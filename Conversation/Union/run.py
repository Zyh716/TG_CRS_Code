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

from models.utils import set_random_seed
from models.utils import create_logger
from models.utils import top_k_top_p_filtering
from models.utils import load_movie

from models.utils import gpt2_setup_args
from models.utils import transformer_setup_args

from models.train_loop import TrainLoop_GPT2
from models.train_loop import TrainLoop_Ours


def main():
    args = gpt2_setup_args()
    # if args.model_type == 'transformer':
    #     args = transformer_setup_args()

    args.use_cuda = torch.cuda.is_available() and args.use_cuda
    args.save_samples_path = args.save_samples_path.format(args.exp_name)
    args.dialogue_model_output_path = args.dialogue_model_output_path.format(
        args.exp_name)
    args.log_path = args.log_path.format(args.exp_name)

    args.train_raw_path = args.train_raw_path.format(args.model_type)
    args.valid_raw_path = args.valid_raw_path.format(args.model_type)
    args.test_raw_path = args.test_raw_path.format(args.model_type)
    args.train_tokenized_path = args.train_tokenized_path.format(
        args.model_type)
    args.valid_tokenized_path = args.valid_tokenized_path.format(
        args.model_type)
    args.test_tokenized_path = args.test_tokenized_path.format(args.model_type)
    args.identity2movieId = args.identity2movieId.format(args.model_type)
    args.identity2topicId = args.identity2topicId.format(args.model_type)
    args.topic_to_id = args.topic_to_id.format(args.model_type)
    args.movieid2name = args.movieid2name.format(args.model_type)

    logger = create_logger(args)
    logger.info(vars(args))

    if args.seed:
        set_random_seed(args.seed, args.use_cuda)

    if args.model_type == 'Ours':
        loop = TrainLoop_Ours(args, logger)
    elif args.model_type == 'GPT2':
        loop = TrainLoop_GPT2(args, logger)

    if args.is_gen:
        loop.generate()
    elif args.do_eval:
        loop.val('test')
    else:
        loop.train()


if __name__ == '__main__':
    main()
