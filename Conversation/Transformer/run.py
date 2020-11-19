import numpy as np
from tqdm import tqdm
from math import exp
import os
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset, CRSdataset
from model import TransformerModel
import torch.nn as nn
from torch import optim
import torch
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu
import nltk
import re
import pickle
import logging
import ipdb

def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


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
    train.add_argument("-exp_name", "--exp_name", type=str, default='v1')

    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)
    train.add_argument("-max_r_length", "--max_r_length", type=int, default=50)
    train.add_argument("-batch_size", "--batch_size", type=int, default=64)
    train.add_argument("-max_count", "--max_count", type=int, default=20)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=False)
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
    return train


def _create_dictionary():
    '''
    word2index
    '''
    pass
    return {}


class TrainLoop_Transformer():
    def __init__(self, opt):
        self.opt = opt

        self.dict = json.load(open(args.bpe2index, encoding='utf-8'))
        self.index2word = {self.dict[key]: key for key in self.dict}

        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']
        self.use_cuda = opt['use_cuda']
        print('self.use_cuda:', self.use_cuda)

        self.device = 'cuda:{}'.format(
            self.opt['gpu']) if self.use_cuda else 'cpu'
        self.opt['device'] = self.device

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))

        # self.metrics_gen = {
        #     "ppl": 0,
        #     "dist1": 0,
        #     "dist2": 0,
        #     "dist3": 0,
        #     "dist4": 0,
        #     "bleu1": 0,
        #     "bleu2": 0,
        #     "bleu3": 0,
        #     "bleu4": 0,
        #     "count": 0
        # }

        self.build_data()
        self.build_model()

        # self.init_optim(
        #     [p for p in self.model.parameters() if p.requires_grad],
        #     optim_states=states.get('optimizer'),
        #     saved_optim_type=states.get('optimizer_type')
        # )
        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad])

    def build_data(self):
        if self.opt['process_data']:
            self.train_dataset = dataset(
                "../../data/data1030/output/train_cut.pkl", self.opt, 'train')
            self.valid_dataset = dataset(
                "../../data/data1030/output/valid_cut.pkl", self.opt, 'valid')
            self.test_dataset = dataset(
                "../../data/data1030/output/test_cut.pkl", self.opt, 'test')

            self.train_processed_set = self.train_dataset.data_process(True)
            self.valid_processed_set = self.valid_dataset.data_process(True)
            self.test_processed_set = self.test_dataset.data_process(True)

            pickle.dump(self.train_processed_set,
                        open('data/train_processed_set.pkl', 'wb'))
            pickle.dump(self.valid_processed_set,
                        open('data/valid_processed_set.pkl', 'wb'))
            pickle.dump(self.test_processed_set,
                        open('data/test_processed_set.pkl', 'wb'))
            logger.info("[Save processed data]")
        else:
            try:
                self.train_processed_set = pickle.load(
                    open('data/train_processed_set.pkl', 'rb'))
                self.valid_processed_set = pickle.load(
                    open('data/valid_processed_set.pkl', 'rb'))
                self.test_processed_set = pickle.load(
                    open('data/test_processed_set.pkl', 'rb'))
            except:
                assert 1 == 0, "No processed data"
            logger.info("[Load processed data]")

    def build_model(self):
        self.model = TransformerModel(self.opt, self.dict)
        # todo
        if self.opt['embedding_type'] != 'random':
            pass

        if self.opt['load_dict'] is not None:
            logger.info('[ Loading existing model params from {} ]'
                        ''.format(self.opt['load_dict']))
            self.model.load_model(self.opt['load_dict'])

        if self.use_cuda:
            self.model.to(self.device)

    def train(self):
        losses = []
        best_val_gen = 1000
        gen_stop = False
        patience = 0
        max_patience = 5
        num = 0

        # file_temp = open('temp.txt', 'w')
        # train_output_file = open(f"output_train_tf.txt", 'w', encoding='utf-8')

        for i in range(self.epoch):
            train_set = CRSdataset(self.train_processed_set,
                                   self.opt['n_entity'], self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(
                dataset=train_set, batch_size=self.batch_size,
                shuffle=True)  # shuffle

            for context,c_lengths,response,r_length,mask_response, \
                    mask_r_length,entity,entity_vector,movie,\
                    concept_mask,dbpedia_mask,concept_vec, \
                    db_vec,rec in tqdm(train_dataset_loader):
                ####################################### 检验输入输出ok
                # file_temp.writelines("[Context] ", self.vector2sentence(context))
                # file_temp.writelines("[Response] ", self.vector2sentence(response))
                # file_temp.writelines("\n")

                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)

                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss= \
                    self.model(context.to(self.device), response.to(self.device), mask_response.to(self.device), concept_mask, dbpedia_mask, seed_sets, movie, \
                        concept_vec, db_vec, entity_vector.to(self.device), rec, test=False)

                ##########################################
                # train_output_file.writelines(
                #     ["Loss per batch = %f\n" % gen_loss.item()])
                # train_output_file.writelines(['[GroundTruth] ' + ' '.join(sen_gt)+'\n' \
                #     + '[Generated] ' + ' '.join(sen_gen)+'\n\n' \
                #     for sen_gt, sen_gen in zip(self.vector2sentence(response.cpu()), self.vector2sentence(preds.cpu()))])

                losses.append([gen_loss])
                self.backward(gen_loss)
                self.update_params()

                if num % 50 == 0:
                    loss = sum([l[0] for l in losses]) / len(losses)
                    ppl = exp(loss)
                    logger.info('gen loss is %f, ppl is %f' % (loss, ppl))
                    losses = []

                num += 1

            output_metrics_gen = self.val(epoch=i)
            _ = self.val(True, epoch=i)

            if best_val_gen < output_metrics_gen["ppl"]:
                patience += 1
                logger.info('Patience = ', patience)
                if patience >= 5:
                    gen_stop = True
            else:
                patience = 0
                best_val_gen = output_metrics_gen["ppl"]
                self.model.save_model(self.opt['model_save_path'])
                logger.info(
                    f"[generator model saved in {self.opt['model_save_path']}"
                    "------------------------------------------------]")

            if gen_stop:
                break

        # train_output_file.close()
        # _ = self.val(is_test=True)

    def val(self, is_test=False, epoch=-1):
        # count是response数量
        self.model.eval()
        if is_test:
            valid_processed_set = self.test_processed_set
        else:
            valid_processed_set = self.valid_processed_set

        val_set = CRSdataset(valid_processed_set, self.opt['n_entity'],
                             self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=self.batch_size, shuffle=False)

        inference_sum = []
        tf_inference_sum = []
        golden_sum = []
        # context_sum = []
        losses = []
        recs = []

        for context, c_lengths, response, r_length, mask_response, mask_r_length, \
                entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec \
                in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)

                # 使用teacher force下的回复生成，
                _, tf_preds, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss = \
                    self.model(context.to(self.device), response.to(self.device), mask_response.to(self.device), concept_mask, dbpedia_mask, \
                        seed_sets, movie, concept_vec, db_vec, entity_vector.to(self.device), rec, test=False)

                # 使用greedy模式下的回复生成，限定maxlen=20？
                # todo
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = \
                    self.model(context.to(self.device), response.to(self.device), mask_response.to(self.device), concept_mask, dbpedia_mask, \
                        seed_sets, movie, concept_vec, db_vec, entity_vector.to(self.device), rec, test=True, maxlen=20, bsz=batch_size)

            golden_sum.extend(self.vector2sentence(response.cpu()))
            inference_sum.extend(self.vector2sentence(preds.cpu()))
            # tf_inference_sum.extend(self.vector2sentence(tf_preds.cpu()))
            # context_sum.extend(self.vector2sentence(context.cpu()))
            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            #logger.info(losses)
            #exit()

        subset = 'valid' if not is_test else 'test'

        # 原版： gen-loss来自teacher force，inference_sum来自greedy
        ppl = exp(sum(loss for loss in losses) / len(losses))
        output_dict_gen = {'ppl': ppl}
        logger.info(f"{subset} set metrics = {output_dict_gen}")
        # logger.info(f"{subset} set gt metrics = {self.metrics_gt}")

        # f=open('context_test.txt','w',encoding='utf-8')
        # f.writelines([' '.join(sen)+'\n' for sen in context_sum])
        # f.close()

        # 将生成的回复输出
        with open(f"output/output_{subset}_gen_epoch_{epoch}.txt",
                  'w',
                  encoding='utf-8') as f:
            f.writelines([
                '[Generated] ' + re.sub('@\d+', '__UNK__', ' '.join(sen)) +
                '\n' for sen in inference_sum
            ])

        # gt shuchu
        with open(f"output/output_{subset}_gt_epoch_{epoch}.txt",
                  'w',
                  encoding='utf-8') as f:
            for sen in golden_sum:
                mask_sen = re.sub('@\d+', '__UNK__', ' '.join(sen))
                mask_sen = re.sub(' ([!,.?])', '\\1', mask_sen)
                f.writelines(['[GT] ' + mask_sen + '\n'])

        # 将生成的回复与gt一起输出
        with open(f"output/output_{subset}_both_epoch_{epoch}.txt",
                  'w',
                  encoding='utf-8') as f:
            f.writelines(['[GroundTruth] ' + re.sub('@\d+', '__UNK__',' '.join(sen_gt))+'\n' \
                + '[Generated] ' + re.sub('@\d+', '__UNK__',' '.join(sen_gen))+'\n\n' \
                for sen_gt, sen_gen in zip(golden_sum, inference_sum)])

        self.save_embedding()

        return output_dict_gen

    def save_embedding(self):
        json.dump(loop.dict, open('output/tf_bpe2index.json', 'w'))

    def vector2sentence(self, batch_sen):
        # 一个batch的sentence 从id换成token
        sentences = []
        for sen in batch_sen.numpy().tolist():
            sentence = []
            for word in sen:
                if word > 3:
                    sentence.append(self.index2word[word])
                elif word == 3:
                    sentence.append('_UNK_')
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

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        # kwargs['amsgrad'] = True
        # kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        logger.info(f'optim_class = {optim_class}')
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum +
                                       1) % update_freq
            if self._number_grad_accum != 0:
                return
        #0.1是不是太小了，原版就是这样
        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.opt['gradient_clip'])

        self.optimizer.step()

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
    print(args.use_cuda)
    print(args.process_data)

    global logger
    logger = create_logger(args)
    logger.info(vars(args))

    loop = TrainLoop_Transformer(vars(args))
    if args.do_eval:
        loop.val(True)
    else:
        loop.train()
