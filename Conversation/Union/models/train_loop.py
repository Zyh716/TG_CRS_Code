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


class TrainLoop_GPT2():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.args.device = 'cuda:{}'.format(
            self.args.gpu) if self.args.use_cuda else 'cpu'
        self.logger.info('using device:{}'.format(self.args.device))

        self.opt = vars(self.args)

        self.batch_size = self.opt['batch_size']
        self.use_cuda = self.opt['use_cuda']
        self.device = self.args.device
        self.multi_gpu = self.args.use_multi_gpu

        # self.movie_ids = pickle.load(open("data/movie_ids.pickle", "rb"))

        self.build_data()
        self.build_model()

    def build_data(self):
        self.tokenizer = BertTokenizer(vocab_file=self.args.vocab_path)
        self.vocab_size = len(self.tokenizer)
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

        # 对原始数据进行预处理,将原始语料转换成对应的token_id
        if self.args.raw:
            for subset in ['train', 'valid', 'test']:
                self.preprocess_raw_data(subset)
        # 加载tokenized data
        self.subset2data = {}
        with open(self.args.test_tokenized_path, "r", encoding="utf8") as f:
            self.subset2data['test'] = f.read()
        if not self.args.do_eval:
            with open(self.args.train_tokenized_path, "r",
                      encoding="utf8") as f:
                self.subset2data['train'] = f.read()
            with open(self.args.valid_tokenized_path, "r",
                      encoding="utf8") as f:
                self.subset2data['valid'] = f.read()
        # 这一步是干啥的
        for subset in self.subset2data:
            self.subset2data[subset] = self.subset2data[subset].split("\n")

        self.logger.info("Train/Valid/Test set has {} convs".format(
            [len(self.subset2data[subset]) for subset in self.subset2data]))

    def build_model(self):
        """

        :param args:
        :param vocab_size:字典大小
        :return:
        """
        if self.args.pretrained_model:
            # 如果指定了预训练的GPT2模型
            self.model = GPT2LMHeadModel.from_pretrained(
                self.args.pretrained_model)
        else:
            # 若没有指定预训练模型，则初始化模型
            model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(
                self.args.model_config)
            self.model = GPT2LMHeadModel(config=model_config)

        # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
        self.model.resize_token_embeddings(self.vocab_size)

        if self.use_cuda:
            self.model.to(self.device)

        self.logger.info('model config:\n{}'.format(
            self.model.config.to_json_string()))

        self.n_ctx = self.model.config.to_dict().get("n_ctx")

        # 建立模型存储路径
        if self.args.is_model_output and not os.path.exists(
                self.args.dialogue_model_output_path):
            os.mkdir(self.args.dialogue_model_output_path)

        # 记录模型参数数量
        num_parameters = 0
        parameters = self.model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        self.logger.info(
            'number of model parameters: {}'.format(num_parameters))

        # 是否使用多块GPU进行并行运算
        if self.args.use_multi_gpu:
            if self.args.use_cuda and torch.cuda.device_count() > 1:
                self.logger.info("Let's use GPUs to train")
                self.model = DataParallel(
                    self.model,
                    device_ids=[int(i) for i in self.args.device.split(',')])
            else:
                self.args.use_multi_gpu = False

    def train(self):
        train_dataset = GPT2Dataset(self.subset2data['train'])
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=self.args.num_workers,
                                      collate_fn=self.collate_fn)

        # 计算所有epoch进行参数优化的总步数total_steps
        self.total_steps = int(train_dataset.__len__() * self.args.epochs /
                               self.args.batch_size /
                               self.args.gradient_accumulation)
        self.logger.info('total training steps = {}'.format(self.total_steps))

        self.init_optim()

        self.logger.info('starting training')
        # 用于统计每次梯度累计的loss
        running_loss = 0
        # 统计一共训练了多少个step
        overall_step = 0
        # 记录tensorboardX
        # tb_writer = SummaryWriter(log_dir=self.args.writer_dir)
        # 记录 out of memory的次数
        oom_time = 0
        # patience
        patience = 0
        max_patience = 2
        best_test_loss = 10000
        # 开始训练
        for epoch in range(self.args.epochs):
            epoch_start_time = datetime.now()
            train_loss = []  # 记录一个epoch里面的train loss
            for batch_idx, (input_ids, mask_r) in enumerate(train_dataloader):
                # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
                # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
                # self.logger.info(input_ids == mask_r)
                # self.logger.info(input_ids)
                # self.logger.info(mask_r)
                # for context in input_ids:
                #     print(tokenizer.convert_ids_to_tokens(int(id) for id in context))
                # ipdb.set_trace()
                self.model.train()
                input_ids = input_ids.to(self.device)
                # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
                try:
                    outputs = self.model.forward(input_ids=input_ids)
                    loss, accuracy = self.calculate_loss_and_accuracy(
                        outputs, input_ids, mask_r, device=self.device)
                    train_loss.append(loss.item())

                    if self.multi_gpu:
                        loss = loss.mean()
                        accuracy = accuracy.mean()
                    if self.args.gradient_accumulation > 1:
                        loss = loss / self.args.gradient_accumulation
                        accuracy = accuracy / self.args.gradient_accumulation
                    loss.backward()
                    # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                    # 进行一定step的梯度累计之后，更新参数
                    if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                        running_loss += loss.item()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()

                        overall_step += 1
                        # 更新日志与tnesorboardX信息
                        if (overall_step + 1) % self.args.log_step == 0:
                            self.logger.info(
                                "batch {} of epoch {}, loss {:.4f}, ppl {:.5f}"
                                .format(batch_idx + 1, epoch + 1, loss,
                                        exp(loss)))
                            # tb_writer.add_scalar('loss', loss.item(), overall_step)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        self.logger.info(
                            "WARNING: ran out of memory,times: {}".format(
                                oom_time))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        self.logger.info(str(exception))
                        raise exception
            train_loss = sum(train_loss) / len(train_loss)
            epoch_finish_time = datetime.now()
            self.logger.info(
                'epoch {}, train loss is {:.4f}, ppl is {:.5f}, spend {} time'.
                format(epoch + 1, train_loss, exp(train_loss),
                       epoch_finish_time - epoch_start_time))
            # val
            # test_loss = val(model, device, test_list, multi_gpu, self.args)
            test_loss = self.val('valid')
            if test_loss <= best_test_loss:
                patience = 0
                best_test_loss = test_loss

                self.logger.info('saving model for epoch {}'.format(epoch + 1))
                model_path = join(self.args.dialogue_model_output_path,
                                  'model')
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                # 这里是什么意思，还不是很懂
                model_to_save = self.model.module if hasattr(
                    self.model, 'module') else self.model
                model_to_save.save_pretrained(model_path)
                self.logger.info("save model to " + str(model_path))
            else:
                patience += 1
                self.logger.info('Patience = ' + str(patience))
                if patience >= max_patience:
                    break
            test_loss = self.val('test')

        # self.logger.info('training finished')

    def val(self, subset):
        # self.logger.info("start evaluating model")
        self.model.eval()
        # self.logger.info('starting evaluating')
        # 记录tensorboardX
        # tb_writer = SummaryWriter(log_dir=self.args.writer_dir)
        test_dataset = GPT2Dataset(self.subset2data[subset])
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=True,
                                     num_workers=self.args.num_workers,
                                     collate_fn=self.collate_fn)
        test_loss = []
        # test_accuracy = []
        with torch.no_grad():
            for batch_idx, (input_ids, mask_r) in enumerate(test_dataloader):
                input_ids = input_ids.to(self.device)
                outputs = self.model.forward(input_ids=input_ids)
                loss, accuracy = self.calculate_loss_and_accuracy(
                    outputs, input_ids, mask_r, device=self.device)
                test_loss.append(loss.item())
                # test_accuracy.append(accuracy)
                if self.multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if self.args.gradient_accumulation > 1:
                    loss = loss / self.args.gradient_accumulation
                    accuracy = accuracy / self.args.gradient_accumulation
                # self.logger.info("val batch {} ,loss {} ,accuracy {}".format(batch_idx, loss, accuracy))
                # tb_writer.add_scalar('loss', loss.item(), overall_step)
        test_loss = sum(test_loss) / len(test_loss)
        self.logger.info("val {} loss {:.4f} , ppl {:.5f}".format(
            subset, test_loss, exp(test_loss)))

        return test_loss

    def generate(self):
        samples_file = open(self.args.save_samples_path, 'w', encoding='utf8')
        convs = pickle.load(open(self.args.test_path, 'rb'))

        for conv in tqdm(convs[:]):
            conv_id = conv['conv_id']
            history = []  # list of id, to model

            for message in conv['messages']:
                message_id, role, content = int(
                    message['local_id']), message['role'], message['content']
                if role == 'Recommender' and message_id != 1:
                    try:
                        if self.args.save_samples_path:
                            samples_file.write(f"[GroundTruth]: {content}\n")
                        input_ids = [
                            self.tokenizer.cls_token_id
                        ] + history[-self.args.max_context_len +
                                    1:]  # 每个input以[CLS]为开头 [SEP]结尾
                        # tensor of [input_token_num]
                        curr_input_tensor = torch.tensor(input_ids).long().to(
                            self.device)
                        generated = []
                        # 最多生成max_len个token
                        for _ in range(self.args.max_len):
                            # (tensor of [input_token_nums, 13317], tuple of 10 tensor)
                            outputs = self.model(
                                input_ids=curr_input_tensor)  #?shape?
                            # tensor of [13317]
                            next_token_logits = outputs[0][-1, :]
                            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                            for id in set(generated):
                                next_token_logits[
                                    id] /= self.args.repetition_penalty
                            next_token_logits = next_token_logits / self.args.temperature
                            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                            next_token_logits[
                                self.tokenizer.convert_tokens_to_ids(
                                    '[UNK]')] = -float('Inf')
                            # 将topk以外的token的概率设置为-inf，然后排序，然后将accum-概率大与topp的token的概率设置为-inf
                            filtered_logits = top_k_top_p_filtering(
                                next_token_logits,
                                top_k=self.args.topk,
                                top_p=self.args.topp)
                            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                            next_token = torch.multinomial(F.softmax(
                                filtered_logits, dim=-1),
                                                           num_samples=1)
                            if next_token == self.tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                                break
                            generated.append(next_token.item())
                            curr_input_tensor = torch.cat(
                                (curr_input_tensor, next_token),
                                dim=0)[-self.n_ctx:]
                        generated_text = self.tokenizer.convert_ids_to_tokens(
                            generated)
                        if self.args.save_samples_path:
                            samples_file.write("[Generated]: {}\n\n".format(
                                "".join(generated_text)))

                    except Exception as e:
                        print(e)
                        print(conv_id, message_id)
                        print(max(input_ids))
                        print('\n')
                history.extend(
                    self.tokenizer.encode(content) +
                    [self.tokenizer.sep_token_id])  #? encode成了啥

        samples_file.close()

    def calculate_loss_and_accuracy(self, outputs, labels, mask_r, device):
        """
        计算非self.pad_id的平均loss和准确率
        :param outputs:
        :param labels:
        :param device:
        :return:
        """
        logits = outputs[
            0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
        # 用前n-1个token，预测出第n个token
        # 用第i个token的prediction_score用来预测第i+1个token。
        # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(device)
        ##################################### shift_labels给mask掉
        mask_shift_labels = mask_r[..., 1:].contiguous().to(device)
        shift_labels = shift_labels * mask_shift_labels
        #######################################

        loss_fct = CrossEntropyLoss(
            ignore_index=self.pad_id,
            reduction='sum')  # 忽略self.pad_id的loss,并对所有的非self.pad_id的loss进行求和
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        _, preds = shift_logits.max(
            dim=-1
        )  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

        # 对非self.pad_id的token的loss进行求平均，且计算出预测的准确率
        not_ignore = shift_labels.ne(
            self.pad_id
        )  # 进行非运算，返回一个tensor，若targets_view的第i个位置为self.pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item(
        )  # 计算target中的非self.pad_id的数量

        correct = (shift_labels
                   == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def preprocess_raw_data(self, subset):
        """
        对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
        :param args:
        :param tokenizer:
        :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
        :return:
        """
        self.logger.info(
            "tokenizing raw data,raw data path:{}, token output path:{}".
            format(args.train_raw_path, args.train_tokenized_path))
        if subset == 'train':
            raw_path = self.args.train_raw_path
        elif subset == 'valid':
            raw_path = self.args.valid_raw_path
        elif subset == 'test':
            raw_path = self.args.test_raw_path

        with open(raw_path, 'rb') as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n\r\n")
        else:
            train_data = data.split("\n\n")
        self.logger.info("there are {} dialogue in raw dataset".format(
            len(train_data)))
        if subset == 'train':
            path = self.args.train_tokenized_path
        elif subset == 'valid':
            path = self.args.valid_tokenized_path
        elif subset == 'test':
            path = self.args.test_tokenized_path
        with open(path, "w", encoding="utf-8") as f:
            for dialogue_index, dialogue in enumerate(tqdm(train_data)):
                if "\r\n" in data:
                    utterances = dialogue.split("\r\n")
                else:
                    utterances = dialogue.split("\n")
                # dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
                dialogue_ids = []  # 每个dialogue以[CLS]开头
                for utterance in utterances:
                    dialogue_ids.extend([
                        self.tokenizer.convert_tokens_to_ids(word)
                        for word in utterance
                    ])
                    dialogue_ids.append(self.tokenizer.sep_token_id
                                        )  # 每个utterance之后添加[SEP]，表示utterance结束
                # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                ###############################m
                dialogue_ids = [self.tokenizer.cls_token_id
                                ] + dialogue_ids[-self.n_ctx + 1:]
                # dialogue_ids = dialogue_ids[:n_ctx]
                for dialogue_id in dialogue_ids:
                    f.write(str(dialogue_id) + ' ')
                # 最后一条记录不添加换行符
                if dialogue_index < len(train_data) - 1:
                    f.write("\n")
        self.logger.info(
            "finish preprocessing raw data,the result is stored in {}".format(
                self.args.train_tokenized_path))

    def collate_fn(self, batch):
        """
        计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
        :param batch:
        :return:
        """
        input_ids = []
        mask_rs = []
        btc_size = len(batch)
        max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
        # 计算该batch中input的最大长度
        # for btc_idx in range(btc_size):
        #     if max_input_len < len(batch[btc_idx]):
        #         max_input_len = len(batch[btc_idx])
        # 使用pad_id对小于max_input_len的input_id进行补全
        # for btc_idx in range(btc_size):
        #     input_len = len(batch[btc_idx])
        #     input_ids.append(batch[btc_idx])
        #     input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))

        # 计算该batch中input的最大长度
        for btc_idx, (inputs, mask_r) in enumerate(batch):
            if max_input_len < len(inputs):
                max_input_len = len(inputs)
        # 使用pad_id对小于max_input_len的input_id进行补全
        for btc_idx, (inputs, mask_r) in enumerate(batch):
            assert len(inputs) == len(mask_r), f"{len(inputs)}, {len(mask_r)}"
            input_len = len(inputs)
            input_ids.append(inputs)
            input_ids[btc_idx].extend([self.pad_id] *
                                      (max_input_len - input_len))
            mask_rs.append(mask_r)
            mask_rs[btc_idx].extend([self.pad_id] *
                                    (max_input_len - input_len))
        # self.logger.info(torch.tensor(input_ids, dtype=torch.long).shape)
        # self.logger.info(torch.tensor(mask_rs, dtype=torch.long).shape)
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(mask_rs, dtype=torch.long))

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
        self.logger.info(optims)
        return optims

    def init_optim(self):
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
        # 设置优化器，并且在初始训练时，使用warmup策略
        self.optimizer = transformers.AdamW(self.model.parameters(),
                                            lr=self.args.lr,
                                            correct_bias=True)
        self.scheduler = transformers.WarmupLinearSchedule(
            self.optimizer,
            warmup_steps=self.args.warmup_steps,
            t_total=self.total_steps)

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


class TrainLoop_Ours(TrainLoop_GPT2):
    def __init__(self, args, logger):
        super(TrainLoop_Ours, self).__init__(args, logger)

    def generate(self):
        samples_file = open(self.args.save_samples_path, 'w', encoding='utf8')

        identity2movieId = json.load(open(self.args.identity2movieId, 'r'))
        identity2topicId = json.load(open(self.args.identity2topicId, 'r'))
        topic_to_id = json.load(open(self.args.topic_to_id, 'r'))
        id2topic = {int(id): topic for topic, id in topic_to_id.items()}
        print(max(id2topic.keys()))
        movieid2name = load_movie(self.args.movieid2name)
        convs = pickle.load(open(self.args.test_path, 'rb'))

        for conv in tqdm(convs[:]):
            conv_id = conv['conv_id']
            history = []  # list of id, to model

            for message in conv['messages']:
                message_id, role, content = int(
                    message['local_id']), message['role'], message['content']
                identity = str(conv_id) + '/' + str(message_id)
                if role == 'Recommender' and message_id != 1:
                    if self.args.save_samples_path:
                        samples_file.write(f"[GroundTruth]: {content}\n")
                    # 组织使用到的对话历史数据
                    # goal_path = conv['goal_path']
                    # goals = goal_path[message_id]
                    kw = ''
                    if identity in identity2topicId:
                        kw = id2topic[int(identity2topicId[identity])]
                    if identity in identity2movieId:
                        kw = '《' + movieid2name[
                            identity2movieId[identity]] + '》'

                    kw_id_list = [
                        self.tokenizer.convert_tokens_to_ids(word)
                        for word in kw
                    ] + [self.tokenizer.sep_token_id]
                    input_ids = [
                        self.tokenizer.cls_token_id
                    ] + kw_id_list + history[
                        -self.args.max_context_len + 1 +
                        len(kw_id_list):]  # 每个input以[CLS]为开头 [SEP]结尾

                    # tensor of [input_token_num]
                    curr_input_tensor = torch.tensor(input_ids).long().to(
                        self.device)
                    generated = []
                    # 最多生成max_len个token
                    for _ in range(self.args.max_len):
                        # (tensor of [input_token_nums, 13317], tuple of 10 tensor)
                        outputs = self.model(input_ids=curr_input_tensor)
                        # tensor of [13317]
                        next_token_logits = outputs[0][-1, :]
                        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                        for id in set(generated):
                            next_token_logits[
                                id] /= self.args.repetition_penalty
                        next_token_logits = next_token_logits / self.args.temperature
                        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                        next_token_logits[self.tokenizer.convert_tokens_to_ids(
                            '[UNK]')] = -float('Inf')
                        # 将topk以外的token的概率设置为-inf，然后排序，然后将accum-概率大与topp的token的概率设置为-inf
                        filtered_logits = top_k_top_p_filtering(
                            next_token_logits,
                            top_k=self.args.topk,
                            top_p=self.args.topp)
                        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                        next_token = torch.multinomial(F.softmax(
                            filtered_logits, dim=-1),
                                                       num_samples=1)
                        if next_token == self.tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                            break
                        generated.append(next_token.item())
                        curr_input_tensor = torch.cat(
                            (curr_input_tensor, next_token),
                            dim=0)[-self.n_ctx:]
                    # generated 生成的utter的ids
                    generated_text = self.tokenizer.convert_ids_to_tokens(
                        generated)
                    if self.args.save_samples_path:
                        samples_file.write("{}\n".format(kw))
                        samples_file.write("[Generated]: {}\n\n".format(
                            "".join(generated_text)))
                history.extend(
                    self.tokenizer.encode(content) +
                    [self.tokenizer.sep_token_id])  #? encode成了啥

        samples_file.close()

    def preprocess_raw_data(self, subset):
        """
        对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
        :param args:
        :param tokenizer:
        :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
        :return:
        """

        self.logger.info(
            "tokenizing raw data,raw data path:{}, token output path:{}".
            format(self.args.train_raw_path, self.args.train_tokenized_path))
        if subset == 'train':
            raw_path = self.args.train_raw_path
        elif subset == 'valid':
            raw_path = self.args.valid_raw_path
        elif subset == 'test':
            raw_path = self.args.test_raw_path

        with open(raw_path, 'rb') as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n\r\n")
        else:
            train_data = data.split("\n\n")
        self.logger.info("there are {} dialogue in raw {} dataset".format(
            len(train_data), subset))

        if subset == 'train':
            path = self.args.train_tokenized_path
        elif subset == 'valid':
            path = self.args.valid_tokenized_path
        elif subset == 'test':
            path = self.args.test_tokenized_path

        with open(path, "w", encoding="utf-8") as f:
            for dialogue_index, dialogue in enumerate(tqdm(train_data)):
                if "\r\n" in data:
                    utterances = dialogue.split("\r\n")
                else:
                    utterances = dialogue.split("\n")
                # dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
                dialogue_ids = []  # 每个dialogue以[CLS]开头
                prompt, utterances = utterances[0], utterances[1:]
                for utterance in utterances:
                    # print(utterance) # string
                    # dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                    word_list = [word for word in utterance]
                    dialogue_ids.extend(
                        self.tokenizer.convert_tokens_to_ids(word_list))
                    dialogue_ids.append(self.tokenizer.sep_token_id
                                        )  # 每个utterance之后添加[SEP]，表示utterance结束
                # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                ###############################m
                dialogue_ids = [self.tokenizer.cls_token_id] + \
                    [self.tokenizer.convert_tokens_to_ids(word) for word in prompt] + \
                    [self.tokenizer.sep_token_id] + \
                    dialogue_ids[-self.n_ctx+2+len(prompt):]
                # dialogue_ids = dialogue_ids[:n_ctx]
                for dialogue_id in dialogue_ids:
                    f.write(str(dialogue_id) + ' ')
                # 最后一条记录不添加换行符
                if dialogue_index < len(train_data) - 1:
                    f.write("\n")
        self.logger.info(
            "finish preprocessing raw data,the result is stored in {}".format(
                self.args.train_tokenized_path))
