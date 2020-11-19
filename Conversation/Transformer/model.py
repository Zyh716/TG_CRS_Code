from models.transformer import TorchGeneratorModel, _build_encoder, _build_decoder, _build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings, _create_entity_embeddings
from models.graph import SelfAttentionLayer, SelfAttentionLayer_batch
# from torch_geometric.nn.conv.rgcn_conv import RGCNConv
# from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json


class TransformerModel(nn.Module):
    def __init__(self,
                 opt,
                 dictionary,
                 padding_idx=0,
                 start_idx=2,
                 end_idx=3,
                 longest_label=1):
        '''
        args:
            dictionary: word2index when gen
            is_finetune: False when gen
        '''
        # self.pad_idx = dictionary[dictionary.null_token]
        # self.start_idx = dictionary[dictionary.start_token]
        # self.end_idx = dictionary[dictionary.end_token]
        super(TransformerModel, self).__init__()

        self.opt = opt
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']


        self.pad_idx = padding_idx
        self.NULL_IDX = padding_idx  #具体怎么用的: 计算loss的时候指定的
        self.END_IDX = end_idx
        # self.register_buffer('START', torch.LongTensor(
        #     [start_idx]))  #应该就是在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出。
        self.START = torch.LongTensor([start_idx])
        self.longest_label = longest_label  #这个是啥

        self.embeddings = _create_embeddings(dictionary, opt['embedding_size'],
                                             self.pad_idx)

        self.kg = pkl.load(open("data/subkg.pkl", "rb"))

        # n_positions 是干啥的，不是很懂下面的的定义
        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,  # 这是干嘛的
                opt.get('text_truncate') or 0,  # 这是干嘛的
                opt.get('label_truncate') or 0  # 这是干嘛的
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        # build model

        self.encoder = _build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction=False,
            n_positions=n_positions,
        )
        # self.back_encoder = _build_encoder_mask(
        #     opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
        #     n_positions=n_positions,
        # )
        self.decoder = _build_decoder4kg(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            n_positions=n_positions,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX,
                                             reduce=False)
        self.criterion_mask = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX,
                                                  reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.representation_bias = nn.Linear(opt['embedding_size'],
                                             len(dictionary) +
                                             4)  # 这个dict不包含4个特殊符号吗

        self.output_mask = nn.Linear(opt['embedding_size'],
                                     len(dictionary) + 4)

        self.embedding_size = opt['embedding_size']
        self.dim = opt['dim']

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1).to(self.opt['device'])

    def decode_greedy(self, encoder_states, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        # v2: encoder_states_kg encoder_states_db 都是None
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, incr_state)
            # batch*1*hidden
            scores = scores[:, -1:, :]

            voc_logits = F.linear(scores, self.embeddings.weight)
            # gate = F.sigmoid(self.gen_gate_norm(scores))

            _, preds = voc_logits.max(dim=-1)

            logits.append(voc_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) >
                            0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        # inputs不包含最后一个token
        inputs = ys.narrow(
            1, 0, seqlen - 1
        )  # 表示取变量input在第dimension维上，从索引start到start+length范围（不包括start+length）的值。
        # 添加start-token-id
        inputs = torch.cat([self._starts(bsz), inputs], 1).to(self.opt['device'])
        # [bs, r_len, hidden_size]
        # latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db) #batch*r_l*hidden
        latent, _ = self.decoder(inputs, encoder_states)  #batch*r_l*hidden

        logits = F.linear(latent, self.embeddings.weight)

        _, preds = logits.max(dim=2)

        # pred只是用的latent form decoder？
        return logits, preds

    def mask_predict_loss(self,
                          m_emb,
                          attention,
                          context,
                          mask,
                          rec,
                          mask_id=3):
        pass

    def forward(self,
                xs,
                ys,
                mask_ys,
                concept_mask,
                db_mask,
                seed_sets,
                labels,
                con_label,
                db_label,
                entity_vector,
                rec,
                test=True,
                cand_params=None,
                prev_enc=None,
                maxlen=None,
                bsz=None):
        # maxlen或者max(self.longest_label, ys.size(1))是生成的回复的最大长度
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        if not test:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            # todo 运行时是多少
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        # xxs = self.embeddings(xs)
        # mask = xs == self.pad_idx
        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)

        if not test:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, mask_ys)
            # print(f"Score.shape = {scores.shape}, preds.shape = {preds.shape} when test==Flase")

            loss = self.compute_loss(scores, mask_ys)
            notnull = mask_ys.ne(self.pad_idx)
            target_tokens = notnull.long().sum().item()
            gen_loss = torch.sum(loss) / target_tokens
            # print("[gen_loss = {}, target_tokens = {}]".format(gen_loss, target_tokens))
        else:
            scores, preds = self.decode_greedy(encoder_states, bsz,
                                               maxlen or self.longest_label)
            # print(f"Score.shape = {scores.shape}, preds.shape = {preds.shape} when test==True")

            gen_loss = None
            # gen_loss = torch.mean(self.compute_loss(scores, mask_ys))

        # return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss
        return scores, preds, None, None, gen_loss, None, None, None

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        # print("shape of output in CrossMode = ", output.shape)  # 960
        # print("shape of scores in CrossMode = ", scores.shape)
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        # shape of loss?
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        # print("shape of loss in CrossMode = ", loss.shape)  # 960
        # print("loss in CrossMode = ", loss)  # 960
        # print("shape of score_view in CrossMode = ", score_view.shape)

        return loss

    def save_model(self, path):
        # torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        # self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))
        print(self.opt['device'])
        load_state = torch.load(path, map_location=self.opt['device'])
        load_state_keys = set(load_state.keys())

        this_state_keys = set(self.state_dict().keys())
        assert this_state_keys.issubset(load_state_keys)
        for key in load_state_keys - this_state_keys:
            del load_state[key]

        self.load_state_dict(load_state)

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        # up_bias = up_bias.unsqueeze(dim=1)
        # output += up_bias
        return output
