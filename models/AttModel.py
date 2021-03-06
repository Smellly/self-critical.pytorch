# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
import visdom
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_ln = getattr(opt, 'use_ln', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        if self.use_ln:
            print('Using LayerNorm')
            self.u_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.att_feat_size),))+
                                    (
                                        nn.Linear(self.scene_feat_size, self.input_encoding_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.att_feat_size),) )))
            self.v_embed = nn.Sequential(
                                    nn.Embedding(self.vocab_size + 1, self.scene_feat_size)
                                    )
            self.fc_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.fc_feat_size),))+
                                    (
                                        nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.fc_feat_size),))))
            self.att_embed = nn.Sequential(*(
                                        ((nn.LayerNorm(self.att_feat_size),))+
                                        (
                                            nn.Linear(self.att_feat_size, self.rnn_size),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))+
                                        ((nn.LayerNorm(self.rnn_size),) )))
        if not self.use_ln and self.use_bn:
            print('Using BatchNorm')
            self.u_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),))+
                                    (
                                        nn.Linear(self.scene_feat_size, self.input_encoding_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.att_feat_size),) )))
            self.v_embed = nn.Sequential(
                                    nn.Embedding(self.vocab_size + 1, self.scene_feat_size)
                                    )
            self.fc_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.fc_feat_size),))+
                                    (
                                        nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.fc_feat_size),))))
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) )+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) )))
        else:
            self.att_embed = nn.Sequential(*(
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))))
            self.u_embed = nn.Sequential(
                                    nn.Linear(self.scene_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
            self.v_embed = nn.Sequential(
                                    nn.Embedding(self.vocab_size + 1, self.scene_feat_size)
                                    )
            self.fc_embed = nn.Sequential(
                                    nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(
                        it, 
                        tmp_fc_feats, 
                        tmp_att_feats, tmp_p_att_feats, tmp_att_masks, 
                        state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

class SceneAttModel(CaptionModel):
    def __init__(self, opt):
        super(SceneAttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.scene_feat_size = opt.scene_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_ln = getattr(opt, 'use_ln', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        # self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
        #                         nn.ReLU(),
        #                         nn.Dropout(self.drop_prob_lm))

        if self.use_ln:
            print('Using LayerNorm')
            self.u_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.scene_feat_size),))+
                                    (
                                        nn.Linear(self.scene_feat_size, self.input_encoding_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.scene_feat_size),) )))

            self.v_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.vocab_size),))+
                                    (
                                        nn.Linear(self.vocab_size,
                                            self.rnn_size))
                                    ((nn.LayerNorm(self.vocab_size),) )))

            self.fc_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.fc_feat_size),))+
                                    (
                                        nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.fc_feat_size),))))

            self.att_embed = nn.Sequential(*(
                                        ((nn.LayerNorm(self.att_feat_size),))+
                                        (
                                            nn.Linear(self.att_feat_size, self.rnn_size),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))+
                                        ((nn.LayerNorm(self.rnn_size),) )))
        if not self.use_ln and self.use_bn:
            print('Using BatchNorm')
            self.u_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),))+
                                    (
                                        nn.Linear(self.scene_feat_size, self.input_encoding_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.att_feat_size),) )))
            self.fc_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.fc_feat_size),))+
                                    (
                                        nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.fc_feat_size),))))
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) )+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) )))
        else:
            self.att_embed = nn.Sequential(*(
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))))
            self.u_embed = nn.Sequential(
                                    nn.Linear(self.scene_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
            self.v_embed = nn.Sequential(
                                    nn.Embedding(self.vocab_size+1, self.scene_feat_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(
                                    nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, scene_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # embed scene feats into diag
        scene_feats = torch.diag_embed(torch.gt(scene_feats, 0)).float()

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks, scene_feats

    def _forward(self, fc_feats, att_feats, scene_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(
                    it, 
                    p_fc_feats, 
                    p_att_feats, pp_att_feats, p_att_masks,
                    scene_feats, 
                    state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(
        self, 
        it, 
        fc_feats, 
        att_feats, p_att_feats, att_masks, 
        scene_feats, 
        state):

        # 'it' contains a word index
        '''
        print('get logProbs state:')
        print(scene_feats.shape)        # (500, 18, 18) torch.float64
        print(it.shape)                 # (500,)
        print(torch.unsqueeze(self.v_embed(it), -1).shape)   # (500, 18, 1) torch.float32
        print(torch.bmm(
                scene_feats, 
                torch.unsqueeze(self.v_embed(it), -1)).shape) # (500, 18, 1)
        print(self.u_embed(
                torch.bmm(
                    scene_feats, 
                    torch.unsqueeze(self.v_embed(it), -1)).squeeze()).shape) # (500, 1000)
        xt = self.embed(it)
        '''
        xt = self.u_embed(
                torch.bmm(
                    scene_feats,
                    torch.unsqueeze(self.v_embed(it), -1)).squeeze())
        output, state = self.core(
                xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, \
                'lets assume this for now, \
                otherwise this corner case causes a few headaches down the road. \
                can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(
                    beam_size, p_fc_feats.size(1))
            # print(p_fc_feats[k:k+1].shape) # (1, 1000)
            # print(tmp_fc_feats.shape) # (2, 1000)
            tmp_att_feats = p_att_feats[k:k+1].expand(
                    *((beam_size,)+p_att_feats.size()[1:])).contiguous()
            # print(p_att_feats[k:k+1].shape) # (1, 196, 1000)
            # print(tmp_att_feats.shape) # (2, 196, 1000) 
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(
                    *((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(
                    *((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            # print(p_scene_feats[k:k+1].shape) # (1, 18, 18)
            # print(p_scene_feats.size())
            # print(p_scene_feats.size(1)) # 18
            tmp_scene_feats = p_scene_feats[k:k+1].expand(
                    *((beam_size,)+p_scene_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(
                        it, 
                        tmp_fc_feats, 
                        tmp_att_feats, tmp_p_att_feats, tmp_att_masks, 
                        tmp_scene_feats,
                        state)

            self.done_beams[k] = self.beam_search(
                    state, 
                    logprobs, 
                    tmp_fc_feats, 
                    tmp_att_feats, tmp_p_att_feats, tmp_att_masks, 
                    tmp_scene_feats,
                    opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, scene_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                    it, 
                    p_fc_feats, 
                    p_att_feats, pp_att_feats, p_att_masks, 
                    p_scene_feats, 
                    state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

# for scene7
class Scene3AttModel(CaptionModel):
    def __init__(self, opt):
        super(Scene3AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.scene_feat_size = opt.scene_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_ln = getattr(opt, 'use_ln', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(
                                nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))

        if self.use_ln:
            print('Using LayerNorm')
            self.fc_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.fc_feat_size),))+
                                    (
                                        # nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.Linear(self.fc_feat_size, self.att_hid_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.att_hid_size),))))

            self.att_embed = nn.Sequential(*(
                                        ((nn.LayerNorm(self.att_feat_size),))+
                                        (
                                            nn.Linear(self.att_feat_size, self.rnn_size),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))+
                                        ((nn.LayerNorm(self.rnn_size),) )))
        if not self.use_ln and self.use_bn:
            print('Using BatchNorm')
            self.fc_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.fc_feat_size),))+
                                    (
                                        # nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.Linear(self.fc_feat_size, self.att_hid_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.att_hid_size),))))
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) )+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) )))
        else:
            self.att_embed = nn.Sequential(*(
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))))
            self.fc_embed = nn.Sequential(
                                    # nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.Linear(self.fc_feat_size, self.att_hid_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, scene_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats.float())
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # embed scene feats into diag
        scene_feats = torch.diag_embed(torch.gt(scene_feats, 0)).float()
        # scene_feats = torch.diag_embed(scene_feats).float()
        # scene_feats = torch.gt(scene_feats, 0).float()

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks, scene_feats

    def _forward(self, fc_feats, att_feats, scene_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs_att = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)
        outputs_lang = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs_lang[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(
                    it, 
                    p_fc_feats, 
                    p_att_feats, pp_att_feats, p_att_masks,
                    scene_feats, 
                    state)

            outputs_att[:, i] = output[0]
            outputs_lang[:, i] = output[1]

        return [outputs_att, outputs_lang]

    def get_logprobs_state(
        self, 
        it, 
        fc_feats, 
        att_feats, p_att_feats, att_masks, 
        scene_feats, 
        state,
        viz=False):

        # 'it' contains a word index
        xt = self.embed(it)
        output, state, weight = self.core(
                xt, 
                fc_feats, att_feats, p_att_feats, scene_feats, 
                state,
                att_masks=att_masks, 
                viz=viz)

        logprobs_att = F.log_softmax(self.logit(output[0]), dim=1)
        logprobs_lang = F.log_softmax(self.logit(output[1]), dim=1)
        # l_att = F.softmax(self.logit(output[0]), dim=1)
        # l_lang = F.softmax(self.logit(output[1]), dim=1)
        # logprobs = l_att * 0.3 + l_lang * 0.7
        # logprobs = torch.stack([F.softmax(self.logit(output[i]), dim=1) 
        #         for i in range(len(output))], 2).mean(2).log()
        # print(logprobs_att.shape, logprobs_lang.shape, logprobs.shape)
        if viz:
            return [logprobs_att, logprobs_lang], state, weight
        else:
            return [logprobs_att, logprobs_lang], state
            # return [logprobs_att, logprobs], state

    def _sample_beam(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, \
                'lets assume this for now, \
                otherwise this corner case causes a few headaches down the road. \
                can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(
                    beam_size, p_fc_feats.size(1))
            # print(p_fc_feats[k:k+1].shape) # (1, 1000)
            # print(tmp_fc_feats.shape) # (2, 1000)
            tmp_att_feats = p_att_feats[k:k+1].expand(
                    *((beam_size,)+p_att_feats.size()[1:])).contiguous()
            # print(p_att_feats[k:k+1].shape) # (1, 196, 1000)
            # print(tmp_att_feats.shape) # (2, 196, 1000) 
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(
                    *((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(
                    *((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            # print(p_scene_feats[k:k+1].shape) # (1, 18, 18)
            # print(p_scene_feats.size())
            # print(p_scene_feats.size(1)) # 18
            tmp_scene_feats = p_scene_feats[k:k+1].expand(
                    *((beam_size,)+p_scene_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs_list, state = self.get_logprobs_state(
                        it, 
                        tmp_fc_feats, 
                        tmp_att_feats, tmp_p_att_feats, tmp_att_masks, 
                        tmp_scene_feats,
                        state)
                logprobs = logprobs_list[-1]

            self.done_beams[k] = self.beam_search(
                    state, 
                    logprobs, 
                    tmp_fc_feats, 
                    tmp_att_feats, tmp_p_att_feats, tmp_att_masks, 
                    tmp_scene_feats,
                    opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, scene_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs_list, state = self.get_logprobs_state(
                    it, 
                    p_fc_feats, 
                    p_att_feats, pp_att_feats, p_att_masks, 
                    p_scene_feats, 
                    state)
            logprobs = logprobs_list[-1]
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _sample_viz(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, scene_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        weight_list = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs_list, state, weight = self.get_logprobs_state(
                    it, 
                    p_fc_feats, 
                    p_att_feats, pp_att_feats, p_att_masks, 
                    p_scene_feats, 
                    state,
                    viz=True)
            logprobs = logprobs_list[-1]
            weight_list.append(weight)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs, weight_list

# for log_vc_0068_scene7_leg3bu1ln_adam_rl
# for scene7, scene8 model
class Scene3TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene3TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.scene_feat_size = opt.scene_feat_size
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.attention = Attention(opt)
        self.vcAttention = VCAttention(opt)

    def forward(self, 
            xt, fc_feats, att_feats, p_att_feats, scene_feats, 
            state, att_masks=None, viz=False):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        if viz:
            att, conv_weight = self.attention(   
                            h_att, att_feats, p_att_feats,
                            att_masks=att_masks,
                            viz=viz)
            vc_att, vc_weight  = self.vcAttention( 
                            h_att, fc_feats, scene_feats, 
                            att_masks=att_masks,
                            viz=viz)
        else:
            att = self.attention(   
                            h_att, att_feats, p_att_feats, att_masks)
            vc_att = self.vcAttention( 
                            h_att, fc_feats, scene_feats)

        # lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        lang_lstm_input = torch.cat([h_att, att, vc_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output_att = F.dropout(h_att, self.drop_prob_lm, self.training)
        output_lang = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        if viz:
            return [output_att, output_lang], state, [conv_weight, vc_weight]
        else:
            return [output_att, output_lang], state, None

'''
# for scene 10
class Scene3TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene3TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.scene_feat_size = opt.scene_feat_size
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.attention = Attention(opt)
        self.vcAttention = VCAttention(opt)

    def forward(self, 
            xt, fc_feats, att_feats, p_att_feats, scene_feats, 
            state, att_masks=None, viz=False):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        if viz:
            vc_att, vc_weight  = self.vcAttention( 
                            prev_h, fc_feats, scene_feats, 
                            att_masks=att_masks,
                            viz=viz)
        else:
            vc_att = self.vcAttention( 
                            prev_h, fc_feats, scene_feats)

        att_lstm_input = torch.cat([prev_h, xt, vc_att], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        if viz:
            att, conv_weight = self.attention(   
                            h_att, att_feats, p_att_feats,
                            att_masks=att_masks,
                            viz=viz)
        else:
            att = self.attention(   
                            h_att, att_feats, p_att_feats, att_masks)

        # lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        lang_lstm_input = torch.cat([h_att, att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output_att = F.dropout(h_att, self.drop_prob_lm, self.training)
        output_lang = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        if viz:
            return [output_att, output_lang], state, [conv_weight, vc_weight]
        else:
            return [output_att, output_lang], state, None
'''

'''
# input: Image Region Features
class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None, viz=False):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        # mapping h_size to attention_size
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size

        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size

        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1

        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        if viz:
            return att_res, weight
        return att_res

# input: Visual Concepts
class VCAttention(nn.Module):
    def __init__(self, opt):
        super(VCAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.scene_feat_size = opt.scene_feat_size
        # ablation exp
        # self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.h2att_u = nn.Linear(self.scene_feat_size, self.att_hid_size)
        self.h2att_v = nn.Linear(self.rnn_size, self.scene_feat_size)
        self.vc2att = nn.Linear(self.att_hid_size, self.rnn_size)
        self.alpha_net = nn.Linear(self.att_hid_size, self.rnn_size)

    # fc_feats here actually is visual concepts
    def forward(self, h, fc_feats, scene_feats, att_masks=None, viz=False):
        # The p_att_feats here is already projected
        # mapping h_size to attention_size
        h_att = self.h2att_u(
                torch.bmm(
                    scene_feats,
                    torch.unsqueeze(self.h2att_v(h), -1)).squeeze())
        # h_att = self.h2att(h)
        dot = h_att + fc_feats                                # batch * rnn_size
        dot = torch.tanh(dot)                               # batch * rnn_size
        dot = self.alpha_net(dot)                         # (batch * fc_size) * 1

        weight = F.softmax(dot, dim=1)                      # batch * rnn_size


        # old vc attention for scene6,7,8,9
        vc_att = self.vc2att(fc_feats)
        fc_res = vc_att.mul(weight)
        # new vc attention for scene10
        if viz:
            return fc_res, weight
        return fc_res
'''

# for both and share scene attention
class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.scene_feat_size = opt.scene_feat_size

        # self.h2att_u = nn.Linear(self.scene_feat_size, self.att_hid_size)
        # self.h2att_v = nn.Linear(self.rnn_size, self.scene_feat_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, att_h, att_feats, p_att_feats, scene_feats,
            att_masks=None, viz=False):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        # mapping h_size to attention_size
        # att_h = self.h2att_u(
        #         torch.bmm(
        #             scene_feats,
        #             torch.unsqueeze(self.h2att_v(h), -1)).squeeze())
        att_h = self.h2att(att_h)
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size

        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size

        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1

        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        
        if viz:
            return att_res, weight
        return att_res

# input: Visual Concepts
class VCAttention(nn.Module):
    def __init__(self, opt):
        super(VCAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.scene_feat_size = opt.scene_feat_size
        # ablation exp
        # self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.h2att_u = nn.Linear(self.scene_feat_size, self.att_hid_size)
        self.h2att_v = nn.Linear(self.rnn_size, self.scene_feat_size)
        self.vc2att = nn.Linear(self.att_hid_size, self.rnn_size)
        self.alpha_net = nn.Linear(self.att_hid_size, self.rnn_size)

    # fc_feats here actually is visual concepts
    def forward(self, h_att, fc_feats, scene_feats, att_masks=None, viz=False):
        # The p_att_feats here is already projected
        # mapping h_size to attention_size
        # h_att = self.h2att_u(
        #         torch.bmm(
        #             scene_feats,
        #             torch.unsqueeze(self.h2att_v(h), -1)).squeeze())
        h_att = self.h2att_u(self.h2att_v(h_att))
        dot = h_att + fc_feats                                # batch * rnn_size
        dot = torch.tanh(dot)                               # batch * rnn_size
        dot = self.alpha_net(dot)                         # (batch * fc_size) * 1

        weight = F.softmax(dot, dim=1)                      # batch * rnn_size


        # old vc attention for scene6,7,8,9
        vc_att = self.vc2att(fc_feats)
        fc_res = vc_att.mul(weight)
        # new vc attention for scene10
        if viz:
            return fc_res, weight
        return fc_res

'''
# for scene7, scene8 model
class Scene3TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene3TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size +\
                opt.scene_feat_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.scene_feat_size = opt.scene_feat_size
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.attention = Attention(opt)
        self.vcAttention = VCAttention(opt)
        # for vc_0068_scene7_leg3ln3_share_adam and fp
        # self.h2att_u = nn.Sequential(*(
        #     ((nn.LayerNorm(self.scene_feat_size),) )+
        #     (
        #         nn.Linear(self.scene_feat_size , self.att_hid_size ),)+
        #     ((nn.LayerNorm(self.att_hid_size),)) ))
        # for vc_0068_scene7_leg3ln2_share2_adam    
        # self.scene_factor = self.att_hid_size
        # self.h2att_u = nn.Linear(self.scene_factor, self.att_hid_size)
        # self.h2att_v = nn.Linear(self.rnn_size, self.scene_factor)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        # self.s2att = nn.Linear(self.scene_feat_size, self.scene_factor)

    def forward(self, 
            xt, fc_feats, att_feats, p_att_feats, scene_feats, 
            state, att_masks=None, viz=False):
        prev_h = state[0][-1]
        
        att_lstm_input = torch.cat([prev_h, xt, scene_feats], 1)
        # att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        # vc_h = self.h2att_u(
        #         torch.bmm(
        #             scene_feats,
        #             torch.unsqueeze(self.h2att_v(h_att), -1)).squeeze())
        # no scene
        # att_h = self.h2att_u(self.h2att_v(h_att))
        # conv_h = self.h2att(h_att)
        att_h = self.h2att(h_att)

        if viz:
            att, conv_weight = self.attention(   
                            att_h, att_feats, p_att_feats, scene_feats,
                            att_masks=att_masks,
                            viz=viz)
            # print('core:', conv_weight.shape)
            vc_att, vc_weight  = self.vcAttention( 
                            att_h, fc_feats, scene_feats, att_masks, viz)
        else:
            # for scene 7 both
            # att     = self.attention(   
            #                 h_att, att_feats, p_att_feats, scene_feats, att_masks)
            att = self.attention(   
                            att_h, att_feats, p_att_feats, att_masks)
            vc_att = self.vcAttention( 
                            att_h, fc_feats, scene_feats)

        # lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        lang_lstm_input = torch.cat([h_att, att, vc_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output_att = F.dropout(h_att, self.drop_prob_lm, self.training)
        output_lang = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        if viz:
            return [output_att, output_lang], state, [conv_weight, vc_weight]
        else:
            return [output_att, output_lang], state, None
'''

# for scene8
class Scene4AttModel(CaptionModel):
    def __init__(self, opt):
        super(Scene4AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.scene_feat_size = opt.scene_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_ln = getattr(opt, 'use_ln', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(
                                nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))

        if self.use_ln:
            print('Using LayerNorm')
            self.fc_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.fc_feat_size),))+
                                    (
                                        # nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.Linear(self.fc_feat_size, self.att_hid_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.fc_feat_size),))))

            self.att_embed = nn.Sequential(*(
                                        ((nn.LayerNorm(self.att_feat_size),))+
                                        (
                                            nn.Linear(self.att_feat_size, self.rnn_size),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))+
                                        ((nn.LayerNorm(self.rnn_size),) )))
            self.img_embed = nn.Sequential(*(
                                    ((nn.LayerNorm(self.fc_feat_size),))+
                                    (
                                        nn.Linear(self.fc_feat_size, self.input_encoding_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.LayerNorm(self.fc_feat_size),))))

        if not self.use_ln and self.use_bn:
            print('Using BatchNorm')
            self.fc_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.fc_feat_size),))+
                                    (
                                        nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.fc_feat_size),))))
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) )+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) )))
        else:
            self.att_embed = nn.Sequential(*(
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))))
            self.fc_embed = nn.Sequential(
                                    # nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.Linear(self.fc_feat_size, self.att_hid_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
            self.img_embed = nn.Sequential(
                                    nn.Linear(self.fc_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, scene_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # embed scene feats into diag
        scene_feats = torch.diag_embed(torch.gt(scene_feats, 0)).float()
        # scene_feats = torch.diag_embed(scene_feats).float()

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks, scene_feats

    def _forward(self, fc_feats, att_feats, scene_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs_att = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)
        outputs_lang = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
                # print('in:', i, seq.size(1))
                # print(type(xt), type(p_fc_feats), type(p_att_feats),
                #         type(pp_att_feats), type(state), type(p_att_masks))
                # print(xt.dtype, p_fc_feats.dtype, p_att_feats.dtype,
                #         pp_att_feats.dtype, p_att_masks.dtype)
                output, state = self.core(
                        xt, 
                        p_fc_feats, 
                        p_att_feats, pp_att_feats, 
                        p_scene_feats, 
                        state, 
                        p_att_masks)
                logprobs_att = F.log_softmax(self.logit(output[0]), dim=1)
                logprobs_lang = F.log_softmax(self.logit(output[1]), dim=1)
                output = [logprobs_att, logprobs_lang]
            else:
                # print('out:', i, seq.size(1))
                if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        prob_prev = torch.exp(outputs_lang[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i-1].clone()          
                # break if all the sequences end
                if i >= 1 and seq[:, i].sum() == 0:
                    break

                output, state = self.get_logprobs_state(
                        it, 
                        p_fc_feats, 
                        p_att_feats, pp_att_feats, p_att_masks,
                        p_scene_feats, 
                        state)

            outputs_att[:, i] = output[0]
            outputs_lang[:, i] = output[1]

        return [outputs_att, outputs_lang]

    def get_logprobs_state(
        self, 
        it, 
        fc_feats, 
        att_feats, p_att_feats, att_masks, 
        scene_feats, 
        state):

        # 'it' contains a word index
        xt = self.embed(it)
        # print(type(xt), type(fc_feats), type(att_feats),
        #         type(p_att_feats), type(state), type(att_masks))
        # print(xt.dtype, fc_feats.dtype, att_feats.dtype,
        #         p_att_feats.dtype, att_masks.dtype)
        output, state = self.core(
                xt, fc_feats, att_feats, p_att_feats, scene_feats, state, att_masks)
        logprobs_att = F.log_softmax(self.logit(output[0]), dim=1)
        logprobs_lang = F.log_softmax(self.logit(output[1]), dim=1)

        return [logprobs_att, logprobs_lang], state

    def _sample_beam(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, \
                'lets assume this for now, \
                otherwise this corner case causes a few headaches down the road. \
                can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(
                    beam_size, p_fc_feats.size(1))
            # print(p_fc_feats[k:k+1].shape) # (1, 1000)
            # print(tmp_fc_feats.shape) # (2, 1000)
            tmp_att_feats = p_att_feats[k:k+1].expand(
                    *((beam_size,)+p_att_feats.size()[1:])).contiguous()
            # print(p_att_feats[k:k+1].shape) # (1, 196, 1000)
            # print(tmp_att_feats.shape) # (2, 196, 1000) 
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(
                    *((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(
                    *((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            # print(p_scene_feats[k:k+1].shape) # (1, 18, 18)
            # print(p_scene_feats.size())
            # print(p_scene_feats.size(1)) # 18
            tmp_scene_feats = p_scene_feats[k:k+1].expand(
                    *((beam_size,)+p_scene_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0: # input <bos>
                    # it = fc_feats.new_zeros([beam_size], dtype=torch.long)
                    # print(fc_feats[k:k+1].shape)
                    t_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats[k:k+1].size(1))
                    xt = self.img_embed(t_fc_feats)
                    # xt = xt.expand(beam_size, xt.size(1))
                    # print(xt.shape)

                output, state = self.core(
                        xt, 
                        tmp_fc_feats, 
                        tmp_att_feats, tmp_p_att_feats, 
                        tmp_scene_feats, 
                        state, 
                        tmp_att_masks)
                logprobs = F.log_softmax(self.logit(output[1]), dim=1)

            self.done_beams[k] = self.beam_search(
                    state, 
                    logprobs, 
                    tmp_fc_feats, 
                    tmp_att_feats, tmp_p_att_feats, tmp_att_masks, 
                    tmp_scene_feats,
                    opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, scene_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, scene_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_scene_feats = \
                self._prepare_feature(fc_feats, att_feats, scene_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: 
                # # input <bos>
                # it = fc_feats.new_zeros(batch_size, dtype=torch.long)
                xt = self.img_embed(fc_feats)
                output, state = self.core(
                        xt, 
                        p_fc_feats, 
                        p_att_feats, pp_att_feats, 
                        p_scene_feats, 
                        state, 
                        p_att_masks)
                logprobs = F.log_softmax(self.logit(output[1]), dim=1)
            else:
                # print('out:', t, self.seq_length)
                logprobs_list, state = self.get_logprobs_state(
                        it, 
                        p_fc_feats, 
                        p_att_feats, pp_att_feats, p_att_masks, 
                        p_scene_feats, 
                        state)
                logprobs = logprobs_list[-1]
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

class myAttModel(CaptionModel):
    def __init__(self, opt):
        super(myAttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_ln = getattr(opt, 'use_ln', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        if self.use_ln:
            print('Using LayerNorm')

            self.att_embed = nn.Sequential(*(
                                        ((nn.LayerNorm(self.att_feat_size),) if self.use_ln else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.LayerNorm(self.rnn_size),) if self.use_ln==2 else ())))
        if not self.use_ln and self.use_bn:
            print('Using BatchNorm')
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        else:
            self.att_embed = nn.Sequential(*(
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        # self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.att_embed(att_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.ctx2att(att_feats)

        # return fc_feats, att_feats, p_att_feats, att_masks
        return fc_feats, att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        p_fc_feats, p_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        p_fc_feats, p_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, \
                'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            # tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                # logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)
                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_att_masks, state)

            # self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            # logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, p_att_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = torch.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = torch.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = torch.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)

        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:,:1], att_masks], 1) # assume one one at the first time step.
            PI = PI / PI.sum(1, keepdim=True)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = torch.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class SceneTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(SceneTopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

# for scene9
class Scene4TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene4TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.vcAttention = VCAttention(opt)
        self.fuse = nn.Sequential(
                    nn.Linear(opt.rnn_size*2, opt.rnn_size),
                    nn.ReLU(),
                    nn.Dropout(self.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, scene_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att     = self.attention(   
                        h_att, att_feats, p_att_feats, att_masks)
        vc_att  = self.vcAttention( 
                        h_att, fc_feats, scene_feats)

        # lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        lang_lstm_input = torch.cat([h_att, att, vc_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output_att = F.dropout(h_att, self.drop_prob_lm, self.training)
        output_lang = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        # plus
        # lambd = 0.7
        # output = lambd * output_lang + (1-lambd) * output_att
        # mul 
        # output = output_lang.mul(output_att)
        # fc
        output = self.fuse(torch.cat((output_att, output_lang), dim=1))

        return output, state

# for scene6 model
class Scene2TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene2TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.vcAttention = VCAttention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, scene_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att     = self.attention(   
                        h_att, att_feats, p_att_feats, att_masks)
        vc_att  = self.vcAttention( 
                        h_att, fc_feats, scene_feats)

        # lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        lang_lstm_input = torch.cat([h_att, att, vc_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state
'''

# for scene5 model
class Scene2TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene2TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.u_embed = nn.Sequential(
                                nn.Linear(
                                    opt.scene_feat_size,
                                    opt.input_encoding_size*2),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.v_embed = nn.Sequential(
                                nn.Linear(opt.rnn_size*2, opt.scene_feat_size))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, scene_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        lstm_input = torch.cat([h_att, fc_feats], 1)
        lstm_input = self.u_embed(
                    torch.bmm(
                        scene_feats,
                        torch.unsqueeze(self.v_embed(lstm_input), -1)).squeeze())

        # lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        lang_lstm_input = torch.cat([att, lstm_input], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state
'''

'''
# for scene4 model
class Scene2TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Scene2TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.u_embed = nn.Sequential(
                                nn.Linear(
                                    opt.scene_feat_size,
                                    opt.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.v_embed = nn.Sequential(
                                nn.Linear(opt.rnn_size, opt.scene_feat_size))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, scene_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        h_att = self.u_embed(
                    torch.bmm(
                        scene_feats,
                        torch.unsqueeze(self.v_embed(h_att), -1)).squeeze())

        lang_lstm_input = torch.cat([att, h_att, fc_feats], 1)
        # lang_lstm_input = torch.cat([att, lstm_input], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state
'''

class myTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(myTopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        # self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, state, att_masks=None):
        prev_h = state[0][-1]

        # print('fc_feats:', fc_feats.shape) (500, 1000)
        # att_lstm_input = self.layernorm(torch.cat([prev_h, fc_feats, xt], 1))
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # print('p_att_feats:', p_att_feats.shape) # (500, 1, 512)
        # att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        # print('att_feats:', att_feats.squeeze().shape) 
        # (500, 1, 1000) -> (500, 1000)
        # print('h_att:', h_att.shape) (500, 1000)
        # p_att_feats.detach_()
        att = att_feats.squeeze()
        # print('att:', att.shape) # (500, 1000)
        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

from .FCModel import LSTMCore
class StackAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1,att_res_2],1), [state[0][2:3], state[1][2:3]])

        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class DenseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size*2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size*3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)),att_res_2],1), [state[0][2:3], state[1][2:3]])

        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class Att2inCore(Att2in2Core):
    def __init__(self, opt):
        super(Att2inCore, self).__init__(opt)
        del self.a2c
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

"""
Note this is my attempt to replicate att2all model in self-critical paper.
However, this is not a correct replication actually. Will fix it.
"""
class Att2all2Core(nn.Module):
    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class AdaAttModel(AttModel):
    def __init__(self, opt):
        super(AdaAttModel, self).__init__(opt)
        self.core = AdaAttCore(opt)

# AdaAtt with maxout lstm
class AdaAttMOModel(AttModel):
    def __init__(self, opt):
        super(AdaAttMOModel, self).__init__(opt)
        self.core = AdaAttCore(opt, True)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class Att2all2Model(AttModel):
    def __init__(self, opt):
        super(Att2all2Model, self).__init__(opt)
        self.core = Att2all2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)

'''
# scene6 scene9
class SceneTopDownModel(Scene2AttModel):
    def __init__(self, opt):
        super(SceneTopDownModel, self).__init__(opt)
        self.num_layers = 2
        # scene6
        # self.core = Scene2TopDownCore(opt)
        # scene9
        self.core = Scene4TopDownCore(opt)

'''
# scene7 scene10
class SceneTopDownModel(Scene3AttModel):
    def __init__(self, opt):
        super(SceneTopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = Scene3TopDownCore(opt)

'''
# scene 8
class SceneTopDownModel(Scene4AttModel):
    def __init__(self, opt):
        super(SceneTopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = Scene3TopDownCore(opt)
'''

class myTopDownModel(myAttModel):
    def __init__(self, opt):
        super(myTopDownModel, self).__init__(opt)
        self.num_layers = 2
        # self.core = TopDownCore(opt)
        self.core = myTopDownCore(opt)

class StackAttModel(AttModel):
    def __init__(self, opt):
        super(StackAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = StackAttCore(opt)

class DenseAttModel(AttModel):
    def __init__(self, opt):
        super(DenseAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = DenseAttCore(opt)

class Att2inModel(AttModel):
    def __init__(self, opt):
        super(Att2inModel, self).__init__(opt)
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
