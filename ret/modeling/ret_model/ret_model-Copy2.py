import torch
from torch import nn, Tensor
from torch.functional import F
from ret.modeling.ret_model.featpool import build_featpool  # downsample 1d temporal features to desired length
from ret.modeling.ret_model.feat2d import \
    build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from ret.modeling.ret_model.loss import build_contrastive_loss
from ret.modeling.ret_model.loss import build_bce_loss
from ret.modeling.ret_model.text_encoder import build_text_encoder
from ret.modeling.ret_model.proposal_conv import build_proposal_conv
import math
from ret.config import cfg as config
from ret.modeling.ret_model.position_encoding import build_position_encoding
import os
import matplotlib.pyplot as plt
import numpy as np
from ret.utils.comm import move_to_cuda
from ret.modeling.ret_model.text_out import build_text_out
from ret.modeling.ret_model.xpos_relative_position import XPOS

import copy
from typing import Optional
class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz).cuda()
        layers = [
            nn.Dropout(dropout).cuda(),
            nn.Linear(in_hsz, out_hsz).cuda(),
        ]
        self.net = nn.Sequential(*layers).cuda()

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x.float())
        x = self.net(x.float())
        if self.relu:
            x = F.relu(x.float(), inplace=True)
        return x  # (N, L, D)


def attention(q, k, v, attn_mask, dropout):
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn *= attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.bmm(attn, v)
    return output, attn



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None


    def forward(self, q, k, v, mask=None):
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(0)

        tgt_len, bsz, embed_dim = q.shape
        src_len, _, _ = k.shape

        q = self.linears[0](q).view(tgt_len, bsz * self.h, self.d_k).transpose(0, 1)
        k = self.linears[1](k).view(k.shape[0], bsz * self.h, self.d_k).transpose(0, 1)
        v = self.linears[2](v).view(v.shape[0], bsz * self.h, self.d_k).transpose(0, 1)
        src_len = k.size(1)

        attn_output, attn_output_weights = attention(q, k, v, attn_mask, self.dropout)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.linears[-1](attn_output)
        attn_output_weights = attn_output_weights.view(bsz, self.h, tgt_len, src_len)
        attn_output_weights = attn_output_weights.sum(dim=1) / self.h

        return attn_output, attn_output_weights


class RetLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dim_feedforward=1024, dropout=0.1, dropout_a=None):
        super().__init__()

        self.d_model = d_model

        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout=dropout_a)
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu
        self.xpos = XPOS(d_model)

        _, self.pos_embed = build_position_encoding('sine', 'sine', d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_vid, src_txt, mask):
        pos_vid_q = self.xpos(src_vid)
        pos_vid_k = self.xpos(src_vid, downscale=True)

        mask_txt = torch.ones(1, src_txt.size(1)).cuda()
        pos_txt = self.pos_embed(src_txt, mask_txt)  # 1,  num, 256

        src = torch.cat([src_vid, src_txt], dim=1)  # (batch_size, L_vid+L_txt, d)
        pos = torch.cat([pos_vid_q, pos_txt], dim=1)
        pos_k = torch.cat([pos_vid_k, pos_txt], dim=1)
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos = pos.permute(1, 0, 2)  # (L, batch_size, d)
        pos_k = pos_k.permute(1, 0, 2)  # (L, batch_size, d)

        src2 = self.norm1(src)

        # ---------------------------------------------------------------------------------------
        # 统一
#         temp = src2
#         src2 = torch.fft.fft(torch.fft.fft(src2, dim=-1), dim=-2).real
#         src2 = self.norm1(src2) + temp
        # ---------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------
        # 不统一
        # vid_fft = torch.fft.fft(torch.fft.fft(src2[:src_vid.size(1), :, :], dim=-1), dim=-2).real
        # txt_fft = torch.fft.fft(torch.fft.fft(src2[src_vid.size(1):, :, :], dim=-1), dim=-2).real
        # src2 = torch.cat([vid_fft, txt_fft], dim=0)
        # ---------------------------------------------------------------------------------------


        # ---------------------------------------------------------------------------------------
        # 统一
#         q = k = src2 + pos
#         src2 = self.self_attn(q, k, src2, mask=mask)[0]
#         src2 = src2 + pos
        # ---------------------------------------------------------------------------------------


        # # ------------------------------------------------------------------------------------
        # 不统一
        q = src2[:src_vid.size(1), :, :] + pos[:src_vid.size(1), :, :]
        k = src2[:src_vid.size(1), :, :] + pos_k[:src_vid.size(1), :, :]
        vid_att, _ = self.self_attn(q, k, src2[:src_vid.size(1), :, :])
        src2 = torch.cat([vid_att, src2[src_vid.size(1):, :, :]], dim=0)
        src2 = src2 + pos
        # ---------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------
        # 统一
#         q = (src2 + pos) @ self.W_Q
#         k = (src2 + pos_k) @ self.W_K
#         v = src2 @ self.W_V
#         att = (q @ k.permute(0, 2, 1)) * mask.unsqueeze(0)   # / math.sqrt(self.d_model)
#         att = F.softmax(att, dim = -1)
#         src2 = att @ v
#         src2 = src2 + pos
        # ---------------------------------------------------------------------------------------

        # # ------------------------------------------------------------------------------------
        # 不统一
#         q = (src2[:, :src_vid.size(1), :] + pos[:, :src_vid.size(1), :]) @ self.W_Q
#         k = (src2[:, :src_vid.size(1), :] + pos_k[:, :src_vid.size(1), :]) @ self.W_K
#         v = src2[:, :src_vid.size(1), :] @ self.W_V
#         att = (q @ k.permute(0, 2, 1)) * mask.unsqueeze(0)
#         att = F.softmax(att, dim = -1)
#         vid_att = att @ v
#         src2 = torch.cat([vid_att, src2[:, src_vid.size(1):, :]], dim=1)
#         src2 = src2 + pos
        # ---------------------------------------------------------------------------------------


        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm3(src)

        src = src.transpose(0, 1)  # (batch_size, L, d)
        out_vid, out_txt = src[:, :src_vid.size(1), :], src[:, src_vid.size(1):, :]

        return out_vid, out_txt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RetTrans(nn.Module):

    def __init__(self, cfg, mask_choice=None, num_layers=1, d_model=512, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.cfg = cfg
        self.gamma=0.9
        self.mask_choice = mask_choice
        self.vid_len = cfg.MODEL.RET.NUM_CLIPS
        encoder_layer = RetLayer(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)

        self.layers = _get_clones(encoder_layer, num_layers)

        self.learn_vid_ret = torch.nn.Embedding(self.vid_len * self.vid_len, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def get_loss_sparsity(self, vid_ret):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(vid_ret)))
        return sparsity_loss

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)  # l, 1  [[0],[1],[2],,,,[l-1]]
        m = torch.arange(sequence_length).unsqueeze(0)  # 1, l  [[0,1,2,,,,l-1]]

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        # 下三角矩阵，斜对角为 gamma^0, 次对角 gamma^1,,,,,左下角  gamma^(L-1)
        D = (self.gamma ** (n - m)) * (n >= m).float()  # this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

#         Bi_D = D + D.T
        # print(Bi_D)
        return D

    def forward(self, src_vid, src_txt):

        vid_len = src_vid.size(1)
        txt_len = src_txt.size(1)
        total_len = vid_len + txt_len
        D = self._get_D(vid_len).cuda()  # 64, 64
        ret_mask = torch.ones(total_len, total_len).cuda()
        ret_mask[:vid_len, :vid_len] = D

        loss_sparsity = torch.tensor(0.0).cuda()
        sparse_mask = torch.ones(total_len, total_len).cuda()
        learn_ret = self.learn_vid_ret.weight.reshape(vid_len, vid_len)
        learn_ret = self.sigmoid(learn_ret)
        diag_mask = torch.diag(torch.ones(vid_len)).cuda()
        video_ret = (1. - diag_mask) * learn_ret
        learn_mask = diag_mask + video_ret
        sparse_mask[:vid_len, :vid_len] = learn_mask

        mask = None
        if self.mask_choice is None:
            mask = torch.ones(vid_len, vid_len).cuda()
        elif self.mask_choice == 'both':

            mask = D * learn_mask
            loss_sparsity += self.get_loss_sparsity(video_ret)

        elif self.mask_choice == 'ret':
            mask = D

        elif self.mask_choice == 'sparse':
            mask = learn_mask
            loss_sparsity += self.get_loss_sparsity(video_ret)
            
#         if self.mask_choice is None:
#             mask = torch.ones(total_len, total_len).cuda()
#         elif self.mask_choice == 'both':

#             mask = ret_mask * sparse_mask
#             loss_sparsity += self.get_loss_sparsity(video_ret)

#         elif self.mask_choice == 'ret':
#             mask = ret_mask

#         elif self.mask_choice == 'sparse':
#             mask = sparse_mask
#             loss_sparsity += self.get_loss_sparsity(video_ret)


        out_vid, out_txt = src_vid, src_txt

        for layer in self.layers:
            out_vid, out_txt = layer(out_vid, out_txt, mask=mask)

        return out_vid, out_txt, loss_sparsity

def build_rettrans(cfg, x):
    return RetTrans(cfg=cfg, mask_choice='both', num_layers=1, d_model=x, n_heads=8, dim_feedforward=1024, dropout=0.1)



class RET(nn.Module):
    def __init__(self, cfg):
        super(RET, self).__init__()

        # configATIC
        self.pos_embed = 'sine'  # choices=['trainable', 'sine', 'learned']

        # other
        self.joint_space = cfg.MODEL.RET.JOINT_SPACE_SIZE

        # video
        self.featpool = build_featpool(cfg)
        self.feat2d = build_feat2d(cfg)

        # audio
        self.encoder_name = cfg.MODEL.RET.TEXT_ENCODER.NAME
        self.text_encoder = build_text_encoder(cfg)
        self.text_out = build_text_out(cfg)

        # use static
        self.lrt_net = build_rettrans(cfg, self.joint_space).cuda()
        # self.xpos = XPOS(self.joint_space)
        #
        # # 只要使用其一就需要预处理，映射
        # self.vid_pos_embed, self.audio_pos_embed = build_position_encoding('sine',
        #                                                                        'sine',
        #                                                                        self.joint_space)

        #             self.audio_encoder = build_audio_encoder(self.joint_space, cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d, self.joint_space)
        self.hidden_dim = self.joint_space
        n_input_proj = 2
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False

            #             self.input_audio_proj = nn.Sequential(*[LinearLayer(self.audio_input, self.hidden_dim, layer_norm=True,
            #                                                             dropout=0.5, relu=relu_args[0]),
            #                                               LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
            #                                                           dropout=0.5, relu=relu_args[1]),
            #                                               LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
            #                                                           dropout=0.5, relu=relu_args[2])][:n_input_proj])

        self.input_vid_proj = nn.Sequential(
                *[LinearLayer(cfg.MODEL.RET.FEATPOOL.HIDDEN_SIZE, self.hidden_dim, layer_norm=True,
                              dropout=0.5, relu=relu_args[0]),
                  LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
                              dropout=0.5, relu=relu_args[1]),
                  LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
                              dropout=0.5, relu=relu_args[2])][:n_input_proj])


        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]

        feats = self.featpool(batches.feats).cuda()  # B x C x T  4,512,64
        mask_feats = torch.ones(feats.size(0), 1, feats.size(2)).cuda()

        sent_feat, sent_feat_iou = self.text_encoder(batches.queries)
        mask_txt = [torch.ones(1, len(sent)) for sent in sent_feat]

        feats = feats.transpose(1, 2)  # B x T x C  48,16,512
        #         print(feats.shape)
        sent_feat, sent_feat_iou = move_to_cuda(sent_feat), move_to_cuda(sent_feat_iou)
        mask_txt = move_to_cuda(mask_txt)

        loss_sparse = 0.
        vid_feats = []
        txt_feats = []
        for i, (vid, txt) in enumerate(zip(feats, sent_feat), start=0):
            src_vid = self.input_vid_proj(vid.cuda()).unsqueeze(0)
            src_txt = txt.unsqueeze(0)

            vid_mem, txt_mem, loss_sp = self.lrt_net(src_vid, src_txt)  # hs: (#layers, batch_size, #queries, d)
            # txt_mem = memory[:, src_vid.shape[1]:]  # (batch_size, L_txt, d)
            # vid_mem = memory[:, :src_vid.shape[1]]  # (batch_size, L_vid, d)
            loss_sparse = loss_sparse + loss_sp

            txt_feats.append(txt_mem.squeeze(0))
            vid_feats.append(vid_mem.squeeze(0))

        vid_feats = torch.stack(vid_feats, dim=0).squeeze(1).transpose(1, 2)  # B  x d x L_vid


        #         print(feats.shape)  #48, 16, 512
        map2d = self.feat2d(
            vid_feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d, map2d_iou = self.proposal_conv(map2d)
        #         print(txt_feats[0].shape)

        txt_feat, txt_feat_iou = self.text_out(txt_feats)
        #         print(txt_feat_iou[0].shape)

        # inference
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()
        for i, sf_iou in enumerate(txt_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou = sf_iou.reshape(-1, self.joint_space)
            sf_iou_norm = F.normalize(sf_iou, dim=1)

            #             content_map = vid_feat_iou_norm  # 1,c,t,t
            #             boundary_map = F.normalize(boundary_map2d[i], dim=0)

            #             score = self.fuse(boundary_map, content_map, self.feat2d.mask2d, sf_iou_norm)

            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T,
                                                                                                                T)  # num_sent x T x T
            iou_scores.append((iou_score * 10).sigmoid() * self.feat2d.mask2d)

        iou_scores = move_to_cuda(iou_scores)
        ious2d = move_to_cuda(list(ious2d))

        # loss
        if self.training:
            loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_vid, loss_sent = self.contrastive_loss(map2d, txt_feats, ious2d, batches.moments)
            return loss_vid, loss_sent, loss_iou, loss_sparse
        else:
            for i, sf in enumerate(txt_feats):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T,
                                                                                                           T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores  # first two maps for visualization
