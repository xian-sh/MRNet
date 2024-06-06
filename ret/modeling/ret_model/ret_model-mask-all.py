import torch
from torch import nn
from torch.functional import F
import math
from ret.modeling.ret_model.featpool import build_featpool  # downsample 1d temporal features to desired length
from ret.modeling.ret_model.feat2d import \
    build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from ret.modeling.ret_model.loss import build_contrastive_loss
from ret.modeling.ret_model.loss import build_bce_loss
from ret.modeling.ret_model.text_encoder import build_text_encoder
from ret.modeling.ret_model.proposal_conv import build_proposal_conv
import copy
from ret.config import cfg as config
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from ret.utils.comm import move_to_cuda
from ret.modeling.ret_model.text_out import build_text_out
from ret.modeling.ret_model.position_encoding import build_position_encoding
from ret.modeling.ret_model.xpos_relative_position import XPOS
# from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
# from torch import nn
# from thop import profile
# from thop import clever_format
import random

def draw_map(map2d, aid_feats, save_dir, batches, filename, cmap='OrRd'):
    for i in range(len(map2d)):
        save_dir1 = os.path.join(save_dir, f'{batches.vid[i]}')
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if aid_feats == None:
            sim_matrices = map2d[i]
        else:
            v_map = map2d[i].view(256, 64 * 64)  # 256, 64, 64
            audio = aid_feats[i].squeeze(0)
            sim_matrices = torch.matmul(audio, v_map).view(audio.size(0), 64, 64)

        for j in range(len(sim_matrices)):
            # 绘制 heatmap 并保存图像
            fig, ax = plt.subplots(figsize=(17, 16))
            s = sim_matrices[j].squeeze().cpu().detach().numpy()
            max_index = np.unravel_index(np.argmax(s), s.shape)
            im = ax.imshow(s, cmap=cmap, interpolation='nearest', aspect='auto',
                           norm=colors.Normalize(vmin=0, vmax=np.max(s, axis=None)))
            ax.set_yticks(range(64))
            ax.set_xticks(range(64))
            ax.set_ylabel('start index', fontsize=12)
            ax.set_xlabel('end index', fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.scatter(max_index[1], max_index[0], marker='*', color='lightgreen', s=1000)
            x = filename.format(j)
            filepath = os.path.join(save_dir1, x+'.pdf')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=300, format='pdf')
            plt.clf()
            
            
def draw_one_map(map2d, aid_feats, save_dir, filename, cmap='Greens'):
#     print('len att:', len(map2d))
    for i in range(len(map2d)):
        save_dir1 = os.path.join(save_dir, f'{filename}')
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        sim_matrices = map2d[i]

        # 绘制 heatmap 并保存图像
        fig, ax = plt.subplots(figsize=(17, 16))
        s = sim_matrices.squeeze().cpu().detach().numpy()
        np.save(os.path.join(save_dir1, str(i)+'_all.npy'), s)
#         s[range(64), range(64)] = 0.0
        max_index = np.unravel_index(np.argmax(s), s.shape)
        im = ax.imshow(s, cmap=cmap, interpolation='nearest', aspect='auto',
                           norm=colors.Normalize(vmin=0, vmax=np.max(s, axis=None)))
        ax.set_yticks(range(64))
        ax.set_xticks(range(64))
        ax.set_ylabel('start index', fontsize=12)
        ax.set_xlabel('end index', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        x = str(i)
        filepath = os.path.join(save_dir1, x+'_all.pdf')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=300, format='pdf')
        plt.clf()
        
        
# def draw_one_map(map2d, aid_feats, save_dir, filename, cmap='Greens'):
#     print('len att:', len(map2d))
#     for i in range(len(map2d)):
#         save_dir1 = os.path.join(save_dir, f'{filename}')
#         if not os.path.exists(save_dir1):
#             os.makedirs(save_dir1)
#         sim_matrices = map2d[i]

#         # 绘制 heatmap 并保存图像
#         fig, ax = plt.subplots(figsize=(17, 16))
#         s = sim_matrices.squeeze().cpu().detach().numpy()
#         np.save(os.path.join(save_dir1, str(i)+'.npy'), s)
#         s[range(64), range(64)] = 0.0
#         max_index = np.unravel_index(np.argmax(s), s.shape)
#         im = ax.imshow(s, cmap=cmap, interpolation='nearest', aspect='auto',
#                            norm=colors.Normalize(vmin=0, vmax=np.max(s, axis=None)))
#         ax.set_yticks(range(64))
#         ax.set_xticks(range(64))
#         ax.set_ylabel('start index', fontsize=12)
#         ax.set_xlabel('end index', fontsize=12)
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         x = str(i)
#         filepath = os.path.join(save_dir1, x+'.pdf')
#         plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=300, format='pdf')
#         plt.clf()
            
            
            
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


def attention(q, k, v, attn_mask, dropout, att_softmax, t_l):
    B, Nt, E = q.shape
    q = q / math.sqrt(E)

    attn = torch.bmm(q, k.transpose(-2, -1))  # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
#     attn_all = F.softmax(attn, dim=-1)
#     hw = torch.nn.Hardswish()
#     attn_all = hw(attn)
    if attn_mask is not None:
        attn *= attn_mask     #  element-wise multiplication
    attn_new = torch.zeros_like(attn)
    if att_softmax:
#         if t_l is not None:
#             attn_qq = F.softmax(attn[:,:t_l,:t_l], dim=-1)
#             attn_qv = F.softmax(attn[:,:t_l,t_l:], dim=-1)
#             attn_vq = F.softmax(attn[:,t_l:,:t_l], dim=-1)
#             attn_vv = F.softmax(attn[:,t_l:,t_l:], dim=-1)
#             attn_new[:,:t_l,:t_l] = attn_qq
#             attn_new[:,:t_l,t_l:] = attn_qv
#             attn_new[:,t_l:,:t_l] = attn_vq
#             attn_new[:,t_l:,t_l:] = attn_vv
#         else:
        attn_new = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn_new = dropout(attn_new)
    output = torch.bmm(attn_new, v)
    return output, attn_new

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, att_softmax=True):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linears = _get_clones(nn.Linear(d_model, d_model), 1)
        self.attn = True
        self.att_softmax = att_softmax
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None


    def forward(self, q, k, v, mask=None, t_l=None):
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(0)

        tgt_len, bsz, embed_dim = q.shape
        src_len, _, _ = k.shape

        q = q.view(tgt_len, bsz * self.h, self.d_k).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.h, self.d_k).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.h, self.d_k).transpose(0, 1)
        src_len = k.size(1)

        attn_output, attn_output_weights = attention(q, k, v, attn_mask, self.dropout, self.att_softmax, t_l)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.linears[-1](attn_output)
        attn_output_weights = attn_output_weights.view(bsz, self.h, tgt_len, src_len)
        attn_output_weights = attn_output_weights.sum(dim=1) / self.h

        return attn_output, attn_output_weights


def expand_matrix(matrix3x3, n):
    # 获取3x3矩阵的大小
    m, _ = matrix3x3.size()

    # 生成随机的行和列索引
    row_indices = torch.randint(0, m, (n, n))
    col_indices = torch.randint(0, m, (n, n))

    # 使用索引从原始的3x3矩阵中选择元素
    expanded_matrix = matrix3x3[row_indices, col_indices]

    return expanded_matrix


class RetLayer(nn.Module):
    def __init__(self,
                 cfg,
                 gamma=0.9,
                 n_heads=8,
                 d_model=256,
                 ):
        super().__init__()


        self.mask_mode = cfg.SOLVER.MASK_MODE
        self.att_mode = cfg.SOLVER.ATT_MODE
        self.att_softmax = cfg.SOLVER.ATT_SOFTMAX
        self.d_model = d_model
        self.gamma = gamma
        self.d_double = cfg.SOLVER.D_DOUBLE
        self.dropout_a = cfg.SOLVER.ATT_DROPOUT

        self.W_vid_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_vid_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_vid_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)

        self.W_txt_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_txt_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_txt_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)


        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout=self.dropout_a, att_softmax=self.att_softmax)
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads)

        self.xpos = XPOS(d_model)  # 1, d, d
        _, self.pos_embed = build_position_encoding('sine', 'sine', d_model)

        self.vid_length = cfg.MODEL.RET.NUM_CLIPS


        self.learn_vid_ret = torch.nn.Embedding(self.vid_length * self.vid_length, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
#         vid_len = 64
#         seq_len = 67
#         txt_len = 3
#         self.mask = torch.ones(seq_len, seq_len).cuda()
        
#         # Decay matrix
#         self.D = self._get_D(vid_len, self.d_double).cuda()
#         self.ret_mask = torch.ones(seq_len, seq_len).cuda()
#         self.ret_mask[txt_len:, txt_len:] = self.D
#         self.mask *= self.ret_mask
        
#         # sparse matrix
#         self.learn_ret = self.learn_vid_ret.weight.reshape(vid_len, vid_len).cuda()
#         self.learn_ret = self.sigmoid(self.learn_ret).cuda()
#         self.diag_mask = torch.diag(torch.ones(vid_len)).cuda()
#         self.video_ret = (1. - self.diag_mask) * self.learn_ret
#         self.learn_mask = self.diag_mask + self.video_ret
        
#         self.sparse_mask = torch.ones(seq_len, seq_len).cuda()
#         self.sparse_mask[txt_len:, txt_len:] = self.learn_mask

#         self.mask *= self.sparse_mask 
        

    def get_loss_sparsity(self, vid_ret):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(vid_ret)))
        return sparsity_loss

    def forward(self, src, txt_len):

        seq_len = src.shape[1]  # m+64
        vid_len = seq_len - txt_len  # 64
        mask_txt = torch.ones(1, txt_len).cuda()

        # Decay matrix
        D = self._get_D(seq_len, self.d_double).cuda()

        # sparse matrix
        learn_ret = torch.ones(seq_len,seq_len).cuda()
        temp = self.learn_vid_ret.weight.reshape(vid_len, vid_len)
        learn_ret = expand_matrix(temp, seq_len) 
#         for i in range(seq_len):
#             for j in range(seq_len):
#                 if i > txt_len and j > txt_len:
#                     continue
#                 else:
#                     learn_ret[i, j] = learn_ret[random.randint(txt_len, seq_len-1), random.randint(txt_len, seq_len-1)]

        learn_ret = self.sigmoid(learn_ret)
        diag_mask = torch.diag(torch.ones(seq_len)).cuda()
        video_ret = (1. - diag_mask) * learn_ret
        learn_mask = diag_mask + video_ret

        # sparse loss
        loss_sparsity = torch.tensor(0.0).cuda()
        mask = None

        if self.att_mode == 'v':
            ret_mask = D
            sparse_mask = learn_mask

            mask = torch.ones(seq_len, seq_len).cuda()

        elif self.att_mode == 'all' or self.att_mode == 'trans':
#             ret_mask = torch.ones(seq_len, seq_len).cuda()
            ret_mask = D

#             sparse_mask = torch.ones(seq_len, seq_len).cuda()
            sparse_mask = learn_mask

            mask = torch.ones(seq_len, seq_len).cuda()
        
        elif self.att_mode == 'no':
            pass

        else:
            raise


        if self.mask_mode == 'no':
            pass
        elif self.mask_mode == 'ret':
            mask *= ret_mask
        elif self.mask_mode == 'sparse':
            mask *= sparse_mask
            loss_sparsity += self.get_loss_sparsity(video_ret)
        elif self.mask_mode == 'both':
            mask *= ret_mask
            mask *= sparse_mask
            loss_sparsity += self.get_loss_sparsity(video_ret)
        else:
            raise


#         s = mask.squeeze().cpu().detach().numpy()
#         np.save(os.path.join('activity/result_1102_loss_ratio/loss/0.25/visual_audio/v_moGDCWEoaK8', 'd.npy'), s)


        if self.att_mode == 'v':
            txt_mem = src[:, :txt_len]
            vid_mem = src[:, txt_len:]

            # Q,K
            txt_Q = (self.pos_embed(txt_mem, mask_txt) + txt_mem) @ self.W_txt_Q
            txt_K = (self.pos_embed(txt_mem, mask_txt) + txt_mem) @ self.W_txt_K
            txt_V = txt_mem @ self.W_txt_V

            vid_Q = self.xpos(vid_mem) @ self.W_vid_Q
            vid_K = self.xpos(vid_mem, downscale=True) @ self.W_vid_K
            vid_V = vid_mem  @ self.W_vid_V  # V: b,64,64
        
            Q = vid_Q.permute(1, 0, 2)
            K = vid_K.permute(1, 0, 2)
            V = vid_V.permute(1, 0, 2)
            ret_V, att  = self.self_attn(Q, K, V, mask=mask, t_l=None)
            ret_V = ret_V.transpose(0,1)
            out = torch.cat([txt_V, ret_V], dim=1)
            

        elif self.att_mode == 'all':
            txt_mem = src[:, :txt_len]
            vid_mem = src[:, txt_len:]

            # Q,K
            txt_Q = (self.pos_embed(txt_mem, mask_txt) + txt_mem) @ self.W_txt_Q
            txt_K = (self.pos_embed(txt_mem, mask_txt) + txt_mem) @ self.W_txt_K
            txt_V = txt_mem @ self.W_txt_V

            vid_Q = self.xpos(vid_mem) @ self.W_vid_Q
            vid_K = self.xpos(vid_mem, downscale=True) @ self.W_vid_K
            vid_V = vid_mem  @ self.W_vid_V  # V: b,64,64
            
            Q = torch.cat([txt_Q, vid_Q], dim=1).permute(1, 0, 2)
            K = torch.cat([txt_K, vid_K], dim=1).permute(1, 0, 2)
            V = torch.cat([txt_V, vid_V], dim=1).permute(1, 0, 2)
            ret_V, att = self.self_attn(Q, K, V, mask=mask, t_l=None)
            out = ret_V.transpose(0, 1)
        
        elif self.att_mode == 'trans':
            txt_mem = src[:, :txt_len]
            vid_mem = src[:, txt_len:]

            # Q,K
            txt_Q = (self.pos_embed(txt_mem, mask_txt) + txt_mem) 
            txt_K = (self.pos_embed(txt_mem, mask_txt) + txt_mem) 
            txt_V = txt_mem 

            vid_Q = self.xpos(vid_mem) 
            vid_K = self.xpos(vid_mem, downscale=True) 
            vid_V = vid_mem 
            
            Q = torch.cat([txt_Q, vid_Q], dim=1).permute(1, 0, 2)
            K = torch.cat([txt_K, vid_K], dim=1).permute(1, 0, 2)
            V = torch.cat([txt_V, vid_V], dim=1).permute(1, 0, 2)
            ret_V, att = self.self_attn(Q, K, V)
            out = ret_V.transpose(0, 1)
        
            
        elif self.att_mode == 'no':
            out = src
            att = None

        else:
            raise 'error! the attention mode only can be \'v\' or \'all\' '


        return out, att, loss_sparsity

    def _get_D(self, sequence_length, d_double):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0
        out_D = D
        if d_double:
            out_D = D + D.T
        
        return out_D


class TransRet(nn.Module):
    def __init__(self,
                 cfg,
                 layers,
                 gamma=0.9,
                 d_model=256,
                 dim_feedforward=1024,
                 n_heads=8,
                 dropout=0.1,
                 ):
        super().__init__()

        self.layers = layers
        self.d_model = d_model

        self.rets = nn.ModuleList([
            RetLayer(cfg, gamma, n_heads, d_model)
            for _ in range(layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, d_model),
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.dropout_1 = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(layers)
        ])
        self.dropout_2 = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(layers)
        ])

    def forward(self, txt_src, vid_src):
        txt_length = txt_src.shape[1]  # m
        src = torch.cat([txt_src, vid_src], dim=1)
        
        att_all = []
        loss_sp = 0.
        for i in range(self.layers):
            temp, att, loss_sparsity = self.rets[i](self.layer_norms_1[i](src), txt_length)
            att_all.append(att)
            loss_sp = loss_sp + loss_sparsity
            src2 = self.dropout_1[i](temp) + src
            src = self.dropout_2[i](self.ffns[i](self.layer_norms_2[i](src2))) + src2

        return self.norm(src), att_all, loss_sp


def build_transret(cfg, x):
    return TransRet(cfg=cfg,
                    layers=cfg.SOLVER.TRANS_LAYER,
                    gamma=cfg.SOLVER.GAMMA,
                    d_model=x,
                    dim_feedforward=1024,
                    n_heads=cfg.SOLVER.N_HEAD,
                    dropout=0.1)





class RET(nn.Module):
    def __init__(self, cfg):
        super(RET, self).__init__()

        # config
        self.cfg = cfg
        self.joint_space = cfg.MODEL.RET.JOINT_SPACE_SIZE

        # video
        self.featpool = build_featpool(cfg)
        self.feat2d = build_feat2d(cfg)

        # text
        self.encoder_name = cfg.MODEL.RET.TEXT_ENCODER.NAME
        self.text_encoder = build_text_encoder(cfg)
        self.text_out = build_text_out(cfg)

        # trans ret
        self.trans_ret = build_transret(cfg, self.joint_space).cuda()

        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d, self.joint_space)

        # pre project
        n_input_proj = 2
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_vid_proj = nn.Sequential(
            *[LinearLayer(cfg.MODEL.RET.FEATPOOL.HIDDEN_SIZE, self.joint_space, layer_norm=True,
                          dropout=0.5, relu=relu_args[0]),
              LinearLayer(self.joint_space, self.joint_space, layer_norm=True,
                          dropout=0.5, relu=relu_args[1]),
              LinearLayer(self.joint_space, self.joint_space, layer_norm=True,
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
#         save_dir = self.cfg.OUTPUT_DIR + '/visual_audio/'
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]

        feats = self.featpool(batches.feats).cuda()  # B x C x T  4,512,64
        feats = feats.transpose(1, 2)  # B x T x C  48,16,512

        sent_feat, sent_feat_iou = self.text_encoder(batches.queries)
        sent_feat, sent_feat_iou = move_to_cuda(sent_feat), move_to_cuda(sent_feat_iou)

        loss_sparse = 0.
        vid_feats = []
        txt_feats = []
        for i, (vid, txt) in enumerate(zip(feats, sent_feat), start=0):
            src_vid = self.input_vid_proj(vid.cuda()).unsqueeze(0)
            src_txt = txt.unsqueeze(0)

            if self.cfg.SOLVER.USE_LRTNET:
                
#                 print(src_txt.shape, src_vid.shape)
#                 flops = FlopCountAnalysis(self.trans_ret, inputs=(src_txt, src_vid)).total()
#                 print(f"flops: {flops / 10 ** 9:.3f} G")
                
#                 params = parameter_count(self.trans_ret)['']
#                 print(f"params: {params / 10 ** 6:.3f} M")
                
                
                memory, att_all, loss_sp = self.trans_ret(src_txt, src_vid)  # hs: (#layers, batch_size, #queries, d)
                loss_sparse = loss_sparse + loss_sp
                txt_mem = memory[:, :src_txt.shape[1]]  # (batch_size, L_txt, d)
                vid_mem = memory[:, src_txt.shape[1]:]  # (batch_size, L_vid, d)
                
#                 draw_one_map(att_all, None, save_dir, batches.vid[i])
                
            else:
                txt_mem = src_txt
                vid_mem = src_vid
                loss_sparse = torch.tensor(0.0).cuda()

            txt_feats.append(txt_mem.squeeze(0))
            vid_feats.append(vid_mem.squeeze(0))

        vid_feats = torch.stack(vid_feats, dim=0).squeeze(1).transpose(1, 2)  # B  x d x L_vid
        map2d = self.feat2d(vid_feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d, map2d_iou = self.proposal_conv(map2d)

        txt_feat, txt_feat_iou = self.text_out(txt_feats)

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

            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T,
                                                                                                                T)  # num_sent x T x T
            iou_scores.append((iou_score * 10).sigmoid() * self.feat2d.mask2d)

        iou_scores = move_to_cuda(iou_scores)
        ious2d = move_to_cuda(list(ious2d))
        #         draw_map(iou_scores, None, save_dir, batches, filename='iou_scores_{}.png')
        #         draw_map(ious2d, None, save_dir, batches, filename='ious2d_{}.png')

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
