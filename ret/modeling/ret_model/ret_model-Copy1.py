import torch
from torch import nn
from torch.functional import F
import math
from ret.modeling.ret_model.featpool import build_featpool  # downsample 1d temporal features to desired length
from ret.modeling.ret_model.feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from ret.modeling.ret_model.loss import build_contrastive_loss
from ret.modeling.ret_model.loss import build_bce_loss
from ret.modeling.ret_model.text_encoder import build_text_encoder
from ret.modeling.ret_model.proposal_conv import build_proposal_conv

from ret.config import cfg as config
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from ret.utils.comm import move_to_cuda
from ret.modeling.ret_model.text_out import build_text_out
from ret.modeling.ret_model.position_encoding import build_position_encoding
from ret.modeling.ret_model.xpos_relative_position import XPOS

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

    
class RetLayer(nn.Module):
    def __init__(self,
                 cfg,
                 gamma=0.9,
                 d_model=256,
                ):
        super().__init__()
        
        self.d_model = d_model
        self.gamma = gamma
        
        self.W_vid_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_vid_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_vid_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        
        self.W_txt_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_txt_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_txt_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)

        self.xpos = XPOS(d_model)  # 1, d, d
        _, self.pos_embed = build_position_encoding('sine', 'sine', d_model)
        
        self.ret_D = None
        
        self.ret_mask = None
        self.vid_length = cfg.MODEL.RET.NUM_CLIPS
        self.learn_mask_enabled = True
        self.sparse_mask_soft2hard = False
        

        if self.learn_mask_enabled == True:
            self.learn_vid_ret = torch.nn.Embedding(self.vid_length*self.vid_length, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
    def get_loss_sparsity(self, vid_ret):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(vid_ret)))
        return sparsity_loss
    
    def forward(self, src, txt_length):

        seq_length = src.shape[1]  # m+64
        vid_length = seq_length - txt_length # 64
        mask_txt = torch.ones(1, txt_length).cuda()
        
        # ret_D
        D = self._get_D(vid_length).to(self.W_vid_Q.device) # 64, 64
        self.ret_D = torch.ones(txt_length+vid_length, txt_length+vid_length).cuda()   
        self.ret_D[txt_length:,txt_length:] = D
        
        # mask_ret
        self.ret_mask = torch.ones(txt_length+vid_length, txt_length+vid_length).cuda()  
        loss_sparsity = 0.0
        learn_att = None
        if self.learn_mask_enabled:
            learn_ret = self.learn_vid_ret.weight.reshape(vid_length, vid_length)
            learn_ret = self.sigmoid(learn_ret)
            diag_mask = torch.diag(torch.ones(vid_length)).cuda()
            video_ret = (1. - diag_mask) * learn_ret
            learn_att = diag_mask + video_ret
            if self.sparse_mask_soft2hard:
                learn_ret = (learn_ret >= 0.5) * 1.0
                learn_ret = learn_ret.cuda()
                learn_ret.requires_grad = False

            loss_sparsity = self.get_loss_sparsity(video_ret) 

        elif self.learn_mask_enabled == False:
            learn_att = torch.ones(vid_length, vid_length).cuda()
            learn_att.requires_grad = False
            loss_sparsity = torch.tensor(0.0).cuda()
        
        self.ret_mask[txt_length:,txt_length:] = learn_ret 
            
        
        txt_mem = src[:, :txt_length]
        vid_mem = src[:, txt_length:]
        
        # Q,K
        txt_Q = (self.pos_embed(txt_mem, mask_txt) + txt_mem) @ self.W_txt_Q
        txt_K = (self.pos_embed(txt_mem, mask_txt) + txt_mem) @ self.W_txt_K
        
        vid_Q = self.xpos(vid_mem @ self.W_vid_Q)   
        vid_K = self.xpos((vid_mem @ self.W_vid_K), downscale=True)
        
        Q = torch.cat([txt_Q, vid_Q], dim=1)  # (batch_size, L_txt+L_vid, d)
        K = torch.cat([txt_K, vid_K], dim=1)  # (batch_size, L_txt+L_vid, d)

        # V
        vid_V = vid_mem @ self.W_vid_V    # V: b,64,64
        txt_V = txt_mem @ self.W_txt_V
        V = torch.cat([txt_V, vid_V], dim=1)  # (batch_size, L_txt+L_vid, d)
        ret = (Q @ K.permute(0, 2, 1)) * self.ret_mask.unsqueeze(0) # * self.ret_D.unsqueeze(0)  * self.ret_mask.unsqueeze(0)

        return ret @ V, loss_sparsity
    
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)  # l, 1  [[0],[1],[2],,,,[l-1]]
        m = torch.arange(sequence_length).unsqueeze(0)  # 1, l  [[0,1,2,,,,l-1]]

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        # 下三角矩阵，斜对角为 gamma^0, 次对角 gamma^1,,,,,左下角  gamma^(L-1)
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    
# class RetLayer(nn.Module):
#     def __init__(self,
#                  cfg,
#                  gamma=0.9,
#                  d_model=256,
#                 ):
#         super().__init__()
        
#         self.d_model = d_model
#         self.gamma = gamma
        

#         self.W_vid_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
#         self.W_vid_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
#         self.W_vid_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        
#         self.W_txt_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
#         self.W_txt_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
#         self.W_txt_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)

#         self.xpos = XPOS(d_model)  # 1, d, d
        
#         self.ret_mask = None

#     def forward(self, src, txt_length):

#         seq_length = src.shape[1]  # m+64
#         vid_length = seq_length - txt_length # 64
        
#         # mask_D
#         D = self._get_D(vid_length).to(self.W_vid_Q.device) # 64, 64
#         self.ret_mask = torch.ones(txt_length+vid_length, txt_length+vid_length).cuda()   
#         self.ret_mask[txt_length:,txt_length:] = D
        
#         txt_mem = src[:, :txt_length]
#         vid_mem = src[:, txt_length:]
        
#         # Q,K
#         txt_Q = self.xpos(txt_mem @ self.W_txt_Q)
#         txt_K = self.xpos(txt_mem @ self.W_txt_K, downscale=True)
        
#         vid_Q = self.xpos(vid_mem @ self.W_vid_Q)   
#         vid_K = self.xpos((vid_mem @ self.W_vid_K), downscale=True)
        
#         Q = torch.cat([txt_Q, vid_Q], dim=1)  # (batch_size, L_txt+L_vid, d)
#         K = torch.cat([txt_K, vid_K], dim=1)  # (batch_size, L_txt+L_vid, d)

#         # V
#         vid_V = vid_mem @ self.W_vid_V    # V: b,64,64
#         txt_V = txt_mem @ self.W_txt_V
#         V = torch.cat([txt_V, vid_V], dim=1)  # (batch_size, L_txt+L_vid, d)
#         ret = Q @ K.permute(0, 2, 1) * self.ret_mask.unsqueeze(0)

#         return ret @ V, torch.tensor(0.0)
    
#     def _get_D(self, sequence_length):
#         n = torch.arange(sequence_length).unsqueeze(1)  # l, 1  [[0],[1],[2],,,,[l-1]]
#         m = torch.arange(sequence_length).unsqueeze(0)  # 1, l  [[0,1,2,,,,l-1]]

#         # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
#         # 下三角矩阵，斜对角为 gamma^0, 次对角 gamma^1,,,,,左下角  gamma^(L-1)
#         D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
#         # fill the NaN with 0
#         D[D != D] = 0

#         return D
    
    
class TransRet(nn.Module):
    def __init__(self, 
                 cfg,
                 layers, 
                 gamma=0.9,
                 d_model=256, 
                 dim_feedforward=1024,
                 dropout=0.1,
                ):
        super().__init__()
        
        self.layers = layers
        self.d_model = d_model

        self.rets = nn.ModuleList([
            RetLayer(cfg, gamma, d_model)
            for _ in range(layers)
        ])
        

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
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
        
        loss_sp = 0.
        for i in range(self.layers):
            temp, loss_sparsity = self.rets[i](self.layer_norms_1[i](src), txt_length)
            loss_sp = loss_sp + loss_sparsity
            src2 = self.dropout_1[i](temp) + src
            src = self.dropout_2[i](self.ffns[i](self.layer_norms_2[i](src2))) + src2

        return self.norm(src), loss_sp
    
    
def build_transret(cfg, x):
    return TransRet(cfg=cfg, layers=6, gamma=0.9, d_model=x, dim_feedforward=1024, dropout=0.1)



def draw_map(map2d, aid_feats, save_dir, batches, filename, cmap='OrRd'):

    for i in range(len(map2d)):
        save_dir1 = os.path.join(save_dir, f'{batches.vid[i]}')
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if aid_feats == None:
            sim_matrices = map2d[i]
        else:
            v_map = map2d[i].view(256, 64*64)  # 256, 64, 64
            audio = aid_feats[i].squeeze(0)
            sim_matrices = torch.matmul(audio, v_map).view(audio.size(0), 64, 64)

        for j in range(len(sim_matrices)):
            # 绘制 heatmap 并保存图像
            fig, ax = plt.subplots(figsize=(17, 16))
            s = sim_matrices[j].squeeze().cpu().detach().numpy()
            max_index = np.unravel_index(np.argmax(s), s.shape)
            im = ax.imshow(s, cmap=cmap, interpolation='nearest', aspect='auto', norm=colors.Normalize(vmin=0, vmax=np.max(s, axis=None)))
            ax.set_yticks(range(64))
            ax.set_xticks(range(64))
            ax.set_ylabel('start index', fontsize=12)
            ax.set_xlabel('end index', fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.scatter(max_index[1], max_index[0], marker='*', color='lightgreen', s=1000)
            x = filename.format(j)
            filepath = os.path.join(save_dir1, x)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.clf()
            

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
        self.input_vid_proj = nn.Sequential(*[LinearLayer(cfg.MODEL.RET.FEATPOOL.HIDDEN_SIZE, self.joint_space, layer_norm=True,
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
            
            memory, loss_sp = self.trans_ret(src_txt, src_vid)  # hs: (#layers, batch_size, #queries, d)
            loss_sparse = loss_sparse + loss_sp
            txt_mem = memory[:, :src_txt.shape[1]]  # (batch_size, L_txt, d)
            vid_mem = memory[:, src_txt.shape[1]:]  # (batch_size, L_vid, d)

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
            sf_iou = sf_iou.reshape(-1,self.joint_space)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)
        
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
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores  # first two maps for visualization

        
        
        
        
class TransRet(nn.Module):
    def __init__(self, 
                 gamma=0.9, 
                 d_model=256, 
                 dim_feedforward=1024, 
                 dropout=0.1
                ):
        super().__init__()
        
        self.d_model = d_model
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) / d_model)

        self.xpos = XPOS(d_model)  # 1, head_size, head_size
        

        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu    # F.relu  F.glu
        self.ret_mask = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, txt_src, vid_src):

        txt_length = txt_src.shape[1]  # m
        vid_length = vid_src.shape[1]  # 64
        
        # mask_D
        D = self._get_D(vid_length).to(self.W_Q.device) # 64, 64
        self.ret_mask = torch.ones(txt_length+vid_length, txt_length+vid_length).cuda()   
        self.ret_mask[txt_length:,txt_length:] = D
        
        # Normlize
        src = torch.cat([txt_src, vid_src], dim=1)
        src2 = self.norm1(src)
        txt_mem = src2[:, :txt_length]
        vid_mem = src2[:, txt_length:]
        
        # Q,K
        txt_Q = self.xpos(txt_mem)
        txt_K = self.xpos(txt_mem, downscale=True)
        
        vid_Q = self.xpos(vid_mem @ self.W_Q)   
        vid_K = self.xpos((vid_mem @ self.W_K), downscale=True)
        
        Q = torch.cat([txt_Q, vid_Q], dim=1)  # (batch_size, L_txt+L_vid, d)
        K = torch.cat([txt_K, vid_K], dim=1)  # (batch_size, L_txt+L_vid, d)

        # V
        vid_V = vid_mem @ self.W_V    # V: b,64,64
        V = torch.cat([txt_mem, vid_V], dim=1)  # (batch_size, L_txt+L_vid, d)
        ret = (Q @ K.permute(0, 2, 1)) * self.ret_mask.unsqueeze(0)
        
        src2 = ret @ V
        
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm3(src)

        return src
    
    
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)  # l, 1  [[0],[1],[2],,,,[l-1]]
        m = torch.arange(sequence_length).unsqueeze(0)  # 1, l  [[0,1,2,,,,l-1]]

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        # 下三角矩阵，斜对角为 gamma^0, 次对角 gamma^1,,,,,左下角  gamma^(L-1)
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    

# def build_transret(x):
#     return TransRet(gamma=0.9, d_model=x, dim_feedforward=1024, dropout=0.1)