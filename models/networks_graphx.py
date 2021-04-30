# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn

from models.graphx import CNN18Encoder, PointCloudEncoder, PointCloudGraphXDecoder, PointCloudDecoder
from models.projection import Projector
from models.edge_detection import EdgeDetector 

from losses.proj_losses import *
from losses.earth_mover_distance import EMD

import utils.network_utils

class Pixel2Pointcloud_GRAPHX(nn.Module):
    def __init__(self, cfg, in_channels, in_instances, activation=nn.ReLU(), optimizer=None, scheduler=None, use_graphx=True, **kwargs):
        super().__init__()
        self.cfg = cfg
        
        # Graphx
        self.img_enc = CNN18Encoder(in_channels, activation)

        out_features = [block[-2].out_channels for block in self.img_enc.children()]
        self.pc_enc = PointCloudEncoder(3, out_features, cat_pc=True, use_adain=True, use_proj=True, 
                                        activation=activation)
        
        deform_net = PointCloudGraphXDecoder if use_graphx else PointCloudDecoder
        self.pc = deform_net(2 * sum(out_features) + 3, in_instances=in_instances, activation=activation)
        
        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)
        self.kwargs = kwargs

        # 2D supervision part
        self.projector = Projector(self.cfg)

        # proj loss
        self.proj_loss = ProjectLoss(self.cfg)

        # emd loss
        self.emd = EMD()
        
        if torch.cuda.is_available():
            self.img_enc = torch.nn.DataParallel(self.img_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc_enc = torch.nn.DataParallel(self.pc_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc = torch.nn.DataParallel(self.pc, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

    def forward(self, input, init_pc):
        img_feats = self.img_enc(input)
        pc_feats = self.pc_enc(img_feats, init_pc)
        return self.pc(pc_feats)

    def loss(self, input, init_pc, gt_pc, view_az, view_el, proj_gt):
        pred_pc = self(input, init_pc)
        
        if self.cfg.SUPERVISION_2D.USE_AFFINITY:
           grid_dist_np = grid_dist(grid_h=self.cfg.PROJECTION.GRID_H, grid_w=self.cfg.PROJECTION.GRID_W).astype(np.float32)
           grid_dist_tensor = utils.network_utils.var_or_cuda(torch.from_numpy(grid_dist_np))
        else:
           grid_dist_tensor = None

        # Use 2D projection loss to train
        proj_pred = {}
        point_gt = {}
        loss_bce = {}
        fwd = {}
        bwd = {}
        loss_fwd = {}
        loss_bwd = {}
        loss_2d = 0.
        if not self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            loss_2d = torch.tensor(loss_2d)

        # For 3d loss
        loss_3d = 0.
        if not self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            loss_3d = torch.tensor(loss_3d)
        
        # for continous 2d supervision
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_2D.PROJ_TYPE == 'CONT':
            for idx in range(0, self.cfg.PROJECTION.NUM_VIEWS):
                proj_pred[idx] = self.projector(pred_pc, view_az[:,idx], view_el[:,idx])
                loss_bce[idx], fwd[idx], bwd[idx] = self.proj_loss(preds=proj_pred[idx], gts=proj_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
                loss_fwd[idx] = 1e-4 * torch.mean(fwd[idx])
                loss_bwd[idx] = 1e-4 * torch.mean(bwd[idx])

                if self.cfg.SUPERVISION_2D.USE_AFFINITY:
                    loss_2d += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx]) +\
                               self.cfg.PROJECTION.LAMDA_AFF_FWD * loss_fwd[idx] +\
                               self.cfg.PROJECTION.LAMDA_AFF_BWD * loss_bwd[idx]
                else:
                    loss_2d += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx])
        
        # for discrete 2d supervision
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_2D.PROJ_TYPE == 'DISC':
            for idx in range(0, self.cfg.PROJECTION.NUM_VIEWS):
                proj_pred[idx] = self.projector(pred_pc, view_az[:,idx], view_el[:,idx])
                point_gt[idx] = torch.mul(proj_pred[idx], proj_gt[:,idx])
                loss_bce[idx], fwd[idx], bwd[idx] = self.proj_loss(preds=proj_pred[idx], gts=point_gt[idx], grid_dist_tensor=grid_dist_tensor)
                loss_2d += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx])

        # for 3d supervision (EMD)
        loss_3d = torch.mean(self.emd(pred_pc, gt_pc))

        # Total loss
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = self.cfg.SUPERVISION_2D.LAMDA_2D_LOSS * (loss_2d/self.cfg.PROJECTION.NUM_VIEWS) +\
                         self.cfg.SUPERVISION_3D.LAMDA_3D_LOSS * loss_3d

        elif self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            total_loss = loss_2d / self.cfg.PROJECTION.NUM_VIEWS

        elif self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = loss_3d
            
        return total_loss, (loss_2d/self.cfg.PROJECTION.NUM_VIEWS), loss_3d, pred_pc

    def learn(self, input, init_pc, gt_pc, view_az, view_el, proj_gt):
        self.train(True)
        self.optimizer.zero_grad()
        total_loss, loss_2d, loss_3d, _ = self.loss(input, init_pc, gt_pc, view_az, view_el, proj_gt)
        total_loss.backward()
        self.optimizer.step()
        total_loss_np = total_loss.detach().item()
        loss_2d_np = loss_2d.detach().item()
        loss_3d_np = loss_3d.detach().item()
        del total_loss, loss_2d, loss_3d
        return total_loss_np, loss_2d_np, loss_3d_np
