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

from models.psgn_fc import PSGN_CONV, PSGN_FC
from models.projection import Projector
from models.edge_detection import EdgeDetector 

from losses.proj_losses import *

import utils.network_utils

class Pixel2Pointcloud_PSGN_FC(nn.Module):
    def __init__(self, cfg, optimizer_conv=None, optimizer_fc=None, scheduler=None):
        super().__init__()
        self.cfg = cfg
        
        # PSGN FC
        self.psgn_conv = PSGN_CONV(self.cfg)
        self.psgn_fc = PSGN_FC(self.cfg)
        
        self.optimizer_conv = None if optimizer_conv is None else optimizer_conv(self.psgn_conv.parameters())
        self.optimizer_fc = None if optimizer_fc is None else optimizer_fc(self.psgn_fc.parameters())
        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)

        # 2D supervision part
        self.projector = Projector(self.cfg)

        # proj loss
        self.proj_loss = ProjectLoss(self.cfg)
        
        if torch.cuda.is_available():
            self.psgn_conv = torch.nn.DataParallel(self.psgn_conv, device_ids=cfg.CONST.DEVICE).cuda()
            self.psgn_fc = torch.nn.DataParallel(self.psgn_fc, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

        # edge detector
        if self.cfg.EDGE_LOSS.USE_EDGE_LOSS:
            self.edge_detector = EdgeDetector(self.cfg)
            self.edge_proj_loss = ProjectLoss(self.cfg)
            if torch.cuda.is_available():
                self.edge_detector = torch.nn.DataParallel(self.edge_detector, device_ids=cfg.CONST.DEVICE).cuda()
                self.edge_proj_loss = torch.nn.DataParallel(self.edge_proj_loss, device_ids=cfg.CONST.DEVICE).cuda()

    def forward(self, input):
        conv_features = self.psgn_conv(input)
        points = self.psgn_fc(conv_features)

        return points

    def loss(self, input, init_pc, view_az, view_el, proj_gt, edge_gt):
        pred_pc = self(input)
        
        grid_dist_np = grid_dist(grid_h=self.cfg.PROJECTION.GRID_H, grid_w=self.cfg.PROJECTION.GRID_W).astype(np.float32)
        grid_dist_tensor = utils.network_utils.var_or_cuda(torch.from_numpy(grid_dist_np))

        # Use 2D projection loss to train
        proj_pred = {}
        loss_bce = {}
        fwd = {}
        bwd = {}
        loss_fwd = {}
        loss_bwd = {}
        loss = 0.
        if not self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            loss_2d = torch.tensor(loss_2d)
        
        # For edge loss
        edge_proj_pred = {}
        edge_loss_bce = {}
        edge_fwd = {}
        edge_bwd = {}
        edge_loss_fwd = {}
        edge_loss_bwd = {}
        edge_loss = 0.

        for idx in range(0, self.cfg.PROJECTION.NUM_VIEWS):
            # Projection
            proj_pred[idx] = self.projector(pred_pc, view_az[:,idx], view_el[:,idx])

            # Projection loss
            loss_bce[idx], fwd[idx], bwd[idx] = self.proj_loss(preds=proj_pred[idx], gts=proj_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
            loss_fwd[idx] = 1e-4 * torch.mean(fwd[idx])
            loss_bwd[idx] = 1e-4 * torch.mean(bwd[idx])

             # Loss = projection loss + edge projection loss
            loss += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx]) +\
                    self.cfg.PROJECTION.LAMDA_AFF_FWD * loss_fwd[idx] +\
                    self.cfg.PROJECTION.LAMDA_AFF_BWD * loss_bwd[idx]

            if self.cfg.EDGE_LOSS.USE_EDGE_LOSS:
                # Edge prediction of projection
                proj_pred[idx] = proj_pred[idx].unsqueeze(1) # (BS, 1, H, W)
                edge_proj_pred[idx] = self.edge_detector(img=proj_pred[idx])
                edge_proj_pred[idx] = edge_proj_pred[idx].squeeze(1) # (BS, H, W)
    
                # Edge projection loss
                edge_loss_bce[idx], edge_fwd[idx], edge_bwd[idx] = self.proj_loss(preds=edge_proj_pred[idx], gts=edge_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
                edge_loss_fwd[idx] = 1e-4 * torch.mean(edge_fwd[idx])
                edge_loss_bwd[idx] = 1e-4 * torch.mean(edge_bwd[idx])

                edge_loss += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(edge_loss_bce[idx]) +\
                             self.cfg.PROJECTION.LAMDA_AFF_FWD * edge_loss_fwd[idx] +\
                             self.cfg.PROJECTION.LAMDA_AFF_BWD * edge_loss_bwd[idx]
        
        if self.cfg.EDGE_LOSS.USE_EDGE_LOSS:
            total_loss = ((loss + edge_loss*self.cfg.EDGE_LOSS.LAMDA_EDGE_LOSS) / self.cfg.PROJECTION.NUM_VIEWS)
        else:
            total_loss = loss / self.cfg.PROJECTION.NUM_VIEWS
        
        return total_loss, pred_pc

    def learn(self, input, init_pc, view_az, view_el, proj_gt, edge_gt):
        self.train(True)
        self.psgn_conv.zero_grad()
        self.psgn_fc.zero_grad()
        total_loss,  _ = self.loss(input, init_pc, view_az, view_el, proj_gt, edge_gt)
        total_loss.backward()
        self.optimizer_conv.step()
        self.optimizer_fc.step()
        total_loss_np = total_loss.detach().item()
        del total_loss
        return total_loss_np

