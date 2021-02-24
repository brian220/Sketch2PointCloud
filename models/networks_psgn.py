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

from layers.graphx import GraphXConv

from models.projection import Projector
from models.edge_detection import EdgeDetector 

# from losses.chamfer_loss import chamfer
# from losses.earth_mover_distance import EMD
from losses.proj_losses import *

import utils.network_utils

Conv = nn.Conv2d

'''
def normalized_chamfer_loss(pred, gt, reduce='sum'):
    loss = chamfer(pred, gt, reduce=reduce)
    return loss if reduce == 'sum' else loss * 3000.
'''

class PSGN_CONV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Encode
        # 32 64 64
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1,1)),
            torch.nn.ReLU(),
        )

        # 64 32 32
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1,1)),
            torch.nn.ReLU(),
        )

        # 128 16 16
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1,1)),
            torch.nn.ReLU(),
        )

        # 256 8 8
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=(2,2)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

    def forward(self, input):
        features = self.layer1(input)
        features = self.layer2(features)
        features = self.layer3(features)
        conv_features = self.layer4(features)

        return conv_features
        

class PSGN_FC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
         # 512 4 4
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(512 * 4 * 4, 128),
            torch.nn.ReLU(),
        )

        # Decode
        # 128
        self.layer6 = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.cfg.CONST.NUM_POINTS*3),
        )

        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        features = self.layer5(input)
        features = self.layer6(features)
        features = features.view(-1, self.cfg.CONST.NUM_POINTS, 3)
        points = self.tanh(features)

        return points


class Pixel2Pointcloud(nn.Module):
    def __init__(self, cfg, in_channels, in_instances, activation=nn.ReLU(), optimizer_encode=None, optimizer_decode=None, scheduler=None, use_graphx=True, **kwargs):
        super().__init__()
        self.cfg = cfg
        
        # PSGN
        self.conv_net = PSGN_CONV(self.cfg)
        self.fc_net = PSGN_FC(self.cfg)
        
        self.optimizer_encode = None if optimizer_encode is None else optimizer_encode(self.rec_net.parameters())
        self.optimizer_decode = None if optimizer_decode is None else optimizer_decode(self.fc_net.parameters())

        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)
        self.kwargs = kwargs
        
        # self.emd = EMD().cuda()

        # 2D supervision part
        self.projector = Projector(self.cfg)

        # edge detector
        self.edge_detector = EdgeDetector(self.cfg)

        # proj loss
        self.proj_loss = ProjectLoss(self.cfg)
        self.edge_proj_loss = ProjectLoss(self.cfg)

        if torch.cuda.is_available():
            self.rec_net = torch.nn.DataParallel(self.rec_net, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.edge_detector = torch.nn.DataParallel(self.edge_detector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

    def forward(self, input):
        points = self.rec_net(input)

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

        edge_proj_pred = {}
        edge_loss_bce = {}
        edge_fwd = {}
        edge_bwd = {}
        edge_loss_fwd = {}
        edge_loss_bwd = {}
        
        loss = 0.

        for idx in range(0, self.cfg.PROJECTION.NUM_VIEWS):
            # Projection
            proj_pred[idx] = self.projector(pred_pc, view_az[:,idx], view_el[:,idx])
            # Projection loss
            loss_bce[idx], fwd[idx], bwd[idx] = self.proj_loss(preds=proj_pred[idx], gts=proj_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
            loss_fwd[idx] = 1e-4 * torch.mean(fwd[idx])
            loss_bwd[idx] = 1e-4 * torch.mean(bwd[idx])

            # Edge prediction of projection
            proj_pred[idx] = proj_pred[idx].unsqueeze(1) # (BS, 1, H, W)
            edge_proj_pred[idx] = self.edge_detector(img=proj_pred[idx])
            edge_proj_pred[idx] = edge_proj_pred[idx].squeeze(1) # (BS, H, W)

            # Edge projection loss
            edge_loss_bce[idx], edge_fwd[idx], edge_bwd[idx] = self.proj_loss(preds=edge_proj_pred[idx], gts=edge_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
            edge_loss_fwd[idx] = 1e-4 * torch.mean(edge_fwd[idx])
            edge_loss_bwd[idx] = 1e-4 * torch.mean(edge_bwd[idx])
             
            # Loss = projection loss + edge projection loss
            loss += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx]) +\
                    self.cfg.PROJECTION.LAMDA_AFF_FWD * loss_fwd[idx] +\
                    self.cfg.PROJECTION.LAMDA_AFF_BWD * loss_bwd[idx] +\
                    self.cfg.PROJECTION.LAMDA_BCE * torch.mean(edge_loss_bce[idx]) +\
                    self.cfg.PROJECTION.LAMDA_AFF_FWD * edge_loss_fwd[idx] +\
                    self.cfg.PROJECTION.LAMDA_AFF_BWD * edge_loss_bwd[idx]
            
            
            """
            loss_bce[idx] = self.proj_loss(preds=proj_pred[idx], 
                                           gts=proj_gt[:,idx], 
                                           grid_dist_tensor=grid_dist_tensor)
            
            loss += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx]) """
 
        loss = (loss / self.cfg.PROJECTION.NUM_VIEWS)
        
        return loss, pred_pc

    def learn(self, input, init_pc, view_az, view_el, proj_gt, edge_gt):
        self.train(True)
        self.optimizer.zero_grad()
        loss, _ = self.loss(input, init_pc, view_az, view_el, proj_gt, edge_gt)
        # loss = loss_dict['total']
        loss.backward()
        self.optimizer.step()
        loss_np = loss.detach().item()
        del loss
        return loss_np
