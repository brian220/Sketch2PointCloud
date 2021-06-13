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

from models.graphx_rec import Graphx_Rec

import cuda.emd.emd_module as emd

import utils.network_utils

class GRAPHX_REC_MODEL(nn.Module):
    def __init__(self, cfg, optimizer=None, scheduler=None):
        super().__init__()
        self.cfg = cfg
        
        # Graphx Reconstructor
        self.reconstructor = Graphx_Rec(
            cfg=cfg,
            in_channels=3,
            in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
            activation=nn.ReLU(),
        )
        
        self.optimizer = None if optimizer is None else optimizer(self.reconstructor.parameters())
        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)
        
        # emd loss
        self.emd_dist = emd.emdModule()

        if torch.cuda.is_available():
            # Reconstructor
            self.reconstructor = torch.nn.DataParallel(self.reconstructor, device_ids=cfg.CONST.DEVICE).cuda()
            # loss
            self.emd_dist = torch.nn.DataParallel(self.emd_dist, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()


    def reconstruction(self, input_imgs, init_pc):
        pred_pc = self.reconstructor(input_imgs, init_pc)
        return pred_pc

    
    def valid_step(self, input_imgs, init_pc, gt_pc):
        # reconstruct the point cloud
        pred_pc = self.reconstruction(input_imgs, init_pc)
        # compute reconstruction loss
        emd_loss, _ = self.emd_dist(
            pred_pc, gt_pc, eps=0.005, iters=50
        )
        rec_loss = torch.sqrt(emd_loss).mean(1).mean()
        
        return rec_loss*1000, pred_pc
    

    def train_step(self, input_imgs, init_pc, gt_pc):
        # reconstruct the point cloud
        pred_pc = self.reconstruction(input_imgs, init_pc)
        # compute reconstruction loss
        emd_loss, _ = self.emd_dist(
            pred_pc, gt_pc, eps=0.005, iters=50
        )
        rec_loss = torch.sqrt(emd_loss).mean(1).mean()

        self.reconstructor_backward(rec_loss)

        rec_loss_np = rec_loss.detach().item()

        return rec_loss_np*1000


    def reconstructor_backward(self, rec_loss):
        self.train(True)
        self.optimizer.zero_grad()
        rec_loss.backward()
        self.optimizer.step()
        del rec_loss
        
