# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

from collections import OrderedDict
from functools import partial
from datetime import datetime as dt
import numpy as np

import torch
import torch.nn as nn

from models.edge_res import EdgeRes
from losses.earth_mover_distance import EMD
import utils.network_utils

class EdgeConv_Refiner(nn.Module):
    """
    inputs:
    - coarse: b x num_dims x npoints2
    outputs:
    - refine_result: b x num_dims x npoints2
    """

    def __init__(self, cfg, optimizer, num_points: int = 2048, use_SElayer: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_points = num_points
        self.residual = EdgeRes(use_SElayer=use_SElayer)
        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        self.emd = EMD()

        if torch.cuda.is_available():
            self.residual = torch.nn.DataParallel(self.residual, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

    def forward(self, coarse):
        delta = self.residual(coarse)  # [batch_size, 3, out_points]
        outs = coarse + delta
        refine_result = outs.transpose(2, 1).contiguous()  # [batch_size, out_points, 3]
        return refine_result

    def loss(self, coarse_pc, gt_pc, view_az, view_el):
        coarse_pc_transpose = coarse_pc.transpose(1, 2).contiguous()  # [b, 3, num_pc]
        refine_pc = self(coarse_pc_transpose)

        loss = torch.mean(self.emd(refine_pc, gt_pc))

        return loss, refine_pc

    def learn(self, coarse_pc, gt_pc, view_az, view_el):
        self.train(True)
        self.optimizer.zero_grad()
        loss, _ = self.loss(coarse_pc, gt_pc, view_az, view_el)
        loss.backward()
        self.optimizer.step()
        loss_np = loss.detach().item()
        del loss
        return loss_np


    

