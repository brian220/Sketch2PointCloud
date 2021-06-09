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

from models.graphx import CNN18Encoder, PointCloudEncoder, PointCloudGraphXDecoder, PointCloudDecoder
from models.edge_res import EdgeRes
from losses.earth_mover_distance import EMD
import utils.network_utils


class Sketch_Rec_Net(nn.Module):
    """
    inputs:
    - input: b x img_dims x img_width x img_length
    - init_pc: b x npoints x 3
    
    outputs:
    - refine_result: b x npoints x num_dims
    """

    def __init__(self, cfg, optimizer, scheduler):
        super().__init__()
        self.cfg = cfg
        
        # Init the generator and refiner
        self.generator = GRAPHX_Generator(cfg=cfg,
                                          in_channels=3, 
                                          in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
                                          use_graphx=cfg.GRAPHX.USE_GRAPHX)

        self.refiner = EdgeConv_Refiner(cfg=cfg, 
                                        num_points=cfg.CONST.NUM_POINTS, 
                                        use_SElayer=True)
        
        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)

        self.emd = EMD()

        if torch.cuda.is_available():
            self.generator = torch.nn.DataParallel(self.generator, device_ids=cfg.CONST.DEVICE).cuda()
            self.refiner = torch.nn.DataParallel(self.refiner, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()
        
        # Load pretrained generator
        if cfg.REFINE.USE_PRETRAIN_GENERATOR:
            print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.REFINE.GENERATOR_WEIGHTS))
            checkpoint = torch.load(cfg.REFINE.GENERATOR_WEIGHTS)
            self.generator.load_state_dict(checkpoint['net'])
            print('Best Epoch: %d' % (checkpoint['epoch_idx']))
        
    def forward(self, input, init_pc):
        coarse_pc = self.generator(input, init_pc) # [b, num_pc, 3]
        coarse_pc_transpose = coarse_pc.transpose(1, 2).contiguous()  # [b, 3, num_pc]
        middle_pc = self.refiner(coarse_pc_transpose)
        middle_pc_transpose = middle_pc.transpose(1, 2).contiguous()
        refine_pc = self.refiner(middle_pc_transpose)
        return coarse_pc, middle_pc, refine_pc

    def loss(self, input, init_pc, gt_pc, view_az, view_el):
        coarse_pc, middle_pc, refine_pc = self(input, init_pc)
        # for 3d supervision (EMD)
        coarse_loss = torch.mean(self.emd(coarse_pc, gt_pc))
        refine_loss = torch.mean(self.emd(refine_pc, gt_pc))
        loss = coarse_loss + refine_loss 
        return loss, coarse_loss, refine_loss, coarse_pc, middle_pc, refine_pc

    def learn(self, input, init_pc, gt_pc, view_az, view_el):
        self.train(True)
        self.optimizer.zero_grad()
        loss, coarse_loss, refine_loss, _, _, _ = self.loss(input, init_pc, gt_pc, view_az, view_el)
        loss.backward()
        self.optimizer.step()
        loss_np = loss.detach().item()
        coarse_loss_np = coarse_loss.detach().item()
        refine_loss_np = refine_loss.detach().item()
        del loss, coarse_loss, refine_loss
        return loss_np, coarse_loss_np, refine_loss_np


class GRAPHX_Generator(nn.Module):
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
        
        self.kwargs = kwargs

        # emd loss
        self.emd = EMD()
        
        if torch.cuda.is_available():
            self.img_enc = torch.nn.DataParallel(self.img_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc_enc = torch.nn.DataParallel(self.pc_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc = torch.nn.DataParallel(self.pc, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()
        

    def forward(self, input, init_pc):
        img_feats = self.img_enc(input)
        pc_feats = self.pc_enc(img_feats, init_pc)
        return self.pc(pc_feats)


class EdgeConv_Refiner(nn.Module):
    """
    inputs:
    - coarse: b x num_dims x npoints2
    outputs:
    - refine_result: b x num_dims x npoints2
    """

    def __init__(self, cfg, num_points: int = 2048, use_SElayer: bool = False):
        super().__init__()
        self.cfg = cfg
        self.num_points = num_points
        self.residual = EdgeRes(use_SElayer=use_SElayer)

    def forward(self, coarse):
        delta = self.residual(coarse)  # [batch_size, 3, out_points]
        outs = coarse + delta
        refine_result = outs.transpose(2, 1).contiguous()  # [batch_size, out_points, 3]
        return refine_result

