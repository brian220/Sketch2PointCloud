# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

from collections import OrderedDict
from functools import partial
from itertools import chain
import random
import numpy as np
import torch
import torch.nn as nn

from models.graphx_rec import Graphx_Rec
from models.projection_discriminator import ProjectionD
from models.projection_depth import ComputeDepthMaps, N_VIEWS_PREDEFINED

import cuda.emd.emd_module as emd
import utils.network_utils

class GRAPHX_GAN_MODEL(nn.Module):
    def __init__(self, 
                 cfg,
                 optimizer_G=None,
                 scheduler_G=None,
                 optimizer_D=None,
                 scheduler_D=None):
        
        super().__init__()
        self.cfg = cfg
        
        # Graphx Generator
        self.model_G = Graphx_Rec(
            cfg=cfg,
            in_channels=3,
            in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
            activation=nn.ReLU(),
        )

        # Projection Discriminator
        self.model_D = ProjectionD(
            num_classes=cfg.DATASET.NUM_CLASSES,
            img_shape=(cfg.RENDER.N_VIEWS + 3, cfg.RENDER.IMG_SIZE, cfg.RENDER.IMG_SIZE),
        )

        # Renderer
        self.renderer = ComputeDepthMaps(
            projection=cfg.RENDER.PROJECTION,
            eyepos_scale=cfg.RENDER.EYEPOS,
            image_size=cfg.RENDER.IMG_SIZE,
        ).float()

        # OptimizerG
        self.optimizer_G = None if optimizer_G is None else optimizer_G(self.model_G.parameters())
        self.scheduler_G = None if scheduler_G or optimizer_G is None else scheduler_G(self.optimizer_G)

        # OptimizerD
        self.optimizer_D = None if optimizer_D is None else optimizer_D(self.model_D.parameters())
        self.scheduler_D = None if scheduler_D or optimizer_D is None else scheduler_D(self.optimizer_D)
        
        # a dict store the losses for each step
        self.loss = {}

        # emd loss
        self.emd_dist = emd.emdModule()
        
        # GAN criterion
        self.criterionD = torch.nn.MSELoss()

        if torch.cuda.is_available():
            # Generator
            self.model_G = torch.nn.DataParallel(self.model_G, device_ids=cfg.CONST.DEVICE).cuda()
            # Discriminator
            self.model_D = torch.nn.DataParallel(self.model_D, device_ids=cfg.CONST.DEVICE).cuda()
            # Renderer
            self.renderer = torch.nn.DataParallel(self.renderer, device_ids=cfg.CONST.DEVICE).cuda()
            # loss
            self.emd_dist = torch.nn.DataParallel(self.emd_dist, device_ids=cfg.CONST.DEVICE).cuda()
            self.criterionD = torch.nn.DataParallel(self.criterionD, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()
    
    
    def train_step(self, input_imgs, init_pc, gt_pc):
        '''
        Input:
            input_imgs: [B, C, img_w, img_h]
            init pc:    [B, N, 3]
            gt pc:      [B, N, 3]

        Output:
            loss
            pred_pc:    [B, N, 3]
        '''

        # create real and fake label
        _batch_size = input_imgs.size(0)
        self.real_label = (
            torch.FloatTensor(_batch_size)
            .resize_([_batch_size, 1])
            .data.fill_(1)
        )
        self.real_label = utils.network_utils.var_or_cuda(self.real_label)
        self.fake_label = (
            torch.FloatTensor(_batch_size)
            .resize_([_batch_size, 1])
            .data.fill_(0)
        )
        self.fake_label = utils.network_utils.var_or_cuda(self.fake_label)

        # reconstruct the point cloud
        pred_pc = self.reconstruction(input_imgs, init_pc)
        # compute reconstruction loss
        emd_loss, _ = self.emd_dist(
            pred_pc, gt_pc, eps=0.005, iters=50
        )
        rec_loss = torch.sqrt(emd_loss).mean(1).mean()
        rendered_pc = pred_pc
        
        # discriminator backward
        errD_real, errD_fake = self.discriminator_backward(input_imgs, gt_pc, rendered_pc)
        
        # generator backward
        errG, errG_D, fm_loss, im_loss = self.generator_backward(rec_loss)

        # store losses
        self.loss["rec_loss"] = rec_loss * 1000
        self.loss["errD_real"] = errD_real
        self.loss["errD_fake"] = errD_fake
        self.loss["errG"] = errG
        self.loss["errG_D"] = errG_D
        self.loss["fm_loss"] = fm_loss
        self.loss["im_loss"] = im_loss
        
        return self.loss

    def valid_step(self, input_imgs, init_pc, gt_pc):
        # reconstruct the point cloud
        pred_pc = self.reconstruction(input_imgs, init_pc)
        # compute reconstruction loss
        emd_loss, _ = self.emd_dist(
            pred_pc, gt_pc, eps=0.005, iters=50
        )
        rec_loss = torch.sqrt(emd_loss).mean(1).mean()
        
        return rec_loss*1000, pred_pc
    
    def reconstruction(self, input_imgs, init_pc):
        pred_pc, _ = self.model_G(input_imgs, init_pc)
        return pred_pc
        
    def discriminator_backward(self, input_imgs, gt_pc, rendered_pc):
        '''
        Input:
            input_imgs:  [B, C, img_w, img_h]
            gt pc:       [B, N, 3]
            rendered_pc: [B, N, 3]

        Output:
            errD_real
            errD_fake
        '''
        
        self.optimizer_D.zero_grad()

        # create rendering imgs (real , fake)
        real_render_imgs_dict = {}
        gen_render_imgs_dict = {}
        random_radius = random.sample(self.cfg.RENDER.radius_list, 1)[0]
        random_view_ids = list(range(0, N_VIEWS_PREDEFINED, 1))

        for _view_id in random_view_ids:
            # get real_imgs, gen_imgs
            real_render_imgs_dict[_view_id] = self.renderer(
                gt_pc, view_id=_view_id, radius_list=[random_radius]
            )
            gen_render_imgs_dict[_view_id] = self.renderer(
                rendered_pc, view_id=_view_id, radius_list=[random_radius]
            )

        _view_id = random_view_ids[0]
        self.real_imgs = real_render_imgs_dict[_view_id]
        self.fake_imgs = gen_render_imgs_dict[_view_id]
        for _index in range(1, len(random_view_ids)):
            _view_id = random_view_ids[_index]
            self.real_imgs = torch.cat(
                (self.real_imgs, real_render_imgs_dict[_view_id]), dim=1
            )
            self.fake_imgs = torch.cat(
                (self.fake_imgs, gen_render_imgs_dict[_view_id]), dim=1
            )

        self.input_imgs = input_imgs

        # forward pass discriminar (D_real_pred , D_fake_pred)
        errD_real = 0.0
        errD_fake = 0.0

        D_real_pred = self.model_D(
            torch.cat((self.input_imgs, self.real_imgs), dim=1).detach()
        )

        D_fake_pred = self.model_D(
            torch.cat((self.input_imgs, self.fake_imgs), dim=1).detach()
        )
        
        # backward discriminar
        errD_real += self.criterionD(D_real_pred, self.real_label)
        errD_fake += self.criterionD(D_fake_pred, self.fake_label)
        errD = errD_real + errD_fake
        errD.backward()
        self.optimizer_D.step()
        return errD_real, errD_fake
    
    
    def generator_backward(self, rec_loss):
        '''
        Input:
            rec_loss:   float

        Output:
            errG
            errG_D

        TO DO:
        (1) Add img feature loss
        (2) Add img depth loss
        '''

        self.optimizer_G.zero_grad()
        
        errG_D = 0.0
        fm_loss = 0.0
        im_loss = 0.0
        
        # forward pass discriminar (D_fake_pred)

        # feature matching loss
        if self.cfg.GAN.USE_FM_LOSS:
            D_fake_pred, D_fake_features = self.model_D(
                    torch.cat((self.input_imgs, self.fake_imgs), dim=1),
                    feat=True
                )
            _, D_real_features = self.model_D(
                torch.cat((self.input_imgs, self.real_imgs), dim=1),
                feat=True
            )
        
            # Feature match loss is weighted by number of feature maps
            map_nums = [feat.shape[1] for feat in D_fake_features]
            feat_weights = [float(i) / sum(map_nums) for i in map_nums]
            for j in range(
                len(D_fake_features)
            ):  # the final loss is the sum of all features
                fm_loss += feat_weights[j] * torch.mean(
                    (D_fake_features[j] - D_real_features[j].detach()) ** 2
                )

        else:
            D_fake_pred = self.model_D(
                torch.cat((self.input_imgs, self.fake_imgs), dim=1)
            )

        errG_D += self.criterionD(D_fake_pred, self.real_label)
        
        # img matching loss
        if self.cfg.GAN.USE_IM_LOSS:  # Get image matching (L1_loss)
            im_loss += torch.nn.L1Loss()(self.fake_imgs, self.real_imgs.detach())
        
        # add reconstruction loss and adv loss
        errG = (
            self.cfg.GAN.WEIGHT_REC * rec_loss + self.cfg.GAN.WEIGHT_GAN * errG_D
        )

        # the sum of recloss and GAN_loss (and feature matching and image matching)
        if self.cfg.GAN.USE_FM_LOSS:
            errG += self.cfg.GAN.WEIGHT_FM * fm_loss
        if self.cfg.GAN.USE_IM_LOSS:
            errG += self.cfg.GAN.WEIGHT_IM * im_loss
        
        # backward pass generator
        errG.backward()
        self.optimizer_G.step()

        return errG, errG_D, fm_loss, im_loss

