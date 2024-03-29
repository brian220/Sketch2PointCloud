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

import utils.network_utils

Conv = nn.Conv2d

def wrapper(func, *args, **kwargs):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.func = func

        def forward(self, input):
            return self.func(input, *args, **kwargs)

    return Wrapper()


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, out_features, cat_pc=True, use_adain=True, use_proj=True, activation=nn.ReLU()):
        super().__init__()
        self.cat_pc = cat_pc
        self.use_adain = use_adain
        self.use_proj = use_proj

        if use_adain:
            self.blocks = nn.Sequential()
            dim = in_features
            for i, out_features_ in enumerate(out_features):
                block = nn.Sequential()
                block.add_module('fc1', nn.Linear(dim, out_features_))
                block.add_module('relu1', activation)
                block.add_module('fc2', nn.Linear(out_features_, out_features_))
                block.add_module('relu2', activation)
                self.blocks.add_module('block%d' % i, block)
                dim = out_features_

            self.fcs = nn.Sequential(*[nn.Linear(block[-2].out_features, out_features_)
                                       for block, out_features_ in zip(self.blocks, out_features)])

        self.concat = wrapper(torch.cat, dim=-1)

    def forward(self, img_feats, init_pc):
        pc_feats = []
        if self.use_adain:
            pc_encoded = []
            out_feat = init_pc
            for block in self.blocks.children():
                out_feat = block(out_feat)
                pc_encoded.append(out_feat)

            pc_feats += [self.adain(pc_feat, img_feat, fc) for pc_feat, img_feat, fc in
                         zip(pc_encoded, img_feats, self.fcs.children())]

        if self.use_proj:
            pc_feats += [self.get_projection(img_feat, init_pc) for img_feat in img_feats]

        if self.cat_pc:
            pc_feats += [init_pc]

        pc_feats_trans = self.concat(pc_feats)
        return pc_feats_trans

    def _project_slow(self, img_feats, xs, ys):
        out = []
        for i in range(list(img_feats.shape)[0]):
            x, y, img_feat = xs[i], ys[i], img_feats[i]
            x1, y1 = torch.floor(x), torch.floor(y)
            x2, y2 = torch.ceil(x), torch.ceil(y)
            q11 = img_feat[..., x1.long(), y1.long()].cuda()
            q12 = img_feat[..., x1.long(), y2.long()].cuda()
            q21 = img_feat[..., x2.long(), y1.long()].cuda()
            q22 = img_feat[..., x2.long(), y2.long()].cuda()

            weights = ((x2 - x) * (y2 - y)).unsqueeze(0)
            q11 *= weights

            weights = ((x - x1) * (y2 - y)).unsqueeze(0)
            q21 *= weights

            weights = ((x2 - x) * (y - y1)).unsqueeze(0)
            q12 *= weights

            weights = ((x - x1) * (y - y1)).unsqueeze(0)
            q22 *= weights
            out.append(q11 + q12 + q21 + q22)
        return torch.stack(out).transpose(2, 1)

    def _project(self, img_feats, xs, ys):
        x, y = xs.flatten(), ys.flatten()
        idb = torch.arange(img_feats.shape[0], device=img_feats.device)
        idb = idb[None].repeat(xs.shape[1], 1).t().flatten().long()

        x1, y1 = torch.floor(x), torch.floor(y)
        x2, y2 = torch.ceil(x), torch.ceil(y)
        q11 = img_feats[idb, :, x1.long(), y1.long()].to(img_feats.device)
        q12 = img_feats[idb, :, x1.long(), y2.long()].to(img_feats.device)
        q21 = img_feats[idb, :, x2.long(), y1.long()].to(img_feats.device)
        q22 = img_feats[idb, :, x2.long(), y2.long()].to(img_feats.device)

        weights = ((x2 - x) * (y2 - y)).unsqueeze(1)
        q11 *= weights

        weights = ((x - x1) * (y2 - y)).unsqueeze(1)
        q21 *= weights

        weights = ((x2 - x) * (y - y1)).unsqueeze(1)
        q12 *= weights

        weights = ((x - x1) * (y - y1)).unsqueeze(1)
        q22 *= weights
        out = q11 + q12 + q21 + q22
        return out.view(img_feats.shape[0], -1, img_feats.shape[1])

    def get_projection(self, img_feat, pc):
        _, _, h_, w_ = tuple(img_feat.shape)
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        h = 248. * Y / Z + 111.5
        w = 248. * -X / Z + 111.5
        h = torch.clamp(h, 0., 223.)
        w = torch.clamp(w, 0., 223.)

        x = (h / (223. / (h_ - 1.))).requires_grad_(False)
        y = (w / (223. / (w_ - 1.))).requires_grad_(False)
        feats = self._project(img_feat, x, y)
        return feats

    def adain(self, pc_feat, img_feat, fc):
        pc_feat = (pc_feat - torch.mean(pc_feat, -1, keepdim=True)) / torch.sqrt(torch.var(pc_feat, -1, keepdim=True) + 1e-8)
        mean, var = torch.mean(img_feat, (2, 3)), torch.var(torch.flatten(img_feat, 2), 2)
        output = (pc_feat + mean[:, None]) * torch.sqrt(var[:, None] + 1e-8)
        return fc(output)


class CNN18Encoder(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU()):
        super().__init__()

        self.block1 = nn.Sequential()
        self.block1.conv1 = Conv(in_channels, 16, 3, padding=1)
        self.block1.relu1 = activation
        self.block1.conv2 = Conv(16, 16, 3, padding=1)
        self.block1.relu2 = activation
        self.block1.conv3 = Conv(16, 32, 3, stride=2, padding=1)
        self.block1.relu3 = activation
        self.block1.conv4 = Conv(32, 32, 3, padding=1)
        self.block1.relu4 = activation
        self.block1.conv5 = Conv(32, 32, 3, padding=1)
        self.block1.relu5 = activation
        self.block1.conv6 = Conv(32, 64, 3, stride=2, padding=1)
        self.block1.relu6 = activation
        self.block1.conv7 = Conv(64, 64, 3, padding=1)
        self.block1.relu7 = activation
        self.block1.conv8 = Conv(64, 64, 3, padding=1)
        self.block1.relu8 = activation

        self.block3 = nn.Sequential()
        self.block3.conv1 = Conv(64, 128, 3, stride=2, padding=1)
        self.block3.relu1 = activation
        self.block3.conv2 = Conv(128, 128, 3, padding=1)
        self.block3.relu2 = activation
        self.block3.conv3 = Conv(128, 128, 3, padding=1)
        self.block3.relu3 = activation

        self.block4 = nn.Sequential()
        self.block4.conv1 = Conv(128, 256, 5, stride=2, padding=2)
        self.block4.relu1 = activation
        self.block4.conv2 = Conv(256, 256, 3, padding=1)
        self.block4.relu2 = activation
        self.block4.conv3 = Conv(256, 256, 3, padding=1)
        self.block4.relu3 = activation

        self.block5 = nn.Sequential()
        self.block5.conv1 = Conv(256, 512, 5, stride=2, padding=2)
        self.block5.relu1 = activation
        self.block5.conv2 = Conv(512, 512, 3, padding=1)
        self.block5.relu2 = activation
        self.block5.conv3 = Conv(512, 512, 3, padding=1)
        self.block5.relu3 = activation
        self.block5.conv4 = Conv(512, 512, 3, padding=1)
        self.block5.relu4 = activation

    def forward(self, input):
        feats = []
        output = input
        for block in self.children():
            output = block(output)
            feats.append(output)
        return feats


class PointCloudDecoder(nn.Sequential):
    def __init__(self, in_features, activation=nn.ReLU(), **kwargs):
        super().__init__()

        self.conv2 = nn.Linear(in_features, 512)
        self.act1 = activation
        self.conv3 = nn.Linear(512, 256)
        self.act2 = activation
        self.conv4 = nn.Linear(256, 128)
        self.act3 = activation
        self.conv6 = nn.Linear(128, 3)
        self.act4 = activation


class PointCloudGraphXDecoder(nn.Sequential):
    def __init__(self, in_features, in_instances, activation=nn.ReLU()):
        super().__init__()

        self.conv2 = GraphXConv(in_features, 512, in_instances=in_instances, activation=activation)
        self.conv3 = GraphXConv(512, 256, in_instances=in_instances, activation=activation)
        self.conv4 = GraphXConv(256, 128, in_instances=in_instances, activation=activation)
        self.conv6 = nn.Linear(128, 3)