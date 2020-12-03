# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390.cs08g@nctu.edu.tw>
# Lot's of codes are borrowed from treeGCN: 
# https://github.com/seowok/TreeGAN

import torch
import torchvision.models
import torch.nn.functional as F
from layers.gcn import TreeGCN

from math import ceil


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.CONST.BATCH_SIZE

        # parameter for tree gcn
        self.features = [256, 512, 256, 256, 128, 128, 64, 3]
        self.degrees = [2, 2, 2, 2, 2, 4, 16]
        self.layer_num = len(self.features) - 1
        assert self.layer_num == len(self.degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        self.vertex_num = 1
        self.support = 10

        self.gcn = torch.nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num - 1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, self.features, self.degrees, 
                                            support=self.support, node=self.vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, self.features, self.degrees, 
                                            support=self.support, node=self.vertex_num, upsample=True, activation=True))
            self.vertex_num = int(self.vertex_num * self.degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]