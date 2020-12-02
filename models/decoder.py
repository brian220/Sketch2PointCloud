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
        super(pc_decoder, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.CONST.BATCH_SIZE

        # parameter for tree gcn
        self.features = [96, 256, 256, 256, 128, 128, 128, 3]
        self.degrees = [1, 2, 2, 2, 2, 2, 64]
        self.layer_num = len(self.features) - 1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num - 1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, self.features, self.degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, self.features, self.degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]