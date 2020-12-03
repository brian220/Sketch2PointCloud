# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models

from models.encoder import Encoder
from models.decoder import Decoder
from models.STN import STN

class sketch2pointcloud_model(torch.nn.Module):
    def __init__(self, cfg):
        super(sketch2pointcloud_model, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.stn = STN()

    def forward(self, rendering_images):
        # view estimation
        transformation = self.stn(rendering_images)
        
        # point cloud generation
        encode_vector = self.encoder(rendering_images)
        original_point_cloud = self.decoder(rendering_images)
        
        # apply transformation on point cloud
        # normalize_point_cloud = 

        return original_point_cloud, normalize_point_cloud 
