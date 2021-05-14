# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models
import os

# Set the path for pretrain weight
os.environ['TORCH_HOME'] = '/media/caig/FECA2C89CA2C406F/sketch3D/pretrain_models'

class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))

        # conv
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )
        
        # linear
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 256),
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False
    
    def forward(self, rendering_image):
        features = self.vgg(rendering_image)
        vgg_features = features
        # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
        features = self.layer1(features)
        # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
        features = self.layer2(features)
        # print(features.size())    # torch.Size([batch_size, 512, 8, 8])
        features = self.layer3(features)
        # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
        features = self.avgpool(features)
        # print(features.size())    # torch.Size([batch_size, 256, 4, 4])
        features = torch.flatten(features, 1)
        # print(features.size())    # torch.Size([batch_size, 256*4*4])
        features = self.linear(features)
        # print(features.size())    # torch.Size([batch_size, 1, 256])

        return vgg_features, torch.unsqueeze(features, 1)