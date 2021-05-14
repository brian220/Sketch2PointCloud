
# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/YoungXIAO13/PoseFromShape/blob/master/auxiliary/model.py

import torch
import torchvision.models


class ViewEstimater(torch.nn.Module):
    def __init__(self, cfg, azi_classes=24, ele_classes=12):
        super(ViewEstimater, self).__init__()
        
        self.cfg = cfg
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))

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

        # RGB image encoder
        self.compress = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout())

        # Pose estimator
        self.fc_cls_azi = torch.nn.Linear(4096, azi_classes)
        self.fc_cls_ele = torch.nn.Linear(4096, ele_classes)

        self.fc_reg_azi = torch.nn.Linear(4096, azi_classes)
        self.fc_reg_ele = torch.nn.Linear(4096, ele_classes)

    def forward(self, vgg_features):    
        # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
        features = self.layer1(vgg_features)
        # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
        features = self.layer2(features)
        # print(features.size())    # torch.Size([batch_size, 512, 8, 8])
        features = self.layer3(features)
        # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
        features = self.avgpool(features)
        # print(features.size())    # torch.Size([batch_size, 256, 4, 4])
        features = torch.flatten(features, 1)
        # concatenate the features obtained from two encoders into one feature
        x = self.compress(features)

        cls_azi = self.fc_cls_azi(x)
        cls_ele = self.fc_cls_ele(x)

        reg_azi = self.fc_reg_azi(x)
        reg_ele = self.fc_reg_ele(x)

        return [cls_azi, cls_ele, reg_azi, reg_ele]

