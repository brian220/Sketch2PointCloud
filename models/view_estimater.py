# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/YoungXIAO13/PoseFromShape/blob/master/auxiliary/model.py

import torch
import torchvision.models


class ViewEstimater(torch.nn.Module):
    def __init__(self, img_feature_dim=1024, azi_classes=24, ele_classes=12, inp_classes=24):
        super(BaselineEstimator, self).__init__()

        # RGB image encoder
        self.compress = nn.Sequential(nn.Linear(img_feature_dim, 800), nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))

        # Pose estimator
        self.fc_cls_azi = nn.Linear(200, azi_classes)
        self.fc_cls_ele = nn.Linear(200, ele_classes)

        self.fc_reg_azi = nn.Linear(200, azi_classes)
        self.fc_reg_ele = nn.Linear(200, ele_classes)

    def forward(self, img_feature):
        # pass the image through image encoder
        img_feature

        # concatenate the features obtained from two encoders into one feature
        x = self.compress(img_feature)

        cls_azi = self.fc_cls_azi(x)
        cls_ele = self.fc_cls_ele(x)

        reg_azi = self.fc_reg_azi(x)
        reg_ele = self.fc_reg_ele(x)
        return cls_azi, cls_ele, reg_azi, reg_ele
        