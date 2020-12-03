from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, cfg):
        super(STN, self).__init__()
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
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 9)
        )

         # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False
        
        def forward(self, rendering_image):
            batchsize = cfg.CONST.BATCH_SIZE

            features = self.vgg(rendering_image)
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)
            transformation = self.linear(features)

            iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
            if features.is_cuda:
                iden = iden.cuda()
            transformation = transformation + iden
            transformation = transformation.view(-1, 3, 3)
         
            return transformation




