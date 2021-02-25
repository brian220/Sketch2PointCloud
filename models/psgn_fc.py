import numpy as np
import torch
import torch.nn as nn

class PSGN_CONV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Encode
        # 32 64 64
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1,1)),
            torch.nn.ReLU(),
        )

        # 64 32 32
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1,1)),
            torch.nn.ReLU(),
        )

        # 128 16 16
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1,1)),
            torch.nn.ReLU(),
        )

        # 256 8 8
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=(2,2)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

    def forward(self, input):
        features = self.layer1(input)
        features = self.layer2(features)
        features = self.layer3(features)
        conv_features = self.layer4(features)

        return conv_features


class PSGN_FC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
         # 512 4 4
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(512 * 4 * 4, 128),
            torch.nn.ReLU(),
        )

        # Decode
        # 128
        self.layer6 = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.cfg.CONST.NUM_POINTS*3),
        )

        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        features = self.layer5(input)
        features = self.layer6(features)
        features = features.view(-1, self.cfg.CONST.NUM_POINTS, 3)
        points = self.tanh(features)

        return points