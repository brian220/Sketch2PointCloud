# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#

import torch
import torchvision.models


class View_Estimater(torch.nn.Module):
    def __init__(self, cfg):
        super(View_Estimater, self).__init__()
        self.cfg = cfg
        