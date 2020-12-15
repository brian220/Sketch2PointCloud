import torch.nn as nn
import torch
CE = nn.CrossEntropyLoss().cuda()


def cross_entropy_loss(pred, target, range):
    binSize = range // pred.size(1)
    trueLabel = target // binSize
    return CE(pred, trueLabel)


class CELoss(nn.Module):
    def __init__(self, range):
        super(CELoss, self).__init__()
        self.__range__ = range
        return

    def forward(self, pred, target):
        return cross_entropy_loss(pred, target, self.__range__)