import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
import os
from PIL import Image

def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch.
    This is a PyTorch implementation computed by myself for calculating
    R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    
    For the original numpy implementation in StarMap, you can refer to:
    https://github.com/xingyizhou/StarMap/blob/26223a6c766eab3c22cddae87c375150f84f804d/tools/EvalCls.py#L20
    """
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)


def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()
    
    # get elevation and inplane-rotation in the right format
    # R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    preds[:, 1] = preds[:, 1] - 180.
    preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1] - 180.
    targets[:, 2] = targets[:, 2] - 180.
    
    # change degrees to radians
    preds = preds * np.pi / 180.
    targets = targets * np.pi / 180.
    
    # get rotation matrix from euler angles
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    
    # compute the angle distance between rotation matrix in degrees
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err


def rotation_acc(preds, targets, th=30.):
    R_err = rotation_err(preds, targets)
    return 100. * torch.mean((R_err <= th).float())
