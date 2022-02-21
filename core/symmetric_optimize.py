import json
import numpy as np
import cv2
import os
import torch
import torch.nn as nn

import scipy.misc as sc
import utils.network_utils
import utils.point_cloud_visualization_old

from layers.graphx import GraphXConv
from tensorboardX import SummaryWriter
from pyntcloud import PyntCloud

from utils.point_cloud_utils import output_point_cloud_ply
from utils.pointnet2_utils import index_points, farthest_point_sample

import cuda.emd.emd_module as emd

def compute_plane_distances(point_clouds, points, normals):
    vectors = point_clouds - points
    normals = normals.unsqueeze(2) # [1, 3, 1]
    distances = torch.bmm(vectors, normals) # [1, 1024, 1]

    return distances

def compute_symmetric_pcs(point_clouds, points, sym_normals):
    distances = compute_plane_distances(point_clouds, points, sym_normals)
    
    sym_normals = sym_normals.unsqueeze(2).permute(0, 2, 1) # [1, 1, 3]
    mul_distances = torch.mul(distances, sym_normals) # [1, 1024, 3]
    symmetric_pcs = point_clouds - 2.*(mul_distances)

    return symmetric_pcs

class PC_OPT_MODEL(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, input_pcs):
        super().__init__()
        self.pred_pcs = nn.Parameter(input_pcs)
        self.emd_dist = emd.emdModule()
            
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        if torch.cuda.is_available():
            self.pred_pcs.cuda()
            self.emd_dist.cuda()
            self.cuda()
        
    def train_step(self, center_points, sym_normals, ori_pcs):
        symmetric_pcs = compute_symmetric_pcs(self.pred_pcs, center_points, sym_normals)
        
        """
        reg_loss, _ = self.emd_dist(
            self.pred_pcs, ori_pcs, eps=0.005, iters=50
        )
        reg_loss = torch.sqrt(reg_loss).mean(1).mean()
        """
        
        sym_loss, _ = self.emd_dist(
            self.pred_pcs, symmetric_pcs, eps=0.005, iters=50
        )
        sym_loss = torch.sqrt(sym_loss).mean(1).mean()
        
        loss = sym_loss
        
        self.opt_backward(loss)
        
        loss_np = loss.detach().item()
        sym_loss_np = sym_loss.detach().item()
        
        return loss_np, sym_loss_np, self.pred_pcs

    def opt_backward(self, loss):
        self.train(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def symmetric_optimize_net(output_dir, ori_point_clouds, center_point, sym_normal):
    if ori_point_clouds.shape[0] != 2048:
        pc_np = np.array([ori_point_clouds]).astype(np.float32)
        pc = torch.from_numpy(pc_np)
        idx = farthest_point_sample(pc, 2048)
        pc = index_points(pc, idx).detach().data.numpy()[0]
    else:
        pc = ori_point_clouds

    ori_pc_np = np.array([pc]).astype(np.float32)
    ori_pcs = torch.from_numpy(ori_pc_np)
    ori_pcs = utils.network_utils.var_or_cuda(ori_pcs)
    
    input_pc_np = np.array([pc]).astype(np.float32)
    input_pcs = torch.from_numpy(input_pc_np)
    input_pcs = utils.network_utils.var_or_cuda(input_pcs)

    center_points = np.array([center_point]).astype(np.float32)
    center_points = torch.from_numpy(center_points)
    center_points = utils.network_utils.var_or_cuda(center_points)
    
    sym_normals = np.array([sym_normal]).astype(np.float32)
    sym_normals = torch.from_numpy(sym_normals)
    sym_normals = utils.network_utils.var_or_cuda(sym_normals)

    pc_opt_model = PC_OPT_MODEL(input_pcs)
    pc_opt_model = pc_opt_model.cuda()
    pc_opt_model.train()
    
    finetune_writer = SummaryWriter(output_dir)

    sym_losses = utils.network_utils.AverageMeter()

    max_step = 200
    for i in range(max_step):
        loss, sym_loss, pred_pcs = pc_opt_model.train_step(center_points, sym_normals, ori_pcs)

        sym_losses.update(sym_loss)

        print('[Steps %d/%d] Sym_loss = %.4f' % (i + 1, max_step, sym_loss))

        finetune_writer.add_scalar('Total/StepLoss_Sym', sym_losses.avg, i + 1)
    
        pred_pc = pred_pcs[0].detach().cpu().numpy()
            
        if i + 1 == max_step:
            return pred_pcs[0].detach().cpu().numpy()
    
