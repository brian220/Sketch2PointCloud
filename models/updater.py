import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models
import os

from models.projection import Projector

import utils.network_utils
from utils.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

from losses.proj_losses import *
from losses.earth_mover_distance import EMD

# Set the path for pretrain weight
os.environ['TORCH_HOME'] = '/media/caig/FECA2C89CA2C406F/sketch3D/pretrain_models'

class PointNet2(nn.Module):
    """
    Point cloud segmentation (set abstraction + feature propagation) in pointnet++
    
    Input:
        xyz: input points position [B, N, 3]

    output:
        point_feature: per-point features encode by pointnet [B, 128, N]
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=64, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=384, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, 1024], group_all=True)
        
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 512 , mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=128 + 256 , mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=0 + 128 , mlp=[128, 128, 128])
    
    def forward(self, xyz):
        xyz = xyz.transpose(2, 1) # [B, C, N]
        
        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        
        return l0_points


class ImgEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImgEncoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # conv
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        
        # linear
        self.linear = torch.nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
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


class DisplacementNet(nn.Module):
    """
    Predict the displacement from pointcloud features and image features

    Input:
        pc_features: poincloud features from pointnet2 [B, D, N]
        img_features: image features from image encoder [B, 1, D']
        noises: noises vector [B, N, n_length]

    Output:
        displacement: perpoint displacement [B, C, N]

    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(416, 384, 1)
        self.bn1 = nn.BatchNorm1d(384)
        self.conv2 = nn.Conv1d(384, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 3, 1)

    def forward(self, img_features, pc_features, noises):
        noises = noises.transpose(2, 1) #[B, n_length, N]
        noises = utils.network_utils.var_or_cuda(noises)

        img_features = img_features.repeat(1, self.cfg.CONST.NUM_POINTS, 1).transpose(2, 1) # [B, D', N]
        refine_features = torch.cat((pc_features, img_features, noises), 1) # concat the img features after each point features

        refine_features = F.relu(self.bn1(self.conv1(refine_features)))
        refine_features = F.relu(self.bn2(self.conv2(refine_features)))
        refine_features = F.relu(self.bn3(self.conv3(refine_features)))
        displacements = self.conv4(refine_features)

        displacements = F.sigmoid(displacements) * self.cfg.UPDATER.RANGE_MAX * 2 - self.cfg.UPDATER.RANGE_MAX
        
        return displacements


class Updater(nn.Module):
    """
    Refine the point cloud based on the input image

    Input:
        xyz: point cloud from reconstruction model

    Ouput:
        update_pc: updated point cloud
    """

    def __init__(self, cfg, optimizer=None):
        super().__init__()
        self.cfg = cfg
        
        self.pointnet2 = PointNet2(cfg)
        self.img_enc = ImgEncoder(cfg)
        self.displacement_net = DisplacementNet(cfg)

        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        
        # 2D supervision part
        self.projector = Projector(self.cfg)

        # proj loss
        self.proj_loss = ProjectLoss(self.cfg)
        
        # emd loss
        self.emd = EMD()

        if torch.cuda.is_available():
            self.img_enc = torch.nn.DataParallel(self.img_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pointnet2 = torch.nn.DataParallel(self.pointnet2, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

    def forward(self, img, xyz):
        _, img_features = self.img_enc(img)
        pc_features = self.pointnet2(xyz)
        noises = torch.normal(mean=0.0, std=1, size=(self.cfg.CONST.BATCH_SIZE, self.cfg.CONST.NUM_POINTS, self.cfg.UPDATER.NOISE_LENGTH))
        displacements = self.displacement_net(img_features, pc_features, noises)
        displacements = displacements.transpose(2, 1)
        refine_pc = xyz + displacements

        return refine_pc

    def loss(self, img, xyz, gt_pc, view_az, view_el, update_proj_gt):
        refine_pc = self(img, xyz)

        if self.cfg.SUPERVISION_2D.USE_AFFINITY:
           grid_dist_np = grid_dist(grid_h=self.cfg.PROJECTION.GRID_H, grid_w=self.cfg.PROJECTION.GRID_W).astype(np.float32)
           grid_dist_tensor = utils.network_utils.var_or_cuda(torch.from_numpy(grid_dist_np))
        else:
           grid_dist_tensor = None
        
        # Use 2D projection loss to train
        proj_pred = {}
        point_gt = {}
        loss_bce = {}
        fwd = {}
        bwd = {}
        loss_fwd = {}
        loss_bwd = {}
        loss_2d = 0.
        if not self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            loss_2d = torch.tensor(loss_2d)

        # For 3d loss
        loss_3d = 0.
        if not self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            loss_3d = torch.tensor(loss_3d)

        # for continous 2d supervision
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_2D.PROJ_TYPE == 'CONT':
            for idx in range(0, self.cfg.PROJECTION.UPDATE_NUM_VIEWS):
                proj_pred[idx] = self.projector(refine_pc, view_az[:,idx], view_el[:,idx])
                loss_bce[idx], fwd[idx], bwd[idx] = self.proj_loss(preds=proj_pred[idx], gts=update_proj_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
                loss_fwd[idx] = 1e-4 * torch.mean(fwd[idx])
                loss_bwd[idx] = 1e-4 * torch.mean(bwd[idx])

                if self.cfg.SUPERVISION_2D.USE_AFFINITY:
                    loss_2d += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx]) +\
                               self.cfg.PROJECTION.LAMDA_AFF_FWD * loss_fwd[idx] +\
                               self.cfg.PROJECTION.LAMDA_AFF_BWD * loss_bwd[idx]
                else:
                    loss_2d += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx])
        
        # for 3d supervision (EMD)
        loss_3d = torch.mean(self.emd(refine_pc, gt_pc))

        # Total loss
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = self.cfg.SUPERVISION_2D.LAMDA_2D_LOSS * (loss_2d/self.cfg.PROJECTION.UPDATE_NUM_VIEWS) +\
                         self.cfg.SUPERVISION_3D.LAMDA_3D_LOSS * loss_3d

        elif self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            total_loss = loss_2d / self.cfg.PROJECTION.UPDATE_NUM_VIEWS

        elif self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = loss_3d

        return total_loss, (loss_2d/self.cfg.PROJECTION.UPDATE_NUM_VIEWS), loss_3d, refine_pc

    def learn(self, img, xyz, gt_pc, view_az, view_el, update_proj_gt):
        self.train(True)
        self.optimizer.zero_grad()
        total_loss, loss_2d, loss_3d, _ = self.loss(img, xyz, gt_pc, view_az, view_el, update_proj_gt)
        total_loss.backward()
        self.optimizer.step()
        total_loss_np = total_loss.detach().item()
        loss_2d_np = loss_2d.detach().item()
        loss_3d_np = loss_3d.detach().item()
        del total_loss, loss_2d, loss_3d
        return total_loss_np, loss_2d_np, loss_3d_np
