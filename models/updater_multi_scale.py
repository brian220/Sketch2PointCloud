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

Conv = nn.Conv2d

def wrapper(func, *args, **kwargs):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.func = func

        def forward(self, input):
            return self.func(input, *args, **kwargs)

    return Wrapper()


class CNN18Encoder(nn.Module):
    """
    Image multi-scale encoder
    
    Input:
        input: input images

    Output:
        feats: Multi-scale image features
    """
    def __init__(self, in_channels, activation=nn.ReLU()):
        super().__init__()

        self.block1 = nn.Sequential()
        self.block1.conv1 = Conv(in_channels, 16, 3, padding=1)
        self.block1.relu1 = activation
        self.block1.conv2 = Conv(16, 16, 3, padding=1)
        self.block1.relu2 = activation
        self.block1.conv3 = Conv(16, 32, 3, stride=2, padding=1)
        self.block1.relu3 = activation
        self.block1.conv4 = Conv(32, 32, 3, padding=1)
        self.block1.relu4 = activation
        self.block1.conv5 = Conv(32, 32, 3, padding=1)
        self.block1.relu5 = activation
        self.block1.conv6 = Conv(32, 64, 3, stride=2, padding=1)
        self.block1.relu6 = activation
        self.block1.conv7 = Conv(64, 64, 3, padding=1)
        self.block1.relu7 = activation
        self.block1.conv8 = Conv(64, 64, 3, padding=1)
        self.block1.relu8 = activation

        self.block3 = nn.Sequential()
        self.block3.conv1 = Conv(64, 128, 3, stride=2, padding=1)
        self.block3.relu1 = activation
        self.block3.conv2 = Conv(128, 128, 3, padding=1)
        self.block3.relu2 = activation
        self.block3.conv3 = Conv(128, 128, 3, padding=1)
        self.block3.relu3 = activation

        self.block4 = nn.Sequential()
        self.block4.conv1 = Conv(128, 256, 5, stride=2, padding=2)
        self.block4.relu1 = activation
        self.block4.conv2 = Conv(256, 256, 3, padding=1)
        self.block4.relu2 = activation
        self.block4.conv3 = Conv(256, 256, 3, padding=1)
        self.block4.relu3 = activation

        self.block5 = nn.Sequential()
        self.block5.conv1 = Conv(256, 512, 5, stride=2, padding=2)
        self.block5.relu1 = activation
        self.block5.conv2 = Conv(512, 512, 3, padding=1)
        self.block5.relu2 = activation
        self.block5.conv3 = Conv(512, 512, 3, padding=1)
        self.block5.relu3 = activation
        self.block5.conv4 = Conv(512, 512, 3, padding=1)
        self.block5.relu4 = activation

    def forward(self, input):
        feats = []
        output = input
        for block in self.children():
            output = block(output)
            feats.append(output)
        return feats


class TransformPC(nn.Module):
    """
    Transform point cloud to camera coordinate

    Input:
        xyz: float tensor, (BS,N_PTS,3); input point cloud
                 values assumed to be in (-1,1)
        az: float tensor, (BS); azimuthal angle of camera in radians
        el: float tensor, (BS); elevation of camera in radians
        
    Output:
        xyz_out: float tensor, (BS,N_PTS,3); output point cloud in camera
                 co-ordinates
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_pts = cfg.CONST.NUM_POINTS
    
    def forward(self, xyz, az, el, update_id):
        # Because there may be different update_id in a same batch
        # Here separate the batch data and compute transform pc one by one
        batch_size = xyz.size(0)
        batch_xyz_out = []
        for batch_id in range(batch_size):
            single_xyz = xyz[batch_id].unsqueeze(0)
            single_update_id = update_id[batch_id]
            single_az = az[batch_id].unsqueeze(0)
            single_el = el[batch_id].unsqueeze(0)

            xyz_out = self.world2cam(single_xyz, single_az[:,single_update_id], single_el[:,single_update_id], N_PTS=self.n_pts)
            batch_xyz_out.append(xyz_out.squeeze(0))
        
        batch_xyz_out = torch.stack(batch_xyz_out)
        return batch_xyz_out

    def world2cam(self, xyz, az, el, N_PTS=1024):
        # y ---> x
        rotmat_az=[
                    [torch.cos(az),torch.sin(az),torch.zeros_like(az)],
                    [-torch.sin(az),torch.cos(az),torch.zeros_like(az)],
                    [torch.zeros_like(az),torch.zeros_like(az), torch.ones_like(az)]
                    ]
        rotmat_az = [ torch.stack(x) for x in rotmat_az ]
        
        # z ---> x, in dataloader, az = original az - 90 degree, which means here is actually x ----> -z 
        rotmat_el=[
                    [torch.cos(el),torch.zeros_like(az), torch.sin(el)],
                    [torch.zeros_like(az),torch.ones_like(az),torch.zeros_like(az)],
                    [-torch.sin(el),torch.zeros_like(az), torch.cos(el)]
                    ]
        rotmat_el = [ torch.stack(x) for x in rotmat_el ]
        
        rotmat_az = torch.stack(rotmat_az, 0) # [3,3,B]
        rotmat_el = torch.stack(rotmat_el, 0) # [3,3,B]
        rotmat_az = rotmat_az.permute(2, 0, 1) # [B,3,3]
        rotmat_el = rotmat_el.permute(2, 0, 1) # [B,3,3]
        rotmat = torch.matmul(rotmat_el, rotmat_az)

        # Transformation(t)
        # Distance of object from camera - fixed to 2
        d = 2.
        # Calculate translation params
        tx, ty, tz = [0, 0, d]
        
        tr_mat = torch.unsqueeze(torch.tensor([tx, ty, tz]), 0).repeat(1,1) # [B,3]
        tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
        tr_mat = tr_mat.permute(0, 2, 1) # [B,1,3]
        tr_mat = tr_mat.repeat(1, N_PTS, 1) # [B,1024,3]
        tr_mat = utils.network_utils.var_or_cuda(tr_mat) # [B,1024,3]

        xyz_out = torch.matmul(rotmat, xyz.permute(0, 2, 1)) - tr_mat.permute(0, 2, 1)

        return xyz_out.permute(0, 2, 1)


class FeatureProjection(nn.Module):
    """
    Project the pointcloud to 2d image and get the corresponding image features at
    the project location
 
    Input:
        img_feats: multi-scale image features 
        pc: input point clouds (in camera coordinate) [B, N, 3]

    Output:
        pc_feats_trans: pointcloud xyz + multi-view image features (by feature ptojection)

    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.concat = wrapper(torch.cat, dim=-1)

    def forward(self, img_feats, pc):
        pc_feats = []
        pc_feats += [self.get_projection(img_feat, pc) for img_feat in img_feats]
        pc_feats_trans = self.concat(pc_feats)
        return pc_feats_trans

    def _project(self, img_feats, xs, ys):
        x, y = xs.flatten(), ys.flatten()
        idb = torch.arange(img_feats.shape[0], device=img_feats.device)
        idb = idb[None].repeat(xs.shape[1], 1).t().flatten().long()

        x1, y1 = torch.floor(x), torch.floor(y)
        x2, y2 = torch.ceil(x), torch.ceil(y)
        q11 = img_feats[idb, :, x1.long(), y1.long()].to(img_feats.device)
        q12 = img_feats[idb, :, x1.long(), y2.long()].to(img_feats.device)
        q21 = img_feats[idb, :, x2.long(), y1.long()].to(img_feats.device)
        q22 = img_feats[idb, :, x2.long(), y2.long()].to(img_feats.device)

        weights = ((x2 - x) * (y2 - y)).unsqueeze(1)
        q11 *= weights

        weights = ((x - x1) * (y2 - y)).unsqueeze(1)
        q21 *= weights

        weights = ((x2 - x) * (y - y1)).unsqueeze(1)
        q12 *= weights

        weights = ((x - x1) * (y - y1)).unsqueeze(1)
        q22 *= weights
        out = q11 + q12 + q21 + q22
        return out.view(img_feats.shape[0], -1, img_feats.shape[1])

    def get_projection(self, img_feat, pc):
        _, _, h_, w_ = tuple(img_feat.shape)
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        w = (420.*X/abs(Z) + (111.5))
        h = (420.*Y/abs(Z) + (111.5))
        w = torch.clamp(w, 0., 223.)
        h = torch.clamp(h, 0., 223.)
    
        x = w / (223. / (w_ - 1.))
        y = h / (223. / (h_ - 1.))
        feats = self._project(img_feat, x, y)
        return feats


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


class DisplacementNet(nn.Module):
    """
    Predict the displacement from pointcloud features and image features

    Input:
        pc_features: poincloud features from pointnet2 [B, D, N]
        proj_features: image features from feature projection [B, N, D']
        noises: noises vector [B, N, n_length]

    Output:
        displacement: perpoint displacement [B, C, N]

    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(1120, 960, 1)
        self.bn1 = nn.BatchNorm1d(960)
        self.conv2 = nn.Conv1d(960, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 64, 1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 3, 1)

    def forward(self, proj_features, pc_features, noises):
        noises = noises.transpose(2, 1) # [B, n_length, N]
        noises = utils.network_utils.var_or_cuda(noises)
        
        proj_features = proj_features.transpose(2, 1) # [B, D', N]
        proj_features = utils.network_utils.var_or_cuda(proj_features)
        
        # concat the img features after each point features
        refine_features = torch.cat((pc_features, proj_features, noises), 1)  # [B, D+D'+n_length, N]
        
        refine_features = F.relu(self.bn1(self.conv1(refine_features)))
        refine_features = F.relu(self.bn2(self.conv2(refine_features)))
        refine_features = F.relu(self.bn3(self.conv3(refine_features)))
        refine_features = F.relu(self.bn4(self.conv4(refine_features)))
        refine_features = F.relu(self.bn5(self.conv5(refine_features)))
        displacements = self.conv6(refine_features)

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

    def __init__(self, cfg, in_channels, activation=nn.ReLU(), optimizer=None):
        super().__init__()
        self.cfg = cfg
        
        self.img_enc = CNN18Encoder(in_channels, activation)
        self.transform_pc = TransformPC(cfg)
        self.feature_projection = FeatureProjection(cfg)
        self.pointnet2 = PointNet2(cfg)
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
            self.transform_pc = torch.nn.DataParallel(self.transform_pc, device_ids=cfg.CONST.DEVICE).cuda()
            self.feature_projection = torch.nn.DataParallel(self.feature_projection, device_ids=cfg.CONST.DEVICE).cuda()
            self.pointnet2 = torch.nn.DataParallel(self.pointnet2, device_ids=cfg.CONST.DEVICE).cuda()
            self.displacement_net = torch.nn.DataParallel(self.displacement_net, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

    def forward(self, img, update_id, xyz, view_az, view_el):
        img_features = self.img_enc(img)
        transform_xyz = self.transform_pc(xyz, view_az, view_el, update_id)
        proj_features = self.feature_projection(img_features, transform_xyz)
        pc_features = self.pointnet2(transform_xyz)
        noises = torch.normal(mean=0.0, std=1, size=(self.cfg.CONST.BATCH_SIZE, self.cfg.CONST.NUM_POINTS, self.cfg.UPDATER.NOISE_LENGTH))
        displacements = self.displacement_net(proj_features, pc_features, noises)
        displacements = displacements.transpose(2, 1)
        refine_pc = xyz + displacements

        return refine_pc

    def loss(self, img, update_id, xyz, gt_pc, view_az, view_el, update_proj_gt):
        refine_pc = self(img, update_id, xyz, view_az, view_el)

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

    def learn(self, img, update_id, xyz, gt_pc, view_az, view_el, update_proj_gt):
        self.train(True)
        self.optimizer.zero_grad()
        total_loss, loss_2d, loss_3d, _ = self.loss(img, update_id, xyz, gt_pc, view_az, view_el, update_proj_gt)
        total_loss.backward()
        self.optimizer.step()
        total_loss_np = total_loss.detach().item()
        loss_2d_np = loss_2d.detach().item()
        loss_3d_np = loss_3d.detach().item()
        del total_loss, loss_2d, loss_3d
        return total_loss_np, loss_2d_np, loss_3d_np
