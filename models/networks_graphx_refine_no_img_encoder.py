import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models
import os

import utils.network_utils
from utils.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

import cuda.emd.emd_module as emd

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
    
    def forward(self, xyz, az, el):
        batch_size = xyz.size(0)
        cam_xyz = self.world2cam(xyz, az, el, batch_size, N_PTS=self.n_pts)
        return cam_xyz

    def world2cam(self, xyz, az, el, batch_size, N_PTS=1024):
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
        
        tr_mat = torch.unsqueeze(torch.tensor([tx, ty, tz]), 0).repeat(batch_size, 1) # [B,3]
        tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
        tr_mat = tr_mat.permute(0, 2, 1) # [B,1,3]
        tr_mat = tr_mat.repeat(1, N_PTS, 1) # [B,N_PTS,3]
        tr_mat = utils.network_utils.var_or_cuda(tr_mat) # [B,N_PTS,3]

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


class LinearDisplacementNet(nn.Module):
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

    def forward(self, transform_xyz, proj_features, pc_features, noises):
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

        displacements = F.sigmoid(displacements) * self.cfg.REFINE.RANGE_MAX * 2 - self.cfg.REFINE.RANGE_MAX
        
        return displacements


class GRAPHX_REFINE_MODEL(nn.Module):
    """
    Refine the point cloud based on the input image

    Input:
        xyz: point cloud from reconstruction model

    Ouput:
        update_pc: updated point cloud
    """

    def __init__(self, cfg, in_channels, optimizer=None):
        super().__init__()
        self.cfg = cfg
        
        # Refinement
        self.transform_pc = TransformPC(cfg)
        self.feature_projection = FeatureProjection(cfg)
        self.pc_encode = PointNet2(cfg)
        self.displacement_net = LinearDisplacementNet(cfg)
        
        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        
        # emd loss
        self.emd_dist = emd.emdModule()

        if torch.cuda.is_available():
            self.transform_pc = torch.nn.DataParallel(self.transform_pc, device_ids=cfg.CONST.DEVICE).cuda()
            self.feature_projection = torch.nn.DataParallel(self.feature_projection, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc_encode = torch.nn.DataParallel(self.pc_encode, device_ids=cfg.CONST.DEVICE).cuda()
            self.displacement_net = torch.nn.DataParallel(self.displacement_net, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd_dist = torch.nn.DataParallel(self.emd_dist, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()
    
    def train_step(self, img_features, xyz, gt_pc, view_az, view_el):
        '''
        Input:
            img_features
            init pc:    [B, N, 3]
            gt pc:      [B, N, 3]
            view_az:    [B]
            view_el:    [B]

        Output:
            loss
            pred_pc:    [B, N, 3]
        '''
        refine_pc = self.refine(img_features, xyz, view_az, view_el)
        # compute reconstruction loss
        emd_loss, _ = self.emd_dist(
            refine_pc, gt_pc, eps=0.005, iters=50
        )
        rec_loss = torch.sqrt(emd_loss).mean(1).mean()

        self.refiner_backward(rec_loss)

        rec_loss_np = rec_loss.detach().item()

        return rec_loss_np*1000

    def valid_step(self, img_features, xyz, gt_pc, view_az, view_el):
        # refine the point cloud
        refine_pc = self.refine(img_features, xyz, view_az, view_el)
        # compute reconstruction loss
        emd_loss, _ = self.emd_dist(
            refine_pc, gt_pc, eps=0.005, iters=50
        )
        rec_loss = torch.sqrt(emd_loss).mean(1).mean()

        return rec_loss*1000, pred_pc

    def refine(self, img_features, xyz, view_az, view_el):
        # img_features = self.img_enc(img)
        transform_xyz = self.transform_pc(xyz, view_az, view_el)
        proj_features = self.feature_projection(img_features, transform_xyz)
        pc_features = self.pc_encode(transform_xyz)
        noises = torch.normal(mean=0.0, std=1, size=(self.cfg.CONST.BATCH_SIZE, self.cfg.CONST.NUM_POINTS, self.cfg.REFINE.NOISE_LENGTH))
        displacements = self.displacement_net(transform_xyz, proj_features, pc_features, noises)
        displacements = displacements.transpose(2, 1)
        refine_pc = xyz + displacements

        return refine_pc

    def refiner_backward(self, rec_loss):
        self.train(True)
        self.optimizer.zero_grad()
        rec_loss.backward()
        self.optimizer.step()

