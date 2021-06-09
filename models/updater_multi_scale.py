import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models
import os

import utils.network_utils
from utils.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from models.graphx import PointCloudGraphXDecoder
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


class EdgeRes(nn.Module):
    """
    input:
    - inp: b x num_dims x num_points
    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(EdgeRes, self).__init__()
        self.k = 8
        self.conv1 = torch.nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv4 = torch.nn.Conv2d(2176, 512, kernel_size=1, bias=False)
        self.conv5 = torch.nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv6 = torch.nn.Conv2d(512, 128, kernel_size=1, bias=False)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer(channel=64)
            self.se2 = SELayer(channel=128)
            self.se4 = SELayer(channel=512)
            self.se5 = SELayer(channel=256)
            self.se6 = SELayer(channel=128)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(1024)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.se1(self.bn1(self.conv1(x))))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.bn1(self.conv1(x)))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.bn2(self.conv2(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]

        x = get_graph_feature(x, k=self.k)  # [bs, 256, num_points, k]
        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 1024, num_points]

        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]

        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.bn5(self.conv5(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.bn6(self.conv6(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        return x

class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)
    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def knn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices

    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k: int = 20, idx=None):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    - idx: neighbor_idx
    outputs:
    - feature: b x npoints1 x (num_dims*2)
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # edge (neighbor - point)
    return feature


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

        displacements = F.sigmoid(displacements) * self.cfg.UPDATER.RANGE_MAX * 2 - self.cfg.UPDATER.RANGE_MAX
        
        return displacements


class GraphxDisplacementNet(nn.Module):
    """
    Predict the displacement from pointcloud features and image features

    Input:
        transform_xyz: pointcloud xyz [B, N, 3]
        pc_features: poincloud features from pointnet2 [B, D, N]
        proj_features: image features from feature projection [B, N, D']
        noises: noises vector [B, N, n_length]

    Output:
        displacement: perpoint displacement [B, C, N]

    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        deform_net = PointCloudGraphXDecoder
        self.graphx = deform_net(in_features=1123, in_instances=cfg.GRAPHX.NUM_INIT_POINTS, activation=nn.ReLU())

    def forward(self, transform_xyz, proj_features, pc_features, noises):
        noises = utils.network_utils.var_or_cuda(noises)
        proj_features = utils.network_utils.var_or_cuda(proj_features)
        pc_features =  pc_features.transpose(2, 1) # [B, N, D]
        refine_features = torch.cat((transform_xyz, pc_features, proj_features, noises), 2)  # [B, N, 3+D+D'+n_length]
        
        displacements = self.graphx(refine_features)
        displacements = displacements.transpose(2, 1)
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

        if cfg.UPDATER.PC_ENCODE_MODULE == 'Pointnet++':
            self.pc_encode = PointNet2(cfg)
        elif cfg.UPDATER.PC_ENCODE_MODULE == 'EdgeRes':
            self.pc_encode = EdgeRes(use_SElayer=True)
        
        if cfg.UPDATER.PC_DECODE_MODULE == 'Linear':
            self.displacement_net = LinearDisplacementNet(cfg)
        elif cfg.UPDATER.PC_DECODE_MODULE == 'Graphx':
            self.displacement_net = GraphxDisplacementNet(cfg)
        
        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        
        # emd loss
        self.emd = EMD()

        if torch.cuda.is_available():
            self.img_enc = torch.nn.DataParallel(self.img_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.transform_pc = torch.nn.DataParallel(self.transform_pc, device_ids=cfg.CONST.DEVICE).cuda()
            self.feature_projection = torch.nn.DataParallel(self.feature_projection, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc_encode = torch.nn.DataParallel(self.pc_encode, device_ids=cfg.CONST.DEVICE).cuda()
            self.displacement_net = torch.nn.DataParallel(self.displacement_net, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

    def forward(self, img, xyz, view_az, view_el):
        img_features = self.img_enc(img)
        transform_xyz = self.transform_pc(xyz, view_az, view_el)
        proj_features = self.feature_projection(img_features, transform_xyz)
        pc_features = self.pc_encode(transform_xyz)
        noises = torch.normal(mean=0.0, std=1, size=(self.cfg.CONST.BATCH_SIZE, self.cfg.CONST.NUM_POINTS, self.cfg.UPDATER.NOISE_LENGTH))
        displacements = self.displacement_net(transform_xyz, proj_features, pc_features, noises)
        displacements = displacements.transpose(2, 1)
        refine_pc = xyz + displacements

        return refine_pc

    def loss(self, img, xyz, gt_pc, view_az, view_el):
        refine_pc = self(img, xyz, view_az, view_el)
        
        # EMD
        loss = torch.mean(self.emd(refine_pc, gt_pc))

        return loss, refine_pc

    def learn(self, img, xyz, gt_pc, view_az, view_el):
        self.train(True)
        self.optimizer.zero_grad()
        loss, _ = self.loss(img, xyz, gt_pc, view_az, view_el)
        loss.backward()
        self.optimizer.step()
        loss_np = loss.detach().item()

        del loss
        return loss_np
