import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
import time
import math

import cdt_utils.pointnet2_utils as pointnet2_utils
from cdt_utils.sampling_grouping import sample_and_group, sample_and_group_all
from cdt_utils.pytorch_utils import SharedRSConv, GloAvgConv

class RSCNN_module_win(nn.Module):
    """Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of points
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int = None,
            radii: List[float] = [None],
            nsamples: List[int],
            mlps: List[List[int]],
            use_xyz: bool = True,
            bias = True,
            init = nn.init.kaiming_normal,
            first_layer = False,
            relation_prior = 1
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        self.radii = radii
        self.nsamples = nsamples
        # initialize shared mapping functions
        C_in = (mlps[0][0] + 3) if use_xyz else mlps[0][0]
        C_out = mlps[0][1]
        
        if relation_prior == 0:
            in_channels = 1
        elif relation_prior == 1 or relation_prior == 2:
            in_channels = 10
        else:
            assert False, "relation_prior can only be 0, 1, 2."
        
        if first_layer:
            mapping_func1 = nn.Conv2d(in_channels = in_channels, out_channels = math.floor(C_out / 2), kernel_size = (1, 1), 
                                      stride = (1, 1), bias = bias)
            mapping_func2 = nn.Conv2d(in_channels = math.floor(C_out / 2), out_channels = 16, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
            xyz_raising = nn.Conv2d(in_channels = C_in, out_channels = 16, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
            init(xyz_raising.weight)
            if bias:
                nn.init.constant(xyz_raising.bias, 0)
        elif npoint is not None:
            mapping_func1 = nn.Conv2d(in_channels = in_channels, out_channels = math.floor(C_out / 4), kernel_size = (1, 1), 
                                      stride = (1, 1), bias = bias)
            mapping_func2 = nn.Conv2d(in_channels = math.floor(C_out / 4), out_channels = C_in, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
        if npoint is not None:
            init(mapping_func1.weight)
            init(mapping_func2.weight)
            if bias:
                nn.init.constant(mapping_func1.bias, 0)
                nn.init.constant(mapping_func2.bias, 0)    
                     
            # channel raising mapping
            cr_mapping = nn.Conv1d(in_channels = C_in if not first_layer else 16, out_channels = C_out, kernel_size = 1, 
                                      stride = 1, bias = bias)
            init(cr_mapping.weight)
            nn.init.constant(cr_mapping.bias, 0)
        
        if first_layer:
            mapping = [mapping_func1, mapping_func2, cr_mapping, xyz_raising]
        elif npoint is not None:
            mapping = [mapping_func1, mapping_func2, cr_mapping]
        
        for i in range(len(radii)):
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if npoint is not None:
                self.mlps.append(SharedRSConv(mlp_spec, mapping = mapping, relation_prior = relation_prior, first_layer = first_layer))
            else:   # global convolutional pooling
                self.mlps.append(GloAvgConv(C_in = C_in, C_out = C_out))

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the points
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new points' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_points descriptors
        """

        new_features_list = []
        if features is not None:
            features = features.permute(0, 2, 1)
        for i in range(len(self.radii)):
            radius = self.radii[i]
            nsample = self.nsamples[i]
            new_xyz, new_features = sample_and_group(self.npoint, radius, nsample, xyz, features) if self.npoint is not None else sample_and_group_all(xyz, features)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
        
        return new_xyz, torch.cat(new_features_list, dim=1)

####################################################################################3
class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the points
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new points' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_points descriptors
        """

        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.npoint is not None:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)  # (B, npoint)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
            fps_idx = fps_idx.data
        else:
            new_xyz = None
            fps_idx = None
        
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features, fps_idx) if self.npoint is not None else self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
        
        return new_xyz, torch.cat(new_features_list, dim=1)


class RSCNN_module(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of points
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int = None,
            radii: List[float] = [None],
            nsamples: List[int],
            mlps: List[List[int]],
            use_xyz: bool = True,
            bias = True,
            init = nn.init.kaiming_normal,
            first_layer = False,
            relation_prior = 1
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        # initialize shared mapping functions
        C_in = (mlps[0][0] + 3) if use_xyz else mlps[0][0]
        C_out = mlps[0][1]
        
        if relation_prior == 0:
            in_channels = 1
        elif relation_prior == 1 or relation_prior == 2:
            in_channels = 10
        else:
            assert False, "relation_prior can only be 0, 1, 2."
        
        if first_layer:
            mapping_func1 = nn.Conv2d(in_channels = in_channels, out_channels = math.floor(C_out / 2), kernel_size = (1, 1), 
                                      stride = (1, 1), bias = bias)
            mapping_func2 = nn.Conv2d(in_channels = math.floor(C_out / 2), out_channels = 16, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
            xyz_raising = nn.Conv2d(in_channels = C_in, out_channels = 16, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
            init(xyz_raising.weight)
            if bias:
                nn.init.constant(xyz_raising.bias, 0)
        elif npoint is not None:
            mapping_func1 = nn.Conv2d(in_channels = in_channels, out_channels = math.floor(C_out / 4), kernel_size = (1, 1), 
                                      stride = (1, 1), bias = bias)
            mapping_func2 = nn.Conv2d(in_channels = math.floor(C_out / 4), out_channels = C_in, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
        if npoint is not None:
            init(mapping_func1.weight)
            init(mapping_func2.weight)
            if bias:
                nn.init.constant(mapping_func1.bias, 0)
                nn.init.constant(mapping_func2.bias, 0)    
                     
            # channel raising mapping
            cr_mapping = nn.Conv1d(in_channels = C_in if not first_layer else 16, out_channels = C_out, kernel_size = 1, 
                                      stride = 1, bias = bias)
            init(cr_mapping.weight)
            nn.init.constant(cr_mapping.bias, 0)
        
        if first_layer:
            mapping = [mapping_func1, mapping_func2, cr_mapping, xyz_raising]
        elif npoint is not None:
            mapping = [mapping_func1, mapping_func2, cr_mapping]
        
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if npoint is not None:
                self.mlps.append(SharedRSConv(mlp_spec, mapping = mapping, relation_prior = relation_prior, first_layer = first_layer))
            else:   # global convolutional pooling
                self.mlps.append(GloAvgConv(C_in = C_in, C_out = C_out))