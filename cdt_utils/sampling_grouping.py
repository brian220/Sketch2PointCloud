import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
# https://blog.csdn.net/weixin_39373480/article/details/88934146


def square_distance(src, dst):
    """
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2     
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
    ----------
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
    ----------
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
    ----------
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
    ----------
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
    ----------
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1,...DN]
    Return:
    ----------
        new_points:, indexed points data, [B, D1,...DN, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
    ----------
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
    ----------
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
    ----------
        npoint: Number of point for FPS
        radius: Radius of ball query
        nsample: Number of point for each ball query
        xyz: Old feature of points position data, [B, N, C]
        points: New feature of points data, [B, N, D]
    Return:
    ----------
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, C+D ,npoint, nsample]
    """
    B, N, C = xyz.shape
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    raw_grouped_xyz = index_points(xyz, idx)
    grouped_xyz = raw_grouped_xyz - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([raw_grouped_xyz, grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = torch.cat([raw_grouped_xyz, grouped_xyz], dim=-1)
    return new_xyz, new_points.permute(0, 3, 1, 2)


def sample_and_group_all(xyz, points):
    """
    Input:
    ----------
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
    ----------
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, C+D, 1, N]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points.permute(0, 3, 1, 2)

