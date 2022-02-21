import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader

from Point2Skeleton.code.SkelPointNet import SkelPointNet
from Point2Skeleton.code.GraphAE import LinkPredNet
import Point2Skeleton.code.DistFunc as DF
import Point2Skeleton.code.FileRW as rw
import Point2Skeleton.code.MeshUtil as util
import Point2Skeleton.code.config as conf

import utils.network_utils

from datetime import datetime
from pyntcloud import PyntCloud
from utils.pointnet2_utils import index_points, farthest_point_sample

load_skelnet_path = '/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/Point2Skeleton/training-weights/weights-skelpoint.pth'
load_gae_path = '/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/Point2Skeleton/training-weights/weights-gae.pth'
save_result_path = '/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/Point2Skeleton/result/'
point_num = 2000
skelpoint_num = 100

def output_results(log_path, input_xyz, skel_xyz, skel_r, skel_faces, skel_edges, A_mesh):

    batch_size = skel_xyz.size()[0]
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    
    for i in range(batch_size):

        save_name_input = log_path + "input.off"
        save_name_sphere = log_path + "sphere" + ".obj"
        save_name_center = log_path + "center" + ".off"
        save_name_f = log_path + "skel_face" + ".obj"
        save_name_e = log_path + "skel_edge" + ".obj"
        save_name_A_mesh = log_path + "mesh_graph" + ".obj"

        rw.save_off_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_off_points(skel_xyz_save[i], save_name_center)
        rw.save_skel_mesh(skel_xyz_save[i], skel_faces[i], skel_edges[i], save_name_f, save_name_e)
        rw.save_graph(skel_xyz_save[i], A_mesh[i], save_name_A_mesh)
        
        # dense_skel_sphere = util.rand_sample_points_on_skeleton_mesh(skel_xyz_save[i], skel_faces[i], skel_edges[i], skel_r_save[i], 10000)
        # rw.save_spheres(dense_skel_sphere[:,0:3], dense_skel_sphere[:,3,None], save_name_sphere)


def predict_skeleton_net(cfg, input_pc):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    #load networks
    model_skel = SkelPointNet(num_skel_points=skelpoint_num, input_channels=0, use_xyz=True)
    model_gae = LinkPredNet()

    model_skel.cuda()
    model_skel.eval()
    model_gae.cuda()
    model_gae.eval()

    model_skel.load_state_dict(torch.load(load_skelnet_path))
    model_gae.load_state_dict(torch.load(load_gae_path))
    
    input_pc_np = np.array([input_pc]).astype(np.float32)
    input_pc = torch.from_numpy(input_pc_np)
    idx = farthest_point_sample(input_pc, point_num)
    input_pc = index_points(input_pc, idx).detach().data.numpy()[0]

    batch_pc_np = np.array([input_pc]).astype(np.float32)
    batch_pcs = torch.from_numpy(batch_pc_np)
    batch_pcs = utils.network_utils.var_or_cuda(batch_pcs)

    # get skeletal points and the node features
    skel_xyz, skel_r, sample_xyz, weights, shape_features, A_init, valid_mask, known_mask = model_skel(
        batch_pcs, compute_graph=True)
    skel_node_features = torch.cat([shape_features, skel_xyz, skel_r], 2)

    # get predicted mesh
    A_pred = model_gae(skel_node_features, A_init)
    A_final = model_gae.recover_A(A_pred, valid_mask)

    skel_faces, skel_edges, A_mesh = util.generate_skel_mesh(batch_pcs, skel_xyz, A_init, A_final)
    skel_r = util.refine_radius_by_mesh(skel_xyz, skel_r, sample_xyz, weights, skel_faces, skel_edges)
    output_results(save_result_path, batch_pcs, skel_xyz, skel_r, skel_faces, skel_edges, A_mesh)
    
    print('skeleton edges:')
    print(skel_edges)

    return skel_edges
