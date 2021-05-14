# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os, sys
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2

from shutil import copyfile

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.networks_psgn import Pixel2Pointcloud_PSGN_FC
from models.networks_graphx import Pixel2Pointcloud_GRAPHX
from models.updater_multi_scale import Updater

from pyntcloud import PyntCloud

def evaluate_multi_view_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    eval_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    # The parameters here need to be set in cfg
    if cfg.NETWORK.REC_MODEL == 'GRAPHX':
        net = Pixel2Pointcloud_GRAPHX(cfg=cfg,
                                      in_channels=3, 
                                      in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
                                      optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.GRAPHX_LEARNING_RATE, weight_decay=cfg.TRAIN.GRAPHX_WEIGHT_DECAY),
                                      scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                                      use_graphx=cfg.GRAPHX.USE_GRAPHX)
    
    update_net = Updater(cfg=cfg,
                         in_channels=3,
                         optimizer=lambda x: torch.optim.Adam(x, lr=cfg.UPDATER.LEARNING_RATE))

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        update_net = torch.nn.DataParallel(update_net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE_MULTI_VIEW.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.EVALUATE_MULTI_VIEW.RECONSTRUCTION_WEIGHTS)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    rec_epoch_id = int(rec_checkpoint['epoch_idx'])

    # Load weight for updater
    print('[INFO] %s Loading update weights from %s ...' % (dt.now(), cfg.EVALUATE_MULTI_VIEW.UPDATE_WEIGHTS))
    update_checkpoint = torch.load(cfg.EVALUATE_MULTI_VIEW.UPDATE_WEIGHTS)
    update_net.load_state_dict(update_checkpoint['net'])
    print('[INFO] Best update result at epoch %d ...' % update_checkpoint['epoch_idx'])
    update_epoch_id = int(update_checkpoint['epoch_idx'])

    net = net.eval()
    update_net = update_net.eval()
    
    # get input img
    input_img_path = cfg.EVALUATE_MULTI_VIEW.INPUT_IMAGE_PATH
    input_img = get_img(input_img_path)
    input_img = eval_transforms(rendering_images=input_img)
    input_img = utils.network_utils.var_or_cuda(input_img)

    # get update imgs
    update_img_folder = cfg.EVALUATE_MULTI_VIEW.UPDATE_IMAGE_FOLDER
    update_imgs = []

    for image_file in os.listdir(update_img_folder):
        update_img_path = os.path.join(update_img_folder, image_file)
        update_img = get_img(update_img_path)
        update_img = eval_transforms(rendering_images=update_img)
        update_img = utils.network_utils.var_or_cuda(update_img)
        update_imgs.append(update_img)
    
    # load init point clouds
    init_pc_np = init_pointcloud_loader(cfg.GRAPHX.NUM_INIT_POINTS)
    init_pc_batch = np.array([init_pc_np])
    init_pc_batch = torch.from_numpy(init_pc_batch)
    init_pc_batch = utils.network_utils.var_or_cuda(init_pc_batch)
    
    out_dir = cfg.EVALUATE_MULTI_VIEW.OUT_DIR
    # run rec
    with torch.no_grad():
        rec_pc = net(input_img, init_pc_batch)
        output_point_cloud_ply(rec_pc, names=['rec'], pc_type='eval', output_dir=out_dir, foldername='ply')
    
    view_az = []
    view_el = []
    radian_x, radian_y = get_view()
    view_az.append(radian_x)
    view_el.append(radian_y)
    view_az = np.array(view_az).astype(np.float32)
    view_el = np.array(view_el).astype(np.float32)
    view_az = torch.from_numpy(view_az)
    view_el = torch.from_numpy(view_el)
    view_az = utils.network_utils.var_or_cuda(view_az)
    view_el = utils.network_utils.var_or_cuda(view_el)

    ids = [0, 1, 2]

    current_pc = rec_pc
    for i, update_img in enumerate(update_imgs):
        with torch.no_grad():
            update_id = []
            update_id.append(ids[i])
            current_pc = update_net(update_img, update_id, current_pc, view_az, view_el)
            output_point_cloud_ply(current_pc, names=['update' + str(i)], pc_type='eval', output_dir=out_dir, foldername='ply')


def get_view():
    angles = [
        [0, 0],
        [45, 0],
        [90, 0],
        [135, 0],
        [180, 0],
        [225, 0],
        [270, 0],
        [325, 0]
    ]

    radian_x = [] # azi
    radian_y = [] # ele
    for angle in angles:
        angle_x = float(angle[0])
        angle_y = float(angle[1])
        # convert angles to radians
        radian_x.append(angle_x*np.pi/180.)
        radian_y.append((angle_y - 90.)*np.pi/180.) # original model face direction: z, change to x
    
    return radian_x, radian_y


def get_update_id(update_id_path):
    with open(update_id_path) as f:
        ids = f.readlines()

    ids = [int(x) for x in ids]
    
    return ids


def get_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_batch = []
    img_batch.append(img)
    img_batch = np.array(img_batch).astype(np.float32)

    return img_batch


def get_point_cloud(point_cloud_file):
    # get data of point cloud
    _, suffix = os.path.splitext(point_cloud_file)

    if suffix == '.ply':
        point_cloud = PyntCloud.from_file(point_cloud_file)
        point_cloud = np.array(point_cloud.points)
    elif suffix == '.npy':
        point_cloud = np.load(point_cloud_file)

    return point_cloud


def output_point_cloud_ply(xyzs, names, pc_type, output_dir, foldername ):

    if not os.path.exists( output_dir ):
        os.mkdir(  output_dir  )

    plydir = output_dir + '/' + foldername

    if not os.path.exists( plydir ):
        os.mkdir( plydir )

    numFiles = len(names)

    for fid in range(numFiles):

        print('write: ' + plydir +'/'+names[fid] + pc_type +'.ply')

        with open( plydir +'/'+names[fid] + pc_type +'.ply', 'w') as f:
            pn = xyzs.shape[1]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f\n' % (xyzs[fid][i][0],  xyzs[fid][i][1],  xyzs[fid][i][2]) )


def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h = np.random.uniform(10., 214., size=(num_points,))
    w = np.random.uniform(10., 214., size=(num_points,))
    X = (w - 111.5) / 248. * -Z
    Y = (h - 111.5) / 248. * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')
