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

from pyntcloud import PyntCloud

def evaluate_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    eval_transforms = utils.data_transforms.Compose([
        # utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        # utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    # The parameters here need to be set in cfg
    if cfg.NETWORK.REC_MODEL == 'GRAPHX':
        net = Pixel2Pointcloud_GRAPHX(cfg=cfg,
                                      in_channels=1, 
                                      in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
                                      optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.GRAPHX_LEARNING_RATE, weight_decay=cfg.TRAIN.GRAPHX_WEIGHT_DECAY),
                                      scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                                      use_graphx=cfg.GRAPHX.USE_GRAPHX)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.EVALUATE.RECONSTRUCTION_WEIGHTS)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    epoch_id = int(rec_checkpoint['epoch_idx'])

    # evaluate the imgs in folder
    # evaluate first ten view
    lines = []
    with open(cfg.EVALUATE.INFO_FILE) as f:
        lines = f.readlines()
     
    for info in lines:
        info = info.split()
        eval_id = info[0]
        sample_name = info[1]
        view_id = int(info[2])
        
        print("Evaluate on img :", eval_id)

        # get img path
        input_img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH% (cfg.EVALUATE.TAXONOMY_ID, sample_name, view_id)

        # get gt pointcloud
        gt_point_cloud_file = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH % (cfg.EVALUATE.TAXONOMY_ID, sample_name)
        gt_point_cloud = get_point_cloud(gt_point_cloud_file)
        
        # evaluate single img
        evaluate_on_img(cfg, net, input_img_path, eval_transforms, eval_id, view_id, epoch_id, gt_point_cloud)


def get_point_cloud(point_cloud_file):
    # get data of point cloud
    _, suffix = os.path.splitext(point_cloud_file)

    if suffix == '.ply':
        point_cloud = PyntCloud.from_file(point_cloud_file)
        point_cloud = np.array(point_cloud.points)
    elif suffix == '.npy':
        point_cloud = np.load(point_cloud_file)

    return point_cloud


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


def evaluate_on_img(cfg, net, input_img_path, eval_transforms, eval_id, view_id, epoch_id, gt_point_cloud):
    # load img
    sample = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

    if cfg.NETWORK.REC_MODEL == 'PSGN_FC':
        sample = cv2.resize(sample, (self.grid_h, self.grid_w))
    elif cfg.NETWORK.REC_MODEL == 'GRAPHX':
        sample = cv2.resize(sample, (224, 224))
    else:
        print('Invalid model name, please check the config.py (NET_WORK.REC_MODEL)')
        sys.exit(2)

    sample = np.expand_dims(sample, -1)
    samples = []
    samples.append(sample)
    samples = np.array(samples).astype(np.float32) 
    rendering_images = eval_transforms(rendering_images=samples)

    # load init point clouds
    init_point_cloud_np = init_pointcloud_loader(cfg.GRAPHX.NUM_INIT_POINTS)
    init_point_clouds = np.array([init_point_cloud_np])
    init_point_clouds = torch.from_numpy(init_point_clouds)

    # load gt pointclouds
    ground_truth_point_clouds = np.array([gt_point_cloud])
    ground_truth_point_clouds = torch.from_numpy(ground_truth_point_clouds)

    # inference model
    with torch.no_grad():
        # Get data from data loader
        input_imgs = utils.network_utils.var_or_cuda(rendering_images)
        init_pc = utils.network_utils.var_or_cuda(init_point_clouds)
        gt_pc = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
        
        #=================================================#
        #           Evaluate the encoder, decoder         #
        #=================================================#
        pred_pc = net(input_imgs, init_pc)

        # Save a copy of image
        evaluate_img_dir = os.path.join(cfg.EVALUATE.INPUT_IMAGE_FOLDER)
        if not os.path.exists(evaluate_img_dir):
            os.mkdir(evaluate_img_dir)
        copyfile(input_img_path, os.path.join(evaluate_img_dir, str(view_id) + '.png'))
        

        # Predict Pointcloud
        g_pc = pred_pc[0].detach().cpu().numpy()
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc,
                                                                                os.path.join(cfg.EVALUATE.OUTPUT_FOLDER, 'reconstruction'),
                                                                                view_id, epoch_id, "reconstruction")
                
        # Groundtruth Pointcloud
        gt_pc = gt_point_cloud
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(gt_pc, 
                                                                                os.path.join(cfg.EVALUATE.OUTPUT_FOLDER, 'ground truth'),
                                                                                view_id, epoch_id, "ground truth")
