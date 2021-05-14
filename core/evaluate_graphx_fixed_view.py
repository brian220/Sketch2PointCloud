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

from models.networks import Pixel2Pointcloud

from pyntcloud import PyntCloud

def evaluate_fixed_view_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    eval_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    # Set up networks
    # The parameters here need to be set in cfg
    net = Pixel2Pointcloud(3, cfg.GRAPHX.NUM_INIT_POINTS,
                          optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY),
                          scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                          use_graphx=cfg.GRAPHX.USE_GRAPHX)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    
    # load samples
    samples = []
    with open(cfg.EVALUATE_FIXED_VIEW.SAMPLE_FILE) as s_f:
        samples = s_f.readlines()

    # evaluate single img on fixed views
    views = []
    with open(cfg.EVALUATE_FIXED_VIEW.VIEW_FILE) as v_f:
        views = v_f.readlines()
    
    for sample in samples:
        sample =  sample.replace('\n','')
        print(sample)

        # create sample dir
        evaluate_sample_dir =  os.path.join(cfg.EVALUATE_FIXED_VIEW.RESULT_DIR, sample)
        if not os.path.exists(evaluate_sample_dir):
            os.mkdir(evaluate_sample_dir)

        # create input folder for sample
        evaluate_sample_input_dir = os.path.join(evaluate_sample_dir, "input_img")
        if not os.path.exists(evaluate_sample_input_dir):
            os.mkdir(evaluate_sample_input_dir)
        
        # create output folder for sample
        evaluate_sample_output_dir = os.path.join(evaluate_sample_dir, "output")
        if not os.path.exists(evaluate_sample_output_dir):
            os.mkdir(evaluate_sample_output_dir)
        
        # create gt dir
        evaluate_sample_output_gt_dir = os.path.join(evaluate_sample_output_dir, "gt")
        if not os.path.exists(evaluate_sample_output_gt_dir):
            os.mkdir(evaluate_sample_output_gt_dir)

        # get gt pointcloud
        gt_point_cloud_file = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH % (cfg.EVALUATE_FIXED_VIEW.TAXONOMY_ID, sample)
        gt_point_cloud = get_point_cloud(gt_point_cloud_file)
        
        # generate gt img
        for view_id, view in enumerate(views):
            fixed_view = view.split()
            fixed_view = [round(float(item)) if float(item) < 359.5 else 0 for item in fixed_view]
            fixed_view = np.array(fixed_view)
                
            # Predict Pointcloud
            rendering_views = utils.point_cloud_visualization.get_point_cloud_image(gt_point_cloud, 
                                                                                    evaluate_sample_output_gt_dir,
                                                                                    view_id, "ground truth", fixed_view)

        # generate fixed view output
        for img_id in range(0, 24):
            # get input img path
            input_img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (cfg.EVALUATE_FIXED_VIEW.TAXONOMY_ID , sample, img_id)
            
            # Save a copy of input image
            copyfile(input_img_path, os.path.join(evaluate_sample_input_dir, str(img_id) + '.png'))

            g_pc = evaluate_on_fixed_view_img(cfg,
                                              net,
                                              gt_point_cloud,
                                              input_img_path,
                                              eval_transforms,
                                              evaluate_sample_dir)

            evaluate_sample_output_img_dir = os.path.join(evaluate_sample_output_dir, str(img_id))
            if not os.path.exists(evaluate_sample_output_img_dir):
                os.mkdir(evaluate_sample_output_img_dir)
           
            for view_id, view in enumerate(views):
                fixed_view = view.split()
                fixed_view = [round(float(item)) if float(item) < 359.5 else 0 for item in fixed_view]
                fixed_view = np.array(fixed_view)
                
                # Predict Pointcloud
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc,
                                                                                        evaluate_sample_output_img_dir,
                                                                                        view_id, "reconstruction", fixed_view)

            
def get_point_cloud(point_cloud_file):
    # get data of point cloud
    _, suffix = os.path.splitext(point_cloud_file)
    if suffix == '.ply':
        point_cloud = PyntCloud.from_file(point_cloud_file)
        point_cloud = np.array(point_cloud.points)

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


def evaluate_on_fixed_view_img(cfg,
                               net,
                               gt_point_cloud,
                               input_img_path,
                               eval_transforms,
                               evaluate_sample_dir):
    # load img
    img_np = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    sample = np.array([img_np])
    rendering_images = eval_transforms(rendering_images=sample)

    # load init point clouds
    init_point_cloud_np = init_pointcloud_loader(cfg.GRAPHX.NUM_INIT_POINTS)
    init_point_clouds = np.array([init_point_cloud_np])
    init_point_clouds = torch.from_numpy(init_point_clouds)
    
    # load gt pointclouds
    ground_truth_point_clouds = np.array([gt_point_cloud])
    ground_truth_point_clouds = torch.from_numpy(ground_truth_point_clouds)
    
    # inference model
    with torch.no_grad():
        # Only one image per sample
        rendering_images = torch.squeeze(rendering_images, 1)
        
         # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
        ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
        
        emd_loss, generated_point_clouds = net.module.loss(rendering_images, init_point_clouds, ground_truth_point_clouds, 'mean')
    
    
    return generated_point_clouds[0].detach().cpu().numpy()

