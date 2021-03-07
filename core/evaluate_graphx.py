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
from models.view_encoder import Encoder
from models.view_estimater import ViewEstimater

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
    # Set up networks
    # The parameters here need to be set in cfg
    net = Pixel2Pointcloud(1, cfg.GRAPHX.NUM_INIT_POINTS,
                          optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY),
                          scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                          use_graphx=cfg.GRAPHX.USE_GRAPHX)
    
    view_encoder = Encoder(cfg)

    azi_classes, ele_classes = int(360 / cfg.CONST.BIN_SIZE), int(180 / cfg.CONST.BIN_SIZE)
    view_estimater = ViewEstimater(cfg, azi_classes=azi_classes, ele_classes=ele_classes)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        view_encoder = torch.nn.DataParallel(view_encoder).cuda()
        view_estimater = torch.nn.DataParallel(view_estimater).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.EVALUATE.RECONSTRUCTION_WEIGHTS)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])

    # Load weight for view encoder
    print('[INFO] %s Loading view estimation weights from %s ...' % (dt.now(), cfg.EVALUATE.VIEW_ENCODER_WEIGHTS))
    view_enc_checkpoint = torch.load(cfg.EVALUATE.VIEW_ENCODER_WEIGHTS)
    view_encoder.load_state_dict(view_enc_checkpoint['encoder_state_dict'])
    print('[INFO] Best view encode result at epoch %d ...' % view_enc_checkpoint['epoch_idx'])

    # Load weight for view estimater
    print('[INFO] %s Loading view estimation weights from %s ...' % (dt.now(), cfg.EVALUATE.VIEW_ESTIMATION_WEIGHTS))
    view_est_checkpoint = torch.load(cfg.EVALUATE.VIEW_ESTIMATION_WEIGHTS)
    view_estimater.load_state_dict(view_est_checkpoint['view_estimator_state_dict'])
    print('[INFO] Best view estimation result at epoch %d ...' % view_est_checkpoint['epoch_idx'])

    # evaluate the imgs in folder
    # evaluate first ten view
    lines = []
    with open(cfg.EVALUATE.INFO_FILE) as f:
        lines = f.readlines()
     
    for info in lines:
        info = info.split()
        eval_id = info[0]
        sample_name = info[1]
        # view_id = int(info[2])
        
        print("Evaluate on img :", eval_id)

        for view_id in range(0, 10):
            # get img path
            input_img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH% (cfg.EVALUATE.TAXONOMY_ID, sample_name, view_id)

            # get gt pointcloud
            gt_point_cloud_file = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH % (cfg.EVALUATE.TAXONOMY_ID, sample_name)
            gt_point_cloud = get_point_cloud(gt_point_cloud_file)

            # get gt view
            gt_view_file = cfg.DATASETS.SHAPENET.VIEW_PATH % (cfg.EVALUATE.TAXONOMY_ID, sample_name)
            gt_view = get_view(gt_view_file, view_id)
        
            # evaluate single img
            evaluate_on_img(cfg,
                            net, view_encoder, view_estimater,
                            input_img_path, view_id,
                            eval_transforms, eval_id,
                            gt_point_cloud, gt_view)


def get_point_cloud(point_cloud_file):
    # get data of point cloud
    _, suffix = os.path.splitext(point_cloud_file)

    if suffix == '.ply':
        point_cloud = PyntCloud.from_file(point_cloud_file)
        point_cloud = np.array(point_cloud.points)
    return point_cloud


def get_view(view_file, view_id):
    with open(view_file) as f:
        lines = f.readlines()
    
    view = lines[view_id]
    view = view.split()
    # get first two (azimuth, elevation)
    view = view[:2]
    view = [round(float(item)) if float(item) < 359.5 else 0 for item in view]
    return view


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


def evaluate_on_img(cfg, 
                    net, view_encoder, view_estimater,
                    input_img_path, view_id,
                    eval_transforms, eval_id,
                    gt_point_cloud, gt_view):
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
        
        #=================================================#
        #           Evaluate the encoder, decoder         #
        #=================================================#
        emd_loss, generated_point_clouds = net.module.loss(rendering_images, init_point_clouds, ground_truth_point_clouds, 'mean')

        #=================================================#
        #          Evaluate the view estimater            #
        #=================================================#
        vgg_features, _ = view_encoder(rendering_images)
        output = view_estimater(vgg_features)
        
        #=================================================#
        #              Get predict view                   #
        #=================================================#
        preds_cls = utils.view_pred_utils.get_pred_from_cls_output([output[0], output[1]])
            
        preds = []
        for n in range(len(preds_cls)):
            pred_delta = output[n + 2]
            delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds_cls[n].long()].tanh() / 2
            preds.append((preds_cls[n].float() + delta_value + 0.5) * cfg.CONST.BIN_SIZE)

        # Save a copy of image
        evaluate_img_dir = os.path.join(cfg.EVALUATE.INPUT_IMAGE_FOLDER, eval_id)
        if not os.path.exists(evaluate_img_dir):
            os.mkdir(evaluate_img_dir)
        copyfile(input_img_path, os.path.join(evaluate_img_dir, str(view_id) + '.png'))
        

        # Predict Pointcloud
        g_pc = generated_point_clouds[0].detach().cpu().numpy()
        pred_view = []
        pred_view.append(preds[0][0].detach().cpu().numpy())
        pred_view.append(preds[1][0].detach().cpu().numpy())
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc,
                                                                                os.path.join(cfg.EVALUATE.OUTPUT_FOLDER, 'reconstruction', eval_id),
                                                                                view_id, "reconstruction", pred_view)
                
        # Groundtruth Pointcloud
        gt_pc = gt_point_cloud
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(gt_pc, 
                                                                                os.path.join(cfg.EVALUATE.OUTPUT_FOLDER, 'ground truth', eval_id),
                                                                                view_id, "ground truth", gt_view)
