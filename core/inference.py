import json
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data

from datetime import datetime as dt
from time import time
from collections import OrderedDict
from shutil import copyfile
import cv2

from models.graphx_rec import Graphx_Rec

import utils.data_transforms
from utils.point_cloud_utils import output_point_cloud_ply

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

def get_green_mask(mask_img_path):
    seg = cv2.imread(mask_img_path)
    hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)
    
    # For arm
    green_min = 35
    green_max = 77
    
    lower_green = np.array([green_min, 100, 100]) 
    upper_green = np.array([green_max, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    return mask

def inference_net(cfg, upload_image=False):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    inference_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    # The parameters here need to be set in cfg
    rec_net = Graphx_Rec(
        cfg=cfg,
        in_channels=3,
        in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
        activation=nn.ReLU(),
    )
    
    if torch.cuda.is_available():
        rec_net = torch.nn.DataParallel(rec_net, device_ids=cfg.CONST.DEVICE).cuda()

    # Load pretrained generator
    print('[INFO] %s Recovering generator from %s ...' % (dt.now(), cfg.INFERENCE.GENERATOR_WEIGHTS))
    rec_net_dict = rec_net.state_dict()
    pretrained_dict = torch.load(cfg.INFERENCE.GENERATOR_WEIGHTS)
    pretrained_weight_dict = pretrained_dict['net']
    new_weight_dict = OrderedDict()
    for k, v in pretrained_weight_dict.items():
        if cfg.REFINE.GENERATOR_TYPE == 'REC':
            name = k[21:] # remove module.reconstructor.
        elif cfg.REFINE.GENERATOR_TYPE == 'GAN':
            name = k[15:] # remove module.model_G.
        if name in rec_net_dict:
            new_weight_dict[name] = v
    rec_net_dict.update(new_weight_dict)
    rec_net.load_state_dict(rec_net_dict)
    rec_net.eval()
    
    input_image = []

    if upload_image:
        # upload image from the file system
        # TO DO
        raise NotImplementedError
        
    else:
        # user draw image on the UI
        screenshot_image_mask = get_green_mask(cfg.INFERENCE.SCREENSHOT_IMAGE_PATH)
        screenshot_image_mask = cv2.bitwise_not(screenshot_image_mask)
        screenshot_image_mask = cv2.resize(screenshot_image_mask, (224, 224), interpolation=cv2.INTER_AREA)
        
        # convert all the pixel > 0 to 255
        screenshot_image_mask[screenshot_image_mask<255] = 0
        sketch_image = cv2.cvtColor(screenshot_image_mask, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(cfg.INFERENCE.SKETCH_IMAGE_PATH, sketch_image)
        input_image_path = cfg.INFERENCE.SKETCH_IMAGE_PATH

    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    samples = []
    samples.append(input_image)
    samples = np.array(samples).astype(np.float32)
    input_images = inference_transforms(rendering_images=samples)
    
    # load init point clouds
    init_point_cloud_np = init_pointcloud_loader(cfg.GRAPHX.NUM_INIT_POINTS)
    init_point_clouds = np.array([init_point_cloud_np])
    init_point_clouds = torch.from_numpy(init_point_clouds)

    with torch.no_grad():
        # generate rough point cloud
        coarse_pc, _ = rec_net(input_images, init_point_clouds)
        output_point_cloud_ply(coarse_pc.detach().cpu().numpy(), ['coarse'], cfg.INFERENCE.CACHE_POINT_CLOUD_PATH)
        return coarse_pc[0].detach().cpu().numpy()
