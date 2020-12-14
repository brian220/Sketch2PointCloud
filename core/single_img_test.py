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

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.STN import STN

def single_img_test_net(cfg,
                        epoch_idx=-1,
                        output_dir=None,
                        encoder=None,
                        decoder=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    img_output_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/single_img_test/'
    
    img_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/rendering/00.png'

    # load image
    # no 1
    # img_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/rendering/01.png'

    # no 2
    # img_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/03001627/1a74a83fa6d24b3cacd67ce2c72c02e/rendering/01.png'
    
    # no 3
    # img_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/03001627/2bf05f8a84f0a6f33002761e7a3ba3bd/rendering/01.png'

    # no 4
    # img_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/03001627/1ab4c6ef68073113cf004563556ddb36/rendering/01.png'

    img_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    sample = np.array([img_np])
 
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
 
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    rendering_images = test_transforms(rendering_images=sample)
    print(rendering_images.size())

    # Set up networks
    if decoder is None or encoder is None or stn is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        stn = STN(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            stn = torch.nn.DataParallel(stn).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        stn.load_state_dict(checkpoint['stn_state_dict'])

    
    # inference model
    with torch.no_grad():
        # test the transformation matrix (STN) 
        transformation = stn(rendering_images)

        image_features = encoder(rendering_images)
        tree = [image_features]
        generated_point_cloud = decoder(tree)

        normalized_point_cloud = torch.bmm(generated_point_cloud, transformation)

        print(generated_point_cloud.size())
    
    print(transformation.detach().cpu().numpy())

    g_pc = generated_point_cloud.detach().cpu().numpy()
    g_pc = g_pc.squeeze(0)
    rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc, img_output_path,
                                                                            epoch_idx, "reconstruction")
    
    n_pc = normalized_point_cloud.detach().cpu().numpy()
    n_pc = n_pc.squeeze(0)
    rendering_views = utils.point_cloud_visualization.get_point_cloud_image(n_pc, img_output_path,
                                                                            epoch_idx, "normalized")
