# 176    39f5eecbfb2470846666a748bda83f67 
# 41753  a58f8f1bd61094b3ff2c92c2a4f65876
# 2603   27c00ec2b6ec279958e80128fd34c2b1
# 37247  484f0070df7d5375492d9da2668ec34c
# 36881  4231883e92a3c1a21c62d11641ffbd35

import json
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data

from shutil import copyfile
import cv2
from datetime import datetime as dt
from collections import OrderedDict

import utils.point_cloud_visualization_old
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from models.graphx_rec import Graphx_Rec
from models.networks_graphx_refine import GRAPHX_REFINE_MODEL

from pyntcloud import PyntCloud


def evaluate_refine_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    eval_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ToTensor(),
    ])

    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    eval_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
                                                   utils.data_loaders.DatasetType.TEST,  eval_transforms),
                                                   batch_size=cfg.EVALUATE.BATCH_SIZE,
                                                   num_workers=1,
                                                   shuffle=False)
                                                   
    # Set up networks
    # The parameters here need to be set in cfg
    rec_net = Graphx_Rec(
        cfg=cfg,
        in_channels=3,
        in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
        activation=nn.ReLU(),
    )
        
    # Refine network
    refine_net = GRAPHX_REFINE_MODEL(
        cfg=cfg,
        in_channels=3,
        optimizer=lambda x: torch.optim.Adam(x, lr=cfg.REFINE.LEARNING_RATE)
    )
    

    if torch.cuda.is_available():
        rec_net = torch.nn.DataParallel(rec_net, device_ids=cfg.CONST.DEVICE).cuda()
        refine_net = torch.nn.DataParallel(refine_net, device_ids=cfg.CONST.DEVICE).cuda()
    
    # Load weight
    # Load pretrained generator
    print('[INFO] %s Recovering generator from %s ...' % (dt.now(), cfg.EVALUATE.GENERATOR_WEIGHTS))
    rec_net_dict = rec_net.state_dict()
    pretrained_dict = torch.load(cfg.EVALUATE.GENERATOR_WEIGHTS)
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

    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Recovering refiner from %s ...' % (dt.now(), cfg.EVALUATE.REFINER_WEIGHTS))
    refine_checkpoint = torch.load(cfg.EVALUATE.REFINER_WEIGHTS)
    refine_net.load_state_dict(refine_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % refine_checkpoint['epoch_idx'])
    epoch_id = int(refine_checkpoint['epoch_idx'])
    
    rec_net.eval()
    refine_net.eval()

    # Testing loop
    for sample_idx, (taxonomy_names, sample_names, rendering_images, update_images,
                     model_x, model_y,
                     init_point_clouds, ground_truth_point_clouds) in enumerate(eval_data_loader):
        
        print("evaluate sample: ", sample_idx)
        with torch.no_grad():
            # Only one image per sample
            rendering_images = torch.squeeze(rendering_images, 1)
            update_images = torch.squeeze(update_images, 1)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            update_images = utils.network_utils.var_or_cuda(update_images)
            model_x = utils.network_utils.var_or_cuda(model_x)
            model_y = utils.network_utils.var_or_cuda(model_y)
            init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
            
            #=================================================#
            #                Test the network                 #
            #=================================================#
            # rec net give out a coarse point cloud
            coarse_pc, _ = rec_net(rendering_images, init_point_clouds)
            # refine net give out a refine result
            loss, pred_pc = refine_net.module.valid_step(update_images, coarse_pc, ground_truth_point_clouds, model_x, model_y)

            img_dir = cfg.EVALUATE.OUTPUT_FOLDER

            azi = model_x[0].detach().cpu().numpy()*180./np.pi
            ele = model_y[0].detach().cpu().numpy()*180./np.pi + 90.
            
            sample_name = sample_names[0]
            taxonomy_name = taxonomy_names[0]
            
            src_sketch_img_path = os.path.join(cfg.DATASETS.SHAPENET.RENDERING_PATH % (taxonomy_name, sample_name, 1))
            src_sketch_empty_img_path = os.path.join(cfg.DATASETS.SHAPENET.UPDATE_PATH % (taxonomy_name, sample_name, 1))
            
            inputs_path = os.path.join(img_dir, str(sample_idx), 'inputs')
            if not os.path.exists(inputs_path):
                os.makedirs(inputs_path)

            # copy the config file
            copyfile(src_sketch_img_path, os.path.join(inputs_path, str(sample_idx) + '_sketch.png'))
            copyfile(src_sketch_empty_img_path, os.path.join(inputs_path, str(sample_idx) + '_empty_sketch.png'))

            # Predict Pointcloud
            p_pc = pred_pc[0].detach().cpu().numpy()
            rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(p_pc, 
                                                                                        os.path.join(img_dir, str(sample_idx), 'rec results'),
                                                                                        sample_idx,
                                                                                        cfg.EVALUATE.VERSION_ID,
                                                                                        "",
                                                                                        view=[azi, ele])
            
            # Groundtruth Pointcloud
            gt_pc = ground_truth_point_clouds[0].detach().cpu().numpy()
            rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(gt_pc,
                                                                                        os.path.join(img_dir, str(sample_idx), 'gt'),
                                                                                        sample_idx,
                                                                                        cfg.EVALUATE.VERSION_ID,
                                                                                        "",
                                                                                        view=[azi, ele])
            
            if sample_idx == 200:
                break

