import json
import numpy as np
import os, sys
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2
from datetime import datetime as dt
from shutil import copyfile

from models.networks_graphx_gan import GRAPHX_GAN_MODEL

import utils.point_cloud_visualization_old
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from utils.point_cloud_utils import output_point_cloud_ply

from pyntcloud import PyntCloud


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


def evaluate_gan_net(cfg):
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
    net = GRAPHX_GAN_MODEL(
        cfg=cfg,
        optimizer_G=lambda x: torch.optim.Adam(x, 
                                               lr=cfg.TRAIN.GENERATOR_LEARNING_RATE, 
                                               weight_decay=cfg.TRAIN.GENERATOR_WEIGHT_DECAY, 
                                               betas=cfg.TRAIN.BETAS),
        scheduler_G=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
        optimizer_D=lambda x: torch.optim.Adam(x, 
                                               lr=cfg.TRAIN.DISCRIMINATOR_LEARNINF_RATE,
                                               weight_decay=cfg.TRAIN.DISCRIMINATOR_WEIGHT_DECAY,
                                               betas=cfg.TRAIN.BETAS),
        scheduler_D=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)
    )
    

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE.WEIGHT_PATH))
    rec_checkpoint = torch.load(cfg.EVALUATE.WEIGHT_PATH)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    epoch_id = int(rec_checkpoint['epoch_idx'])
    
    net.eval()

    # Testing loop
    for sample_idx, (taxonomy_names, sample_names, rendering_images,
                    model_azi, model_ele,
                    init_point_clouds, ground_truth_point_clouds) in enumerate(eval_data_loader):
        
        print("evaluate sample: ", sample_idx)
        with torch.no_grad():
            # Only one image per sample
            rendering_images = torch.squeeze(rendering_images, 1)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            model_azi = utils.network_utils.var_or_cuda(model_azi)
            model_ele = utils.network_utils.var_or_cuda(model_ele)
            init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)

            loss, pred_pc = net.module.valid_step(rendering_images, init_point_clouds, ground_truth_point_clouds)

            azi = model_azi[0].detach().cpu().numpy()*180./np.pi
            ele = model_ele[0].detach().cpu().numpy()*180./np.pi + 90.
            
            taxonomy_name = taxonomy_names[0]
            sample_name = sample_names[0]
            
            # Save sample dir
            output_dir = cfg.EVALUATE.OUTPUT_FOLDER
            sample_dir = os.path.join(output_dir, str(sample_idx))
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            sample_name_dir = os.path.join(sample_dir, sample_name)
            if not os.path.exists(sample_name_dir):
                os.makedirs(sample_name_dir)
    
            # Rendering image
            sketch_path = os.path.join(sample_name_dir, 'sketch')
            if not os.path.exists(sketch_path):
                os.makedirs(sketch_path)
            src_sketch_img_path = os.path.join(cfg.DATASETS.SHAPENET.RENDERING_PATH % (taxonomy_name, sample_name, 1))
            copyfile(src_sketch_img_path, os.path.join(sketch_path, 'sketch.png'))

            # Predict Pointcloud
            p_pc = pred_pc[0].detach().cpu().numpy()
            rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(p_pc, 
                                                                                        os.path.join(sample_name_dir, 'rec results'),
                                                                                        sample_idx,
                                                                                        cfg.EVALUATE.VERSION_ID,
                                                                                        "",
                                                                                        view=[azi, ele])

            # Groundtruth Pointcloud
            gt_pc = ground_truth_point_clouds[0].detach().cpu().numpy()
            rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(gt_pc,
                                                                                        os.path.join(sample_name_dir, 'gt'),
                                                                                        sample_idx,
                                                                                        cfg.EVALUATE.VERSION_ID,
                                                                                        "",
                                                                                        view=[azi, ele])
            
            # save ply file
            pc_dir = os.path.join(sample_name_dir, 'pc')
            if not os.path.exists(pc_dir):
                os.makedirs(pc_dir)
            output_point_cloud_ply(np.array([p_pc]), ['pred_pc'], pc_dir)
            
            if sample_idx == 200:
                break

