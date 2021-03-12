# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
# https://github.com/hzxie/Pix2Vox
#

import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.utils

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import numpy as np
import cv2

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from datetime import datetime as dt

from models.networks_psgn import Pixel2Pointcloud_PSGN_FC
from models.networks_graphx import Pixel2Pointcloud_GRAPHX

from losses.chamfer_loss import ChamferLoss
from losses.earth_mover_distance import EMD
from utils.point_cloud_utils import Scale, Scale_one

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

def test_opt_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    test_transforms = utils.data_transforms.Compose([
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
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.TEST_OPT.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.TEST_OPT.RECONSTRUCTION_WEIGHTS)
    net.load_state_dict(rec_checkpoint['net'])
    epoch_idx = rec_checkpoint['epoch_idx']
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    
    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.TEST_OPT.OUT_PATH, '%s')
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    opt_writer = SummaryWriter(os.path.join(log_dir, 'opt'))

    input_img_path = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/capnet_data/data/ShapeNet_sketch/03001627/4231883e92a3c1a21c62d11641ffbd35/render_0.png'
    input_mask_path = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/capnet_data/data/ShapeNet_sketch/03001627/4231883e92a3c1a21c62d11641ffbd35/depth_0.png'
    init_point_cloud_np = init_pointcloud_loader(cfg.GRAPHX.NUM_INIT_POINTS)
    init_point_clouds = np.array([init_point_cloud_np])
    init_point_clouds = torch.from_numpy(init_point_clouds)
    azi = 150.
    ele = 3.

    # Load input image
    sample = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

    if cfg.NETWORK.REC_MODEL == 'PSGN_FC':
        sample = cv2.resize(sample, (64, 64))
    elif cfg.NETWORK.REC_MODEL == 'GRAPHX':
        sample = cv2.resize(sample, (224, 224))
    else:
        print('Invalid model name, please check the config.py (NET_WORK.REC_MODEL)')
        sys.exit(2)

    sample = np.expand_dims(sample, -1)
    samples = []
    samples.append(sample)
    samples = np.array(samples).astype(np.float32) 
    rendering_images = test_transforms(rendering_images=samples)

    # Load model_gt (input mask)
    ip_proj = cv2.imread(input_mask_path)[:,:,0]
    ip_proj = cv2.resize(ip_proj, (64,64))
    ip_proj[ip_proj<254] = 1
    ip_proj[ip_proj>=254] = 0
    batch_model_gt = []
    model_gt = []
    model_gt.append(ip_proj)
    batch_model_gt.append(model_gt)
    batch_model_gt = np.array(batch_model_gt).astype(np.float32)
    batch_model_gt = torch.from_numpy(batch_model_gt)

    # Convert azi ele to radian
    batch_model_x = []
    batch_model_y = []
    model_x = []
    model_y = []
    model_x.append(azi*np.pi/180.)
    model_y.append((ele - 90.)*np.pi/180.)
    batch_model_x.append(model_x)
    batch_model_y.append(model_y)
    batch_model_x = np.array(batch_model_x).astype(np.float32)
    batch_model_y = np.array(batch_model_y).astype(np.float32)
    batch_model_x = torch.from_numpy(batch_model_x)
    batch_model_y = torch.from_numpy(batch_model_y)
    
    edge_gt = None

    # Switch models to train mode
    net.train()
    
    reconstruction_losses = utils.network_utils.AverageMeter()
    loss_2ds = utils.network_utils.AverageMeter()
    loss_3ds = utils.network_utils.AverageMeter()
    
    test_opt_model_path = output_dir % 'rec_model.npy'
    rec_pc = net(rendering_images, init_point_clouds)
    np_rec_pc = rec_pc[0].detach().cpu().numpy()
    print(np_rec_pc.shape)
    np.save(test_opt_model_path, np_rec_pc)
    
    batch_pc = []
    pc = np.load(test_opt_model_path)
    batch_pc.append(pc)
    batch_pc = np.array(batch_pc).astype(np.float32)
    batch_pc = torch.from_numpy(batch_pc)

    # optimize loop
    for opt_idx in range(100):

        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        batch_model_gt = utils.network_utils.var_or_cuda(batch_model_gt)
        batch_model_x = utils.network_utils.var_or_cuda(batch_model_x)
        batch_model_y = utils.network_utils.var_or_cuda(batch_model_y)
        init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
        batch_pc = utils.network_utils.var_or_cuda(batch_pc)
        
        total_loss, loss_2d, loss_3d = net.module.learn(rendering_images, init_point_clouds, batch_pc, batch_model_x, batch_model_y, batch_model_gt, edge_gt)
        pred_pc = net(rendering_images, init_point_clouds)

        reconstruction_losses.update(total_loss)
        loss_2ds.update(loss_2d)
        loss_3ds.update(loss_3d)

        # Append epoch loss to TensorBoard
        opt_writer.add_scalar('Total/EpochLoss_Rec', reconstruction_losses.avg, opt_idx + 1)
        opt_writer.add_scalar('2D/EpochLoss_Loss_2D', loss_2ds.avg, opt_idx + 1)
        opt_writer.add_scalar('3D/EpochLoss_Loss_3D', loss_3ds.avg, opt_idx + 1)
        
        img_dir = output_dir % 'images'
        # Predict Pointcloud
        g_pc = pred_pc[0].detach().cpu().numpy()
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc, os.path.join(img_dir, 'test opt'),
                                                                                        opt_idx, epoch_idx, "reconstruction")
        opt_writer.add_image('Test Opt Sample#%02d/Point Cloud Reconstructed' % opt_idx, rendering_views, epoch_idx)







