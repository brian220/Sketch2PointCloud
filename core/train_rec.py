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
from shutil import copyfile

import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.valid_rec import valid_rec_net
from models.networks_psgn import Pixel2Pointcloud_PSGN_FC
from models.networks_graphx_rec import GRAPHX_REC_MODEL

def train_rec_net(cfg):
    print("cuda is available?", torch.cuda.is_available())
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        # utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
                                                    utils.data_loaders.DatasetType.TRAIN, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
                                                  utils.data_loaders.DatasetType.VAL, val_transforms),
                                                  batch_size=cfg.CONST.BATCH_SIZE,
                                                  num_workers=cfg.TEST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=True)

    # Set up networks
    # The parameters here need to be set in cfg
    net = GRAPHX_REC_MODEL(
        cfg=cfg,
        optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.GRAPHX_LEARNING_RATE, weight_decay=cfg.TRAIN.GRAPHX_WEIGHT_DECAY),
        scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
    )

    if torch.cuda.is_available():
       net = torch.nn.DataParallel(net, device_ids=cfg.CONST.DEVICE).cuda()

    print(net)

    init_epoch = 0
    # best_emd =  10000 # less is better
    best_loss = 100000
    best_epoch = -1
    
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        net.load_state_dict(checkpoint['net'])

        print('[INFO] %s Recover complete. Current epoch #%d at epoch #%d.' %
              (dt.now(), init_epoch, cfg.TRAIN.NUM_EPOCHES))
    

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s')
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    
    # copy the config file
    copyfile(cfg.DIR.CONFIG_PATH, os.path.join(cfg.DIR.OUT_PATH, 'config-backup.py'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()
    
        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        reconstruction_losses = utils.network_utils.AverageMeter()

        net.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        model_azi, model_ele,
                        init_point_clouds, ground_truth_point_clouds) in enumerate(train_data_loader):

            # Measure data time
            data_time.update(time() - batch_end_time)
    
            # Only one image per batch
            rendering_images = torch.squeeze(rendering_images, 1)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            model_azi = utils.network_utils.var_or_cuda(model_azi)
            model_ele = utils.network_utils.var_or_cuda(model_ele)
            init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
            
            # net give out a point cloud
            loss = net.module.train_step(rendering_images, init_point_clouds, ground_truth_point_clouds)
            
            reconstruction_losses.update(loss)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
        
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) \
                 Total_loss = %.4f'
                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val, data_time.val, loss))
              
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('Total/EpochLoss_Rec', reconstruction_losses.avg, epoch_idx + 1)
       

        # Validate the training models
        current_loss = valid_rec_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, net)
        
        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)), 
                                                 epoch_idx + 1, 
                                                 net,
                                                 best_loss, best_epoch)
        
        # Save best check point for cd
        if current_loss < best_loss:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_loss = current_loss
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-rec-ckpt.pth'), 
                                                 epoch_idx + 1, 
                                                 net,
                                                 best_loss, best_epoch)
    
        

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


        






        
