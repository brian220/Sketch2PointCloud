# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Reference to Pix2Vox: 
# https://github.com/hzxie/Pix2Vox
#

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.view_pred_utils
import utils.rotation_eval

from datetime import datetime as dt

from models.networks_graphx_rec import GRAPHX_REC_MODEL

from losses.chamfer_loss import ChamferLoss
import cuda.emd.emd_module as emd

def test_rec_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ToTensor(),
    ])
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
                                                   utils.data_loaders.DatasetType.TEST,  test_transforms),
                                                   batch_size=cfg.TEST.BATCH_SIZE,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Set up networks
    # The parameters here need to be set in cfg
    net = GRAPHX_REC_MODEL(
        cfg=cfg,
        optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.GRAPHX_LEARNING_RATE, weight_decay=cfg.TRAIN.GRAPHX_WEIGHT_DECAY),
        scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
    )
    
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.TEST.WEIGHT_PATH))
    rec_checkpoint = torch.load(cfg.TEST.WEIGHT_PATH)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])

    # Set up loss functions
    emd_dist = emd.emdModule()
    cd = ChamferLoss().cuda()
    
    # Batch average meterics
    cd_distances = utils.network_utils.AverageMeter()
    emd_distances = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    net.eval()

    n_batches = len(test_data_loader)

    # Testing loop
    for sample_idx, (taxonomy_names, sample_names, rendering_images,
                    model_azi, model_ele,
                    init_point_clouds, ground_truth_point_clouds) in enumerate(test_data_loader):
        with torch.no_grad():
            # Only one image per sample
            rendering_images = torch.squeeze(rendering_images, 1)
            
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            model_azi = utils.network_utils.var_or_cuda(model_azi)
            model_ele = utils.network_utils.var_or_cuda(model_ele)
            init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
            
            #=================================================#
            #           Test the encoder, decoder             #
            #=================================================#
            loss, pred_pc = net.module.valid_step(rendering_images, init_point_clouds, ground_truth_point_clouds)

            # Compute CD, EMD
            cd_distance = cd(pred_pc, ground_truth_point_clouds) / cfg.TEST.BATCH_SIZE / cfg.CONST.NUM_POINTS
            
            # compute reconstruction loss
            emd_loss, _ = emd_dist(
                pred_pc, ground_truth_point_clouds, eps=0.005, iters=50
            )
            emd_distance = torch.sqrt(emd_loss).mean(1).mean()

            # Append loss and accuracy to average metrics
            cd_distances.update(cd_distance.item())
            emd_distances.update(emd_distance.item())

            print("Test on [%d/%d] data, CD: %.4f EMD %.4f" % (sample_idx + 1,  n_batches, cd_distance.item(), emd_distance.item()))
    
    # print result
    print("Reconstruction result:")
    print("CD result: ", cd_distances.avg)
    print("EMD result", emd_distances.avg)
    logname = cfg.TEST.RESULT_PATH 
    with open(logname, 'a') as f:
        f.write('Reconstruction result: \n')
        f.write("CD result: %.8f \n" % cd_distances.avg)
        f.write("EMD result: %.8f \n" % emd_distances.avg)
            


