# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
# https://github.com/hzxie/Pix2Vox
#

import json
import numpy as np
import cv2
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import scipy.misc as sc
from torchvision.utils import save_image

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.view_pred_utils

from datetime import datetime as dt

def valid_refine_net(
    cfg,
    epoch_idx=-1,
    output_dir=None,
    test_data_loader=None,
    test_writer=None,
    rec_net=None,
    refine_net=None):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    n_samples = len(test_data_loader)
    
    losses = utils.network_utils.AverageMeter()

    rec_net.eval()
    refine_net.eval()

    # Testing loop
    for sample_idx, (taxonomy_names, sample_names, rendering_images,
                    model_x, model_y,
                    init_point_clouds, ground_truth_point_clouds) in enumerate(test_data_loader):

        with torch.no_grad():
            # Only one image per sample
            rendering_images = torch.squeeze(rendering_images, 1)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            model_x = utils.network_utils.var_or_cuda(model_x)
            model_y = utils.network_utils.var_or_cuda(model_y)
            init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)

            #=================================================#
            #                Test the network                 #
            #=================================================#
            # rec net give out a coarse point cloud
            coarse_pc, img_feats = rec_net(rendering_images, init_point_clouds)
            # refine net give out a refine result
            loss, refine_pc = refine_net.module.valid_step(img_feats, coarse_pc, ground_truth_point_clouds, model_x, model_y)

            loss = loss.cpu().detach().data.numpy()

            # Append loss and accuracy to average metrics
            losses.update(loss)

            # Append generated point clouds to TensorBoard
            if output_dir and sample_idx < 6:
                
                img_dir = output_dir % 'images'
                
                # Coarse Pointcloud
                c_pc = coarse_pc[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(c_pc, os.path.join(img_dir, 'test'),
                                                                                            sample_idx, epoch_idx, "coarse")
                test_writer.add_image('Test Sample#%02d/Coarse Pointcloud' % sample_idx, rendering_views, epoch_idx)

                # Refine Pointcloud
                r_pc = refine_pc[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(r_pc, os.path.join(img_dir, 'test'),
                                                                                            sample_idx, epoch_idx, "refine")
                test_writer.add_image('Test Sample#%02d/Refine Pointcloud' % sample_idx, rendering_views, epoch_idx)

                # Ground Truth Pointcloud
                gt_pc = ground_truth_point_clouds[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(gt_pc, os.path.join(img_dir, 'test'),
                                                                                        sample_idx, epoch_idx, "ground truth")
                test_writer.add_image('Test Sample#%02d/GroundTruth Point Cloud' % sample_idx, rendering_views, epoch_idx)


    if test_writer is not None:
        test_writer.add_scalar('Total/EpochLoss_Rec', losses.avg, epoch_idx)
        
    return losses.avg