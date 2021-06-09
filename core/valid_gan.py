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

import utils.point_cloud_visualization_old
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.view_pred_utils

from datetime import datetime as dt

def valid_gan_net(cfg,
                  epoch_idx=-1,
                  output_dir=None,
                  test_data_loader=None,
                  test_writer=None,
                  net=None):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    n_samples = len(test_data_loader)

    rec_losses = utils.network_utils.AverageMeter()

    net.eval()

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

            rec_loss, pred_pc = net.module.valid_step(rendering_images, init_point_clouds, ground_truth_point_clouds)

            # Append loss and accuracy to average metrics
            rec_loss = rec_loss.cpu().detach().data.numpy()
            rec_losses.update(rec_loss)

            # Append generated point clouds to TensorBoard
            if output_dir and sample_idx < 6:
                
                img_dir = output_dir % 'images'
                
                # Predict Pointcloud
                p_pc = pred_pc[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(p_pc, os.path.join(img_dir, 'test'),
                                                                                        sample_idx, epoch_idx, "reconstruction")
                test_writer.add_image('Test Sample#%02d/Point Cloud Reconstructed' % sample_idx, rendering_views, epoch_idx)
                
                # Groundtruth Pointcloud
                gt_pc = ground_truth_point_clouds[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(gt_pc, os.path.join(img_dir, 'test'),
                                                                                        sample_idx, epoch_idx, "ground truth")
                test_writer.add_image('Test Sample#%02d/Point Cloud GroundTruth' % sample_idx, rendering_views, epoch_idx)

    if test_writer is not None:
        test_writer.add_scalar('Total/EpochLoss_Rec', rec_losses.avg, epoch_idx)

    return rec_losses.avg
