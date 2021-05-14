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

idtodegree = [
    [0, 0],
    [45, 0],
    [90, 0],
    [135, 0],
    [180, 0],
    [225, 0],
    [270, 0],
    [325, 0]
]

def valid_stage2_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             net=None,
             update_net=None):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    n_samples = len(test_data_loader)

    reconstruction_losses = utils.network_utils.AverageMeter()

    net.eval()
    update_net.eval()

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
            # net give out a point cloud
            pred_pc = net(rendering_images, init_point_clouds)
            # *** Here put the same images to reconstruction and update model
            rec_loss, update_pc = update_net.module.loss(rendering_images, pred_pc, ground_truth_point_clouds, model_x, model_y)

            reconstruction_loss = rec_loss.cpu().detach().data.numpy()

            # Append loss and accuracy to average metrics
            reconstruction_losses.update(reconstruction_loss)

            # Append generated point clouds to TensorBoard
            if output_dir and sample_idx < 6:
                
                img_dir = output_dir % 'images'
                sketch_dir = output_dir % 'sketchs'
                
                if  not os.path.exists(sketch_dir):
                    os.makedirs(sketch_dir)
                
                # rendering imgs
                r_img = rendering_images[0]
                r_img_path = os.path.join(sketch_dir, str(epoch_idx) + '_' + str(sample_idx) + '_' + 'render.png')
                save_image(r_img, r_img_path)
                r_img = cv2.imread(r_img_path)
                test_writer.add_image('Test Sample#%02d/Img Reconstructed' % sample_idx, r_img, epoch_idx)
                
                # Plot in update view
                degree_x = model_x[0].item()*180./np.pi
                degree_y = (model_y[0].item()*180./np.pi) + 90.

                update_view = (degree_x, degree_y)
                # Predict Pointcloud
                p_pc = pred_pc[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(p_pc, os.path.join(img_dir, 'test'),
                                                                                        sample_idx, epoch_idx, "reconstruction", view=update_view)
                test_writer.add_image('Test Sample#%02d/Point Cloud Reconstructed' % sample_idx, rendering_views, epoch_idx)
                
                # Update Pointcloud
                u_pc = update_pc[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(u_pc, os.path.join(img_dir, 'test'),
                                                                                        sample_idx, epoch_idx, "update", view=update_view)
                test_writer.add_image('Test Sample#%02d/Point Cloud Updated' % sample_idx, rendering_views, epoch_idx)
                
                # Groundtruth Pointcloud
                gt_pc = ground_truth_point_clouds[0].detach().cpu().numpy()
                # ground_truth_view = ground_truth_views[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(gt_pc, os.path.join(img_dir, 'test'),
                                                                                        sample_idx, epoch_idx, "ground truth", view=update_view)
                test_writer.add_image('Test Sample#%02d/Point Cloud GroundTruth' % sample_idx, rendering_views, epoch_idx)

    if test_writer is not None:
        test_writer.add_scalar('Total/EpochLoss_Rec', reconstruction_losses.avg, epoch_idx)

    return reconstruction_losses.avg