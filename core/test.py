# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

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

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.STN import STN

from losses.chamfer_loss import ChamferLoss
from losses.earth_mover_distance import EMD
from losses.transform_loss import feature_transform_regularizer

def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             stn=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

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

    # Set up loss functions
    emd = EMD().cuda()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    reconstruction_losses = utils.network_utils.AverageMeter()
    transformation_losses = utils.network_utils.AverageMeter()
    total_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    stn.eval()
    
    for sample_idx, (taxonomy_names, sample_names, rendering_images,
                    ground_truth_point_clouds) in enumerate(test_data_loader):
        with torch.no_grad():
            # Only one image per batch
            rendering_images = torch.squeeze(rendering_images, 1)
          
             # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
            
            # test the transformation matrix (STN) 
            transformation = stn(rendering_images)

            # Test the encoder, decoder
            image_features = encoder(rendering_images)
            tree = [image_features]
            generated_point_clouds = decoder(tree)

            normalized_point_clouds = torch.bmm(generated_point_clouds, transformation)
            
            # loss computation
            reconstruction_loss = torch.mean(emd(normalized_point_clouds, ground_truth_point_clouds))
            transformation_loss = feature_transform_regularizer(transformation)

            total_loss = reconstruction_loss + cfg.TRAIN.TRANS_LAMDA * transformation_loss

            # Append loss and accuracy to average metrics
            reconstruction_losses.update(reconstruction_loss.item())
            transformation_losses.update(transformation_loss.item())
            total_losses.update(total_loss.item())

            # Append generated point clouds to TensorBoard
            if output_dir and sample_idx < 3:
                img_dir = output_dir % 'images'
                
                # Point cloud Visualization
                g_pc = generated_point_clouds[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc, os.path.join(img_dir, 'test'),
                                                                                        epoch_idx, "reconstruction")
                test_writer.add_image('Test Sample#%02d/Point Cloud Reconstructed' % sample_idx, rendering_views, epoch_idx)

                n_pc = normalized_point_clouds[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(n_pc, os.path.join(img_dir, 'test'),
                                                                                        epoch_idx, "normalize")
                test_writer.add_image('Test Sample#%02d/Point Cloud Normalized' % sample_idx, rendering_views, epoch_idx)
 
                gt_pc = ground_truth_point_clouds[0].detach().cpu().numpy()
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(gt_pc, os.path.join(img_dir, 'test'),
                                                                                        epoch_idx, "ground truth")
                test_writer.add_image('Test Sample#%02d/Point Cloud GroundTruth' % sample_idx, rendering_views, epoch_idx)

    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss_Rec', reconstruction_losses.avg, epoch_idx)
        test_writer.add_scalar('EncoderDecoder/EpochLoss_Trans', transformation_losses.avg, epoch_idx)
        test_writer.add_scalar('EncoderDecoder/EpochLoss_Total', total_losses.avg, epoch_idx)

    return reconstruction_losses.avg


