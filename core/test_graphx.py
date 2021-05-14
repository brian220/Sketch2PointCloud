# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
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

from models.networks import Pixel2Pointcloud
from models.view_encoder import Encoder
from models.view_estimater import ViewEstimater

from losses.chamfer_loss import ChamferLoss
from losses.earth_mover_distance import EMD
from losses.cross_entropy_loss import CELoss
from losses.delta_loss import DeltaLoss

def test_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
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
                                                   utils.data_loaders.DatasetType.TEST,  test_transforms),
                                                   batch_size=cfg.CONST.BATCH_SIZE,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Set up networks
    # Set up networks
    # The parameters here need to be set in cfg
    net = Pixel2Pointcloud(3, cfg.GRAPHX.NUM_INIT_POINTS,
                        optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY),
                        scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                        use_graphx=cfg.GRAPHX.USE_GRAPHX)
    
    view_encoder = Encoder(cfg)

    azi_classes, ele_classes = int(360 / cfg.CONST.BIN_SIZE), int(180 / cfg.CONST.BIN_SIZE)
    view_estimater = ViewEstimater(cfg, azi_classes=azi_classes, ele_classes=ele_classes)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        view_encoder = torch.nn.DataParallel(view_encoder).cuda()
        view_estimater = torch.nn.DataParallel(view_estimater).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.TEST.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.TEST.RECONSTRUCTION_WEIGHTS)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])

    # Load weight for view encoder
    print('[INFO] %s Loading view estimation weights from %s ...' % (dt.now(), cfg.TEST.VIEW_ENCODER_WEIGHTS))
    view_enc_checkpoint = torch.load(cfg.TEST.VIEW_ENCODER_WEIGHTS)
    view_encoder.load_state_dict(view_enc_checkpoint['encoder_state_dict'])
    print('[INFO] Best view encode result at epoch %d ...' % view_enc_checkpoint['epoch_idx'])

    # Load weight for view estimater
    print('[INFO] %s Loading view estimation weights from %s ...' % (dt.now(), cfg.TEST.VIEW_ESTIMATION_WEIGHTS))
    view_est_checkpoint = torch.load(cfg.TEST.VIEW_ESTIMATION_WEIGHTS)
    view_estimater.load_state_dict(view_est_checkpoint['view_estimator_state_dict'])
    print('[INFO] Best view estimation result at epoch %d ...' % view_est_checkpoint['epoch_idx'])

    # Set up loss functions
    emd = EMD().cuda()
    cd = ChamferLoss().cuda()
    
    # Batch average meterics
    cd_distances = utils.network_utils.AverageMeter()
    emd_distances = utils.network_utils.AverageMeter()
    pointwise_emd_distances = utils.network_utils.AverageMeter()

    test_preds = torch.zeros([1, 3], dtype=torch.float).cuda()
    test_ground_truth_views = torch.zeros([1, 3], dtype=torch.long).cuda()

    # Switch models to evaluation mode
    net.eval()
    view_encoder.eval()
    view_estimater.eval()

    n_batches = len(test_data_loader)

    # Testing loop
    for sample_idx, (taxonomy_names, sample_names, rendering_images,
                    init_point_clouds, ground_truth_point_clouds, ground_truth_views) in enumerate(test_data_loader):
        with torch.no_grad():
            # Only one image per sample
            rendering_images = torch.squeeze(rendering_images, 1)
          
             # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            init_point_clouds = utils.network_utils.var_or_cuda(init_point_clouds)
            ground_truth_views = utils.network_utils.var_or_cuda(ground_truth_views)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
            
            #=================================================#
            #           Test the encoder, decoder             #
            #=================================================#
            emd_loss, generated_point_clouds = net.module.loss(rendering_images, init_point_clouds, ground_truth_point_clouds, 'mean')
            
            # Compute CD, EMD
            cd_distance = cd(generated_point_clouds, ground_truth_point_clouds) / cfg.CONST.BATCH_SIZE / cfg.CONST.NUM_POINTS
            emd_distance = emd_loss
            pointwise_emd_distance = emd_loss / cfg.CONST.NUM_POINTS
            
            #=================================================#
            #          Test the view estimater                #
            #=================================================#
            # output[0]:prediction of azi class
            # output[1]:prediction of ele class
            # output[2]:prediction of azi regression
            # output[3]:prediction of ele regression
            vgg_features, _  = view_encoder(rendering_images)
            output = view_estimater(vgg_features)
    
            #=================================================#
            #              Get predict view                   #
            #=================================================#
            preds_cls = utils.view_pred_utils.get_pred_from_cls_output([output[0], output[1]])
            
            preds = []
            for n in range(len(preds_cls)):
                pred_delta = output[n + 2]
                delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds_cls[n].long()].tanh() / 2
                preds.append((preds_cls[n].float() + delta_value + 0.5) * cfg.CONST.BIN_SIZE)
            # In this experiment, expect all the object has a fixed rotaion angle 0 and with diferent view point
            # Add the zero inplane term for the rotation acc calculation
            zero_inplane = torch.zeros_like(preds[0])
            # Add fixed inplane rotation to  pred view
            test_pred = torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), zero_inplane.unsqueeze(1)), 1)
            # Add fixed inplane rotation to ground_truth_views
            zero_inplane = zero_inplane.unsqueeze(1)
            ground_truth_views = torch.cat((ground_truth_views, zero_inplane.long()), 1)
            
            # Append loss and accuracy to average metrics
            cd_distances.update(cd_distance.item())
            emd_distances.update(emd_distance.item())
            pointwise_emd_distances.update(pointwise_emd_distance.item())
            
            # concatenate results and labels for view estimation
            test_preds = torch.cat((test_preds, test_pred), 0)
            test_ground_truth_views = torch.cat((test_ground_truth_views, ground_truth_views), 0)

            print("Test on [%d/%d] data, CD: %.4f Point EMD: %.4f Total EMD %.4f" % (sample_idx + 1,  n_batches, cd_distance.item(), pointwise_emd_distance.item(), emd_distance.item()))
    
    test_preds = test_preds[1:, :]
    test_ground_truth_views = test_ground_truth_views[1:, :]

    # calculate the rotation errors between prediction and ground truth
    test_errs = utils.rotation_eval.rotation_err(test_preds, test_ground_truth_views.float()).cpu().numpy()
    Acc = 100. * np.mean(test_errs <= 30)
    Med = np.median(test_errs)
    
    # print result
    print("Reconstruction result:")
    print("CD result: ", cd_distances.avg)
    print("Pointwise EMD result: ", pointwise_emd_distances.avg)
    print("Total EMD result", emd_distances.avg)
    print("View estimation result:")
    print('Med_Err is %.2f, and Acc_pi/6 is %.2f \n \n' % (Med, Acc))
    logname = cfg.TEST.RESULT_PATH 
    with open(logname, 'a') as f:
        f.write('Reconstruction result: \n')
        f.write("CD result: %.8f \n" % cd_distances.avg)
        f.write("Pointwise EMD result: %.8f \n" % pointwise_emd_distances.avg)
        f.write("Total EMD result: %.8f \n" % emd_distances.avg)
        f.write('View estimation result: \n')
        f.write('Med_Err is %.2f, and Acc_pi/6 is %.2f \n \n' % (Med, Acc))
            


