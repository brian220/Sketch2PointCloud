# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
# https://github.com/hzxie/Pix2Vox

import json
import numpy as np
import os, sys
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2

from shutil import copyfile

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.view_estimater import ViewEstimater

from pyntcloud import PyntCloud

def evaluate_hand_draw_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    eval_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    azi_classes, ele_classes = int(360 / cfg.CONST.BIN_SIZE), int(180 / cfg.CONST.BIN_SIZE)
    view_estimater = ViewEstimater(cfg, azi_classes=azi_classes, ele_classes=ele_classes)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        view_estimater = torch.nn.DataParallel(view_estimater).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE_HAND_DRAW.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.EVALUATE_HAND_DRAW.RECONSTRUCTION_WEIGHTS)
    encoder.load_state_dict(rec_checkpoint['encoder_state_dict'])
    decoder.load_state_dict(rec_checkpoint['decoder_state_dict'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])

    # Load weight for view estimater
    print('[INFO] %s Loading view estimation weights from %s ...' % (dt.now(), cfg.EVALUATE_HAND_DRAW.VIEW_ESTIMATION_WEIGHTS))
    view_checkpoint = torch.load(cfg.EVALUATE_HAND_DRAW.VIEW_ESTIMATION_WEIGHTS)
    view_estimater.load_state_dict(view_checkpoint['view_estimator_state_dict'])
    print('[INFO] Best view estimation result at epoch %d ...' % view_checkpoint['epoch_idx'])

    for img_path in os.listdir(cfg.EVALUATE_HAND_DRAW.INPUT_IMAGE_FOLDER):
        eval_id = int(img_path[:-4])
        input_img_path = os.path.join(cfg.EVALUATE_HAND_DRAW.INPUT_IMAGE_FOLDER, img_path)
        print(input_img_path)
        evaluate_hand_draw_img(cfg, 
                               encoder, decoder, view_estimater,
                               input_img_path,
                               eval_transforms, eval_id)


def evaluate_hand_draw_img(cfg, 
                           encoder, decoder, view_estimater,
                           input_img_path,
                           eval_transforms, eval_id):
    # load img
    img_np = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    sample = np.array([img_np])
    rendering_images = eval_transforms(rendering_images=sample)
    print(rendering_images.size())

    # inference model
    with torch.no_grad():
        # Only one image per sample
        rendering_images = torch.squeeze(rendering_images, 1)
        
         # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        
        #=================================================#
        #           Evaluate the encoder, decoder         #
        #=================================================#
        vgg_features, image_code = encoder(rendering_images)
        image_code = [image_code]
        generated_point_clouds = decoder(image_code)

        #=================================================#
        #          Evaluate the view estimater            #
        #=================================================#
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

        # Predict Pointcloud
        g_pc = generated_point_clouds[0].detach().cpu().numpy()
        pred_view = []
        pred_view.append(preds[0][0].detach().cpu().numpy())
        pred_view.append(preds[1][0].detach().cpu().numpy())
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc, cfg.EVALUATE_HAND_DRAW.OUTPUT_FOLDER,
                                                                                eval_id, "reconstruction", pred_view)