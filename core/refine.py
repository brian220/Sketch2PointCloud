import json
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data

from datetime import datetime as dt
from time import time
from collections import OrderedDict
from shutil import copyfile
import cv2

from models.networks_graphx_refine import GRAPHX_REFINE_MODEL
import utils.data_transforms
from utils.point_cloud_utils import output_point_cloud_ply

def get_red_mask(mask_img_path):
    seg = cv2.imread(mask_img_path)
    hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)

    # For arm
    red_min0 = 0
    red_max0 = 10
    red_min1 = 156
    red_max1 = 180

    lower_red0 = np.array([red_min0, 100, 100])
    upper_red0 = np.array([red_max0, 255, 255])
    lower_red1 = np.array([red_min1, 100, 100])
    upper_red1 = np.array([red_max1, 255, 255])

    mask0 = cv2.inRange(hsv, lower_red0 , upper_red0)
    mask1 = cv2.inRange(hsv, lower_red1 , upper_red1)
    mask = cv2.bitwise_or(mask0, mask1)
    
    return mask

def combine_sketch_empty(sketch_img, empty_img):
    hsv = cv2.cvtColor(empty_img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0]) 
    upper_black = np.array([180, 255, 46])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    sketch_img_bg = cv2.bitwise_and(sketch_img, sketch_img, mask = mask)
    sketch_empty_img = cv2.add(sketch_img_bg, empty_img)

    return sketch_empty_img

def refine_net(cfg, coarse_pc, azi, ele):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    refine_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ToTensor(),
    ])

    refine_net = GRAPHX_REFINE_MODEL(
        cfg=cfg,
        in_channels=3,
        optimizer=lambda x: torch.optim.Adam(x, lr=cfg.REFINE.LEARNING_RATE)
    )

    if torch.cuda.is_available():
        refine_net = torch.nn.DataParallel(refine_net, device_ids=cfg.CONST.DEVICE).cuda()
    
    # Load weight for refiner
    print('[INFO] %s Recovering refiner from %s ...' % (dt.now(), cfg.INFERENCE.REFINER_WEIGHTS))
    refine_checkpoint = torch.load(cfg.INFERENCE.REFINER_WEIGHTS)
    pretrained_dict = refine_checkpoint['net']
    refine_net.load_state_dict(pretrained_dict)
    refine_net.eval()
    
    refine_image_mask = get_red_mask(cfg.INFERENCE.REFINE_IMAGE_PATH)
    refine_image_mask = cv2.bitwise_not(refine_image_mask)
    refine_image_mask = cv2.resize(refine_image_mask, (224, 224), interpolation=cv2.INTER_AREA)
    
    # find the empty parts by connected component
    ret, thresh = cv2.threshold(refine_image_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels, status, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    
    empty_img = np.zeros( (224, 224, 3), np.uint8 )
    
    for i in range(0, num_labels):
        # check back ground
        s = status[i]
        if not (s[0] == 0 and s[1] == 0 and s[2] == 224 and s[3] == 224):
            mask = labels == i
            empty_img[:, :, 0][mask] = 109
            empty_img[:, :, 1][mask] = 0
            empty_img[:, :, 2][mask] = 216
    
    cv2.imwrite(cfg.INFERENCE.EMPTY_IMAGE_PATH, empty_img)
    
    sketch_img = cv2.imread(cfg.INFERENCE.SKETCH_IMAGE_PATH)
    empty_img = cv2.imread(cfg.INFERENCE.EMPTY_IMAGE_PATH)
    
    sketch_empty_img = combine_sketch_empty(sketch_img, empty_img)
    cv2.imwrite(cfg.INFERENCE.UPDATE_IMAGE_PATH, sketch_empty_img)
    
    # load update image
    update_image = cv2.imread(cfg.INFERENCE.UPDATE_IMAGE_PATH, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    samples = []
    samples.append(update_image)
    samples = np.array(samples).astype(np.float32) 
    update_images = refine_transforms(rendering_images=samples)
    
    # set view
    azi = float(azi)
    rec_radian_azi = azi*np.pi/180.
    batch_azi = []
    batch_azi.append(rec_radian_azi)
    batch_azi = np.array(batch_azi).astype(np.float32)
    batch_azi = torch.from_numpy(batch_azi)

    ele = float(ele)
    rec_radian_ele = (ele - 90.)*np.pi/180.
    batch_ele = []
    batch_ele.append(rec_radian_ele)
    batch_ele = np.array(batch_ele).astype(np.float32)
    batch_ele = torch.from_numpy(batch_ele)

    coarse_pc = np.array([coarse_pc])
    coarse_pc = torch.from_numpy(coarse_pc).cuda()
    
    with torch.no_grad():
        # generate refine point cloud
        refine_pc = refine_net.module.refine(update_images, coarse_pc, batch_azi, batch_ele)
        output_point_cloud_ply(refine_pc.detach().cpu().numpy(), ['refine'], cfg.INFERENCE.CACHE_POINT_CLOUD_PATH)
        return refine_pc[0].detach().cpu().numpy()
