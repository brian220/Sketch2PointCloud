# 176    39f5eecbfb2470846666a748bda83f67 
# 41753  a58f8f1bd61094b3ff2c92c2a4f65876
# 2603   27c00ec2b6ec279958e80128fd34c2b1
# 37247  484f0070df7d5375492d9da2668ec34c
# 36881  4231883e92a3c1a21c62d11641ffbd35

import json
import numpy as np
import os, sys
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2
from datetime import datetime as dt

from models.networks_graphx import Pixel2Pointcloud_GRAPHX

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

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


def part_rec(cfg, part_img_path, part_weight_path):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    eval_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ToTensor(),
    ])
    
    # Set up networks
    # The parameters here need to be set in cfg
    if cfg.NETWORK.REC_MODEL == 'GRAPHX':
        net = Pixel2Pointcloud_GRAPHX(cfg=cfg,
                                      in_channels=3, 
                                      in_instances=cfg.GRAPHX.NUM_INIT_POINTS,
                                      optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.GRAPHX_LEARNING_RATE, weight_decay=cfg.TRAIN.GRAPHX_WEIGHT_DECAY),
                                      scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                                      use_graphx=cfg.GRAPHX.USE_GRAPHX)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), part_weight_path))
    rec_checkpoint = torch.load(part_weight_path)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    epoch_id = int(rec_checkpoint['epoch_idx'])
    
    net.eval()

    # load img
    sample = cv2.imread(part_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)

    samples = []
    samples.append(sample)
    samples = np.array(samples).astype(np.float32) 
    rendering_images = eval_transforms(rendering_images=samples)

    # load init point clouds
    init_point_cloud_np = init_pointcloud_loader(cfg.GRAPHX.NUM_INIT_POINTS)
    init_point_clouds = np.array([init_point_cloud_np])
    init_point_clouds = torch.from_numpy(init_point_clouds)

    # inference model
    with torch.no_grad():
        # Get data from data loader
        input_imgs = utils.network_utils.var_or_cuda(rendering_images)
        init_pc = utils.network_utils.var_or_cuda(init_point_clouds)
        
        pred_pc = net(input_imgs, init_pc)
        
        # Predict Pointcloud
        g_pc = pred_pc[0].detach().cpu().numpy()
    
    return g_pc


def evaluate_part_rec_net(cfg):
    components = ['arm', 'back', 'base', 'seat']

    sample_id = cfg.EVALUATE_PART_REC.SAMPLE_ID
    sample_img_path = cfg.EVALUATE_PART_REC.IMG_FOLDER
    
    weight_dict = {}
    weight_dict['arm'] = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v23/checkpoints/best-rec-ckpt.pth'
    weight_dict['back'] = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v24/checkpoints/best-rec-ckpt.pth'
    weight_dict['base'] = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v25/checkpoints/best-rec-ckpt.pth'
    weight_dict['seat'] = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v26/checkpoints/best-rec-ckpt.pth'
    
    for view_id in range(3):
        img_dict = {}
        img_dict['arm'] = os.path.join(sample_img_path ,  'Chair_Arm', str(sample_id), 'render_' + str(view_id) + '.png')
        img_dict['back'] = os.path.join(sample_img_path ,  'Chair_Back', str(sample_id), 'render_' + str(view_id) + '.png')
        img_dict['base'] = os.path.join(sample_img_path ,  'Chair_Base', str(sample_id), 'render_' + str(view_id) + '.png')
        img_dict['seat'] = os.path.join(sample_img_path ,  'Chair_Seat', str(sample_id), 'render_' + str(view_id) + '.png')
    
        g_pcs = []
        for component in components:
            if os.path.exists(img_dict[component]):    
                g_pcs.append(part_rec(cfg, img_dict[component], weight_dict[component]))
        
        g_pcs = tuple(g_pcs)
        g_pc = np.concatenate(g_pcs, axis=0)
        print(g_pc.shape)
    
        output_path = cfg.EVALUATE_PART_REC.OUT_PATH
        outfolder = os.path.join(output_path, str(sample_id))
        rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc,
                                                                                outfolder,
                                                                                int(sample_id), view_id, "part rec result", view=[ 45*int(view_id + 1), 25])
    