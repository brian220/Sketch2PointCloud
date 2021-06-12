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

import utils.point_cloud_visualization_old
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from pyntcloud import PyntCloud

partnet2shapenet = {
        '176'  :  '39f5eecbfb2470846666a748bda83f67',
        '41753':  'a58f8f1bd61094b3ff2c92c2a4f65876',
        '2603' :  '27c00ec2b6ec279958e80128fd34c2b1',
        '37247':  '484f0070df7d5375492d9da2668ec34c',
        '36881':  '4231883e92a3c1a21c62d11641ffbd35'
    }


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


def rec(cfg, img_path, weight_path):
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
                                      in_instances=2048,
                                      optimizer=lambda x: torch.optim.Adam(x, lr=cfg.TRAIN.GRAPHX_LEARNING_RATE, weight_decay=cfg.TRAIN.GRAPHX_WEIGHT_DECAY),
                                      scheduler=lambda x: MultiStepLR(x, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA),
                                      use_graphx=cfg.GRAPHX.USE_GRAPHX)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), weight_path))
    rec_checkpoint = torch.load(weight_path)
    net.load_state_dict(rec_checkpoint['net'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    epoch_id = int(rec_checkpoint['epoch_idx'])
    
    net.eval()

    # load img
    sample = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)

    samples = []
    samples.append(sample)
    samples = np.array(samples).astype(np.float32) 
    rendering_images = eval_transforms(rendering_images=samples)

    # load init point clouds
    init_point_cloud_np = init_pointcloud_loader(2048)
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


def generate_gt(cfg, sample_id, view_id):

    shapenet_id = partnet2shapenet[sample_id]

    gt_pc_path = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/%s.ply' % (shapenet_id)
    gt_pc = PyntCloud.from_file(gt_pc_path)
    gt_pc = np.array(gt_pc.points).astype(np.float32)

    output_path = cfg.EVALUATE.OUT_PATH
    outfolder = os.path.join(output_path, str(sample_id))
    rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(gt_pc,
                                                                                outfolder,
                                                                                int(sample_id), view_id, "gt", view=[ 45*int(view_id + 1), 25])


def evaluate_net(cfg):
    sample_img_folder = cfg.EVALUATE.IMG_FOLDER
    weight_path = cfg.EVALUATE.WEIGHT_PATH
    
    for partnet_id in partnet2shapenet.keys():
        sample_id = partnet_id
        for view_id in range(3):
            img_path = os.path.join(sample_img_folder, str(sample_id), 'render_' + str(view_id) + '.png')
    
            g_pc = rec(cfg, img_path, weight_path)
            print(g_pc.shape)
        
            output_path = cfg.EVALUATE.OUT_PATH
            outfolder = os.path.join(output_path, str(sample_id))
            rendering_views = utils.point_cloud_visualization_old.get_point_cloud_image(g_pc,
                                                                                        outfolder,
                                                                                        int(sample_id), view_id, "rec result", view=[ 45*int(view_id + 1), 25])
            
            generate_gt(cfg, sample_id, view_id)
