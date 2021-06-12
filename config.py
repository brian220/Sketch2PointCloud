# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/rec.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/caig/423ECD443ECD3229/new_dataset/shapenet_24_fix/%s/%s/render_%d.png'
__C.DATASETS.SHAPENET.POINT_CLOUD_PATH      = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/%s/%s.ply'
__C.DATASETS.SHAPENET.VIEW_PATH             = '/media/caig/423ECD443ECD3229/new_dataset/shapenet_24_fix/%s/%s/view.txt'

__C.DATASETS.SHAPENET.HAND_DRAW_IMG_PATH    = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/hand_draw_img'

# __C.DATASETS.SHAPENET.UPDATE_PATH           = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/sketch_eight_view/%s/%s/render_%d.png'
# __C.DATASETS.SHAPENET.UPDATE_DEPTH_PATH     = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/sketch_eight_view/%s/%s/depth_%d.png'
# __C.DATASETS.SHAPENET.UPDATE_VIEW_PATH      = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/sketch_eight_view/%s/%s/view.txt'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.RENDER_VIEWS                    = 24

__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNetFix'
__C.DATASET.TEST_DATASET                    = 'ShapeNetFix'
__C.DATASET.CLASS                           = 'chair'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = [0]
__C.CONST.DEVICE_NUM                        = 1
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.BATCH_SIZE                        = 4
__C.CONST.BIN_SIZE                          = 15
__C.CONST.NUM_POINTS                        = 2048
__C.CONST.WEIGHTS                           = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/output/checkpoints/ckpt-epoch-0600.pth'

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/output'
__C.DIR.CONFIG_PATH                         = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/config.py'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.REC_MODEL                       = 'GRAPHX' # GRAPHX or PSGN_FC
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False

#
# GraphX
#
__C.GRAPHX                                 = edict()
__C.GRAPHX.USE_GRAPHX                      = True
__C.GRAPHX.NUM_INIT_POINTS                 = 2048

#
# Updater
#
__C.UPDATER                               = edict()
__C.UPDATER.RANGE_MAX                     = 0.2
__C.UPDATER.LEARNING_RATE                 = 1e-3
__C.UPDATER.NOISE_LENGTH                  = 32
__C.UPDATER.PC_ENCODE_MODULE              = 'Pointnet++'
__C.UPDATER.PC_DECODE_MODULE              = 'Graphx'


#
# Updater
#
__C.REFINE                                 = edict()
__C.REFINE.USE_PRETRAIN_GENERATOR          = True
__C.REFINE.GENERATOR_WEIGHTS               = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v27/checkpoints/best-rec-ckpt.pth'
__C.REFINE.START_EPOCH                     = 1000
__C.REFINE.EDGERES_LEARNING_RATE           = 1e-4
__C.REFINE.EDGERES_WEIGHT_DECAY            = 0
__C.REFINE.EDGERES_BETAS                   = (0.0, 0.9)

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 1100
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam

# train parameters for graphx
__C.TRAIN.GRAPHX_LEARNING_RATE              = 5e-5
__C.TRAIN.GRAPHX_WEIGHT_DECAY               = 1e-5

# train parameters for psgn fc
__C.TRAIN.PSGN_FC_LEARNING_RATE             = 5e-5
__C.TRAIN.PSGN_FC_CONV_WEIGHT_DECAY         = 1e-5
__C.TRAIN.PSGN_FC_FC_WEIGHT_DECAY           = 1e-3

# __C.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE      = 5e-4
__C.TRAIN.MILESTONES                        = [400]
__C.TRAIN.VIEW_ESTIMATOR_LR_MILESTONES      = [400]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .3
__C.TRAIN.SAVE_FREQ                         = 100            # weights will be overwritten every save_freq epoch
__C.TRAIN.TRANS_LAMDA                       = 10

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
__C.TEST.NUM_WORKER                         = 4             # number of data workers
__C.TEST.RECONSTRUCTION_WEIGHTS             = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v20/checkpoints/best-rec-ckpt.pth'
__C.TEST.RESULT_PATH                        = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/test.txt'

#
# Evaluating options
#
__C.EVALUATE                                = edict()
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/eval_output_fix/'
__C.EVALUATE.WEIGHT_PATH                    = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v27/checkpoints/best-rec-ckpt.pth'
__C.EVALUATE.BATCH_SIZE                     = 1

#
# Evaluate on the true habd draw image
#
__C.EVALUATE_HAND_DRAW                                = edict()
__C.EVALUATE_HAND_DRAW.INPUT_IMAGE_FOLDER             = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/evaluate_4_10/v9/evaluate_hand_draw/hand_draw_input_img'
__C.EVALUATE_HAND_DRAW.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/evaluate_4_10/v9/evaluate_hand_draw/hand_draw_output/'
__C.EVALUATE_HAND_DRAW.RECONSTRUCTION_WEIGHTS         = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v9/checkpoints/best-reconstruction-ckpt.pth'
# __C.EVALUATE_HAND_DRAW.VIEW_ESTIMATION_WEIGHTS        = '/media/itri/Files_2tb/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-view-ckpt.pth'

#
# Evaluating options
#
__C.EVALUATE_FIXED_VIEW                             = edict()
__C.EVALUATE_FIXED_VIEW.TAXONOMY_ID                 = '03001627'
__C.EVALUATE_FIXED_VIEW.RESULT_DIR                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/'
__C.EVALUATE_FIXED_VIEW.SAMPLE_FILE                 = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/evaluate_sample.txt'
__C.EVALUATE_FIXED_VIEW.VIEW_FILE                   = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/fixed_view.txt'
__C.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS      = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v4/checkpoints/best-reconstruction-ckpt.pth'

#
# Evaluate multi-view
# 
__C.EVALUATE_MULTI_VIEW                             = edict()
__C.EVALUATE_MULTI_VIEW.INPUT_IMAGE_PATH            = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/evaluate_multi_view/input_imgs/render_0.png'
__C.EVALUATE_MULTI_VIEW.UPDATE_IMAGE_FOLDER         = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/evaluate_multi_view/update_imgs/'
__C.EVALUATE_MULTI_VIEW.RECONSTRUCTION_WEIGHTS      = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v1/checkpoints/best-reconstruction-ckpt.pth'
__C.EVALUATE_MULTI_VIEW.UPDATE_WEIGHTS              = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v15/checkpoints/best-update-ckpt.pth'
__C.EVALUATE_MULTI_VIEW.OUT_DIR                     = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/evaluate_multi_view/outputs'

#
# Evaluate part
# 
__C.EVALUATE_PART_REC                               = edict()
__C.EVALUATE_PART_REC.IMG_FOLDER                    = '/media/caig/423ECD443ECD3229/dataset/partnet_eval_data'
__C.EVALUATE_PART_REC.SAMPLE_ID                     = 41753
__C.EVALUATE_PART_REC.OUT_PATH                      = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/part_eval_output'

#
# Test time optimization
#
__C.TEST_OPT                                        = edict()
__C.TEST_OPT.RECONSTRUCTION_WEIGHTS                 = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v1/checkpoints/best-reconstruction-ckpt.pth'
__C.TEST_OPT.OUT_PATH                               = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/test_opt'

