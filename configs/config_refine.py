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
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/caig/423ECD443ECD3229/sketch_3d_dataset/img/shapenet_24_fix/%s/%s/render_%d.png'
__C.DATASETS.SHAPENET.UPDATE_PATH           = '/media/caig/423ECD443ECD3229/sketch_3d_dataset/img/shapenet_24_sketch_empty/%s/%s/render_empty_%d.png'
# __C.DATASETS.SHAPENET.UPDATE_PATH           = '/media/caig/423ECD443ECD3229/sketch_3d_dataset/img/shapenet_24_fix/%s/%s/render_%d.png'
__C.DATASETS.SHAPENET.POINT_CLOUD_PATH      = '/media/caig/423ECD443ECD3229/sketch_3d_dataset/shape_net_core_uniform_samples_2048/%s/%s.ply'
__C.DATASETS.SHAPENET.VIEW_PATH             = '/media/caig/423ECD443ECD3229/sketch_3d_dataset/img/shapenet_24_fix/%s/%s/view.txt'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.RENDER_VIEWS                    = 24

__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNetFixRefine'
__C.DATASET.TEST_DATASET                    = 'ShapeNetFixRefine'
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
__C.CONST.BATCH_SIZE                        = 8
__C.CONST.BIN_SIZE                          = 15
__C.CONST.NUM_POINTS                        = 2048
__C.CONST.WEIGHTS                           = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/output/checkpoints/ckpt-epoch-0600.pth'

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/output'
__C.DIR.CONFIG_PATH                         = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/configs/config_refine.py'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False

#
# GraphX
#
__C.GRAPHX                                 = edict()
__C.GRAPHX.USE_GRAPHX                      = True
__C.GRAPHX.NUM_INIT_POINTS                 = 2048
__C.GRAPHX.RETURN_IMG_FEATURES             = False

#
# Refiner
#
__C.REFINE                                 = edict()
__C.REFINE.USE_PRETRAIN_GENERATOR          = True
__C.REFINE.GENERATOR_WEIGHTS               = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v39/checkpoints/best-gan-ckpt.pth'
# __C.REFINE.GENERATOR_TYPE                = 'GAN'
__C.REFINE.GENERATOR_TYPE                  = 'REC'
__C.REFINE.START_EPOCH                     = 1000
__C.REFINE.RANGE_MAX                       = 0.2
__C.REFINE.LEARNING_RATE                   = 1e-3
__C.REFINE.NOISE_LENGTH                    = 32

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
__C.TRAIN.GRAPHX_LEARNING_RATE              = 1e-4
__C.TRAIN.GRAPHX_WEIGHT_DECAY               = 0

# train parameters for psgn fc
__C.TRAIN.PSGN_FC_LEARNING_RATE             = 5e-5
__C.TRAIN.PSGN_FC_CONV_WEIGHT_DECAY         = 1e-5
__C.TRAIN.PSGN_FC_FC_WEIGHT_DECAY           = 1e-3

# __C.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE      = 5e-4
__C.TRAIN.MILESTONES                        = [400]
__C.TRAIN.VIEW_ESTIMATOR_LR_MILESTONES      = [400]
__C.TRAIN.BETAS                             = (0.0, 0.9)
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 100            # weights will be overwritten every save_freq epoch
__C.TRAIN.TRANS_LAMDA                       = 10

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
__C.TEST.NUM_WORKER                         = 4             # number of data workers
__C.TEST.BATCH_SIZE                         = 1
__C.TEST.GENERATOR_WEIGHTS                  = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v39/checkpoints/best-gan-ckpt.pth'
__C.TEST.REFINER_WEIGHTS                    = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v46/checkpoints/best-refine-ckpt.pth'
__C.TEST.RESULT_PATH                        = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/eval_6_28_meeting/eval_v46/test_v46.txt'

#
# Evaluating options
#

'''
# v44
__C.EVALUATE                                = edict()
__C.EVALUATE.VERSION_ID                     = 44
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/eval_v44/'
__C.EVALUATE.GENERATOR_WEIGHTS              = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v39/checkpoints/best-gan-ckpt.pth'
__C.EVALUATE.REFINER_WEIGHTS                = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v44/checkpoints/best-refine-ckpt.pth'
__C.EVALUATE.BATCH_SIZE                     = 1
'''

# v45
__C.EVALUATE                                = edict()
__C.EVALUATE.VERSION_ID                     = 45
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/eval_v45/'
__C.EVALUATE.GENERATOR_WEIGHTS              = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v38/checkpoints/best-rec-ckpt.pth'
__C.EVALUATE.REFINER_WEIGHTS                = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v45/checkpoints/best-refine-ckpt.pth'
__C.EVALUATE.BATCH_SIZE                     = 1


'''
# v46
__C.EVALUATE                                = edict()
__C.EVALUATE.VERSION_ID                     = 46
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/eval_v46/'
__C.EVALUATE.GENERATOR_WEIGHTS              = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v39/checkpoints/best-gan-ckpt.pth'
__C.EVALUATE.REFINER_WEIGHTS                = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v46/checkpoints/best-refine-ckpt.pth'
__C.EVALUATE.BATCH_SIZE                     = 1
'''
