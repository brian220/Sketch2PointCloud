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

__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/caig/FECA2C89CA2C406F/sketch3D/drc/cachedir/blenderRenderPreprocess/%s/%s/render_%d.png'
__C.DATASETS.SHAPENET.DEPTH_PATH            = '/media/caig/FECA2C89CA2C406F/sketch3D/drc/cachedir/blenderRenderPreprocess/%s/%s/depth_%d.png'
__C.DATASETS.SHAPENET.POINT_CLOUD_PATH      = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/%s/%s.ply'
__C.DATASETS.SHAPENET.VIEW_PATH             = '/media/caig/FECA2C89CA2C406F/sketch3D/drc/cachedir/blenderRenderPreprocess/%s/%s/view.txt'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.TOTAL_VIEWS                     = 10
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
__C.DATASET.CLASS                           = 'chair'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.BATCH_SIZE                        = 1
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D
__C.CONST.BIN_SIZE                          = 15
__C.CONST.NUM_POINTS                        = 2048

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = '/media/caig/FECA2C89CA2C406F/sketch3D/2d_supervision/output'
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

#
# Continuous Projection
#
__C.PROJECTION                             = edict()
__C.PROJECTION.GRID_H                      = 64
__C.PROJECTION.GRID_W                      = 64
__C.PROJECTION.SIGMA_SQ                    = 0.5
__C.PROJECTION.NUM_VIEWS                   = 4
__C.PROJECTION.LAMDA_BCE                   = 1.
__C.PROJECTION.LAMDA_AFF_FWD               = 1.
__C.PROJECTION.LAMDA_AFF_BWD               = 1.

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 1000
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.LEARNING_RATE                     = 5e-5
# __C.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE      = 5e-4
__C.TRAIN.MILESTONES                        = [400]
__C.TRAIN.VIEW_ESTIMATOR_LR_MILESTONES      = [400]
__C.TRAIN.WEIGHT_DECAY                      = 1e-5
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
__C.TEST.RECONSTRUCTION_WEIGHTS             = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v4/checkpoints/best-reconstruction-ckpt.pth'
__C.TEST.VIEW_ENCODER_WEIGHTS               = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-reconstruction-ckpt.pth'
__C.TEST.VIEW_ESTIMATION_WEIGHTS            = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-view-ckpt.pth'
__C.TEST.RESULT_PATH                        = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/test.txt'

#
# Evaluating options
#
__C.EVALUATE                                = edict()
__C.EVALUATE.TAXONOMY_ID                    = '03001627'
__C.EVALUATE.INPUT_IMAGE_FOLDER             = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate/evaluate_input_img/'
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate/evaluate_output/'
__C.EVALUATE.INFO_FILE                      = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate/eval_chair.txt'
__C.EVALUATE.RECONSTRUCTION_WEIGHTS         = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v4/checkpoints/best-reconstruction-ckpt.pth'
__C.EVALUATE.VIEW_ENCODER_WEIGHTS           = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-reconstruction-ckpt.pth'
__C.EVALUATE.VIEW_ESTIMATION_WEIGHTS        = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-view-ckpt.pth'

#
# Evaluate on the true habd draw image
#
__C.EVALUATE_HAND_DRAW                                = edict()
__C.EVALUATE_HAND_DRAW.INPUT_IMAGE_FOLDER             = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/hand_draw_input_img/'
__C.EVALUATE_HAND_DRAW.OUTPUT_FOLDER                  = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/hand_draw_output/'
__C.EVALUATE_HAND_DRAW.RECONSTRUCTION_WEIGHTS         = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-reconstruction-ckpt.pth'
__C.EVALUATE_HAND_DRAW.VIEW_ESTIMATION_WEIGHTS        = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-view-ckpt.pth'

#
# Evaluating options
#
__C.EVALUATE_FIXED_VIEW                             = edict()
__C.EVALUATE_FIXED_VIEW.TAXONOMY_ID                 = '03001627'
__C.EVALUATE_FIXED_VIEW.RESULT_DIR                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/'
__C.EVALUATE_FIXED_VIEW.SAMPLE_FILE                 = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/evaluate_sample.txt'
__C.EVALUATE_FIXED_VIEW.VIEW_FILE                   = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/fixed_view.txt'
__C.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS      = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v4/checkpoints/best-reconstruction-ckpt.pth'