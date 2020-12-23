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
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.POINT_CLOUD_PATH      = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/shape_net_core_uniform_samples_2048/%s/%s.ply'
__C.DATASETS.SHAPENET.VIEW_PATH             = '/media/itri/Files_2TB/chaoyu/pointcloud3d/dataset/ShapeNetRendering_copy/%s/%s/rendering/rendering_metadata.txt'

#
# Dataset
#
__C.DATASET                                 = edict()
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
__C.DIR.OUT_PATH                            = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'
__C.DIR.RESULT_PATH                         = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/test.txt'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False

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
__C.TRAIN.ENCODER_LEARNING_RATE             = 5e-4
__C.TRAIN.DECODER_LEARNING_RATE             = 5e-4
__C.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE      = 5e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [200]
__C.TRAIN.DECODER_LR_MILESTONES             = [200]
__C.TRAIN.VIEW_ESTIMATOR_LR_MILESTONES      = [200]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.TRANS_LAMDA                       = 10


#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
__C.TEST.NUM_WORKER                         = 4             # number of data workers
__C.TEST.RECONSTRUCTION_WEIGHTS             = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-reconstruction-ckpt.pth'
__C.TEST.VIEW_ESTIMATION_WEIGHTS            = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-view-ckpt.pth'

#
# Evaluating options
#
__C.EVALUATE                                = edict()
__C.EVALUATE.TAXONOMY_ID                    = '03001627'
__C.EVALUATE.INPUT_IMAGE_FOLDER             = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/evaluate_input_img/'
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/evaluate_output/'
__C.EVALUATE.INFO_FILE                      = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/eval_chair.txt'
__C.EVALUATE.RECONSTRUCTION_WEIGHTS         = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-reconstruction-ckpt.pth'
__C.EVALUATE.VIEW_ESTIMATION_WEIGHTS        = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-view-ckpt.pth'


