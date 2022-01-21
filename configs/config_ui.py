# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# GraphX
#
__C.GRAPHX                                 = edict()
__C.GRAPHX.USE_GRAPHX                      = True
__C.GRAPHX.NUM_INIT_POINTS                 = 2048
__C.GRAPHX.RETURN_IMG_FEATURES             = False

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
# Refiner
#
__C.REFINE                                 = edict()
__C.REFINE.USE_PRETRAIN_GENERATOR          = True
__C.REFINE.GENERATOR_WEIGHTS               = '/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v39/checkpoints/best-gan-ckpt.pth'
__C.REFINE.GENERATOR_TYPE                  = 'GAN'
__C.REFINE.START_EPOCH                     = 1000
__C.REFINE.RANGE_MAX                       = 0.2
__C.REFINE.LEARNING_RATE                   = 1e-3
__C.REFINE.NOISE_LENGTH                    = 32

#
# Application Settings
# 

__C.INFERENCE                               = edict()

# Ten references
reference1_id = '1e283319d1f2782ff2c92c2a4f65876'
reference2_id = '3b88922c44e2311519fb4103277a6b93'
reference3_id = '27c00ec2b6ec279958e80128fd34c2b1'
reference4_id = '4231883e92a3c1a21c62d11641ffbd35'
reference5_id = 'f1dac1909107c0eef51f77a6d7299806'
reference6_id = '1ab4c6ef68073113cf004563556ddb36'
reference7_id = '1aeb17f89e1bea954c6deb9ede0648df'
reference8_id = '484f0070df7d5375492d9da2668ec34c'
reference9_id = '8f2cc8ff68f3208ec935dd3bb5739fc'
reference10_id = '717e28c855c935c94d2d89cc1fd36fca'

__C.INFERENCE.REFERENCE1_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference1_id + ".ply"
__C.INFERENCE.REFERENCE2_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference2_id + ".ply"
__C.INFERENCE.REFERENCE3_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference3_id + ".ply"
__C.INFERENCE.REFERENCE4_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference4_id + ".ply"
__C.INFERENCE.REFERENCE5_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference5_id + ".ply"
__C.INFERENCE.REFERENCE6_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference6_id + ".ply"
__C.INFERENCE.REFERENCE7_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference7_id + ".ply"
__C.INFERENCE.REFERENCE8_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference8_id + ".ply"
__C.INFERENCE.REFERENCE9_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference9_id + ".ply"
__C.INFERENCE.REFERENCE10_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/" + reference10_id + ".ply"

__C.INFERENCE.REFERENCE1_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference1_id + "/render_0.png"
__C.INFERENCE.REFERENCE2_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference2_id + "/render_5.png"
__C.INFERENCE.REFERENCE3_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference3_id + "/render_17.png"
__C.INFERENCE.REFERENCE4_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference4_id + "/render_4.png"
__C.INFERENCE.REFERENCE5_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference5_id + "/render_12.png"
__C.INFERENCE.REFERENCE6_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference6_id + "/render_17.png"
__C.INFERENCE.REFERENCE7_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference7_id + "/render_8.png"
__C.INFERENCE.REFERENCE8_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference8_id + "/render_11.png"
__C.INFERENCE.REFERENCE9_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference9_id + "/render_3.png"
__C.INFERENCE.REFERENCE10_ICON_PATH = "/media/caig/FECA2C89CA2C406F/sketch3D/drc/shapenet_obj_render/blenderRenderPreprocess/03001627/" + reference10_id + "/render_12.png"


__C.INFERENCE.SKETCH_3D_UI_PATH              = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/"
__C.INFERENCE.TEST_SAMPLES_FOLDER            = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/test_samples/"
__C.INFERENCE.ICONS_PATH                     = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/icons/"

# Cache path
__C.INFERENCE.CACHE_IMAGE_PATH               = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/imgs/"
__C.INFERENCE.SCREENSHOT_IMAGE_PATH          = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/imgs/screenshot_image.png"
__C.INFERENCE.SKETCH_IMAGE_PATH              = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/imgs/sketch_image.png"
__C.INFERENCE.REFINE_IMAGE_PATH              = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/imgs/refine_image.png"
__C.INFERENCE.EMPTY_IMAGE_PATH               = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/imgs/empty_image.png"
__C.INFERENCE.UPDATE_IMAGE_PATH              = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/imgs/update_image.png"

__C.INFERENCE.CACHE_POINT_CLOUD_PATH         = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/cache/pcs/"

__C.INFERENCE.USER_EVALUATION_PATH           = "/media/caig/FECA2C89CA2C406F/sketch3D_final/sketch_part_rec/sketch_3d_ui/user_evaluation/"

__C.INFERENCE.GENERATOR_WEIGHTS              = "/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v39/checkpoints/best-gan-ckpt.pth"
__C.INFERENCE.REFINER_WEIGHTS                = "/media/caig/FECA2C89CA2C406F/sketch3D/results/outputs/output_v44/checkpoints/best-refine-ckpt.pth"
