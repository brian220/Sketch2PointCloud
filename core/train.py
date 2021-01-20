# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
# https://github.com/hzxie/Pix2Vox
#

import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.valid import valid_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.view_estimater import ViewEstimater

from losses.chamfer_loss import ChamferLoss
from losses.earth_mover_distance import EMD
from losses.cross_entropy_loss import CELoss
from losses.delta_loss import DeltaLoss

def train_net(cfg):
    print("cuda is available?", torch.cuda.is_available())
    print("train")
    
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        # utils.data_transforms.RandomFlip(), # Disable the random flip to avoid problem in view estimation
        # utils.data_transforms.RandomPermuteRGB(), # Sketch data is gray scale image, no need to permute RGB
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
                                                    utils.data_loaders.DatasetType.TRAIN, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
                                                  utils.data_loaders.DatasetType.VAL, val_transforms),
                                                  batch_size=cfg.CONST.BATCH_SIZE,
                                                  num_workers=cfg.TEST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=True)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)

    azi_classes, ele_classes = int(360 / cfg.CONST.BIN_SIZE), int(180 / cfg.CONST.BIN_SIZE)
    view_estimater = ViewEstimater(cfg, azi_classes=azi_classes, ele_classes=ele_classes)

    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in View Estimator: %d.' % (dt.now(), utils.network_utils.count_parameters(view_estimater)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    # decoder.apply(utils.network_utils.init_weights) # Something need to change!

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        view_estimater_solver = torch.optim.Adam(view_estimater.parameters(),
                                          lr=cfg.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        view_estimater_solver = torch.optim.SGD(view_estimater.parameters(),
                                         lr=cfg.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    view_estimater_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(view_estimater_solver,
                                                                milestones=cfg.TRAIN.VIEW_ESTIMATOR_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        view_estimater = torch.nn.DataParallel(view_estimater).cuda()

    # Set up loss functions
    chamfer = ChamferLoss().cuda()
    emd = EMD().cuda()
    criterion_cls_azi = CELoss(360)
    criterion_cls_ele = CELoss(180)
    criterion_reg = DeltaLoss(cfg.CONST.BIN_SIZE)

    # Load pretrained model if exists
    init_epoch = 0
    # best_CD =  10000 # less is better
    best_emd =  10000 # less is better
    best_view_loss = 10000
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        # best_CD = checkpoint['least_CD']
        # best_EMD = checkpoint['least_EMD']
        # best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d at epoch #%d.' %
              (dt.now(), init_epoch, cfg.TRAIN.NUM_EPOCHES))

    # Summary writer for TensorBoard
    output_dir = cfg.DIR.OUT_PAT
    log_dir = output_dir  + 'logs'
    ckpt_dir = output_dir  + 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        reconstruction_losses = utils.network_utils.AverageMeter()
        cls_azi_losses = utils.network_utils.AverageMeter()
        cls_ele_losses = utils.network_utils.AverageMeter()
        reg_losses = utils.network_utils.AverageMeter()
        view_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        view_estimater.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_point_clouds, ground_truth_views) in enumerate(train_data_loader):
            
            # Only one image per batch
            rendering_images = torch.squeeze(rendering_images, 1)

            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_point_clouds = utils.network_utils.var_or_cuda(ground_truth_point_clouds)
            ground_truth_views = utils.network_utils.var_or_cuda(ground_truth_views)
            
            #=================================================#
            #           Train the encoder, decoder            #
            #=================================================#
            vgg_features, image_code = encoder(rendering_images)
            image_code = [image_code]
            generated_point_clouds = decoder(image_code)
            
            # loss computation
            reconstruction_loss = torch.mean(emd(generated_point_clouds, ground_truth_point_clouds))

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            reconstruction_loss.backward()
            encoder_solver.step()
            decoder_solver.step()
            
            #=================================================#
            #          Train the view estimater               #
            #=================================================#
            # output[0]:prediction of azi class
            # output[1]:prediction of ele class
            # output[2]:prediction of azi regression
            # output[3]:prediction of ele regression
            output = view_estimater(vgg_features)

            # loss computation
            loss_cls_azi = criterion_cls_azi(output[0], ground_truth_views[:, 0])
            loss_cls_ele = criterion_cls_ele(output[1], ground_truth_views[:, 1])
            loss_reg = criterion_reg(output[2], output[3], ground_truth_views.float())
            view_loss = loss_cls_azi + loss_cls_ele + loss_reg
            
            # Gradient decent
            view_estimater.zero_grad()
            view_loss.backward()
            view_estimater_solver.step()

            #=================================================#
            #              Show result                        #
            #=================================================#
            reconstruction_losses.update(reconstruction_loss.item())
            cls_azi_losses.update(loss_cls_azi.item())
            cls_ele_losses.update(loss_cls_ele.item())
            reg_losses.update(loss_reg.item())
            view_losses.update(view_loss.item())

            # Append loss to TensorBoard
            # n_itr = epoch_idx * n_batches + batch_idx
            # train_writer.add_scalar('EncoderDecoder/BatchLoss_Rec', reconstruction_loss.item(), n_itr)
            # train_writer.add_scalar('EncoderDecoder/BatchLoss_View', view_loss.item(), n_itr)
            
            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) \
                 REC_Loss = %.4f VIEW_Loss =  %.4f'
                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                   data_time.val, reconstruction_loss.item(), view_loss.item()))
        
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss_Rec', reconstruction_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('EncoderDecoder/EpochLoss_Cls_azi', cls_azi_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('EncoderDecoder/EpochLoss_Cls_ele', cls_ele_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('EncoderDecoder/EpochLoss_Reg', reg_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('EncoderDecoder/EpochLoss_View', view_losses.avg, epoch_idx + 1)
        
        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        view_estimater_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) REC_Loss = %.4f VIEW_Loss =  %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, 
              reconstruction_losses.avg, view_losses.avg))

        # Validate the training models
        current_emd, current_view_loss = valid_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, view_estimater)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)), 
                                                 epoch_idx + 1, 
                                                 encoder, encoder_solver, decoder, decoder_solver, view_estimater, view_estimater_solver,
                                                 best_emd, best_view_loss, best_epoch)
        
        # Save best check point for emd
        if current_emd < best_emd:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_emd = current_emd
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-reconstruction-ckpt.pth'), 
                                                 epoch_idx + 1, 
                                                 encoder, encoder_solver, decoder, decoder_solver, view_estimater, view_estimater_solver,
                                                 best_emd, best_view_loss ,best_epoch)
        
        # Save best check point for view
        if current_view_loss < best_view_loss:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_view_loss = current_emd
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-view-ckpt.pth'), 
                                                 epoch_idx + 1, 
                                                 encoder, encoder_solver, decoder, decoder_solver, view_estimater, view_estimater_solver,
                                                 best_emd, best_view_loss, best_epoch)
        
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
