#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from configs.config_rec import cfg as cfg_rec
from configs.config_gan import cfg as cfg_gan

# from config import cfg
from core.train_rec import train_rec_net

from core.train_gan import train_gan_net


from core.test_rec import test_rec_net
from core.test_gan import test_gan_net

from core.evaluate_rec import evaluate_rec_net
from core.evaluate_gan import evaluate_gan_net

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Sketch To Pointcloud')
    
    parser.add_argument('--train_rec', dest='train_rec', help='Train the reconstruction model', action='store_true')
    parser.add_argument('--train_gan', dest='train_gan', help='Train the GAN model', action='store_true')
    parser.add_argument('--train_refine', dest='train_refine', help='Train the update model', action='store_true')

    parser.add_argument('--test_rec', dest='test_rec', help='Test neural networks', action='store_true')
    parser.add_argument('--test_gan', dest='test_gan', help='Test neural networks', action='store_true')
    parser.add_argument('--test_refine', dest='test_refine', help='Test neural networks', action='store_true')

    parser.add_argument('--evaluate_rec', dest='evaluate_rec', help='Evaluate neural networks', action='store_true')
    parser.add_argument('--evaluate_gan', dest='evaluate_gan', help='Evaluate neural networks (GAN version)', action='store_true')
    parser.add_argument('--evaluate_refine', dest='evaluate_refine', help='Evaluate neural networks (GAN version)', action='store_true')

    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Print config
    print('Use config:')
    if args.train_rec or args.test_rec or args.evaluate_rec:
        model_cfg = cfg_rec
    elif args.train_gan or args.test_gan or args.evaluate_gan:
        model_cfg = cfg_gan

    # Print config 
    pprint(model_cfg)
    
    # Set GPU to use
    if type(model_cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = model_cfg.CONST.DEVICE

    # Start train/test process
    # Train
    if args.train_rec:
        train_rec_net(model_cfg)
    elif args.train_gan:
        train_gan_net(model_cfg)
    # Test
    elif args.test_rec:
        test_rec_net(model_cfg)
    elif args.test_gan:
        test_gan_net(model_cfg)
    # Evaluate
    elif args.evaluate_rec:
        evaluate_rec_net(model_cfg)
    elif args.evaluate_gan:
        evaluate_gan_net(model_cfg)
        
    else:
        print("Please specify the arguments (--train, --test, --evaluate)")


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
