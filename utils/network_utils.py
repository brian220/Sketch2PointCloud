# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# most of codes are borrowed from Pix2Vox: 
# https://github.com/hzxie/Pix2Vox

import torch

from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, 
                     encoder, encoder_solver, decoder, decoder_solver, view_estimator, view_estimator_solver, 
                     best_emd, best_view_loss, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict(),
        'view_estimator_state_dict': view_estimator.state_dict(),
        'view_estimator_solver_state_dict': view_estimator_solver.state_dict(),
        'best_emd': best_emd,
        'best_view_loss': best_view_loss,
        'best_epoch': best_epoch
    }

    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
