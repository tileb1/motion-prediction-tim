#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[1])
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(ckpt_path, file_name[0])
        torch.save(state, file_path)


def save_model(model, name):
    create_dir = True
    for f in os.listdir('.'):
        if os.path.isdir(f) and f == 'models':
            create_dir = False
            break

    if create_dir:
        os.mkdir('./models')

    torch.save(model.state_dict(), './models/' + name)


def load_model(model, name):
    model.load_state_dict(torch.load('./models/' + name, map_location=lambda storage, loc: storage))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
