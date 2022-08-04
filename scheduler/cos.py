#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch


def Scheduler(optimizer, **kwargs):

    sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min = 1e-7)

    lr_step = 'iteration'

    print('Initialised step cos scheduler')

    return sche_fn, lr_step
