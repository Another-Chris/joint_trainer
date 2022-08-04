#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, step_size=test_interval, gamma=lr_decay)

	lr_step = 'step'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step
