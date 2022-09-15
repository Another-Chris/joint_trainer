from re import S
from models import PASE, Head
from trainSSL import MAX_FRAMES
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from utils import Config

import torch
import sys

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
sys.path.append('..')


MODEL_NAME = "PASE"


class SupTrainer(ModelTrainer):
    def __init__(self):
        super().__init__(MODEL_NAME)

        # model
        self.encoder = PASE('./configs/PASE+.cfg')
        dim_in = self.encoder(
            torch.zeros(size=(1,1,MAX_FRAMES * 160 + 240))).size(2)
        self.model = Head(self.encoder, dim_in=dim_in, feat_dim=Config.NUM_CLASSES)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_network(self, loader=None, epoch=None):

        self.encoder.train()

        pbar = tqdm(enumerate(loader), total=len(loader))
        loss_epoch = 0

        for step, (data, label) in pbar:

            outp = self.model(data[2].to(Config.DEVICE))
            loss = F.cross_entropy(outp, torch.LongTensor(
                np.eye(Config.NUM_CLASSES, dtype='uint8')[label]).to(Config.DEVICE))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss = loss.detach().cpu().item()
            loss_epoch += loss

            pbar.set_description(f'{loss = :.4f}')

        return loss_epoch / step
