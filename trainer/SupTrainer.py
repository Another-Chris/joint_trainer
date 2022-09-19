from re import S
from turtle import forward
from models import PASE, Head
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


class PASE_with_statspool(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.pase = PASE('./configs/PASE+.cfg')
        
    def forward(self, x):
        x = self.pase(x)
        # std = torch.std(x, dim = 2)
        # mean = torch.mean(x, dim = 2)
        # cat = torch.cat([std, mean], dim = 1)
        print(x.shape)
        exit()
        return torch.mean(x, dim = 2)
        

class SupTrainer(ModelTrainer):
    def __init__(self):
        super().__init__(MODEL_NAME)

        # model
        self.encoder = PASE_with_statspool().to(Config.DEVICE)
        dim_in = self.encoder(
            torch.zeros(size=(1,1,Config.MAX_FRAMES * 160 + 240)).to(Config.DEVICE)).size(1)
        self.model = Head(self.encoder, dim_in=dim_in, feat_dim=Config.NUM_CLASSES)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def train_network(self, loader=None, epoch=None):

        self.encoder.train()

        pbar = tqdm(enumerate(loader), total=len(loader))
        loss_epoch = 0

        for step, (data, label) in pbar:

            outp = self.model(data[2].to(Config.DEVICE))
            loss = F.cross_entropy(outp, label.to(Config.DEVICE))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss = loss.detach().cpu().item()
            loss_epoch += loss

            pbar.set_description(f'{loss = :.4f}')

        return loss_epoch / (step + 1)
