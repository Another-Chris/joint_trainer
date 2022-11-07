from models.ECAPA_TDNN import ECAPA_TDNN
from models.common import Discriminator

from tqdm import tqdm
from loss import SupCon, AAMsoftmax
from utils import Config
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

import torch
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.append('..')
    
class Workers(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = ECAPA_TDNN(C=Config.C, embed_size=Config.EMBED_SIZE)
        self.supcon = SupCon()
        self.aamsoftmax = AAMsoftmax(m = 0.2, s = 30, n_class = Config.NUM_CLASSES, n_embed = Config.EMBED_SIZE)
        self.discriminator = Discriminator(dim_in = Config.EMBED_SIZE, feat_dim = 1, hidden_size=1024)
    
    def forward(self, x, domain):
        return F.normalize(self.encoder(x))
        
    def start_train(self, ds_gen, alpha):
        data, label = next(ds_gen)
        losses = {}
        
        """source domain""" 
        # source_anchor, source_pos = data['source_data']['anchor'],data['source_data']['pos']
        # source_f1 = self.encoder(source_anchor.to(Config.DEVICE))
        # source_f2 = self.encoder(source_pos.to(Config.DEVICE))
        # spk_loss = self.aamsoftmax(source_f1, label['source_label'].to(Config.DEVICE))
        
        source_data = data['source_data']
        source_feat = self.encoder(source_data.to(Config.DEVICE))
        spk_loss = self.aamsoftmax(source_feat, label['source_label'].to(Config.DEVICE))
        losses['spk_loss'] = spk_loss

        """target domain"""     
        # target_anchor, target_pos = data['target_data']['anchor'],data['target_data']['pos']
        # target_f1 = self.encoder(target_anchor.to(Config.DEVICE))
        # target_f2 = self.encoder(target_pos.to(Config.DEVICE))
        
        # target_feat = torch.cat([F.normalize(target_f1).unsqueeze(1), F.normalize(target_f2).unsqueeze(1)], dim = 1)
        # source_feat = torch.cat([F.normalize(source_f1).unsqueeze(1), F.normalize(source_f2).unsqueeze(1)], dim = 1)
        # # feat = torch.cat([source_feat, target_feat], dim = 0)
        # simCLR = (self.supcon(source_feat) + self.supcon(target_feat)) / 2
        # losses['simCLR'] = simCLR
        
        target_data = data['target_data']
        target_feat = self.encoder(target_data.to(Config.DEVICE))
        
        """DANN"""
        # dann = F.cross_entropy(dann_out, label['target_genre'].to(Config.DEVICE))
        
        bz = source_feat.shape[0]
        dann_feat = F.normalize(torch.cat([source_feat, target_feat]))
        dann_out = self.discriminator(dann_feat, alpha)
        labels = torch.cat([torch.zeros(bz,1), torch.ones(bz,1)]).to(Config.DEVICE)
        dann = F.binary_cross_entropy_with_logits(dann_out, labels)
        losses['language_DAT'] = dann
        
        return losses


class Trainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()

        self.writer = SummaryWriter(
            log_dir=f"./logs/{exp_name}/{dt.now().strftime('%Y-%m-%d %H.%M.%S')}")
        
        self.model = Workers()
        self.model.to(Config.DEVICE)

        self.optim = optim.Adam(self.model.parameters(), lr = Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.95)

    def train_network(self, ds_gen, epoch):

        self.model.train()
        loss_val_dict = {}

        steps = 256
        pbar = tqdm(range(steps))

        for step in pbar:
            p = float(step + epoch * steps) / 200 / steps
            alpha = min(2. / (1. + np.exp(-10 * p)) - 1, 0.5)
            
            losses = self.model.start_train(ds_gen, alpha)
            loss = torch.sum(torch.stack(list(losses.values())))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = f"alpha={alpha :.4f}"
            for key, val in losses.items():
                val = val.detach().cpu()
                loss_val_dict[key] = (loss_val_dict.get(key, 0) + val)
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss)

            desc += f" loss = {loss:.3f}"
            pbar.set_description(desc)

        loss_val_dict = {key: value/steps for key,
                         value in loss_val_dict.items()}
        return loss_val_dict
