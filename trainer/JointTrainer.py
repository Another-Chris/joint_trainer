from models import Head, ECAPA_TDNN_WITH_FBANK, BigHead
from tqdm import tqdm
from loss import SupConLoss, AAMsoftmax
from models.encoders.ECAPA_TDNN import ECAPA_TDNN_WITH_DSBN
from utils import Config, get_pair
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
    def __init__(self, encoder, embed_size) -> None:
        super().__init__()

        self.encoder = encoder
        self.supcon = SupConLoss()
        self.aamsoftmax = AAMsoftmax(m = 0.2, s = 30, n_class = Config.NUM_CLASSES, n_embed = embed_size)
    
    def forward(self, x, domain):
        # return F.normalize(self.target_head(F.normalize(self.encoder(x, domain))))
        return F.normalize(self.encoder(x, domain))
    
    def forward_simCLR(self, f1, f2, label = None):
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        return self.supcon(feat, label)
        
    def start_train(self, ds_gen):
        data, label = next(ds_gen)
                
        """source domain""" 
        # source_data = torch.cat([data['source_data']['anchor'], data['source_data']['pos']], dim = 0)
        source_data = data['source_data']
        source_feat = self.encoder(source_data.to(Config.DEVICE), 'source', aug = True)       
        spk_loss = self.aamsoftmax(source_feat, label['source_label'].to(Config.DEVICE))
        
        """target domain"""     
        # target_data = data['target_data']
        target_anchor, target_pos = get_pair(data['target_data'])
        target_anchor = F.normalize(self.encoder(target_anchor.to(Config.DEVICE), domain = 'target', aug = True))
        target_pos = F.normalize(self.encoder(target_pos.to(Config.DEVICE), domain = 'target', aug = True))
        
        # target_anchor = F.normalize(self.target_head(F.normalize(target_anchor)))
        # target_pos = F.normalize(self.target_head(F.normalize(target_pos)))
        
        simCLR = self.forward_simCLR(target_anchor,target_pos)
        
        return {
            'spk_loss': spk_loss,
            'simCLR': simCLR
        }


class JointTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()

        self.writer = SummaryWriter(
            log_dir=f"./logs/{exp_name}/{dt.now().strftime('%Y-%m-%d %H.%M.%S')}")

        self.encoder = ECAPA_TDNN_WITH_DSBN(
            C=Config.C, embed_size=Config.EMBED_SIZE)
        self.model = Workers(self.encoder, embed_size=Config.EMBED_SIZE)
        self.model.to(Config.DEVICE)

        self.optim = optim.Adam(self.model.parameters(), lr = Config.LEARNING_RATE)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.95)

    def train_network(self, ds_gen, epoch):

        self.model.train()
        loss_val_dict = {}

        steps = 256
        pbar = tqdm(range(steps))

        for step in pbar:
            
            losses = self.model.start_train(ds_gen)
            loss = torch.sum(torch.stack(list(losses.values())))
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = f""
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
