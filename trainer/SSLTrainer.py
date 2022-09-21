from models import Head, ECAPA_TDNN_WITH_FBANK
from tqdm import tqdm
from loss import SupConLoss
from utils import Config

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append('..')


class Workers(nn.Module):
    def __init__(self, encoder, embed_size):
        super().__init__()
        
        self.encoder = encoder
        self.proj = Head(dim_in = embed_size, feat_dim = 128)
        self.supConLoss = SupConLoss()
        
    def forward_supCon(self, anchor, pos):
        bz = anchor.shape[0]
        feat = F.normalize(self.proj(F.normalize(torch.cat([anchor, pos], dim = 0))))
        f1, f2 = torch.split(feat, [bz,bz], dim = 1)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(2)], dim = 1)
        return self.supConLoss(feat)     
    
    def forward(self, batch):
        data, _ = batch
        feat = [self.encoder(d.to(Config.DEVICE)) for d in data]
        
        return {
            'SupCon': self.forward_supCon(feat['anchor'], feat['pos']),
        }


class SSLTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__(exp_name)

        # model
        self.encoder = ECAPA_TDNN_WITH_FBANK()
        self.model = Workers(self.encoder,embed_size=192)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(),
                                lr=Config.LEARNING_RATE)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        steps = len(loader)
        pbar = tqdm(enumerate(loader),total = len(loader))

        for step, batch in pbar:

            losses = self.model(batch)
            loss = torch.sum(torch.stack(list(losses.values())))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = ""
            for key, val in losses.items():
                val = val.detach().cpu()
                loss_val_dict[key] = (loss_val_dict.get(key, 0) + val)
                self.writer.add_scalar(
                    f"step/{key}", val, epoch * steps + step)
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            self.writer.add_scalar(
                f"step/loss", loss, epoch * steps + step)
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss)

            desc += f" {loss = :.3f}"
            pbar.set_description(desc)

        loss_val_dict = {key: value/steps for key,
                         value in loss_val_dict.items()}
        return loss_val_dict
