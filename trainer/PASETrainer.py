from models import Head, PASE
from tqdm import tqdm
from loss import SupConLoss
from utils import Config

import torch
import importlib
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
sys.path.append('..')


class Workers(nn.Module):
    def __init__(self, encoder, embed_size, pase_frames):
        super().__init__()
        
        self.encoder = encoder
        self.pase = PASE('./configs/PASE+.cfg')

        self.dis_gim = Head(dim_in = pase_frames, feat_dim = 128)
        self.dis_lim = Head(dim_in = pase_frames, feat_dim = 128)
        self.proj = Head(dim_in = embed_size, feat_dim = 128)
        self.encoder_loss = SupConLoss()
        
    def forward_LIM(self, anchor, pos, neg):
        bz = anchor.size()[0]
        
        randi = np.random.choice(anchor.shape[2], 3)
        anchor_frame = anchor[:, :, randi[0]]
        pos_frame = pos[:, :, randi[1]]
        neg_frame = neg[:, :, randi[2]]

        pos = torch.cat([anchor_frame, pos_frame], dim = 1)
        neg = torch.cat([anchor_frame, neg_frame], dim = 1)
        z1z2 = torch.cat([pos, neg], dim = 0)
        target = self.dis_lim(z1z2)    
        label = torch.cat([torch.ones(size = (bz, target.shape[1])), torch.zeros(size = (bz,target.shape[1]))], dim = 0).to(self.device)
        return F.binary_cross_entropy_with_logits(target, label)
        
    def forward_GIM(self, anchor, pos, neg):
        
        bz = anchor.size()[0]
        pos = torch.cat([anchor, pos], dim = 1)
        neg = torch.cat([anchor, neg], dim = 1)
        z1z2 = torch.mean(torch.cat([pos, neg], dim = 0), dim = 2)
        target = self.dis_gim(z1z2)    
        label = torch.cat([torch.ones(size = (bz, target.shape[1])), torch.zeros(size = (bz,target.shape[1]))], dim = 0).to(self.device)
        return F.binary_cross_entropy_with_logits(target, label)
        
    def forward_encoder(self, anchor, pos, neg, anchor_label, neg_label):
         
        feat = F.normalize(self.proj(F.normalize(self.encoder(torch.cat([anchor, pos, neg], dim = 0)))))
        label = torch.cat(2 * [anchor_label] + [neg_label], dim = 0).to(Config.DEVICE)
        
        return self.encoder_loss(feat, label)
    
    def forward(self, batch):
        data, label = batch
        feat = [self.pase(d.to(Config.DEVICE)) for d in data]
        
        return {
            'LIM': self.forward_LIM(feat['anchor'], feat['same'], feat['diff']),
            'GIM': self.forward_GIM(feat['anchor'], feat['same'], feat['diff']),
            'encoder': self.forward_encoder(feat['anchor'], feat['same'], feat['diff'], label['anchor'], label['diff'])
        }


class PASETrainer(torch.nn.Module):
    def __init__(self, exp_name, model_name):
        super().__init__(exp_name)

        # model
        self.encoder = importlib.import_module(
            'models').__getattribute__(model_name)()
        
        self.model = Workers(embed_size=192, pase_frames=601)
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

    def save_parameters(self, path):
        torch.save(self.model.encoder.state_dict(), f'{path}/encoder.model')
        torch.save(self.model.pase.state_dict(), f'{path}/pase.model')
    
    def load_parameters(self, path):
        self.model.encoder.load_state_dict(torch.load(f'{path}/encoder.model'))
        self.model.pase.load_state_dict(torch.load(f'{path}/pase.model'))
        