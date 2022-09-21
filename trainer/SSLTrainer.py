from models import Head, ECAPA_TDNN_WITH_FBANK
from tqdm import tqdm
from loss import SupConLoss
from utils import Config
from torch.utils.tensorboard import SummaryWriter

import torch
import time
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append('..')


def get_grad(params):
    grads = [0]
    for p in params:
        if p.grad is None: continue
        grads.append(torch.mean(p.grad).item())
    return max(grads)

class Workers(nn.Module):
    def __init__(self, encoder, embed_size):
        super().__init__()
        
        self.encoder = encoder
        self.proj = Head(dim_in = embed_size, feat_dim = 128)
        self.supConLoss = SupConLoss()
        
    def forward_supCon(self, anchor, pos, label = None):
        bz = anchor.shape[0]
        feat = F.normalize(self.proj(F.normalize(torch.cat([anchor, pos], dim = 0))))
        f1, f2 = torch.split(feat, [bz,bz], dim = 0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        return self.supConLoss(feat, label)     
    
    def forward(self, ds_gen):
        data, label = next(ds_gen)
        feat = {key: self.encoder(d.to(Config.DEVICE), aug = True) for key,d in data.items()}
        
        return {
            'SupCon': self.forward_supCon(feat['anchor'], feat['pos']),
        }


class SSLTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()
        
        self.writer = SummaryWriter(log_dir=f"./logs/{exp_name}/{time.time()}")
        
        self.encoder = ECAPA_TDNN_WITH_FBANK()
        self.model = Workers(self.encoder,embed_size=192)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(),
                                lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size = 5, gamma = 0.95)

    def train_network(self, ds_gen, epoch):

        self.model.train()
        loss_val_dict = {}

        steps = 1024
        pbar = tqdm(range(steps))

        for step in pbar:

            losses = self.model(ds_gen)
            loss = torch.sum(torch.stack(list(losses.values())))
                    
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = f""
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
