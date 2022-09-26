from turtle import forward
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



class Workers(nn.Module):
    def __init__(self, encoder, embed_size) -> None:
        super().__init__()
        
        self.encoder = encoder 
        self.proj_local = Head(dim_in = embed_size, feat_dim = 128)
        self.discriminator = Head(dim_in = 2 * embed_size, feat_dim = 1)
        self.supCon = SupConLoss()
        
    def forward_lim(self, anchor, pos, diff):
        bz = anchor.shape[0]
        feat = self.encoder(torch.cat([anchor, pos, diff], dim = 0).to(Config.DEVICE))
        feat_anchor, feat_pos, feat_diff = torch.split(feat, 3 * [bz])
        
        X1 = torch.cat([feat_anchor, feat_pos], dim = 1)
        X2 = torch.cat([feat_anchor, feat_diff], dim = 1)
        X = self.discriminator(torch.cat([X1,X2], dim = 0))
        
        slen = X.shape[1]
        y = torch.cat([torch.ones(size = (X1.shape[0], slen)), torch.zeros(size = (X2.shape[0], slen))]).to(Config.DEVICE)

        return F.binary_cross_entropy_with_logits(X, y)
        
    
    def forward_local_supCon(self, anchor, pos, label = None):
        bz = anchor.shape[0]
        feat = F.normalize(self.proj_local(F.normalize(self.encoder(torch.cat([anchor, pos], dim = 0).to(Config.DEVICE), aug = True))))
        f1, f2 = torch.split(feat, [bz,bz], dim = 0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        return self.supCon(feat, label)       
    
    
    def forward(self, ds_gen):
        data, _ = next(ds_gen)
        
        source_data, source_label = data['source_data'], data['source_label']
        target_data, target_label = data['target_data'], data['target_label']
        
        return {
            # 'supCon': self.forward_local_supCon(data['anchor'], data['pos']),

        }


class JointTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()
        
        self.writer = SummaryWriter(log_dir=f"./logs/{exp_name}/{time.time()}")
        
        self.encoder = ECAPA_TDNN_WITH_FBANK(C = 512, embed_size = Config.EMBED_SIZE)
        self.model = Workers(self.encoder,embed_size = Config.EMBED_SIZE)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(),
                                lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size = 5, gamma = 0.95)

    def train_network(self, ds_gen, epoch):

        self.model.train()
        loss_val_dict = {}

        steps = 512
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
