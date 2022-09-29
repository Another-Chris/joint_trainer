from turtle import forward
from models import Head, ECAPA_TDNN_WITH_FBANK
from tqdm import tqdm
from loss import SupConLoss
from utils import Config
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append('..')


class Workers(nn.Module):
    def __init__(self, encoder, embed_size) -> None:
        super().__init__()

        self.encoder = encoder
        self.supCon = SupConLoss()
        # self.head = Head(dim_in = 2 * embed_size, feat_dim = 1)
        
    def forward_supcon(self, feat,bz, label=None):
        feat = F.normalize(feat)
        f1, f2 = torch.split(feat, [bz, bz], dim=0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        return self.supCon(feat, label)

    # def forward_lim(self, feat, bz):
    #     anchor, pos, diff = torch.split(feat, 3*[bz])
        
    #     x1 = torch.cat([anchor, pos], dim = 1)
    #     x2 = torch.cat([anchor, diff], dim = 1)
    #     X = self.discriminator(torch.cat([x1, x2], dim = 0))
        
    #     slen = X.shape[1]
    #     label = torch.cat([torch.ones(x1.shape[0], slen), torch.zeros(x2.shape[0], slen)])
    #     return F.binary_cross_entropy_with_logits(X, label)
        

    def forward(self, ds_gen):
        data, _ = next(ds_gen)
        
        bz = data['anchor'].shape[0]
        feat = self.encoder(torch.cat([data['anchor'], data['pos']], dim=0).to(Config.DEVICE), aug=True)
        
        return {
            'simCLR': self.forward_supcon(feat, bz),
            # 'LIM': self.forward_lim(data['anchor'], data['pos'], data['diff'])
        }


class SSLTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()

        self.writer = SummaryWriter(
            log_dir=f"./logs/{exp_name}/{dt.now().strftime('%Y-%m-%d %H.%M.%S')}")

        self.encoder = ECAPA_TDNN_WITH_FBANK(C=Config.C, embed_size=Config.EMBED_SIZE)
        self.model = Workers(self.encoder, Config.EMBED_SIZE)
        self.model.to(Config.DEVICE)
        
        self.optim = optim.Adam(self.model.parameters(), lr = Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.95)

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
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss)

            desc += f" {loss = :.3f}"
            pbar.set_description(desc)

        loss_val_dict = {key: value/steps for key,
                         value in loss_val_dict.items()}
        return loss_val_dict
