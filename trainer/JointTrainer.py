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
        self.estimator_s = Head(dim_in=embed_size, feat_dim=1)
        self.estimator_t = Head(dim_in=embed_size, feat_dim=1)
        self.projector = Head(dim_in=embed_size, feat_dim=128)
        self.discriminator = Head(dim_in = embed_size, feat_dim = 128)

        # loss
        self.supCon = SupConLoss()
        self.kl = nn.KLDivLoss(reduction = 'batchmean')
        
    def forward_kl(self, source_data, target_data):
        bz = source_data['anchor'].shape[0]

        data = torch.cat(
            [d for _, d in source_data.items() + target_data.items()], dim=0)
        embed = self.encoder(data)
        s_anchor, s_pos, t_anchor, t_pos = torch.split(embed, 4 * [bz])
        
        s_dist = F.logsigmoid(self.estimator_s(torch.cat([s_anchor, s_pos], dim = 0)))
        t_dist = F.logsigmoid(self.estimator_t(torch.cat([t_anchor, t_pos], dim = 0)))
        
        return self.kl(s_dist, t_dist) + self.kl(t_dist, s_dist)
        

    def forward_MI(self, source_data, target_data):
        bz = source_data['anchor'].shape[0]

        data = torch.cat(
            [d for _, d in source_data.items() + target_data.items()], dim=0)
        embed = self.encoder(data)
        s_anchor, s_pos, t_anchor, t_pos = torch.split(embed, 4 * [bz])

        same_lan = torch.cat([
            torch.cat([s_anchor, s_pos], dim=1),
            torch.cat([t_anchor, t_pos], dim=1)
        ], dim=0)

        diff_lan = torch.cat([
            torch.cat([s_anchor, t_anchor], dim=1),
            torch.cat([s_anchor, t_pos], dim=1),
            torch.cat([s_pos, t_pos], dim=1),
            torch.cat([s_pos, t_pos], dim=1),
        ], dim=0)
        
        target = self.discriminator(torch.cat([same_lan, diff_lan], dim = 0))
        
        label = torch.cat([
            torch.ones(size = (same_lan.shape[0], target.shape[1])),
            torch.zeros(size = (diff_lan.shape[0], target.shape[1])),
        ])
        
        return F.binary_cross_entropy_with_logits(target, label)

    def forward_lim(self, anchor, pos, diff):
        bz = anchor.shape[0]
        feat = self.encoder(
            torch.cat([anchor, pos, diff], dim=0).to(Config.DEVICE))
        feat_anchor, feat_pos, feat_diff = torch.split(feat, 3 * [bz])

        X1 = torch.cat([feat_anchor, feat_pos], dim=1)
        X2 = torch.cat([feat_anchor, feat_diff], dim=1)
        X = self.discriminator(torch.cat([X1, X2], dim=0))

        slen = X.shape[1]
        y = torch.cat([torch.ones(size=(X1.shape[0], slen)), torch.zeros(
            size=(X2.shape[0], slen))]).to(Config.DEVICE)

        return F.binary_cross_entropy_with_logits(X, y)

    def forward_local_supCon(self, anchor, pos, label=None):
        bz = anchor.shape[0]
        feat = F.normalize(self.proj_local(F.normalize(self.encoder(
            torch.cat([anchor, pos], dim=0).to(Config.DEVICE), aug=True))))
        f1, f2 = torch.split(feat, [bz, bz], dim=0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        return self.supCon(feat, label)

    def forward(self, ds_gen):
        data, label = next(ds_gen)

        source_data = data['source_data']
        target_data = data['target_data']

        anchor = torch.cat(
            [source_data['anchor'], target_data['anchor']], dim=0)
        pos = torch.cat([source_data['pos'], target_data['pos']], dim=0)

        return {
            'supCon': self.forward_local_supCon(anchor, pos),
        }


class JointTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()

        self.writer = SummaryWriter(log_dir=f"./logs/{exp_name}/{time.time()}")

        self.encoder = ECAPA_TDNN_WITH_FBANK(
            C=512, embed_size=Config.EMBED_SIZE)
        self.model = Workers(self.encoder, embed_size=Config.EMBED_SIZE)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(),
                                lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optim, step_size=5, gamma=0.95)

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
