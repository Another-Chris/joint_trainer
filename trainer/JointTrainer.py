from models import BigHead, ECAPA_TDNN_WITH_FBANK
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
        self.feature_extractor = BigHead(dim_in = embed_size, feat_dim=128)

        # loss
        self.supCon = SupConLoss()
        
    def forward_MI(self, source_data, target_data):
        bz = source_data['anchor'].shape[0]

        data = torch.cat(
            [d for _, d in list(source_data.items()) + list(target_data.items())], dim=0)
        embed = self.encoder(data.to(Config.DEVICE))
        s_anchor, s_pos, t_anchor, t_pos = torch.split(embed, 4 * [bz])

        same_lan = torch.cat([
            torch.cat([s_anchor, s_pos], dim=1),
            torch.cat([t_anchor, t_pos], dim=1)
        ], dim=0)

        diff_lan = torch.cat([
            torch.cat([s_anchor, t_pos], dim=1),
            torch.cat([s_pos, t_anchor], dim=1),
            # torch.cat([s_anchor, t_anchor], dim=1),
            # torch.cat([s_pos, t_pos], dim=1),
        ], dim=0)
        
        target = self.discriminator(torch.cat([same_lan, diff_lan], dim = 0))
        
        label = torch.cat([
            torch.ones(size = (same_lan.shape[0], target.shape[1])),
            torch.zeros(size = (diff_lan.shape[0], target.shape[1])),
        ])
        
        return F.binary_cross_entropy_with_logits(target, label.to(Config.DEVICE))

    def forward_supcon(self, source_data, target_data, label=None):
        
        bz = source_data['anchor'].shape[0]

        data = torch.cat(
            [d for _, d in list(source_data.items()) + list(target_data.items())], dim=0)
        embed = F.normalize(self.encoder(data.to(Config.DEVICE)))
        s_anchor, s_pos, t_anchor, t_pos = torch.split(embed, 4 * [bz])
        
        feat_source = torch.cat([s_anchor.unsqueeze(1), s_pos.unsqueeze(1)], dim = 1)
        feat_target = torch.cat([t_anchor.unsqueeze(1), t_pos.unsqueeze(1)], dim = 1)
        
        return (self.supCon(feat_source) + self.supCon(feat_target)) / 2

    def forward(self, ds_gen):
        data, label = next(ds_gen)

        source_data = data['source_data']
        target_data = data['target_data']
        
        return {
            'supcon': self.forward_supcon(source_data, target_data),
            # 'MI': self.forward_MI(source_data, target_data)
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
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss)

            desc += f" loss = {loss:.3f}"
            pbar.set_description(desc)

        loss_val_dict = {key: value/steps for key,
                         value in loss_val_dict.items()}
        return loss_val_dict
