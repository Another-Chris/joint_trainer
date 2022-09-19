from models import GIM, LIM, Head
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from loss import SubConLoss
from utils import Config

import torch
import importlib
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append('..')


def make_labels(y):
    bsz = y.size(0) // 2
    slen = y.size(1)
    label = torch.cat((torch.ones(bsz, slen, requires_grad=False), torch.zeros(bsz, slen, requires_grad=False)), dim=0)
    return label

class Workers(nn.Module):
    def __init__(self, encoder, ds_gen, embed_size) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.ds_gen = ds_gen

        self.gim = GIM(embed_size=embed_size, device=Config.DEVICE)
        self.lim = LIM(embed_size=embed_size, device=Config.DEVICE)
        self.proj = Head(dim_in=embed_size, feat_dim=512)

        self.sup_loss = SubConLoss()

    def forward_SUP(self, anchor, same, label):
        bz = anchor.shape[0]
        feat = F.normalize(self.proj(torch.cat([anchor, same], dim = 0)))
        f1, f2 = torch.split(feat, [bz, bz], dim = 0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        return self.sup_loss(feat, label.to(Config.DEVICE))

    def forward_LIM(self, feat_anchor, feat_pos, feat_neg):
        return self.lim(feat_anchor, feat_pos, feat_neg)

    def forward_GIM(self, feat_anchor, feat_pos, feat_neg):
        return self.gim(feat_anchor, feat_pos, feat_neg)

    def forward_encoder(self, data):
        embeds = {}
        for key, value in data.items():
            if 'gim' in key:  
                value = value[:Config.BATCH_SIZE // Config.GIM_SEGS]              
                value = torch.cat(torch.split(value, Config.GIM_SEGS * [1], dim = 1), dim = 0)
            embeds[key] = self.encoder(value.to(Config.DEVICE))
        return embeds

    def start_train(self):

        data, label = next(self.ds_gen)
        embeds = self.forward_encoder(data)

        return {
            'SUP': self.forward_SUP(embeds['source_anchor'], embeds['source_same'], label['source_anchor']),
            'LIM': self.forward_LIM(embeds['target_anchor'], embeds['target_same'], embeds['target_diff']),
            'GIM': self.forward_GIM(embeds['gim_anchor'], embeds['gim_same'], embeds['gim_diff']),
        }


class JointTrainer(ModelTrainer):
    def __init__(self, exp_name, model_name, ds_gen):
        super().__init__(exp_name)

        # model
        self.encoder = importlib.import_module(
            'models').__getattribute__(model_name)()
        self.model = Workers(self.encoder, embed_size=192, ds_gen=ds_gen)
        self.model.to(Config.DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(),
                                lr=Config.LEARNING_RATE)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        steps = 512
        pbar = tqdm(range(steps))

        for step in pbar:

            losses = self.model.start_train()

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
