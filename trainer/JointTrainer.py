from xml.sax.handler import feature_external_ges
from models import Cls, GIM, LIM, Head
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from loss import SubConLoss, AngleProtoLoss
from utils import Config

import torch
import importlib
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append('..')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Workers(nn.Module):
    def __init__(self, encoder, ds_gens, embed_size) -> None:
        super().__init__()

        self.encoder = encoder
        self.ds_gens = ds_gens

        self.gim = GIM(embed_size=embed_size, device=Config.DEVICE)
        self.lim = LIM(embed_size=embed_size, device=Config.DEVICE)
        self.proj = Head(embed_size=embed_size, feat_dim=128)

        self.sup_loss = AngleProtoLoss()

    def train_SUP(self, source_data):
        data_anchor, data_pos = source_data[0].to(
            Config.DEVICE), source_data[1].to(Config.DEVICE)
        data = torch.cat([data_anchor, data_pos], dim=0)
        feat = self.proj(self.encoder(data))
        bz = data_anchor.size()[0]

        feat_anchor, feat_pos = torch.split(feat, [bz, bz], dim=0)
        feat = torch.cat([feat_anchor.unsqueeze(
            1), feat_pos.unsqueeze(1)], dim=1)

        return self.sup_loss(feat)[0]

    def train_LIM(self, target_data):
        data_anchor, data_pos, data_neg = target_data[0].to(
            Config.DEVICE), target_data[1].to(Config.DEVICE), target_data[2].to(Config.DEVICE)

        feat_anchor, feat_pos, feat_neg = self.encoder(
            data_anchor), self.encoder(data_pos), self.encoder(data_neg)
        return self.lim(feat_anchor, feat_pos, feat_neg)

    def train_GIM(self, gim_data):
        data_anchor, data_pos, data_neg = gim_data[0].to(
            Config.DEVICE), gim_data[1].to(Config.DEVICE), gim_data[2].to(Config.DEVICE)

        feat_anchor, feat_pos, feat_neg = self.encoder(
            data_anchor), self.encoder(data_pos), self.encoder(data_neg)
        return self.gim(feat_anchor, feat_pos, feat_neg)

    def train_subCon(self, segs, augs, bz, nviews):
        outp_segs = self.worker_subcon(self.get_fbank(segs, aug=False))
        outp_augs = self.worker_subcon(self.get_fbank(augs, aug=True))

        outp = torch.cat([outp_segs, outp_augs], dim=0)
        outp = torch.cat([d.unsqueeze(1)
                         for d in torch.split(outp, [bz] * nviews, dim=0)], dim=1)
        loss = self.loss_subcon(outp)
        return loss

    def start_train(self):

        source_gen, target_gen, gim_gen = self.ds_gens[
            'source_gen'], self.ds_gens['target_gen'], self.ds_gens['gim_gen']

        source_data, _ = next(source_gen)
        target_data, _ = next(target_gen)
        gim_data, _ = next(gim_gen)

        return {
            'SUP': self.train_SUP(source_data),
            'LIM': self.train_LIM(target_data),
            'GIM': self.train_GIM(gim_data),
        }


class JointTrainer(ModelTrainer):
    def __init__(self, model_name, source_gen, target_gen):
        super().__init__(model_name)

        # model
        self.encoder = importlib.import_module(
            'models').__getattribute__(model_name)
        self.model = Workers(self.encoder, embed_size=192,
                             source_gen=source_gen, target_gen=target_gen)
        self.model.to(DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        steps = 1024
        pbar = tqdm(range(steps))

        for step in pbar:

            losses = self.model.start_train()

            loss = torch.mean(torch.stack(list(losses.values())))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = ""
            for key, val in losses.items():
                val = val.detach().cpu()
                loss_val_dict[key] = (loss_val_dict.get(key, 0) + val)
                self.writer.add_scalar(
                    f"step/{key}", val, epoch * len(loader) + step)
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            self.writer.add_scalar(
                f"step/loss", loss, epoch * len(loader) + step)
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss)

            desc += f" {loss = :.3f}"
            pbar.set_description(desc)

        load_val_dict = {key: value/steps for key,
                         value in load_val_dict.items()}
        return loss_val_dict
