from models import Cls, GIM, LIM
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from loss import SubConLoss
from utils import Config

import torch
import importlib
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append('..')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Workers(nn.Module):
    def __init__(self, encoder, embed_size) -> None:
        super().__init__()

        self.gim = GIM(encoder, embed_size=embed_size, proj_size=128)
        self.lim = LIM(encoder, embed_size=embed_size, proj_size=128)
        self.cls = Cls(encoder, embed_size=embed_size, num_classes=Config.NUM_CLASSES)
        
    def train_cls(self, x, y):
        return self.cls.loss(self.cls(x), y)
        
    def train_subCon(self, segs, augs, bz, nviews):
        outp_segs = self.worker_subcon(self.get_fbank(segs, aug=False))
        outp_augs = self.worker_subcon(self.get_fbank(augs, aug=True))

        outp = torch.cat([outp_segs, outp_augs], dim=0)
        outp = torch.cat([d.unsqueeze(1)
                         for d in torch.split(outp, [bz] * nviews, dim=0)], dim=1)
        loss = self.loss_subcon(outp)
        return loss
    
    def train_GIM(self, data):
        return self.gim.loss(self.gim(data)[0], self.gim(data)[1])
    
    def train_LIM(self, data):
        return self.lim.loss(self.lim(data)[0], self.lim(data)[1])

    def start_train(self):

        source_data, source_label = next(self.source_gen)
        target_data, _ = next(self.target_gen)
        return {
            'cls': self.train_cls(source_data[0], source_label),
            'LIM': self.train_LIM(target_data),
            'GIM': self.train_GIM(target_data),
        }


class JointTrainer(ModelTrainer):
    def __init__(self, model_name, source_gen, target_gen):
        super().__init__(model_name)

        # model
        self.encoder = importlib.import_module('models').__getattribute__(model_name)
        self.model = Workers(self.encoder, embed_size = 192, source_gen = source_gen, target_gen = target_gen)
        self.model.to(DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        pbar = tqdm(enumerate(range(1024)), total=len(loader))

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

        return loss_val_dict
