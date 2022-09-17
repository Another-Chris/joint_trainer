from utils import plot_batch
from models import ECAPA_TDNN, Head, GIM, LIM,  Spec, MFCC, FbankAug, Torchfbank, TorchMFCC
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from loss import SubConLoss

import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append('..')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Workers(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()

        self.gim = GIM(encoder, embed_size=256, proj_size=128)
        self.lim = LIM(encoder, embed_size=256, proj_size=128)
        
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

    def start_train(self, data):

        return {
            'LIM': self.train_LIM(data),
            'GIM': self.train_GIM(data),
        }


class SSLTrainer(ModelTrainer):
    def __init__(self):
        super().__init__("ECAPA_TDNN")

        # model
        self.encoder = ECAPA_TDNN()
        self.model = Workers(self.encoder)
        self.model.to(DEVICE)

        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        pbar = tqdm(enumerate(loader), total=len(loader))

        for step, (data, _) in pbar:

            losses = self.model.start_train(data)

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
