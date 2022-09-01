from models import ECAPA_TDNN, Head, Spec, MFCC, FbankAug, Torchfbank, TorchMFCC
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
        
        self.fbank_aug = FbankAug()

        self.worker_subcon = Head(encoder, dim_in=192, feat_dim=128)
        self.loss_subcon = SubConLoss(temperature=0.5)

        self.worker_channel = Head(encoder, dim_in=192, feat_dim=128)
        self.worker_spec = Spec(encoder, feat_dim=202)  # 202 frames
        self.worker_mfcc = MFCC(encoder, feat_dim=202)

    def get_fbank(self, x, aug):
        with torch.no_grad():
            x = Torchfbank(x)+1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.fbank_aug(x)
        return x

    def get_mfcc(self, x):
        with torch.no_grad():
            x = TorchMFCC(x)
        return x

    def train_subCon(self, segs, augs, bz):
        outp_segs = self.worker_subcon(self.get_fbank(segs, aug=False))
        outp_augs = self.worker_subcon(self.get_fbank(augs, aug=True))

        embed = outp_segs.shape[1]
        outp = torch.cat([outp_segs, outp_augs], dim=0).reshape(bz, -1, embed)
        loss = self.loss_subcon(outp)
        return loss

    def train_channel(self, segs, augs):
        return -F.cosine_similarity(
            self.worker_channel(self.get_fbank(augs, aug=True)),
            self.worker_channel(self.get_fbank(segs, aug=False))
        ).mean()

    def train_spec(self, segs, augs):
        fbank_segs = self.get_fbank(segs, aug=False)
        fbank_reconstruct = self.worker_spec(self.get_fbank(augs, aug=True))

        return F.mse_loss(fbank_reconstruct, fbank_segs)

    def train_mfcc(self, segs, augs):
        mfcc_segs = self.get_mfcc(segs)
        mfcc_reconstruct = self.worker_mfcc(self.get_fbank(augs, aug = True))
        return F.mse_loss(mfcc_segs, mfcc_reconstruct)

    def start_train(self, data):
        segs = torch.cat([d for d in data[:2]], dim=0).squeeze(1).to(DEVICE)
        augs = torch.cat([d for d in data[2:]], dim=0).squeeze(1).to(DEVICE)
        bz = data[0].shape[0]

        return {
            'spec': self.train_spec(segs, augs),
            'mfcc': self.train_mfcc(segs, augs),
            'subCon': self.train_subCon(segs, augs, bz),
            'channel': self.train_channel(segs, augs),
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
            run = step + 1

            losses = self.model.start_train(data)

            loss = torch.mean(torch.stack(list(losses.values())))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = ""
            for key, val in losses.items():
                val = val.detach().cpu()
                loss_val_dict[key] = (loss_val_dict.get(key, 0) + val) / run
                self.writer.add_scalar(
                    f"step/{key}", val, epoch * len(loader) + step)
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            self.writer.add_scalar(
                f"step/loss", loss, epoch * len(loader) + step)
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss) / run

            desc += f" {loss = :.3f}"
            pbar.set_description(desc)

        return loss_val_dict
