from utils import plot_batch
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
        self.loss_subcon = SubConLoss()

        self.worker_channel = Head(encoder, dim_in=192, feat_dim=128)
        self.worker_spec = Spec(encoder, feat_dim=202)  # 202 frames
        self.worker_mfcc = MFCC(encoder, feat_dim=202)
        self.worker_LIM = Head(encoder, dim_in=192, feat_dim=128)

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

    def train_LIM(self, same_anchor, same_pos, diff_anchor):
        data = torch.cat([torch.cat([same_anchor, same_pos], dim=1),
                         torch.cat([same_anchor, diff_anchor], dim=1)])
        label = torch.cat([torch.ones(size=(same_anchor.shape[0], data.shape[1])),
                          torch.zeros(size=(same_anchor.shape[0], data.shape[1]))])
        return F.binary_cross_entropy_with_logits(data, label)

    def train_subCon(self, segs, augs, bz, nviews):
        outp = torch.cat([
            torch.cat(
                [
                    d.unsqueeze(1) for d in torch.split(
                        self.worker_subcon(
                            self.get_fbank(data[0], aug=data[1])),
                        [bz] * nviews,
                        dim=0
                    )
                ], dim=1) for data in [(segs, False), (augs, True)]
        ], dim=0)

        loss = self.loss_subcon(outp)
        return loss

    def train_channel(self, segs, augs):
        return -F.cosine_similarity(
            self.worker_channel(self.get_fbank(augs, aug=True)),
            self.worker_channel(self.get_fbank(segs, aug=False))
        ).mean()

    def train_spec(self, segs, bz):

        seg1 = self.get_fbank(segs[:bz], aug=False)
        seg1_aug = self.get_fbank(segs[:bz], aug=True)

        seg2 = self.get_fbank(segs[bz:], aug=False)
        seg2_aug = self.get_fbank(segs[bz:], aug=True)

        mse_seg1 = F.mse_loss(seg1, self.worker_spec(
            seg1_aug))  # the first segment
        mse_seg2 = F.mse_loss(seg2, self.worker_spec(
            seg2_aug))  # the second segment

        loss = (mse_seg1 + mse_seg2).mean()

        return loss

    def train_mfcc(self, segs, bz):
        seg1 = self.get_mfcc(segs[:bz])
        seg1_aug = self.get_fbank(segs[:bz], aug=True)

        seg2 = self.get_mfcc(segs[bz:])
        seg2_aug = self.get_fbank(segs[bz:], aug=True)

        mse_seg1 = F.mse_loss(seg1, self.worker_mfcc(
            seg1_aug))  # the first segment
        mse_seg2 = F.mse_loss(seg2, self.worker_mfcc(
            seg2_aug))  # the second segment

        loss = (mse_seg1 + mse_seg2).mean()

        return loss

    def start_train(self, data, nviews, bz):

        return {
            # 'spec': self.train_spec(segs, bz),
            # 'mfcc': self.train_mfcc(segs, bz),
            'subCon': self.train_subCon(
                segs=torch.cat([data['same_anchor'], data['diff']],
                               dim=0).squeeze(1).to(DEVICE),
                augs=torch.cat([data['same_anchor_aug'], data['diff_aug']], dim=0).squeeze(
                    1).to(DEVICE),
                bz=bz,
                nviews=nviews),
            'LIM': self.train_LIM(
                *(d.squeeze(1).to(DEVICE) for d in [data['same_anchor'], data['same_pos'], data['diff']])
            )
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

            losses = self.model.start_train(
                data, nviews=2, bz=loader.batch_size)

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
