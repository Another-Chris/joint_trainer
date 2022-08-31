from models import ECAPA_TDNN, Head
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from loss import SubConLoss

import torch
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append('..')


class Workers(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = ECAPA_TDNN()

        self.worker_subcon = Head(self.encoder, dim_in=192, feat_dim=128)
        self.loss_subcon = SubConLoss(temperature=0.5)

    def train_subcon(self, data):
        data = torch.cat([d for d in data], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.worker_subcon(data)
        outp = outp.reshape(len(data), -1, outp.size()
                            [-1]).transpose(1, 0).squeeze(1)
        loss = self.loss_subcon(outp)
        return loss

    def start_train(self, data):
        return {
            'subCon': self.train_subCon(data),
        }


class SSLTrainer(ModelTrainer):
    def __init__(self):
        super().__init__("ECAPA_TDNN")

        # model
        self.model = Workers()
        self.model.cuda()

        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        pbar = tqdm(enumerate(loader))

        for step, (data, label) in pbar:
            losses = self.model.start_train(data)

            loss = torch.mean(losses.values())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = ""
            for key, val in losses:
                val = val.detach().cpu()
                loss_val_dict[key] = (loss_val_dict.get(key, 0) + val) / step
                self.writer.add_scalar(
                    f"step/{key}", val, epoch * len(loader) + step)
                desc += "{key} = {val}"

            loss = loss.detach().cpu()
            self.writer.add_scalar(
                f"step/loss", loss, epoch * len(loader) + step)
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss) / step
            desc += f"{loss = }"

        return loss_val_dict
