from models import ECAPA_TDNN, Head
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
from loss import SubConLoss
from torchsummary import summary

import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append('..')


class Workers(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()

        self.worker_subcon = Head(encoder, dim_in=192, feat_dim=128)
        self.loss_subcon = SubConLoss(temperature=0.5)
        
        self.worker_channel = Head(encoder, dim_in = 192, feat_dim=128)
        

    def train_subCon(self, data):
        data = torch.cat([d for d in data], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.worker_subcon(data)
        outp = outp.reshape(len(data), -1, outp.size()
                            [-1]).transpose(1, 0).squeeze(1)
        loss = self.loss_subcon(outp)
        return loss
    
    def train_channel(self, data):
        segs = torch.cat([d for d in data[:2]], dim=0).squeeze(1).cuda()
        augs = torch.cat([d for d in data[2:]], dim=0).squeeze(1).cuda()
        
        return F.mse_loss(
            self.worker_channel(augs), 
            self.worker_channel(segs)
        )
    
    def start_train(self, data):
        return {
            'subCon': self.train_subCon(data),
            'channel': self.train_channel(data)
        }


class SSLTrainer(ModelTrainer):
    def __init__(self):
        super().__init__("ECAPA_TDNN")

        # model
        self.encoder = ECAPA_TDNN()
        self.model = Workers(self.encoder)
        self.model.cuda()
    
        # optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_network(self, loader=None, epoch=None):

        self.model.train()
        loss_val_dict = {}

        pbar = tqdm(enumerate(loader), total = len(loader))

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
