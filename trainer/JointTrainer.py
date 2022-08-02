from .ModelTrainer import ModelTrainer
from tqdm import tqdm

import torch
import importlib


class JointTrainer(ModelTrainer):
    def __init__(self, supervised_loss, ssl_loss, supervised_gen, **kwargs):
        super().__init__(**kwargs)

        supervised_loss_fn = importlib.import_module(
            'loss.' + supervised_loss).__getattribute__('LossFunction')

        self.supervised_loss = supervised_loss_fn(**kwargs)

        ssl_loss_fn = importlib.import_module(
            'loss.' + ssl_loss).__getattribute__('LossFunction')

        self.ssl_loss = ssl_loss_fn(**kwargs)
        
        self.supervised_gen = supervised_gen

    def train_network(self, loader):
        
        device = torch.device('cuda')
        self.__model__.to(device)
        self.__model__.encoder.to(device)
        self.ssl_loss.to(device)
        self.supervised_loss.to(device)
        self.__model__.train()

        counter = 0
        loss = 0

        pbar = tqdm(loader)
        for data in pbar:
            self.__model__.zero_grad()
            
            ################# SSL #################
            data = torch.cat([data[0], data[1]], dim=0)

            data = data.transpose(1, 0)
            data = data.reshape(-1, data.size()[-1]).cuda()
            outp = self.__model__.forward(data)
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            bsz = outp.size()[0] // 2
            f1, f2 = torch.split(outp, [bsz, bsz], dim=0)
            outp = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            nloss = self.ssl_loss.forward(outp)

            ################# supervised #################
            supervised_data, supervised_label = next(self.supervised_gen)
            supervised_label = supervised_label.float().cuda()
            
            supervised_data = supervised_data.transpose(1, 0)
            supervised_data = supervised_data.reshape(-1, supervised_data.size()[-1]).cuda()
            
            pred = self.__model__.encoder(supervised_data)
            pred = pred.reshape(self.nPerSpeaker, -1, pred.size()[-1]).transpose(1, 0).squeeze(1)
            
            sloss, _ = self.supervised_loss(pred, supervised_label)

            ################# backward pass  #################
            loss_val = nloss + sloss
            loss_val.backward()
            self.__optimizer__.step()

            # record
            loss += loss_val.detach().cpu().item()
            counter += 1

            pbar.set_description(f'loss: {loss / counter :.3f}')
            pbar.total = len(loader)

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / (counter + 1e-6)
