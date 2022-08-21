from models import DomainAdaptor, ModelWithHead
from .ModelTrainer import ModelTrainer
from tqdm import tqdm

import torch
import sys
import importlib
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')


class Composer(nn.Module):
    def __init__(self, nOut, supervised_gen, ssl_gen, sup_loss, ssl_loss, encoder, nPerSpeaker, **kwargs) -> None:
        super().__init__()

        self.ssl_model = ModelWithHead(
            encoder, dim_in=192, feat_dim=nOut, head='mlp')
        # because the loss already includes the learnable w and b
        self.sup_model = ModelWithHead(
            encoder, dim_in=192, feat_dim=nOut, head='mlp')
        self.regressor = ModelWithHead(
            encoder, dim_in=192, feat_dim=nOut, head='mlp')
        self.domain_adaptor = DomainAdaptor(in_dim=2 * nOut)

        self.supervised_gen = supervised_gen
        self.ssl_gen = ssl_gen
        self.nOut = nOut
        self.nPerSpeaker = nPerSpeaker
        SupLoss = importlib.import_module(
            'loss.' + sup_loss).__getattribute__('LossFunction')
        self.sup_loss = SupLoss(nOut=nOut, temperature=0.5, **kwargs)

        SSLLoss = importlib.import_module(
            'loss.' + ssl_loss).__getattribute__('LossFunction')
        self.ssl_loss = SSLLoss(**kwargs)
        self.ssl_loss_name = ssl_loss


    def train_ssl(self):
        data, _ = next(self.ssl_gen)
        data = torch.cat([d for d in data], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.ssl_model(data)
        outp = outp.reshape(len(data), -1, outp.size()
                            [-1]).transpose(1, 0).squeeze(1)

        loss = self.ssl_loss(outp)

        if type(loss) == tuple:
            loss = loss[0]

        return loss, outp

    def train_sup(self):
        data, label = next(self.supervised_gen)

        data = data.transpose(1, 0)
        data = data.reshape(-1, data.size()[-1]).cuda()

        label = label.long().cuda()

        outp = self.sup_model(data)

        # if some dim has more than 1 cols, squeeze has no use
        outp = outp.reshape(self.nPerSpeaker, -1,
                            outp.size()[-1]).transpose(1, 0).squeeze(1)
        loss = self.sup_loss(outp, label)

        if type(loss) == tuple:
            loss = loss[0]

        return loss, outp

    def train_channel(self):
        data, _ = next(self.ssl_gen)
        seg1, seg2, aug1, aug2 = data

        mse_seg = F.mse_loss(self.regressor(seg1), self.regressor(seg2))
        mse_aug = F.mse_loss(self.regressor(aug1), self.regressor(aug2))

        return (mse_aug + mse_seg) / 2

    def train_language(self):
        ssl_data, _ = next(self.ssl_gen)
        sup_data, _ = next(self.supervised_gen)

        ssl_aug1, ssl_aug2 = ssl_data[2], ssl_data[3]
        pos = self.domain_adaptor(torch.cat([ssl_aug1, ssl_aug2], dim=1))
        neg = self.domain_adaptor(torch.cat([ssl_aug1, sup_data], dim=1))

        return torch.mean(torch.log(pos + 1e-6) + torch.log(1-neg + 1e-6), dim=0)

    def start_train(self):
        return {
            'sup': self.train_sup(),
            'ssl': self.train_ssl(),
            'lan':    self.train_language(),
            'channel':  self.train_channel()
        }


class JointTrainer(ModelTrainer):
    def __init__(self, nOut, **kwargs):
        super().__init__(**kwargs)

        # model
        ModelFn = importlib.import_module(
            'models.' + self.model).__getattribute__('MainModel')
        self.encoder = ModelFn()
        self.model = Composer(encoder=self.encoder, nOut=nOut, **kwargs)
        self.model.cuda()

        # optimizer
        Optim = importlib.import_module(
            'optimizer.' + self.optimizer).__getattribute__('Optimizer')
        self.optim = Optim(
            self.model.parameters(),
            lr=kwargs['lr'],
            weight_decay=kwargs['weight_decay'])

        Scheduler = importlib.import_module(
            'scheduler.' + self.scheduler).__getattribute__('Scheduler')
        del kwargs['optimizer']
        self.scheduler, self.lr_step = Scheduler(
            optimizer=self.optim, **kwargs)

    def train_network(self, loader=None, epoch=None):

        steps_per_epoch = 256
        self.model.train()
        loss_epoch = 0

        pbar = tqdm(range(steps_per_epoch))
        for step in pbar:
            losses = self.model.start_train()
            
            loss = torch.mean(losses.values())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = ""
            for key, val in losses:
                val = val.detach().cpu()
                self.writer.add_scalar(
                    f"step/{key}", val, epoch * steps_per_epoch + step)
                desc += "{key} = {val}"
                
            loss =  loss.detach().cpu()
            self.writer.add_scalar(
                    f"step/loss", loss, epoch * steps_per_epoch + step)
            desc += f"{loss = }"
            
            loss_epoch += loss

            if self.lr_step == 'iteration':
                self.scheduler.step(epoch + step / len(loader))

        if self.lr_step == 'epoch':
            self.scheduler.step()

        step += 1

        return loss_epoch

