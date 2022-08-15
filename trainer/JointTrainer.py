from .ModelWithHead import ModelWithHead
from models import ResNetSE34L
from loss import AngleProtoLoss, SubConLoss
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
import torch.optim as optim
import torch
import sys
import importlib
sys.path.append('..')


class JointTrainer(ModelTrainer):
    def __init__(
        self, 
        supervised_gen, 
        ssl_gen,
        nOut, 
        **kwargs):
        super().__init__(**kwargs)

        # loss function
        self.sup_loss = AngleProtoLoss(**kwargs)
        self.ssl_loss = SubConLoss(**kwargs)

        # encoder
        ModelFn = importlib.import_module('models.' + self.model).__getattribute__('MainModel')
        self.encoder = ModelFn(nOut = nOut, **kwargs)

        self.ssl_model = ModelWithHead(self.encoder, dim_in=nOut)
        self.sup_model = ModelWithHead(self.encoder, dim_in=nOut)
        
        # optimizer
        Optim = importlib.import_module('optimizer.' + self.optimizer).__getattribute__('Optimizer')
        self.optim = Optim(list(self.ssl_model.parameters()) + list(self.sup_model.parameters()), **kwargs)
        
        Scheduler = importlib.import_module('scheduler.' + self.scheduler).__getattribute__('Scheduler')
        del kwargs['optimizer']
        self.scheduler, self.lr_step = Scheduler(optimizer = self.optim, **kwargs)

        self.supervised_gen = supervised_gen
        self.ssl_gen = ssl_gen

    def put_to_device(self):
        device = torch.device('cuda')
        self.ssl_model.to(device)
        self.sup_model.to(device)
        self.ssl_loss.to(device)
        self.sup_loss.to(device)

    def split_and_cat(self, outp):
        fs = torch.split(outp, outp.size()[0] // self.nPerSpeaker, dim=0)
        outpcat = torch.cat([f.unsqueeze(1) for f in fs], dim=1)
        return outpcat

    def train_ssl(self):
        data, label = next(self.ssl_gen)
        data = torch.cat([d for d in data[:2]], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.ssl_model(data)
        fs = torch.split(outp, outp.size()[0] // self.nPerSpeaker, dim=0)
        outp = torch.cat([f.unsqueeze(1) for f in fs], dim=1)
        ssl_loss_val = self.ssl_loss(outp)

        return ssl_loss_val

    def train_sup(self):
        sup_data, sup_label = next(self.supervised_gen)

        sup_data = sup_data.transpose(1, 0)
        sup_data = sup_data.reshape(-1, sup_data.size()[-1]).cuda()

        sup_label = sup_label.long().cuda()

        outp = self.sup_model(sup_data)

        outp = outp.reshape(self.nPerSpeaker, -1,
                            outp.size()[-1]).transpose(1, 0).squeeze(1)
        sup_loss_val = self.sup_loss(outp, sup_label)

        if type(sup_loss_val) == tuple:
            sup_loss_val = sup_loss_val[0]

        return sup_loss_val

    def train_network(self, loader, epoch=None):

        self.put_to_device()
        self.sup_model.train()
        self.ssl_model.train()

        loss = 0
        sup_loss = 0
        ssl_loss = 0

        steps_per_epoch = 512

        pbar = tqdm(range(steps_per_epoch))
        for step in pbar:
            sup_loss_val = self.train_sup()
            ssl_loss_val = self.train_ssl()

            loss = sup_loss_val + ssl_loss_val
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            ssl_loss_step = ssl_loss_val.detach().cpu()
            sup_loss_step = sup_loss_val.detach().cpu()
            loss_step = ssl_loss_step + sup_loss_step

            self.writer.add_scalar(
                "step/loss", loss_step, epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/ssl_loss", ssl_loss_step,
                                epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/sup_loss", sup_loss_step,
                                epoch * steps_per_epoch + step)

            pbar.set_description(
                f'{loss_step =:.3f} {ssl_loss_step =:.3f} {sup_loss_step =:.3f}')

            if self.lr_step == 'iteration':
                self.scheduler.step(epoch + step / len(loader))
                
            sup_loss += ssl_loss_step
            ssl_loss += sup_loss_step
            loss += loss_step

        if self.lr_step == 'epoch':
            self.scheduler.step()

        return loss / step, ssl_loss / step, sup_loss / step
