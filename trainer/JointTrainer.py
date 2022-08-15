import ssl
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
        sup_loss,
        ssl_loss,
        nOut, 
        **kwargs):
        super().__init__(**kwargs)

        # loss function
        SupLoss = importlib.import_module('loss.' + sup_loss).__getattribute__('LossFunction')
        self.sup_loss = SupLoss(nOut = nOut, temperature = 0.5, **kwargs)
        
        SSLLoss = importlib.import_module('loss.' + ssl_loss).__getattribute__('LossFunction')
        self.ssl_loss = SSLLoss(**kwargs)
        self.ssl_loss_name = ssl_loss

        # encoder
        ModelFn = importlib.import_module('models.' + self.model).__getattribute__('MainModel')
        self.encoder = ModelFn(nOut = nOut, **kwargs)

        self.ssl_model = ModelWithHead(self.encoder, dim_in=nOut, feat_dim =nOut, head = 'mlp')
        self.sup_model = ModelWithHead(self.encoder, dim_in=nOut, feat_dim= nOut, head = 'mlp')
        
        # optimizer
        Optim = importlib.import_module('optimizer.' + self.optimizer).__getattribute__('Optimizer')
        self.optim = Optim(
            list(self.encoder.parameters()) + 
            list(self.sup_model.head.parameters()) + 
            list(self.ssl_model.head.parameters()), 
            lr = kwargs['lr'], 
            weight_decay = kwargs['weight_decay'])
        
        Scheduler = importlib.import_module('scheduler.' + self.scheduler).__getattribute__('Scheduler')
        del kwargs['optimizer']
        self.scheduler, self.lr_step = Scheduler(optimizer = self.optim, **kwargs)

        self.supervised_gen = supervised_gen
        self.ssl_gen = ssl_gen
        self.nOut = nOut
        
        self.put_to_device()

    def put_to_device(self):
        device = torch.device('cuda')
        self.ssl_model.to(device)
        self.sup_model.to(device)
        self.ssl_loss.to(device)
        self.sup_loss.to(device)

    def train_ssl(self):
        data, _ = next(self.ssl_gen)
        data = torch.cat([d for d in data], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.ssl_model(data)
        outp = outp.reshape(2, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)        
        
        ssl_loss_val = self.ssl_loss(outp)
        
        if type(ssl_loss_val) == tuple:
            ssl_loss_val = ssl_loss_val[0]

        return ssl_loss_val

    def train_sup(self):
        sup_data, sup_label = next(self.supervised_gen)

        sup_data = sup_data.transpose(1, 0)
        sup_data = sup_data.reshape(-1, sup_data.size()[-1]).cuda()

        sup_label = sup_label.long().cuda()

        outp = self.sup_model(sup_data)

        # if some dim has more than 1 cols, squeeze has no use
        outp = outp.reshape(self.nPerSpeaker, -1,
                            outp.size()[-1]).transpose(1, 0).squeeze(1)
        sup_loss_val = self.sup_loss(outp, sup_label)
       
        if type(sup_loss_val) == tuple:
            sup_loss_val = sup_loss_val[0]

        return sup_loss_val

    def train_network(self, loader = None, epoch=None):

        self.sup_model.train()
        self.ssl_model.train()

        sup_loss = 0
        ssl_loss = 0

        steps_per_epoch = 256

        pbar = tqdm(range(steps_per_epoch))
        for step in pbar:
            sup_loss_val = self.train_sup()
            ssl_loss_val = self.train_ssl()
            
            # # scale the loss
            # if sup_loss_val > ssl_loss_val:
            #     lamb = torch.div(sup_loss_val, ssl_loss_val, rounding_mode='trunc') 
            #     ssl_loss_val = ssl_loss_val * lamb
            # else:
            #     lamb = torch.div(ssl_loss_val, sup_loss_val ,rounding_mode='trunc') 
            #     sup_loss_val = sup_loss_val * lamb
                
            loss = ssl_loss_val + sup_loss_val
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            ssl_loss_val = ssl_loss_val.detach().cpu()
            sup_loss_val = sup_loss_val.detach().cpu()
            loss = ssl_loss_val + sup_loss_val

            self.writer.add_scalar(
                "step/loss", loss, epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/ssl_loss", ssl_loss_val,
                                epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/sup_loss", sup_loss_val,
                                epoch * steps_per_epoch + step)

            pbar.set_description(
                f'{loss = :.3f} {ssl_loss_val =:.3f} {sup_loss_val =:.3f}')

            if self.lr_step == 'iteration':
                self.scheduler.step(epoch + step / len(loader))
                
            ssl_loss += ssl_loss_val
            sup_loss += sup_loss_val
            
            break

        if self.lr_step == 'epoch':
            self.scheduler.step()

        step+=1
        return (ssl_loss + sup_loss) / step, ssl_loss / step, sup_loss / step