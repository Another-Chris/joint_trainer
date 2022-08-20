import ssl
from .ModelWithHead import ModelWithHead
from models import ResNetSE34L,DomainAdaptor
from loss import AngleProtoLoss, SubConLoss
from .ModelTrainer import ModelTrainer
from tqdm import tqdm
import torch.optim as optim
import torch
import sys
import importlib
import torch.nn.functional as F
import torch.nn as nn

sys.path.append('..')


class Composer(nn.Module):
    def __init__(self, nOut, supervised_gen, ssl_gen, sup_loss, ssl_loss, encoder, nPerSpeaker, **kwargs) -> None:
        super().__init__()

        self.ssl_model = ModelWithHead(encoder, dim_in=nOut, feat_dim =nOut, head = 'mlp')
        self.sup_model = encoder #because the loss already includes the learnable w and b
        self.domain_adaptor = DomainAdaptor(in_dim = 2 * nOut)
        
        self.supervised_gen = supervised_gen
        self.ssl_gen = ssl_gen
        self.nOut = nOut
        self.nPerSpeaker = nPerSpeaker
        
        SupLoss = importlib.import_module('loss.' + sup_loss).__getattribute__('LossFunction')
        self.sup_loss = SupLoss(nOut = nOut, temperature = 0.5, **kwargs)
        
        SSLLoss = importlib.import_module('loss.' + ssl_loss).__getattribute__('LossFunction')
        self.ssl_loss = SSLLoss(**kwargs)
        self.ssl_loss_name = ssl_loss
        self.put_to_device()
        

    def put_to_device(self):
        device = torch.device('cuda')
        self.ssl_model.to(device)
        self.sup_model.to(device)
        self.ssl_loss.to(device)
        self.sup_loss.to(device)
        self.domain_adaptor.to(device)

    def train_ssl(self):
        data, _ = next(self.ssl_gen)
        data = torch.cat([d for d in data], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.ssl_model(data)
        outp = outp.reshape(2, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)        
        
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
        
    def train(self):
        sup_loss_val, sup_embed = self.train_sup()
        ssl_loss_val, ssl_embed = self.train_ssl()
        
        diff_lan = []
        for v1 in range(sup_embed.size()[1]):
            for v2 in range(ssl_embed.size()[1]):
                diff_lan.append(
                    torch.cat([sup_embed[:, v1, :], ssl_embed[:, v2, :]], axis = -1)
                )
        same_lan = torch.cat([sup_embed.reshape(sup_embed.size()[0], -1),ssl_embed.reshape(ssl_embed.size()[0], -1)], dim = 0)
        diff_lan = torch.cat(diff_lan, dim = 0)
        
        concats = torch.cat([diff_lan, same_lan], dim = 0)
        bce_label = torch.cat([torch.ones(size = (diff_lan.size()[0], 1)), torch.zeros(size = (same_lan.size()[0], 1))]).cuda()
        bce_input = self.domain_adaptor(concats)
        bce_loss_val = F.binary_cross_entropy(bce_input, bce_label)
                        
        return ssl_loss_val, sup_loss_val, bce_loss_val

class JointTrainer(ModelTrainer):
    def __init__(self, nOut, **kwargs):
        super().__init__(**kwargs)

        # model
        ModelFn = importlib.import_module('models.' + self.model).__getattribute__('MainModel')
        self.encoder = ModelFn(nOut = nOut, **kwargs)
        self.model = Composer(encoder = self.encoder, nOut = nOut, **kwargs)
        
        # optimizer
        Optim = importlib.import_module('optimizer.' + self.optimizer).__getattribute__('Optimizer')
        self.optim = Optim(
            self.model.parameters(),
            lr = kwargs['lr'], 
            weight_decay = kwargs['weight_decay'])
        
        Scheduler = importlib.import_module('scheduler.' + self.scheduler).__getattribute__('Scheduler')
        del kwargs['optimizer']
        self.scheduler, self.lr_step = Scheduler(optimizer = self.optim, **kwargs)


    def train_network(self, loader = None, epoch=None):

        sup_loss = 0
        ssl_loss = 0
        steps_per_epoch = 256
        self.model.train()

        pbar = tqdm(range(steps_per_epoch))
        for step in pbar:
            ssl_loss_val, sup_loss_val, bce_loss_val  = self.model.train()
            loss = ssl_loss_val + sup_loss_val + bce_loss_val

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            ssl_loss_val = ssl_loss_val.detach().cpu()
            sup_loss_val = sup_loss_val.detach().cpu()
            bce_loss_val = bce_loss_val.detach().cpu()
            
            loss = ssl_loss_val + sup_loss_val + bce_loss_val

            self.writer.add_scalar(
                "step/loss", loss, epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/ssl_loss", ssl_loss_val,
                                epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/sup_loss", sup_loss_val,
                                epoch * steps_per_epoch + step)
            self.writer.add_scalar("step/bce_loss", bce_loss_val,
                                epoch * steps_per_epoch + step)

            pbar.set_description(
                f'{loss = :.3f} {ssl_loss_val =:.3f} {sup_loss_val =:.3f} {bce_loss_val = :.8f}')

            if self.lr_step == 'iteration':
                self.scheduler.step(epoch + step / len(loader))
                
            ssl_loss += ssl_loss_val
            sup_loss += sup_loss_val
                        
        if self.lr_step == 'epoch':
            self.scheduler.step()
            

        step+=1
        
        return (ssl_loss + sup_loss) / step, ssl_loss / step, sup_loss / step
    
    
# # scale the loss
# if sup_loss_val > ssl_loss_val:
#     lamb = torch.div(sup_loss_val, ssl_loss_val, rounding_mode='trunc') 
#     ssl_loss_val = ssl_loss_val * lamb
# else:
#     lamb = torch.div(ssl_loss_val, sup_loss_val ,rounding_mode='trunc') 
#     sup_loss_val = sup_loss_val * lamb