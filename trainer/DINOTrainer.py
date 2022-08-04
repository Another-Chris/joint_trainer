from .ModelTrainer import ModelTrainer
from tqdm import tqdm

import torch
import torch.nn.functional as F


class SupervisedTrainer(ModelTrainer):
    def __init__(self, tps, tpt, **kwargs):
        super().__init__(**kwargs)
        
        self.tps = tps 
        self.tpt = tpt
        

    def loss_fn(self, t, s):
        t = t.detach() # stop gradients 
        s = F.softmax(s / self.tps, dim=1)
        t = F.softmax((t - C) / self.tpt, dim=1) # center + sharpen
        return - (t*F.log(s)).sum(dim=1).mean()
        

    def train_network(self, loader):
        self.__model__.train()
        self.__model__.to(torch.device('cuda'))

        counter = 0
        loss = 0

        pbar = tqdm(loader)
        for data, label in pbar:
            data = data.transpose(1, 0)
            label = label.float()
            label = label.cuda()

            self.__model__.zero_grad()

            ################# supervised #################
            data = data.reshape(-1, data.size()[-1]).cuda()
            pred = self.__model__.encoder(data)
            pred = pred.reshape(self.nPerSpeaker, -1, pred.size()[-1]).transpose(1, 0).squeeze(1)
            
            nloss, _ = self.__L__(pred, label)

            ################# backward pass  #################
            nloss.backward()
            self.__optimizer__.step()

            # record
            loss += nloss.detach().cpu().item()
            counter += 1

            pbar.set_description(f'loss: {loss / counter :.3f}')
            pbar.total = len(loader)

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / (counter + 1e-6)
