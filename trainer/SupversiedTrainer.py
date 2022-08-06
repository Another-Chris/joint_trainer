from yaml import load
from .ModelTrainer import ModelTrainer
from tqdm import tqdm

import torch

class SupervisedTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train_network(self, loader, epoch):
        self.encoder.train()
        self.encoder.to(torch.device('cuda'))
        self.__L__.cuda()

        counter = 0
        loss = 0

        pbar = tqdm(enumerate(loader), total = len(loader))
        for step, (data, label) in pbar:
            label = label.float()
            label = label.cuda()

            self.encoder.zero_grad()

            ################# supervised #################
            data = data.transpose(1, 0)
            data = data.reshape(-1, data.size()[-1]).cuda()

            label = label.long().cuda()

            outp = self.encoder(data)

            outp = outp.reshape(self.nPerSpeaker, -1,
                                outp.size()[-1]).transpose(1, 0).squeeze(1)
            sup_loss_val = self.__L__(outp, label)

            if type(sup_loss_val) == tuple:
                sup_loss_val  = sup_loss_val[0]

            ################# backward pass  #################
            sup_loss_val.backward()
            self.__optimizer__.step()

            # record
            loss += sup_loss_val.detach().cpu().item()
            counter += 1

            pbar.set_description(f'loss: {loss / counter :.3f}')
            pbar.total = len(loader)

            if self.lr_step == 'iteration':
                self.__scheduler__.step(epoch + step / len(loader))

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / (counter + 1e-6)
