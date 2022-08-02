from .ModelTrainer import ModelTrainer
from tqdm import tqdm

import torch


class SupervisedTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
