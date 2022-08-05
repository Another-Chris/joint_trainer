from yaml import load
from .ModelTrainer import ModelTrainer
from .ModelWithHead import ModelWithHead
from tqdm import tqdm

import torch

class SupervisedTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.source_model = ModelWithHead(
            self.encoder, dim_in=kwargs['nOut'], head = 'linear', feat_dim=128)

    def train_network(self, loader, epoch):
        self.source_model.train()
        self.source_model.to(torch.device('cuda'))
        self.__L__.cuda()

        counter = 0
        loss = 0

        pbar = tqdm(enumerate(loader), total = len(loader))
        for step, (data, label) in pbar:
            label = label.float()
            label = label.cuda()

            self.source_model.zero_grad()

            ################# supervised #################
            data = torch.cat([data[0], data[1]], dim=0)
            data = data.squeeze(1).cuda()
            
            outp = self.source_model(data)
 
            bsz = outp.size()[0] // 2
            f1, f2 = torch.split(outp, [bsz, bsz], dim=0)

            # outpcat: bz, ncrops, dim
            outp = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            nloss = self.__L__(outp, label)

            ################# backward pass  #################
            nloss.backward()
            self.__optimizer__.step()

            # record
            loss += nloss.detach().cpu().item()
            counter += 1

            pbar.set_description(f'loss: {loss / counter :.3f}')
            pbar.total = len(loader)

            if self.lr_step == 'iteration':
                self.__scheduler__.step(epoch + step / len(loader))

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / (counter + 1e-6)
