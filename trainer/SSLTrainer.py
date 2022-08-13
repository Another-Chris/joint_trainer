from .ModelTrainer import ModelTrainer
from .ModelWithHead import ModelWithHead
from tqdm import tqdm 

import torch 

class SSLTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.source_model = ModelWithHead(
            self.encoder, dim_in=kwargs['nOut'], feat_dim=256)
        
        
    def train_network(self, loader):
        self.source_model.train()
        self.source_model.to(torch.device('cuda'))

        counter = 0
        loss = 0

        pbar = tqdm(loader)
        for data in pbar:
            self.source_model.zero_grad()

            # forward pass
            data = torch.cat([data[0], data[1]], dim=0)
            data = data.squeeze(1).cuda()
            outp = self.source_model.forward(data)
            
            bsz = outp.size()[0] // 2
            f1, f2 = torch.split(outp, [bsz, bsz], dim=0)
            outp = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            nloss = self.__L__.forward(outp)

            # backward pass
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