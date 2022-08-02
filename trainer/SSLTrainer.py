from .ModelTrainer import ModelTrainer
from tqdm import tqdm 

import torch 

class SSLTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
    def train_network(self, loader):
        self.__model__.train()
        self.__model__.to(torch.device('cuda'))

        counter = 0
        loss = 0

        pbar = tqdm(loader)
        for data in pbar:
            self.__model__.zero_grad()

            # forward pass
            data = torch.cat([data[0], data[1]], dim=0)
            
            # batch, 1, len
            # 1, batch len
            # batch * 1, len
            data = data.transpose(1, 0)
            data = data.reshape(-1, data.size()[-1]).cuda()
            outp = self.__model__.forward(data)
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            
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