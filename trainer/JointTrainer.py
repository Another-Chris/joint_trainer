from tqdm import tqdm
from .ModelWithHead import ModelWithHead
from .ModelTrainer import ModelTrainer
import torch
import importlib


class JointTrainer(ModelTrainer):
    def __init__(self, supervised_loss, ssl_loss, supervised_gen, **kwargs):
        super().__init__(**kwargs)

        sup_lossfn = importlib.import_module(
            'loss.' + supervised_loss).__getattribute__('LossFunction')

        ssl_lossfn = importlib.import_module(
            'loss.' + ssl_loss).__getattribute__('LossFunction')

        self.sup_loss = sup_lossfn(**kwargs)
        self.ssl_loss = ssl_lossfn(**kwargs)

        embed_size = kwargs['nOut']
        self.sup_model = self.encoder

        self.ssl_model = ModelWithHead(
            self.encoder, dim_in=embed_size, head='mlp', feat_dim=128)

        self.supervised_gen = supervised_gen

    def put_to_device(self):
        device = torch.device('cuda')
        self.sup_model.to(device)
        self.ssl_model.to(device)
        self.ssl_loss.to(device)
        self.sup_loss.to(device)

    def split_and_cat(self, outp, nPerSpeaker):
        fs = torch.split(outp, outp.size()[0] // nPerSpeaker, dim=0)
        outpcat = torch.cat([f.unsqueeze(1) for f in fs], dim=1)
        return outpcat

    def train_network(self, loader, epoch=None):
        
        self.put_to_device()
        self.sup_model.train()
        self.ssl_model.train()
        
        counter = 0
        loss = 0
        sup_loss = 0
        ssl_loss = 0

        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, (data, _) in pbar:

            nPerSpeaker = len(data)  # data: [segs]

            ################# supervised #################
            self.sup_model.zero_grad()

            sup_data, sup_label = next(self.supervised_gen)

            sup_data = sup_data.transpose(1, 0)
            sup_data = sup_data.reshape(-1, sup_data.size()[-1]).cuda()

            sup_label = sup_label.float().cuda()

            outp = self.sup_model(sup_data)

            outp = outp.reshape(self.nPerSpeaker, -1,
                                outp.size()[-1]).transpose(1, 0).squeeze(1)
            sup_loss_val = self.sup_loss(outp, sup_label)

            if type(sup_loss_val) == tuple:
                sup_loss_val  = sup_loss_val[0]

            ################# SSL #################
            self.ssl_model.zero_grad()

            data = torch.cat([d for d in data], dim=0)

            data = data.squeeze(1).cuda()
            outp = self.ssl_model(data)
            ssl_loss_val = self.ssl_loss(
                self.split_and_cat(outp, nPerSpeaker))

            ################# backward pass  #################
            loss_val = ssl_loss_val + sup_loss_val
            loss_val.backward()
            self.__optimizer__.step()

            # record
            loss += loss_val.detach().cpu().item()
            ssl_loss += ssl_loss_val.detach().cpu().item()
            sup_loss += sup_loss_val.detach().cpu().item()

            counter += 1

            pbar.set_description(
                f'{loss/counter=:.3f} {ssl_loss/counter=:.3f} {sup_loss/counter=:.2f}')

            if self.lr_step == 'iteration':
                self.__scheduler__.step(epoch + step / len(loader))
        

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / counter, ssl_loss / counter, sup_loss / counter
