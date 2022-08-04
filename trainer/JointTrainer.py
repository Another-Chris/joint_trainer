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
        self.source_model = ModelWithHead(
            self.encoder, dim_in=embed_size, feat_dim=embed_size)
        self.target_model = ModelWithHead(
            self.encoder, dim_in=embed_size, feat_dim=embed_size)

        self.supervised_gen = supervised_gen

    def put_to_device(self):
        device = torch.device('cuda')
        self.source_model.to(device)
        self.target_model.to(device)
        self.ssl_loss.to(device)
        self.sup_loss.to(device)

    def forward_pairs(self, data):
        data = torch.cat([data[0], data[1]], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.target_model(data)

        bsz = outp.size()[0] // 2
        f1, f2 = torch.split(outp, [bsz, bsz], dim=0)

        # outpcat: bz, ncrops, dim
        outpcat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        return outpcat

    def train_network(self, loader, epoch=None):

        self.put_to_device()
        self.source_model.train()
        self.target_model.train()

        counter = 0
        loss = 0
        sup_loss = 0
        ssl_loss = 0

        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, (data, _) in pbar:

            ################# SSL #################
            self.target_model.zero_grad()
            outp = self.forward_pairs(data)
            ssl_loss_val = self.ssl_loss(outp)

            ################# supervised #################
            self.source_model.zero_grad()

            sup_data, sup_label = next(self.supervised_gen)
            sup_label = sup_label.float().cuda()
            outp = self.forward_pairs(sup_data)
            sup_loss_val = self.sup_loss(outp, sup_label)

            ################# backward pass  #################
            loss_val = ssl_loss_val + 0.5 * sup_loss_val
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
                # cosine scheduler with warm restart
                self.__scheduler__.step(epoch + step / len(loader))
            
        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / counter, ssl_loss / counter, sup_loss / counter
