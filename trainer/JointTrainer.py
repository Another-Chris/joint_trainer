from .ModelTrainer import ModelTrainer
from .ModelWithHead import ModelWithHead
from tqdm import tqdm

import torch
import importlib


class JointTrainer(ModelTrainer):
    def __init__(self, supervised_loss, ssl_loss, supervised_gen, **kwargs):
        super().__init__(**kwargs)

        supervised_loss_fn = importlib.import_module(
            'loss.' + supervised_loss).__getattribute__('LossFunction')

        self.supervised_loss = supervised_loss_fn(**kwargs)

        ssl_loss_fn = importlib.import_module(
            'loss.' + ssl_loss).__getattribute__('LossFunction')

        self.ssl_loss = ssl_loss_fn(**kwargs)

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
        self.supervised_loss.to(device)
        
    def forward_pairs(self, data, label = None):
        data = torch.cat([data[0], data[1]], dim=0)

        data = data.squeeze(1).cuda()
        outp = self.target_model(data)

        bsz = outp.size()[0] // 2
        f1, f2 = torch.split(outp, [bsz, bsz], dim=0)

        # outpcat: bz, ncrops, dim
        outpcat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        return self.ssl_loss(outpcat, label)
        

    def train_network(self, loader, epoch=None):

        self.put_to_device()
        self.source_model.train()
        self.target_model.train()

        counter = 0
        loss = 0

        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, (data, _) in pbar:

            ################# SSL #################
            self.target_model.zero_grad()

            data = torch.cat([data[0], data[1]], dim=0)

            data = data.squeeze(1).cuda()
            outp = self.target_model(data)

            bsz = outp.size()[0] // 2
            f1, f2 = torch.split(outp, [bsz, bsz], dim=0)

            # outpcat: bz, ncrops, dim
            outpcat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            nloss = self.ssl_loss.forward(outpcat)

            ################# supervised #################
            self.source_model.zero_grad()

            supervised_data, supervised_label = next(self.supervised_gen)
            supervised_label = supervised_label.float().cuda()

            supervised_data = supervised_data.transpose(1, 0)
            supervised_data = supervised_data.reshape(
                -1, supervised_data.size()[-1]).cuda()

            pred = self.source_model(supervised_data)
            pred = pred.reshape(self.nPerSpeaker, -1,
                                pred.size()[-1]).transpose(1, 0).squeeze(1)

            sloss, _ = self.supervised_loss(pred, supervised_label)

            ################# language adaptation #################

            same_lan = torch.cat(
                [f1.unsqueeze(1), torch.flip(f2, dims=(0,)).unsqueeze(1)], dim=1)
            diff_lan = torch.cat([f1.unsqueeze(1), pred.unsqueeze(1)], dim=1)
            data_lan = torch.cat([same_lan, diff_lan], dim=0)

            labels = torch.cat([torch.zeros(size=(bsz, 1)),
                               torch.ones(size=(bsz, 1))], dim=0).cuda()
            lloss = self.ssl_loss.forward(data_lan, labels)

            ################# backward pass  #################
            loss_val = nloss + 0.5 * sloss + 0.5 * lloss
            loss_val.backward()
            self.__optimizer__.step()

            # record
            loss += loss_val.detach().cpu().item()
            counter += 1

            pbar.set_description(
                f'loss: {loss / counter :.3f} ssl_loss: {nloss :.3f} sup_loss: {sloss :.3f} lan_loss: {lloss :.3f}')

            if self.lr_step == 'iteration':
                # cosine scheduler with warm restart
                self.__scheduler__.step(epoch + step / len(loader))

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / (counter + 1e-6)
