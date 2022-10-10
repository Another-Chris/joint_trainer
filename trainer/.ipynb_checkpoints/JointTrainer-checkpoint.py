from models import Head, ECAPA_TDNN_WITH_FBANK, BigHead
from tqdm import tqdm
from loss import SupConLoss, AAMsoftmax
from utils import Config
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

import torch
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
sys.path.append('..')


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def grad_reverse(x, alpha):
    return ReverseLayerF.apply(x,alpha)
    
    
    
class Workers(nn.Module):
    def __init__(self, encoder, embed_size) -> None:
        super().__init__()

        self.encoder = encoder
        # self.speaker_predictor = Head(dim_in = 3072, feat_dim = 1024, head = 'linear')
        self.language_predictor = BigHead(dim_in = embed_size, feat_dim = 1, hidden_size = 512)
        
        self.supcon = SupConLoss()
        self.aamsoftmax = AAMsoftmax(m = 0.2, s = 30, n_class = Config.NUM_CLASSES, n_embed = embed_size)
        # self.bn_source = nn.BatchNorm1d(embed_size)
        # self.bn_target = nn.BatchNorm1d(embed_size)
        # self.bn =  nn.BatchNorm1d(embed_size)
        self.dp = nn.Dropout(0.5)
    
    def forward(self, x):
        # return F.normalize(self.encoder(x))
        return self.encoder(x)
    
    def predict(self, feat, label):
        # aamsoftmax
        # feat = self.speaker_predictor(feat)
        return self.aamsoftmax(feat, label.to(Config.DEVICE))
        
        ## cross entropy
        # return F.cross_entropy(self.speaker_predictor(feat), label.to(Config.DEVICE))
    
        ## supcon
        # feat = F.normalize(self.speaker_predictor(feat))
        # bz = feat.shape[0] // 2
        # f1, f2 = torch.split(feat, [bz,bz])
        # feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        # return self.supcon(feat, label)
    
    def language(self, feat, label):
        # bce style
        label = label.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(self.language_predictor(feat), label.to(Config.DEVICE))
        
        ## nll style
        # feat = F.log_softmax(self.language_predictor(feat), dim = 1)
        # label = label.to(Config.DEVICE)
        # return F.nll_loss(feat, label) 

    def start_train(self, ds_gen, alpha):
        data, label = next(ds_gen)
                
        """source domain""" 
        # source_data = torch.cat([data['source_data']['anchor'], data['source_data']['pos']], dim = 0)
        source_data = data['source_data']
        source_feat = self.forward(source_data.to(Config.DEVICE))       
        spk_loss = self.predict(source_feat, label['source_label'])
        # if type(source_feat) == tuple:
        #     source_feat, source_embed = source_feat
        #     spk_loss = self.predict(source_embed), label['source_label']
            
        
        """target domain"""     
        # target_data = torch.cat([data['target_data']['anchor'], data['target_data']['pos']], dim = 0)
        target_data = data['target_data']
        target_feat = self.forward(target_data.to(Config.DEVICE))
        # if type(target_feat) == tuple:
        #     target_feat, _ = target_feat        
        
        """language loss"""
        source_lan_label = torch.zeros(size = (source_feat.shape[0], ))
        target_lan_label = torch.ones(size = (target_feat.shape[0], ))
        # source_lan_loss = self.language(grad_reverse(F.relu(self.bn_source(source_feat)), alpha), source_lan_label)
        # target_lan_loss = self.language(grad_reverse(F.relu(self.bn_target(target_feat)), alpha), target_lan_label)
        lan_feat = grad_reverse(torch.cat([source_feat, target_feat], dim = 0), alpha)
        lan_label = torch.cat([source_lan_label, target_lan_label], dim = 0)
        lan_loss = self.language(lan_feat, lan_label)
        
        return {
            'spk_loss': spk_loss,
            # 'source_lan': source_lan_loss,
            # 'target_lan': target_lan_loss,
            'lan_loss': lan_loss
        }


class JointTrainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()

        self.writer = SummaryWriter(
            log_dir=f"./logs/{exp_name}/{dt.now().strftime('%Y-%m-%d %H.%M.%S')}")

        self.encoder = ECAPA_TDNN_WITH_FBANK(
            C=Config.C, embed_size=Config.EMBED_SIZE)
        self.model = Workers(self.encoder, embed_size=Config.EMBED_SIZE)
        self.model.to(Config.DEVICE)
        
        # self.optim = optim.Adam([
        #     {'params': self.model.speaker_predictor.parameters(), 'lr': 1e-3},
        #     {'params': self.model.encoder.parameters(), 'lr': 5e-4},
        #     {'params': self.model.language_predictor.parameters(), 'lr': 1e-3}
        # ], lr = Config.LEARNING_RATE)
        
        self.optim = optim.Adam(self.model.parameters(), lr = Config.LEARNING_RATE)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.95)

    def train_network(self, ds_gen, epoch):

        self.model.train()
        loss_val_dict = {}

        steps = 256
        pbar = tqdm(range(steps))

        for step in pbar:
            
            p = float(step + epoch * steps) / 200 / steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            if alpha > 0.6:
                alpha = 0.6
            
            losses = self.model.start_train(ds_gen, alpha)
            loss = torch.sum(torch.stack(list(losses.values())))
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            desc = f"alpha = {alpha :.4f}"
            for key, val in losses.items():
                val = val.detach().cpu()
                loss_val_dict[key] = (loss_val_dict.get(key, 0) + val)
                desc += f" {key} = {val :.4f}"

            loss = loss.detach().cpu().item()
            loss_val_dict['loss'] = (
                loss_val_dict.get('loss', 0) + loss)

            desc += f" loss = {loss:.3f}"
            pbar.set_description(desc)

        loss_val_dict = {key: value/steps for key,
                         value in loss_val_dict.items()}
        return loss_val_dict
