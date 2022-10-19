from models.ECAPA_TDNN_WITH_DSBN import ECAPA_TDNN_WITH_DSBN
from models.ResNet34_DSBN import ResNet34_DSBN 
from tqdm import tqdm
from loss import SupCon, AAMsoftmax, AngleProto
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

scaler = torch.cuda.amp.GradScaler()


def get_pair(data):
    
    anchor, pos = data['anchor'], data['pos']
    
    if len(anchor.shape) > 2:
        anchor = torch.squeeze(anchor, dim = 1)
        pos = torch.squeeze(pos, dim = 1)
    
    hoplen = 100 * 160 + 240
    min_hop = 2
    anchor_len = np.random.randint(min_hop, 4)
    pos_len = np.random.randint(min_hop, 6 - anchor_len)
    
    # anchor_len = 2
    # pos_len = 2

    anchor_segs = []
    pos_segs = []
    for i in range(5):
        anchor_seg = anchor[:, i*hoplen:(i+1)*hoplen]
        pos_seg = pos[:, i*hoplen:(i+1)*hoplen]
        
        if len(anchor_segs) == anchor_len and len(pos_segs) == pos_len: break
        
        if len(anchor_segs) >= anchor_len:
            pos_segs.append(pos_seg)    
        else:
            anchor_segs.append(anchor_seg)
    
    anchor = torch.cat(anchor_segs, dim = 1)
    pos =  torch.cat(pos_segs, dim = 1)
    
    if anchor.shape[1] < min_hop * hoplen or pos.shape[1] < min_hop * hoplen:
        raise ValueError('anchor or pos < min_hop in data loader')
    
    return anchor, pos 

    
class Workers(nn.Module):
    def __init__(self, encoder, embed_size) -> None:
        super().__init__()

        self.encoder = encoder
        self.supcon = SupCon()
        # self.angleproto = AngleProto()
        self.aamsoftmax = AAMsoftmax(m = 0.2, s = 30, n_class = Config.NUM_CLASSES, n_embed = embed_size)
    
    def forward(self, x, domain):
        # return F.normalize(self.target_head(F.normalize(self.encoder(x, domain))))
        return F.normalize(self.encoder(x, domain))
    
    def forward_simCLR(self, f1, f2, label = None):
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        return self.supcon(feat, label)
        
    def start_train(self, ds_gen):
        data, label = next(ds_gen)
        
        """source domain""" 
        # source_data = torch.cat([data['source_data']['anchor'], data['source_data']['pos']], dim = 0)
        source_data = data['source_data']
        source_feat = self.encoder(source_data.to(Config.DEVICE), 'target')       
        spk_loss = self.aamsoftmax(source_feat, label['source_label'].to(Config.DEVICE))

        """target domain"""     
        # target_data = data['target_data']
        # spk_feat = self.encoder(target_data.to(Config.DEVICE), domain = 'empty', aug = True)
        # spk_loss = self.aamsoftmax(spk_feat, label['target_label'].to(Config.DEVICE))
        
        target_anchor, target_pos = data['target_data']['anchor'],data['target_data']['pos']
        target_anchor = F.normalize(self.encoder(target_anchor.to(Config.DEVICE), domain = 'target'))
        target_pos = F.normalize(self.encoder(target_pos.to(Config.DEVICE), domain = 'target'))

        simCLR = self.forward_simCLR(target_anchor,target_pos)
    
        return {
            'spk_loss': spk_loss,
            'simCLR': simCLR
        }


class Trainer(torch.nn.Module):
    def __init__(self, exp_name):
        super().__init__()

        self.writer = SummaryWriter(
            log_dir=f"./logs/{exp_name}/{dt.now().strftime('%Y-%m-%d %H.%M.%S')}")

        # self.encoder = ResNet34_DSBN(nOut = 512, encoder_type = 'ASP')
        # self.model = Workers(self.encoder, embed_size=512)
        
        self.encoder = ECAPA_TDNN_WITH_DSBN(C=Config.C, embed_size=Config.EMBED_SIZE)
        self.model = Workers(self.encoder, embed_size=Config.EMBED_SIZE)
        
        self.model.to(Config.DEVICE)

        self.optim = optim.Adam(self.model.parameters(), lr = Config.LEARNING_RATE)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.95)

    def train_network(self, ds_gen, epoch):

        self.model.train()
        loss_val_dict = {}

        steps = 256
        pbar = tqdm(range(steps))

        for step in pbar:
            
            losses = self.model.start_train(ds_gen)
            loss = torch.sum(torch.stack(list(losses.values())))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # scaler.scale(loss).backward()
            # scaler.step(self.optim)
            # scaler.update()

            desc = f""
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
