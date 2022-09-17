import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import make_samples, make_labels
from Head import Head


class GIM(nn.Module):
    def __init__(self, encoder, embed_size = 256, proj_size = 128) -> None:
        
        self.encoder = encoder
        self.minion = Head(encoder, dim_in = embed_size, dim_out = proj_size)
        self.loss = nn.BCEWithLogitsLoss()
        
    
    def forward(self, x, alpha=1, device=None):
        x_pos, x_neg = make_samples(x)
        x = torch.cat((x_pos, x_neg), dim=0).to(device)
        x = torch.mean(x, dim=2, keepdim=True)
        y = self.minion(x, alpha)
        label = make_labels(y).to(device)
        return y, label
    
class LIM(nn.Module):
    def __init__(self, encoder, embed_size = 256, proj_size = 128) -> None:
        
        self.encoder = encoder
        self.minion = Head(encoder, dim_in = embed_size, dim_out = proj_size)
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, alpha=1, device=None):
        x_pos, x_neg = make_samples(x,self.augment)
        x = torch.cat((x_pos, x_neg), dim=0).to(device)
        y = self.minion(x, alpha)
        label = make_labels(y).to(device)
        return y, label