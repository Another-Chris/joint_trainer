import torch.nn as nn
import torch.nn.functional as F
import torch

from Head import Head
from loss import AngleProtoLoss


class Sup(nn.Module):
    def __init__(self, num_classes, embed_size = 256) -> None:
        
        self.head = Head(dim_in = embed_size, dim_out = 128)
        self.loss = AngleProtoLoss()
    
    def forward(self, anchor, pos):
        feat_anchor = self.head(anchor)
        feat_pos = self.head(pos)
        feat = torch.cat([feat_anchor, feat_pos], dim = 0)
        
        
        
        