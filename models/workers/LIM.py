import torch.nn as nn

import torch

from Head import Head


class LIM(nn.Module):
    def __init__(self, embed_size, device) -> None:
        
        self.discriminator = Head(dim_in = 2 * embed_size, feat_dim=1)
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device
        
    def forward(self, embed_anchor, embed_pos, embed_neg):

        bz = embed_anchor.size()[0]
        pos = torch.cat([embed_anchor, embed_pos], dim = 1)
        neg = torch.cat([embed_anchor, embed_neg], dim = 1)
        z1z2 = torch.cat([pos, neg], dim = 0)
        target = self.discriminator(z1z2)    
        label = torch.cat([torch.ones(size = (bz // 2, 1)), torch.zeros(size = (bz // 2,1))], dim = 0).to(self.device)
        return self.loss(target, label)
        