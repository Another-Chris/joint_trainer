import torch.nn as nn

from Head import Head


class Cls(nn.Module):
    def __init__(self, encoder,num_classes, embed_size = 256) -> None:
        
        self.encoder = encoder
        self.minion = Head(encoder, dim_in = embed_size, dim_out = num_classes)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x, alpha=1, device=None):
        return self.minion(x)