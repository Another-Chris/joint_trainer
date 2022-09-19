import torch.nn as nn

from .Head import Head


class Cls(nn.Module):
    def __init__(self, num_classes, embed_size = 256) -> None:
        
        super().__init__()
        
        self.head = Head(dim_in = embed_size, feat_dim = num_classes)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.head(x)