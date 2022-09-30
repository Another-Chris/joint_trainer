import torch.nn as nn
import torch.nn.functional as F

class BigHead(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in, head='mlp', feat_dim=128, **kwargs):
        super().__init__(**kwargs)
        
        self.layer1 = nn.Linear(dim_in, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
