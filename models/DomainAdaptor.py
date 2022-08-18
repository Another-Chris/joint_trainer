import torch.nn as nn
import torch.nn.functional as F
import torch


class DomainAdaptor(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        
        self.l1 = nn.Linear(in_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = torch.sigmoid(x)
        return x
        