import torch.nn as nn
import torch.nn.functional as F

class BigHead(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in, head='mlp', feat_dim=128, hidden_size = 1024, **kwargs):
        super().__init__(**kwargs)
        
        self.layer1 = nn.Linear(dim_in, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, feat_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        # x = F.relu(self.bn1(self.layer1(x)))
        # x = F.relu(self.bn2(self.layer2(x)))
        
        x = F.relu(self.dp1(self.layer1(x)))
        x = F.relu(self.dp2(self.layer2(x)))  
        
        return self.layer3(x)
