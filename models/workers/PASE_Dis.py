import torch.nn as nn
import torch.nn.functional as F

class PASE_Dis(nn.Module):
    def __init__(self, in_channels) -> None:
        
        super().__init__()
        
        self.l1 = nn.Conv1d(in_channels = in_channels, out_channels= 256)
        self.l2 = nn.Conv1d(in_channels = 256, out_channels=1)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
        