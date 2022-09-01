import torch.nn as nn
import torch.nn.functional as F
import torch


class MFCC(nn.Module):

    def __init__(self, encoder, feat_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

        self.transp1 = nn.ConvTranspose1d(1, 10, 2, 1)
        self.transp2 = nn.ConvTranspose1d(10, 20, 2, 1)
        self.fc1 = nn.Linear(194, 128)
        self.fc2 = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.unsqueeze(x, dim = 1)

        x = self.transp1(x)
        x = F.relu(x)
        x = self.transp2(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
