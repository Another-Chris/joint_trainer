import torch.nn as nn
import torch.nn.functional as F
import torch


class Spec(nn.Module):

    def __init__(self, encoder, feat_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

        self.transp1 = nn.ConvTranspose1d(1, 20, 2, 1)
        self.transp2 = nn.ConvTranspose1d(20, 40, 2, 1)
        self.transp3 = nn.ConvTranspose1d(40, 80, 2, 1)
        self.fc1 = nn.Linear(195, 128)
        self.fc2 = nn.Linear(128, feat_dim)

    def forward(self, x, aug):
        x = self.encoder(x, aug)
        x = torch.unsqueeze(x, dim = 1)

        x = self.transp1(x)
        x = F.relu(x)
        x = self.transp2(x)
        x = F.relu(x)
        x = self.transp3(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
