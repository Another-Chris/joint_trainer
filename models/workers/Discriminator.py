import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """backbone + projection head"""

    def __init__(self, encoder, dim_in, head='mlp', feat_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1)
        return feat
