import torch.nn as nn


class Head(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in, head='mlp', feat_dim=128, **kwargs):
        super().__init__(**kwargs)
                
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        return self.head(x)
