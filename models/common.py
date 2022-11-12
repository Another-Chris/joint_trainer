import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function



"""DSBN"""
class DSBN1d(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()

        self.bn_source = nn.BatchNorm1d(size)
        self.bn_target = nn.BatchNorm1d(size)

    def forward(self, x, domain):
        if domain == 'source':
            return self.bn_source(x)
        elif domain == 'target':
            return self.bn_target(x)
        else:
            raise ValueError("please specify a domain")


class DSBN2d(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()

        self.bn_source = nn.BatchNorm2d(size)
        self.bn_target = nn.BatchNorm2d(size)

    def forward(self, x, domain):
        if domain == 'source':
            return self.bn_source(x)
        elif domain == 'target':
            return self.bn_target(x)
        else:
            raise ValueError("please specify a domain")

"""head"""
class Head(nn.Module):
    def __init__(self, dim_in, head='mlp', feat_dim=128, hidden_size = 256):
        super().__init__()
        
        self.l1 = nn.Linear(dim_in, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, feat_dim)
        self.head = head
        
    def forward(self, x, domain):
        if self.head == 'linear': 
            x = self.l2(x)
        if self.head == 'mlp':
            x = F.relu(self.bn(self.l1(x)))
            x = self.l2(x)
        return x
    
""" discriminator """

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class Discriminator(nn.Module):
    def __init__(self, dim_in, feat_dim=128, hidden_size = 256):
        super().__init__()
        
        self.l1 = nn.Linear(dim_in, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dp1 = nn.Dropout(0.5)
        
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dp2 = nn.Dropout(0.5)
        
        self.l3 = nn.Linear(hidden_size, feat_dim)
        
    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        # x = self.dp1(F.relu((self.l1(x))))
        # x = self.dp2(F.relu((self.l2(x))))
        return self.l3(x)
        
#     def forward(self, x, alpha):
#         x = ReverseLayerF.apply(x, alpha)
        
#         bz = x.shape[0] // 2
#         x_source = self.one_branch(x[:bz], 'source')
#         x_target = self.one_branch(x[bz:], 'target')
#         return torch.cat([x_source, x_target])
    
    
"""other"""
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        # expected shape: [bz, 1, siglen]
        if len(input.size()) == 2:
            input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)
    
    
class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(
            0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x
