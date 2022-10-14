import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from .common import DSBN1d, PreEmphasis, DSBN2d


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = DSBN2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = DSBN2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        domain = None
        if type (x) == tuple:
            x, domain = x
        
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out, domain)

        out = self.conv2(out)
        out = self.bn2(out, domain)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x, domain)

        out += residual
        out = self.relu(out)
        return out, domain


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(dim_in, 128, kernel_size=1)
        self.bn = DSBN1d(128)
        self.conv2 = nn.Conv1d(128, dim_out, kernel_size=1)

    def forward(self, x, domain):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn(x, domain)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = F.softmax(x, dim=2)
        return x
    
    
class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out, stride):
        super().__init__()
        
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)
        self.bn = DSBN2d(dim_out)
        
    def forward(self, x, domain):
        x = self.conv2d(x)
        x = self.bn(x, domain)
        return x    
    

class ResNet34(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=64, log_input=True, **kwargs):
        super(ResNet34, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = DSBN2d(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
                )

        outmap_size = int(self.n_mels/8)

        dim = num_filters[3] * outmap_size
        self.attention = Attention(dim, dim)

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)
        self.bn_last = DSBN1d(nOut)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, domain):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x, domain)

        x = self.layer1((x, domain))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if type(x) == tuple:
            x = x[0]
        

        x = x.reshape(x.size()[0],-1,x.size()[-1])

        w = self.attention(x, domain)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def ResNet34_DSBN(nOut=256, **kwargs):
    # Number of filters
    num_blocks = [3, 4, 6, 3]
    num_filters = [32, 64, 128, 256]
    model = ResNet34(SEBasicBlock, num_blocks, num_filters, nOut, **kwargs)
    return model
