import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from .common import DSBN1d, PreEmphasis, FbankAug
from .ECAPA_TDNN import SEModule


class Bottle2neck_with_DSBN(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck_with_DSBN, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1 = DSBN1d(width*scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(
                width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(DSBN1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3 = DSBN1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x, domain):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out, domain)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp, domain)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out, domain)

        out = self.se(out)
        out += residual
        return out


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(4608, 256, kernel_size=1)
        self.bn = DSBN1d(256)
        self.conv2 = nn.Conv1d(256, 1536, kernel_size=1)

    def forward(self, x, domain):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn(x, domain)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = F.softmax(x, dim=2)
        return x


class ECAPA_TDNN_WITH_DSBN(nn.Module):
    def __init__(self, C, embed_size) -> None:

        super().__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation

        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = DSBN1d(C)
        self.layer1 = Bottle2neck_with_DSBN(
            C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_with_DSBN(
            C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_with_DSBN(
            C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = Attention()

        self.bn5 = DSBN1d(3072)
        self.fc6 = nn.Linear(3072, embed_size)
        self.bn6 = DSBN1d(embed_size)

    def forward(self, x, domain, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x, domain)

        x1 = self.layer1(x, domain)
        x2 = self.layer2(x+x1, domain)
        x3 = self.layer3(x+x1+x2, domain)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))

        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t), torch.sqrt(
            torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x, domain)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x, domain)
        x = self.fc6(x)
        x = self.bn6(x, domain)
        return x
