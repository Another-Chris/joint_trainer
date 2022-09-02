import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
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


melkwargs = {
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "f_min": 20,
    "f_max": 7600,
    "window_fn": torch.hamming_window,
    "n_mels": 80
}

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

Torchfbank = torch.nn.Sequential(
    PreEmphasis(),
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, **melkwargs),
).to(DEVICE)

TorchMFCC = torch.nn.Sequential(
    PreEmphasis(),
    torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=20, log_mels=True, melkwargs=melkwargs)
).to(DEVICE)
