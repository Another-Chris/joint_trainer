import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
EMBED_SIZE = 192

MAX_FRAMES = 200

inp = torch.normal(0, 1, (BATCH_SIZE, EMBED_SIZE)).unsqueeze(1)

spec = nn.Sequential(
    nn.ConvTranspose1d(1, 20, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose1d(20, 40, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose1d(40, 80, 2, 1),
    nn.ReLU(),
    nn.Linear(195, 200)
)


