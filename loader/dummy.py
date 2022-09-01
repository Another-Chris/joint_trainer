import torch 
from torch.utils.data import Dataset

class DummyLoader(Dataset):
    def __init__(self, siglen = (1, 32147)):
        self.siglen = siglen

    def __getitem__(self, index):
        segs, augs = [], []
        for _ in range(2):
            segs.append(torch.FloatTensor(torch.normal(0,1,size = self.siglen)))
            augs.append(torch.FloatTensor(torch.normal(0,1,size = self.siglen)))
            
            
        return segs + augs, 0

    def __len__(self):
        return 1024 * 1024
