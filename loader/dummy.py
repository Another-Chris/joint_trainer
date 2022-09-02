import torch 
from torch.utils.data import Dataset

class DummyLoader(Dataset):
    def __init__(self, siglen = (1, 32147)):
        self.siglen = siglen

    def __getitem__(self, index):
        data = {}
        
        
        data['same_anchor'] = torch.normal(mean = 0, std = 1, size = (32210, ))
        data['same_anchor_aug'] = torch.normal(mean = 0, std = 1, size = (32210, ))
        data['same_pos'] = torch.normal(mean = 0, std = 1, size = (32210, ))
        
        data['diff'] = torch.normal(mean = 0, std = 1, size = (32210, ))
        data['diff_aug'] = torch.normal(mean = 0, std = 1, size = (32210, ))
        
        for key,value in data.items():
            data[key] = torch.FloatTensor(value)
        
        nviews = 2
        return data, 0

    def __len__(self):
        return 1024 * 1024
