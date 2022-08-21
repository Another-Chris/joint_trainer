import torch 
import os 
from torch.utils.data import Dataset

from .utils import load_wav


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_path = test_path
        self.test_list = test_list

    def __getitem__(self, index):
        audio = load_wav(os.path.join(self.test_path, self.test_list[index]), self.max_frames, evalmode=True,
                         num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)
