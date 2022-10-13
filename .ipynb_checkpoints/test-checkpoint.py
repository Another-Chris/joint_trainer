import torch
from utils import Config
import numpy as np
from scipy.io import wavfile
from data_loader import DsLoader
from utils import get_pair

SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'
TARGET_PATH = './data/cnceleb/data/'
TARGET_LIST = './data/cnceleb_train_gt5.txt'

ds = DsLoader(
    source_list=SOURCE_LIST,
    source_path=SOURCE_PATH,
    target_list=TARGET_LIST,
    target_path=TARGET_PATH,
    augment=True,
    musan_path=Config.MUSAN_PATH,
    rir_path=Config.RIR_PATH,
    max_frames=Config.MAX_FRAMES
)
loader = torch.utils.data.DataLoader(
    ds,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)


if __name__ == '__main__':
    for data, label in loader:
        target_anchor, target_pos = get_pair(data['target_data'])
        print(target_anchor.shape, target_pos.shape)

        break
