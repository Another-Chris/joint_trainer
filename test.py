import sys
import torch 

from loader import FullSupConDataLoader
from utils import Config
from scipy.io.wavfile import write


SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'

ds = FullSupConDataLoader(
    source_list=SOURCE_LIST,
    source_path=SOURCE_PATH,
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
    for data in loader:
        for i in range(Config.GIM_SEGS):
            write(f'anchor-{i}.wav', data = data[0]['gim_anchor'][0][i].numpy(), rate = 16000)
            write(f'pos-{i}.wav', data = data[0]['gim_pos'][0][i].numpy(), rate = 16000)
        break