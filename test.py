import torch
from utils import Config
from loader import JointLoader
import numpy as np
from scipy.io import wavfile
SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'
TARGET_PATH = './data/cnceleb/data/'
TARGET_LIST = './data/cnceleb_train_gt5.txt'

ds = JointLoader(
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


def get_pair(data):
    anchor_len = np.random.randint(1, 5)
    if anchor_len == 4: 
        pos_len = 1
    else:
        pos_len = np.random.randint(1, 5-anchor_len)
        
    anchor = []
    pos = []
    for i in range(5):
        
        if len(anchor) == anchor_len and len(pos) == pos_len: break
        
        d = data[:, i, :]
        
        if len(anchor) == anchor_len:
            pos.append(d)
        elif len(pos) == pos_len:
            anchor.append(d)
        else:
            if np.random.random() < 0.5:
                anchor.append(d)
            else:
                pos.append(d)
    
    
    anchor = np.concatenate(anchor, axis = 1)
    pos = np.concatenate(pos, axis = 1)
    return anchor, pos
    

if __name__ == '__main__':
    for data, label in loader:
        target = data['target_data']
        anchor, pos = get_pair(target)
        print(anchor.shape, pos.shape)
        
        wavfile.write('anchor.wav', 16000, anchor[0])
        wavfile.write('pos.wav', 16000, pos[0])

        break
