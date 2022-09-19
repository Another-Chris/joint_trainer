from .utils import load_wav
from .augment import AugmentWAV
from utils import Config


import random
import os

import numpy as np


def get_data_from_file(data_root, filename):

    with open(filename) as dataset_file:
        lines = dataset_file.readlines()

    dictkeys = list(set([x.split()[0] for x in lines]))
    dictkeys.sort()
    dictkeys = {key: li for li, key in enumerate(dictkeys)}

    data_list = [] 
    data_label = []

    for line in lines:
        data = line.strip().split()

        speaker_label = dictkeys[data[0]]  # data[0]: index
        filename = os.path.join(data_root, data[1])

        data_label.append(speaker_label)
        data_list.append(filename)
        
    return data_list, data_label


class DsLoader():
    def __init__(
            self, 
            source_list,
            source_path,
            augment, 
            musan_path, 
            rir_path, 
            max_frames,
            target_list = None,
            target_path = None, 
            ):
        self.augment_wav = AugmentWAV(
            musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        self.source_data, self.source_label = get_data_from_file(source_path, source_list)
        
        if target_list is not None and target_path is not None:
            self.target_data, self.target_label = get_data_from_file(target_path, target_list)
        
        self.max_frames = max_frames
        self.augment = augment

    def augment_audio(self, audio):
        augtype = random.randint(0, 4)
        if augtype == 1:
            audio = self.augment_wav.reverberate(audio)
        elif augtype == 2:
            audio = self.augment_wav.additive_noise('music', audio)
        elif augtype == 3:
            audio = self.augment_wav.additive_noise('speech', audio)
        elif augtype == 4:
            audio = self.augment_wav.additive_noise('noise', audio)
        return audio

    def __len__(self):
        return len(self.source_data)
    
    def get_triplet(self, idx, data_list, label_list, eval_mode, num_eval):

        diff_idx = np.random.choice([i for i in range(len(data_list)) if i != idx], 1)[0]
        
        anchor = self.augment_audio(load_wav(
            data_list[idx], self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        same = self.augment_audio(load_wav(
            data_list[idx], self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        diff = self.augment_audio(load_wav(
            data_list[diff_idx], self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        
        return anchor, same, diff, label_list[idx], label_list[diff_idx]   
    
        
class JointLoader():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        source_anchor, source_same, source_diff, source_anchor_label, source_diff_label\
            = self.get_triplet(idx, self.source_data, self.source_label, eval_mode=False, num_eval = 1)
                    
        gim_idx = np.random.choice(range(len(self.target_data)),1)[0]
        gim_anchor, gim_same, gim_diff, gim_anchor_label, gim_diff_label\
            = self.get_gim_triplet(gim_idx, self.target_data, self.target_label, eval_mode = True, num_eval = Config.GIM_SEGS)
        
        target_idx = np.random.choice(range(len(Config.GIM_SEGS)),1)[0]
        
        data =  {
            'source_anchor': source_anchor,
            'source_same': source_same,
            'source_diff': source_diff,
   
            'target_anchor': gim_anchor[target_idx],
            'target_same': gim_same[target_idx],
            'target_diff': gim_diff[target_idx],
            
            'gim_anchor': gim_anchor,
            'gim_same': gim_same,
            'gim_diff': gim_diff,
        }
        
        label =  {
            'source_anchor': source_anchor_label,
            'source_diff': source_diff_label,
 
            'target_anchor': gim_anchor_label,
            'target_diff': gim_diff_label,
            
            'gim_anchor': gim_anchor_label,
            'gim_diff': gim_diff_label,
        }
        return data, label
        