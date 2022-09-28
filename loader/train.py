from .utils import load_wav
from .augment import AugmentWAV
from utils import Config

import os
import torch

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
            target_list=None,
            target_path=None,
    ):
        self.augment_wav = AugmentWAV(
            musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        self.source_data, self.source_label = get_data_from_file(
            source_path, source_list)

        if target_list is not None and target_path is not None:
            self.target_data, self.target_label = get_data_from_file(
                target_path, target_list)

        self.max_frames = max_frames
        self.augment = augment

    def augment_audio(self, audio, return_type=False):

        augtype = [0]
        if np.random.random() < 0.8:
            audio = self.augment_wav.reverberate(audio)
            augtype.append(1)

        randi = np.random.randint(0, 3)
        if randi == 0:
            audio = self.augment_wav.additive_noise('music', audio)
            augtype.append(2)
        if randi == 1:
            audio = self.augment_wav.additive_noise('speech', audio)
            augtype.append(3)
        if randi == 2:
            audio = self.augment_wav.additive_noise('noise', audio)
            augtype.append(4)

        if return_type:
            return audio, sum(augtype)
        return audio

    def __len__(self):
        return len(self.source_data)

    def get_triplet(self, idx, data_list, label_list, eval_mode, num_eval):

        diff_idx = np.random.randint(0, len(data_list))
        while diff_idx == idx:
            diff_idx = np.random.randint(0, len(data_list))
            
        anchor = self.augment_audio(load_wav(data_list[idx], self.max_frames, evalmode=eval_mode, num_eval = num_eval))
        pos = self.augment_audio(load_wav(data_list[idx], self.max_frames, evalmode=eval_mode, num_eval = num_eval))
        diff = self.augment_audio(load_wav(data_list[diff_idx], self.max_frames, evalmode=eval_mode, num_eval = num_eval))
        
        return {
            'anchor': anchor, 'pos': pos, 'diff': diff
        }, {
            'anchor': label_list[idx], 'diff': label_list[diff_idx]
        }

    
    def get_tuple(self, idx, data_list, label_list, eval_mode, num_eval = Config.GIM_SEGS):
        anchor = self.augment_audio(load_wav(data_list[idx], self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        pos = self.augment_audio(load_wav(data_list[idx], self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        
        anchor = torch.FloatTensor(anchor)
        pos = torch.FloatTensor(pos)
        
        return {'anchor': anchor, 'pos': pos}, label_list[idx]


class JointLoader(DsLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, idx):
       
        source_data, source_label = self.get_tuple(idx, self.source_data, self.source_label, eval_mode=False)
        
        tidx = np.random.randint(0, len(self.target_data))
        target_data, target_label = self.get_tuple(tidx, self.target_data, self.target_label, eval_mode=False)

        return {
            'source_data': source_data,
            'target_data': target_data,
        }, {
            'source_label': source_label,
            'target_label': target_label,
        }


class SourceDataLoader(DsLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        anchor, same, diff, anchor_label, diff_label\
            = self.get_triplet(idx, self.data, self.label, eval_mode=False, num_eval=1)

        data = {
            'anchor': anchor,
            'same': same,
            'diff': diff,
        }

        label = {
            'anchor': anchor_label,
            'diff': diff_label,
        }
        return {key: torch.FloatTensor(value) for key, value in data.items()}, label

# a pair of augmented data
class SimpleDataLoader(DsLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        
        # diff_idx = np.random.randint(0, len(self.source_data))
        # while diff_idx == idx:
        #     diff_idx = np.random.randint(0, len(self.source_data))

        return {
            'anchor': torch.FloatTensor(self.augment_audio(load_wav(self.source_data[idx], self.max_frames, evalmode=False))),
            'pos': torch.FloatTensor(self.augment_audio(load_wav(self.source_data[idx], self.max_frames, evalmode=False))),
            # 'diff': torch.FloatTensor(self.augment_audio(load_wav(self.source_data[diff_idx], self.max_frames, evalmode=False)))
        }, self.source_label[idx]


class FullSupConDataLoader(DsLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # anchor and same are two versions of augmentations
    def __getitem__(self, idx):
        return {
            'anchor': torch.FloatTensor(self.augment_audio(load_wav(self.source_data[idx], self.max_frames, evalmode=False))),
            'pos': torch.FloatTensor(self.augment_audio(load_wav(self.source_data[idx], self.max_frames, evalmode=True, num_eval=Config.GIM_SEGS)))
        }, self.source_label[idx]

class AugTypeDataLoader(DsLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):

        anchor, anchor_aug = self.augment_audio(
            load_wav(self.source_data[idx], self.max_frames, evalmode=False), return_type=True)
        pos, pos_aug = self.augment_audio(
            load_wav(self.source_data[idx], self.max_frames, evalmode=False), return_type=True)

        anchor = torch.FloatTensor(anchor)
        pos = torch.FloatTensor(pos)

        return {
            'anchor': anchor,
            'pos': pos
        }, self.source_label[idx], {
            'anchor': anchor_aug,
            'pos': pos_aug
        }
