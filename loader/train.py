from .utils import load_wav
from .augment import AugmentWAV
from torch.utils.data import Dataset

import random
import torch
import os

import numpy as np


class TrainLoader(Dataset):
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, **kwargs):
        self.augment_wav = AugmentWAV(
            musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)
        self.train_list = train_list
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment

        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # make a dictionary of speaker labels and indices
        # usage: label encoder
        # line: idxxxxx path
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: li for li, key in enumerate(dictkeys)}

        # parse the training list into file names and ID indices
        self.data_list = []  # store the filename
        self.data_label = []

        for line in lines:
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]  # data[0]: index
            filename = os.path.join(train_path, data[1])

            self.data_label.append(speaker_label)
            self.data_list.append(filename)

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
        return len(self.data_list)


class ssl_dataset_loader(TrainLoader):

    def __init__(self, nPerSpeaker,  **kwargs):
        super().__init__(**kwargs)
        self.nPerSpeaker = nPerSpeaker
        print(f'ssl dataset train list: {self.train_list}, {nPerSpeaker = }')

    def __getitem__(self, idx):

        segs, augs = [], []
        for _ in range(2):
            seg = load_wav(
                self.data_list[idx], self.max_frames, evalmode=False)
            
            segs.append(torch.FloatTensor(seg))
            augs.append(torch.FloatTensor(self.augment_audio(seg)))

        return segs + augs, self.data_label[idx]


class train_dataset_loader(TrainLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'sup dataset train list: {self.train_list}')

    def __getitem__(self, indices):

        feat = []

        for index in indices:

            audio = load_wav(self.data_list[index],
                             self.max_frames, evalmode=False)

            audio = self.augment_audio(audio)

            feat.append(audio)

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
