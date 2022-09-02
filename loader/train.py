from .utils import load_wav
from .augment import AugmentWAV
from torch.utils.data import Dataset

import random
import torch
import os

import numpy as np


class TrainDatasetLoader():

    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path):
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

    def __getitem__(self, idx):

        data = {}
        
        same_utt = self.data_list[idx]
        diff_utt = self.data_list[np.random.choice(np.arange(0, idx), 1)]
        
        data['same_anchor'] = load_wav(same_utt, self.max_frames, evalmode=False)
        data['same_anchor_aug'] = self.augment_audio(data['same_anchor'])
        data['same_pos'] = load_wav(same_utt, self.max_frames, evalmode=False)
        
        data['diff'] = load_wav(diff_utt, self.max_frames, evalmode=False)
        data['diff_aug'] = self.augment_audio(data['diff'])
        
        for key,value in data.items():
            data[key] = torch.FloatTensor(value)
        
        return data, self.data_label[idx]
