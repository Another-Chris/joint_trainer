from utils import Config
from scipy import signal as ss

import os
import torch
import glob
import random

import numpy as np
import soundfile as sf


import numpy as np


""" functions """
def load_wav(filename, max_frames=None, max_audio=None, evalmode=False, num_eval=10):

    if max_audio is None and max_frames is None:
        raise ValueError('please specify either max_frames or max_audio')

    if max_audio is None:
        max_audio = max_frames * 160 + 240

    audio, _ = sf.read(filename)
    audiosize = audio.shape[0]

    # pad short audio frames
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    # random seg; if short, startframe = 0
    # num_eval: take segments in a linearly increasing fashion
    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array(
            [np.int64(random.random() * (audiosize - max_audio))])

    # actually take the segment
    feats = []
    for asf in startframe:
        feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat



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


""" augmentation """
class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240  # signal length

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [0, 15],
                         'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        # repr(data\d) --> data\\d, so there is actually only one \
        augment_files = [f.replace('\\', '/') for f in augment_files]
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []

            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    # noisecat: noise category: noise, speech, music
    def additive_noise(self, noisecat, audio, max_audio=None, max_frames=None):        
        if max_audio is None:
            if max_frames is None:
                max_audio = self.max_frames * 160 + 240
            else:
                max_audio = max_frames * 160 + 240

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = load_wav(noise, max_audio=max_audio, evalmode=False)
            noise_snr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio, max_frames=None, max_audio=None):

        if max_audio is None:
            if max_frames is None:
                max_audio = self.max_frames * 160 + 240
            else:
                max_audio = max_frames * 160 + 240

        rir_file = random.choice(self.rir_files)
        rir, _ = sf.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))

        return ss.convolve(audio, rir, mode='full')[:, :max_audio]

""" loader """
class DsLoader(torch.utils.data.Dataset):
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

    def augment_audio(self, audio, max_frames = None, max_audio = None):

        randi = np.random.randint(0, 6)
        
        if randi == 0:
            audio = audio
        if randi == 1:
            audio = self.augment_wav.additive_noise('music', audio, max_frames = max_frames, max_audio = max_audio)
        if randi == 2:
            audio = self.augment_wav.additive_noise('speech', audio, max_frames = max_frames, max_audio = max_audio)
        if randi == 3:
            audio = self.augment_wav.additive_noise('noise', audio, max_frames = max_frames, max_audio = max_audio)
        if randi == 4:
            audio = self.augment_wav.reverberate(audio, max_frames = max_frames, max_audio = max_audio)
        if randi == 5:
            audio = self.augment_wav.additive_noise('speech', audio, max_frames = max_frames, max_audio = max_audio)
            audio = self.augment_wav.additive_noise('music', audio, max_frames = max_frames, max_audio = max_audio)

        return audio

    def __len__(self):
        return len(self.source_data)
    

    def get_tuple(self, idx, data_list, label_list, eval_mode, num_eval=Config.GIM_SEGS):
        anchor = self.augment_audio(load_wav(data_list[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        pos = self.augment_audio(load_wav(data_list[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval))

        anchor = torch.FloatTensor(anchor)
        pos = torch.FloatTensor(pos)

        return {'anchor': anchor, 'pos': pos}, label_list[idx]
    
    def __getitem__(self, idx):
        eval_mode = False
        num_eval = 5
        
        ## two segments
        # source_data, source_label = self.get_tuple(idx, self.source_data, self.source_label, eval_mode=False)
        tidx = np.random.randint(0, len(self.target_data))
        target_data, target_label = self.get_tuple(tidx, self.target_data, self.target_label, eval_mode=False)
        
        ## one segment
        source_data = torch.FloatTensor(
            self.augment_audio(load_wav(self.source_data[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval)))
        source_label = self.source_label[idx]
        
        # tidx = np.random.randint(0, len(self.target_data))
        # target_data = torch.FloatTensor(
        #     self.augment_audio(load_wav(self.target_data[tidx], self.max_frames, evalmode=eval_mode, num_eval=num_eval)))
        # target_label = self.target_label[tidx]
        
        # tidx = np.random.randint(0, len(self.target_data))
        # max_audio = 5 * (100 * 160 + 240)
        # target_audio = load_wav(self.target_data[tidx], max_audio = max_audio)
        # target_data = {
        #     'anchor': torch.FloatTensor(self.augment_audio(target_audio, max_audio=max_audio)),
        #     'pos': torch.FloatTensor(self.augment_audio(target_audio, max_audio=max_audio))
        # }
        # target_label = self.target_label[tidx]

        return {
            'source_data': source_data,
            'target_data': target_data,
        }, {
            'source_label': source_label,
            'target_label': target_label,
        }

class test_dataset_loader(torch.utils.data.Dataset):
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
