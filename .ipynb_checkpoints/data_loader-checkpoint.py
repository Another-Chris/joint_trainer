from utils import Config
from scipy import signal as ss
from pathlib import PurePath

import os
import torch
import glob
import random

import soundfile as sf
import numpy as np



GENRE_MAP = {name: i for i, name in enumerate([
    'entertainment','interview','singing','live_broadcast','recitation','speech','play','advertisement', 'drama','movie','vlog'
])}



""" functions """
def load_wav(filename, max_frames, evalmode=False, num_eval=10):

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
    def additive_noise(self, noisecat, audio):        
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = load_wav(noise,max_frames=self.max_frames,evalmode=False)
            noise_snr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, _ = sf.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))

        return ss.convolve(audio, rir, mode='full')[:, :self.max_audio]

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

    def augment_audio(self, audio):

        randi = np.random.randint(0, 6)
        
        if randi == 0:
            audio = audio
        if randi == 1:
            audio = self.augment_wav.additive_noise('music', audio)
        if randi == 2:
            audio = self.augment_wav.additive_noise('speech', audio)
        if randi == 3:
            audio = self.augment_wav.additive_noise('noise', audio)
        if randi == 4:
            audio = self.augment_wav.reverberate(audio)
        if randi == 5:
            audio = self.augment_wav.additive_noise('speech', audio)
            audio = self.augment_wav.additive_noise('music', audio)

        return audio

    def __len__(self):
        return len(self.source_data)
    

    def get_tuple(self, idx, data_list, label_list, eval_mode, num_eval=Config.GIM_SEGS):
        # anchor = self.augment_audio(load_wav(data_list[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        # pos = self.augment_audio(load_wav(data_list[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval))
        # anchor = torch.FloatTensor(anchor)
        # pos = torch.FloatTensor(pos)
        
        utt = load_wav(data_list[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval)
        aug1 = self.augment_audio(utt)
        aug2 = self.augment_audio(utt)
        
        middle = aug1.shape[1] // 2
        anchor = torch.FloatTensor(aug1[:, :middle])
        pos = torch.FloatTensor(aug2[:, middle:])
        
        return {'anchor': anchor, 'pos': pos}, label_list[idx]
    
    def __getitem__(self, idx):
        eval_mode = False
        num_eval = 5
        data = {}
        label = {}
        
        ## two segments
        # source_data, source_label = self.get_tuple(idx, self.source_data, self.source_label, eval_mode=False)
        # tidx = np.random.randint(0, len(self.target_data))
        # target_data, target_label = self.get_tuple(tidx, self.target_data, self.target_label, eval_mode=False)

        ## one segment
        source_data = torch.FloatTensor(
            self.augment_audio(load_wav(self.source_data[idx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval)))
        source_label = self.source_label[idx]
        
        tidx = np.random.randint(0, len(self.target_data))
        target_data = torch.FloatTensor(
            self.augment_audio(load_wav(self.target_data[tidx], max_frames = self.max_frames, evalmode=eval_mode, num_eval=num_eval)))
        target_label = self.target_label[tidx]
        
        data['source_data'] = source_data
        data['target_data'] = target_data
        label['source_label'] = source_label
        label['target_label'] = target_label
        # path = PurePath(self.target_data[tidx])
        # genre = path.name.split('-')[0]
        # label['target_genre'] = genre
        return data, label

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
