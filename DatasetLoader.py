import random
import glob
import os

import numpy as np
import soundfile as sf
import torch

from scipy import signal as ss
from torch.utils.data import Dataset


def round_down(num, divisor): return num - (num % divisor)


def worker_init_fn(worker_id): np.random.seed(
    np.random.get_state()[1][0] + worker_id)


def load_wav(filename, max_frames, evalmode=True, num_eval=10):
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
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat


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
            noiseaudio = load_wav(noise, self.max_frames, evalmode=False)
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

    def __len__(self):
        return len(self.data_list)


class train_dataset_loader(TrainLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def __getitem__(self, idx):

        segs = []
        for _ in range(2):
            seg = load_wav(
                self.data_list[idx], self.max_frames, evalmode=False)
            seg = self.augment_audio(seg)
            segs.append(torch.FloatTensor(seg))

        return segs



class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
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


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label = data_source.data_label
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)

        # Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        def lol(lst, sz): return [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(
                min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(numpy.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        # Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        # Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        # Divide data to each GPU
        if self.distributed:
            total_size = round_down(
                len(mixed_list), self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) /
                              dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) /
                            dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch =
