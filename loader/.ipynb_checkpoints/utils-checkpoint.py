import random 

import numpy as np
import soundfile as sf 

from audiomentations import AddGaussianNoise,  PitchShift, Shift, BandStopFilter





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


def transform(signal):
    sr = 16000
    signal = signal.astype(np.float32)
    t0 = AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.03, p=1)
    t1 = BandStopFilter(p = 1)
    t2 = PitchShift(min_semitones=-2, max_semitones=1, p=1)
    t3 = Shift(min_fraction=-0.5, max_fraction=0.5, p=1)
    
    aug_type = np.random.randint(0,4)
    if aug_type == 0:
        signal = t0(signal, sample_rate = sr)
    if aug_type == 1:
        signal = t1(signal, sample_rate = sr)
    if aug_type == 2:
        signal = t2(signal, sample_rate = sr)
    if aug_type == 3:
        signal = t3(signal, sample_rate = sr)
    return signal, aug_type

