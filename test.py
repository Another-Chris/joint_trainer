from random import sample
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift

import numpy as np


def transform(signal):
    sr = 16000
    t0 = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1)
    t1 = TimeStretch(min_rate=0.9, max_rate=1.1, p=1)
    t2 = PitchShift(min_semitones=-2, max_semitones=1, p=1)
    t3 = Shift(min_fraction=-0.3, max_fraction=0.3, p=1)
    
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
        