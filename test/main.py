import random 

import numpy as np
import soundfile as sf 


import scipy.signal as ss 


d = np.random.random(size = (10, 100))
rir = np.random.random(size = (1, 100))

c1 = ss.convolve(d, rir)

c2 = []
for dd in d:
    c2.append(ss.convolve(dd[None, ...], rir))
c2 = np.concatenate(c2)

print((c1 == c2).all())