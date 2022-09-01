import librosa 
import numpy as np 
y = np.random.rand(32,12240) 

for yy in y:
    print(librosa.yin(yy, fmin=440, fmax=880).shape)

f = librosa.yin(y, fmin=440, fmax=880)
print(f"{y.shape = }")
print(f"{f.shape = }")