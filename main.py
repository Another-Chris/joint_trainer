from models import PASE, ECAPA_TDNN

import torch
import librosa.display

import matplotlib.pyplot as plt


feature_extractor = PASE("./configs/PASE+.cfg")

data = torch.normal(size = (1,1,10000), mean = 0, std = 1)
feature = feature_extractor(data)

model = ECAPA_TDNN()
feature = torch.normal(size = (32,256,1000), mean = 1, std = 1)
embedding = model(feature, aug = False)
print(embedding.shape)
# librosa.display.specshow(feature.detach().numpy()[0])
# plt.show()
