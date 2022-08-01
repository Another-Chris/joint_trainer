import pickle

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# d = '../save/RawNet3_SSL/feature'
d = '../save/ResNetSE34L_SSL/feature'
with open(f'{d}/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open(f'{d}/labels.pkl', 'rb') as f:
    labels = pickle.load(f)


spks = [f'id0080{i}' for i in range(6)]

for i in range(200, len(labels)):
    if labels[i] not in spks:
        break

labels = labels[200:i]
features = features[200:i]

features = np.stack(features, axis=0)

proj = TSNE(learning_rate='auto', init='pca').fit_transform(features)

df = pd.DataFrame({'x1': proj[:, 0], 'x2': proj[:, 1], 'label': labels})

plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='x1', y='x2', hue='label')
plt.show()
