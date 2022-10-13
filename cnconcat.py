import glob 
import re
import soundfile as sf 
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

tlen = 5 * 16000
files = glob.glob('./data/cnceleb/data/*/*.flac')

dev_list = []
with open('./data/cnceleb/dev/dev.lst') as f:
    for line in f:
        dev_list.append(line)

cat = np.array([])
curr_label = None
prev_label = None

for i,file in enumerate(tqdm(files)):
    
    curr_label = re.findall(r'(id\d+)', file)[0]
    if curr_label not in dev_list: continue
    
    audio, _ = sf.read(file)
    
    if curr_label != prev_label:
        cat = np.array([])
    
    if len(audio) < tlen:
        cat = np.concatenate([cat, audio])
    
    if len(cat) >= tlen:
        fname = f'./data/cnceleb/data/{curr_label}/{curr_label}-concat-{i}.flac'
        wavfile.write(fname, rate = 16000, data = cat)
        cat = np.array([])
        
    prev_label = curr_label
        
        
        