import glob 
import re
import soundfile as sf 
from tqdm import tqdm

tlen = 5 * 16000
files = glob.glob('./data/cnceleb/data/*/*.flac')

dev_list = []
with open('./data/cnceleb/dev/dev.lst') as f:
    for line in f:
        dev_list.append(line.strip())
        
l = []
for i,file in enumerate(tqdm(files)):
    curr_label = re.findall(r'(id\d+)', file)[0]
    if curr_label not in dev_list: continue
    
    if 'concat' in file: continue
    
    fpath = file.replace('./data/cnceleb/data\\', '')
    l.append(f"{curr_label} {fpath}")

with open('./data/cnceleb_train_whole.txt', 'w') as f:
    for line in l:
        f.write(line + '\n')