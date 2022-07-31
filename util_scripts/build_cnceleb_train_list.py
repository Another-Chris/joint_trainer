import glob
import re
import multiprocessing
import os 

from tqdm import tqdm

import soundfile as sf

def write(files, lines):
    
    for file in tqdm(files):

        signal, sr = sf.read(file)
        if len(signal) // sr < 5:
            continue
        
        label = re.findall(r'(id\d+)', file)[0]
        file = file.replace('../data/cnceleb/data\\', '')
        lines.append(f'{label} {file}\n')
            

files = glob.glob('../data/cnceleb/data/*/*.flac')


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    lines = manager.list()
    nprocess = 8
    per_process = len(files) // nprocess + 1
    ps = []
    for i in range(nprocess):
        fileseg = files[i*per_process: (i+1)*per_process]

        p = multiprocessing.Process(target=write, args=(fileseg,lines))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
        
    
    if os.path.exists('../data/train_list_cnceleb.txt'):
        os.unlink('../data/train_list_cnceleb.txt')
        
    with open('../data/train_list_cnceleb.txt', 'w') as f:
        for line in lines:
            f.write(line)
