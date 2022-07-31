from pydub import AudioSegment
from tqdm import tqdm

import glob
import re
import os
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='train')
args = parser.parse_args()

fpath = './data/voxceleb2/*/*/*' if args.type == 'train' else './data/voxceleb1_test/*/*/*'

files = glob.glob(fpath)


def convert_audio(files):

    for file in tqdm(files):

        if '.flac' in file:
            continue

        ofile = re.sub(r'\.(\w+)', '.flac', file)
        AudioSegment.from_file(file).export(ofile, format='flac')

        os.unlink(file)


if __name__ == '__main__':
    nprocess = 8
    files_per_process = len(files) // 8 + 1

    processes = []
    for i in range(nprocess):
        
        fileseg = files[i*files_per_process: (i+1)*files_per_process]
        print(f"{len(fileseg) = }")
        p = multiprocessing.Process(
            target=convert_audio,
            args=(fileseg,)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
