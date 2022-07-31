import numpy as np
import soundfile as sf

import re
import glob
import multiprocessing

from scipy.io import wavfile
from tqdm import tqdm

sr = 16000
nsamples = sr * 5  # 5s


def write_file(label, signal, fdir, index):
    fname = f"{label}-concat-{index}.flac"
    directory = "/".join(fdir.split('/')[:-1])
    new_fdir = f"{directory}/{fname}"
    wavfile.write(new_fdir, sr, signal)


def convert_process(files):
    print(len(files))

    combine = None
    prev_label = None

    for i, file in enumerate(tqdm(files)):
        file = file.replace('\\', '/')

        signal, sr = sf.read(file)
        length = len(signal) // sr
        label = re.findall(r'(id\d+)', file)[0]

        if prev_label != label:
            combine = None

        if length < 5:
            if combine is None:
                combine = signal
            else:
                if len(signal) < nsamples:
                    combine = np.concatenate([combine, signal])
                    if len(combine) > nsamples:
                        write_file(label, combine, file, i)
                        combine = None

        prev_label = label


files = glob.glob('../data/cnceleb/*/*/*.flac')

if __name__ == '__main__':
    nprocess = 8
    items_per_process = len(files) // nprocess + 1

    ps = []
    for i in range(nprocess):
        p = multiprocessing.Process(target=convert_process, args=(
            files[i*items_per_process:(i+1)*items_per_process], ))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
