from pathlib import PurePath
from pydub import AudioSegment
from tqdm import tqdm
from scipy.io import wavfile
from collections import defaultdict

import glob
import multiprocessing
import os
import re
import webrtcvad
import struct
import argparse
import shutil

import numpy as np
import soundfile as sf


parser = argparse.ArgumentParser(description = "CN-Celeb data handler")
parser.add_argument('--convert', action = 'store_true', dest = 'convert', help = 'convert flac to wav')
parser.add_argument('--concat', action = 'store_true', dest = 'concat', help = 'concat short utterances')
parser.add_argument('--vad', action = 'store_true', dest = 'vad', help = 'perform vad')
parser.add_argument('--build_train', action = 'store_true', dest = 'build_train', help = 'perform vad')
args = parser.parse_args()

dev_list = []
with open('./data/cnceleb/dev/dev.lst') as f:
    for line in f:
        dev_list.append(line.strip())

""" convert flac to wav"""
def convert_files(files):
    
    for file in tqdm(files):
        spk = re.findall(r'(id\d+)', file)[0]
        
        if spk not in dev_list: continue
        if 'concat' in file: continue
            
        file_path = PurePath(file)
        flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
        flac_tmp_audio_data.export(file.replace('.flac', '.wav'), format="wav") 
        os.unlink(file)
        

def concat_cn(files, trange, target_path):
    cat = np.array([])
    prev_label = None
    prev_genre = None
    cat_dict = defaultdict(list)
    
    print('concat files...')
    for i, file in enumerate(tqdm(files)):
        
        if 'concat' in file: 
            os.unlink(file)
            continue
        
        curr_label = re.findall(r'(id\d+)', file)[0]
        if curr_label not in dev_list: continue
        
        audio, _ = sf.read(file)
        curr_path = PurePath(file)
        curr_genre = curr_path.name.split('-')[0]            
        
        if trange[0] < len(audio) < trange[1]:
            if curr_label == prev_label and curr_genre == prev_genre:
                cat = np.concatenate([cat, audio])
                if len(cat) >= trange[1]:
                    fname = f'{target_path}/{curr_label}/{curr_genre}-concat-{i}.wav'
                    wavfile.write(fname, rate = 16000, data = cat)
                    cat_dict[(curr_label, curr_genre)].append(fname)
                    cat = np.array([])
            else:
                if len(cat) > 0:
                    prev_cat_name = cat_dict[(prev_label, prev_genre)]
                    if prev_cat_name:
                        prev_cat_name = prev_cat_name[-1]
                        prev_cat, _ = sf.read(prev_cat_name)
                        wavfile.write(prev_cat_name, rate = 16000, data = np.concatenate([prev_cat, cat]))
                    else:
                        print(prev_label, prev_genre)
                    cat = np.array([])
                cat = np.concatenate([cat, audio])
                        
        prev_label = curr_label
        prev_genre = curr_genre
    return cat_dict



def vad_files(files):
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    for file in tqdm(files):
        
        curr_label = re.findall(r'(id\d+)', file)[0]
        if curr_label not in dev_list: continue
        if 'concat' in file: continue

        sample_rate, samples = wavfile.read(file)
                
        try:
            raw_samples = struct.pack("%dh" % len(samples), *samples)
        except:
            samples = np.int16(samples)
            raw_samples = struct.pack("%dh" % len(samples), *samples)

        bytes_per_sample = len(raw_samples) / len(samples)
        samples_per_window = int(30e-3 * bytes_per_sample * sample_rate)

        vaild_points = []
        for start in range(0, len(raw_samples) - samples_per_window, samples_per_window):
            seg = raw_samples[start:start+samples_per_window]
            is_speech = vad.is_speech(seg, sample_rate = sample_rate)
            if is_speech:
                vaild_points.append(int(start // bytes_per_sample))

        f = np.array([])
        for p in vaild_points:
            f = np.concatenate([f, samples[p:p+int(samples_per_window//bytes_per_sample)]])
            
        spk = re.findall(r'(id\d+)', file)[0]
        if not os.path.exists(f'./data/cnceleb_vad/{spk}'):
            os.mkdir(f'./data/cnceleb_vad/{spk}')
            
        fname = os.path.split(file)[1]
        wavfile.write(data = f, filename =  f'./data/cnceleb_vad/{spk}/{fname}', rate = 16000)
        
        
def build_train_list(files, target, tlen):
    lines = []
    
    for file in tqdm(files):
        
        curr_label = re.findall(r'(id\d+)', file)[0]
        if curr_label not in dev_list: continue
        
        audio, _ = sf.read(file)
        if len(audio) >= tlen:
            sufix = os.path.split(file)[1]
            lines.append(f'{curr_label} {curr_label}/{sufix}\n')
    
    with open(target, 'w') as f:
        f.writelines(lines)
    print(f'writing complete. total files: {len(lines)}')    

        
def start_multiprocesssing(files, target, args, segs = 8):
    items_per_segs = len(files) // segs + 1
    ps = []
    for s in range(segs):
        p = multiprocessing.Process(target = target, args = args)
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
        
if __name__ == '__main__':
    
    
    if args.convert:
        files = glob.glob('./data/cnceleb/data/*/*.flac')
        segs = 8
        items_per_segs = len(files) // segs + 1
        ps = []
        for s in range(segs):
            p = multiprocessing.Process(target = convert_files, args = (files[s*items_per_segs:(s+1)*items_per_segs], ))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
        
    if args.vad:
        shutil.rmtree('./data/cnceleb_vad',ignore_errors=True)
        os.mkdir('./data/cnceleb_vad')
        files = glob.glob('./data/cnceleb/data/*/*.wav')
        # vad_files(files)
        segs = 8
        items_per_segs = len(files) // segs + 1
        ps = []
        for s in range(segs):
            p = multiprocessing.Process(target = vad_files, args = (files[s*items_per_segs:(s+1)*items_per_segs], ))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    
    # >= tlen
    if args.build_train:
        # files = glob.glob('./data/cnceleb_vad/*/*.wav') # build train for vad
        # target = './data/cnceleb_train_vad.txt'
        
        files = glob.glob('./data/cnceleb_wav/data/*/*.wav') # build train for wav
        target = './data/cnceleb_train_wav.txt'
        
        tlen = 500 * 160 + 240 
        build_train_list(files, target, tlen = tlen)
        
        
    # save to the same root path; concat < tlen
    if args.concat:
        files = glob.glob('./data/cnceleb_wav/data/*/*.wav')
        # files = glob.glob('./data/cnceleb_vad/*/*.wav')
        tlen = 500 * 160 + 240
        concat_cn(files, tlen)
    
        
        

    