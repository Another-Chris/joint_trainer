import argparse
import subprocess
import tarfile
import os
import hashlib
import glob

from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
from pydub import AudioSegment

parser = argparse.ArgumentParser(description = "VoxCeleb downloader")

parser.add_argument('--save_path', type = str, default = "data", help = "target directory")
parser.add_argument('--user', type = str, default = 'voxceleb1912', help = 'username')
parser.add_argument('--password', type = str, default = '0s42xuw6', help = 'password')

parser.add_argument('--download', dest = 'download', action = 'store_true', help = 'enable download')
parser.add_argument('--extract', dest = 'extract', action = 'store_true', help = 'enable extract')
parser.add_argument('--convert', dest = 'convert', action = 'store_true', help = 'enalbe convert')
parser.add_argument('--augment', dest = 'augment', action = 'store_true', help = 'download and extract augmentation files')
parser.add_argument('--build_train_list', dest = 'build_train_list', action = 'store_true', help = 'build training list')

args = parser.parse_args()


## ========== ==========
## MD5SUM
## ========== ==========
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

## ========== ==========
## Download with wget
## ========== ==========
def download(args, lines):

    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split('/')[-1]

        ## Download files
        if os.path.exists(
            os.path.join(args.save_path, outfile)
        ):
            print('file already exists')
            continue
        else:
            out = subprocess.call('wget %s --user %s --password %s -O %s/%s'%(url, args.user, args.password, args.save_path, outfile), shell = True)

        ## Check MD5
        md5ck = md5('%s/%s'%(args.save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s'%outfile)
        else:
            raise Warning('Checksum failed %s'%outfile)


## ========== ==========
## Concatenate file parts
## ========== ==========
def concatenate(args, lines):
    for line in lines:
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        ## concatenate files
        out = subprocess.call('cat %s/%s > %s/%s' %(args.save_path,infile,args.save_path,outfile), shell = True)

        ## check md5
        md5ck = md5('%s/%s'%(args.save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        else:
            raise Warning('Checksum failed %s.'%outfile)

        out = subprocess.call('rm %s/%s'%(args.save_path, infile), shell = True)

## ========== ==========
## Extract zip files
## ========== ==========
def full_extract(args, fname):
    print('Extracting %s'%fname)
    if fname.endswith('.tar.gz'):
        with tarfile.open(fname, 'r:gz') as tar:
            tar.extractall(args.save_path)
    elif fname.endswith('.zip'):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(args.save_path)

## ========== ==========
## Partially extract zip files
## ========== ==========
def part_extract(args, fname, target):
    print('Extracting %s'%fname)
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            # extract these:
            # ['RIRS_NOISES/simulated_rirs/mediumroom', ...]
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)

## ========== ==========
## convert
## ========== ==========
def convert(args):
    files = glob.glob('%s/voxceleb2/*/*/*.m4a'%args.save_path)
    files = [f.replace('\\', '/') for f in files]
    files.sort()

    print('Converting files from AAC to WAV')
    for fname in tqdm(files):
        outfile = fname.replace('.m4a', '.wav')
        AudioSegment.from_file(fname).export(outfile, format = "wav")
        os.unlink(fname)



## ========== ==========
## split MUSAN for faster random access
## ========== ==========
def split_musan(args):
    files = glob.glob('%s/musan/*/*/*.wav'%args.save_path)
    files = [file.replace("\\", "/") for file in files]

    audlen = 16000 * 5 # segment length: 5s
    audstr = 16000 * 3 # steps: 3s


    # MUSAN provides a wavfile
    # this code split that wav file
    # then store the segments into a folder
    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/', '/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + '/%05d.wav'%(st/fs), fs, aud[st:st+audlen])

        print(idx, file)

## ========== ==========
## Build train list
## ========== ==========
def build_train_list():
    files = glob.glob("./data/voxceleb2/*/*.wav")
    print(files)



## ========== ==========
## main
## ========== ==========
if __name__ == '__main__':
    exists = os.path.exists(args.save_path)


    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    f = open('lists/fileparts.txt', 'r')
    fileparts = f.readlines()
    f.close()

    f = open('lists/files.txt', 'r')
    files = f.readlines()
    f.close()

    f = open('lists/augment.txt', 'r')
    augfiles = f.readlines()
    f.close()

    if args.augment:
        download(args, augfiles)
        part_extract(args, os.path.join(args.save_path, 'rirs_noises.zip'),
                ['RIRS_NOISES/simulated_rirs/mediumroom', 'RIRS_NOISES/simulated_rirs/smallroom'])
        full_extract(args, os.path.join(args.save_path, 'musan.tar.gz'))
        split_musan(args)

    if args.download:
        download(args, fileparts)

    if args.extract:
        concatenate(args, files)
        for file in files:
            full_extract(args, os.path.join(args.save_path, file.split()[1]))
        out = subprocess.call('mv %s/dev/aac/* %s/acc/ && rm -r %s/dev'%(args.save_path, args.save_path, args.save_path), shell = True)
        out = subprocess.call('mv %s/aac %s/voxceleb2'%(args.save_path, args.save_path), shell = True)

    if args.build_train_list:
        build_train_list()

    if args.convert:
        convert(args)















