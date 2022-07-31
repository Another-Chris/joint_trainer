import glob
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='train')
args = parser.parse_args()

fpath = "../data/voxceleb2/*/*/*.flac" if args.type == 'train' else "../data/voxceleb2_test/*/*/*.flac"
lpath = '../data/train_list.txt' if args.type == 'train' else '../data/test_list.txt'
r = '../data/voxceleb2\\' if args.type == 'train' else '../data/voxceleb2_test\\'

files = glob.glob(fpath)

with open (lpath, 'w') as f:
    for file in files:
        label = re.findall(r'(id\d+)', file)[0]
        file = file.replace(r, "")
        line = f'{label} {file}\n'
        f.write(line)
