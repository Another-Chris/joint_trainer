from glob import glob
from tqdm import tqdm
from data_loader import test_dataset_loader
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from utils import Config
from trainer import JointTrainer

import torch.nn.functional as F
import random
import torch
import itertools
import sys
import multiprocessing
import os
import glob

import numpy as np
import pandas as pd

sys.path.append("..")

TEST_LIST = Config.TEST_LIST
TEST_PATH = Config.TEST_PATH
MODEL_NAME = 'ECAPA_TDNN'
PRE_TRAINED = './save/ECAPA_TDNN_DAT/model-22.model' # ~80 epochs
# PRE_TRAINED = './pre_trained/ECAPA_TDNN.model'
NUM_WORKERS = 1
N_PROCESS = 3


def compute_one_score(lines, feats, num_eval, position):
    
    all_scores = []
    all_labels = []
    all_trails = []

    for line in tqdm(lines, position = position):
        data = line.split()

        # Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = feats[data[1]].cuda()
        com_feat = feats[data[2]].cuda()

        # normalize feature
        ref_feat = F.normalize(ref_feat, p=2, dim=1)
        com_feat = F.normalize(com_feat, p=2, dim=1)

        # euclidean dis
        dist = torch.cdist(ref_feat.reshape(
            num_eval, -1), com_feat.reshape(num_eval, -1)).cpu().numpy()

        score = -1 * np.mean(dist)

        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trails.append(data[1] + " " + data[2])
        
    df = pd.DataFrame({'all_scores': all_scores, 'all_labels': all_labels, 'all_trails': all_trails})
    df.to_csv(f'./{position}.csv', index = False)


def evaluateFromList(encoder, test_list, test_path, num_eval=10):

    encoder.eval()
    encoder.to(Config.DEVICE)

    lines = []
    files = []
    feats = {}

    # Read all lines
    with open(test_list) as f:
        lines = f.readlines()
        
    # Get a list of unique file names
    files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
    setfiles = list(set(files))
    setfiles.sort()

    # Define test data loader
    test_dataset = test_dataset_loader(
        setfiles,
        test_path,
        num_eval=num_eval,
        eval_frames=400
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        sampler=None
    )

    ########## extract features ##########
    print('--- extract features ---')
    pbar = tqdm(test_loader, total=len(test_loader))

    for data in pbar:
        inp1 = data[0][0].cuda()

        with torch.no_grad():
            ref_feat = encoder(inp1.unsqueeze(1))
            if type(ref_feat) == tuple:
                ref_feat = ref_feat[1]
                
        feats[data[1][0]] = ref_feat.detach().cpu()

    ########## compute the scores ##########
    items_per_p = len(lines) // N_PROCESS + 1
    ps = []
    for n in range(N_PROCESS):
        line_seg = lines[n*items_per_p : (n+1) * items_per_p]
        p = multiprocessing.Process(target = compute_one_score, args = (line_seg, feats, num_eval, n))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
            

    score_files = glob.glob("*.csv")
    df = pd.DataFrame({'all_scores': [], 'all_labels': [], 'all_trails': []})
    for file in score_files:
        df = pd.concat([df, pd.read_csv(file)], axis = 0)
        os.unlink(file)
    
    all_scores = df['all_scores'].values
    all_labels = df['all_labels'].values
    all_trails = df['all_trails'].values
    
    return (all_scores, all_labels, all_trails)


def evaluate(encoder):
    sc, lab, _ = evaluateFromList(
        encoder = encoder, 
        test_list=TEST_LIST, 
        test_path=TEST_PATH, 
        num_eval=10
    )
    _, eer, _, _ = tuneThresholdfromScore(sc, lab, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    return eer, mindcf


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    trainer = JointTrainer(exp_name='eval')
    
    if PRE_TRAINED is not None:
        trainer.model.load_state_dict(torch.load(PRE_TRAINED))
        print('pre-trained weight loaded!')
    
    eer, mindcf = evaluate(trainer.speaker_predictor)
    print(f'eer = {eer:.4f}, mindcf = {mindcf:.4f}')
    
