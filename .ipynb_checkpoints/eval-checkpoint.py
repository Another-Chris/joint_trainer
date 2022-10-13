from glob import glob
from tqdm import tqdm
from data_loader import test_dataset_loader
from utils import Config
from trainer import Trainer
from sklearn import metrics
from operator import itemgetter

import torch.nn.functional as F
import numpy as np
import pandas as pd

import random
import torch
import itertools
import sys
import multiprocessing
import os
import glob

sys.path.append("..")

TEST_LIST = Config.TEST_LIST
TEST_PATH = Config.TEST_PATH
MODEL_NAME = 'ECAPA_TDNN'
PRE_TRAINED = './save/ECAPA_TDNN_DSBN_variable_length/model-10.model' # ~80 epochs
# PRE_TRAINED = './pre_trained/ECAPA_TDNN.model'
NUM_WORKERS = 1
N_PROCESS = 2


def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label = 1)
    fnr = 1 - tpr

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute(tfr - fnr))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute(tfa - fpr))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute(fnr - fpr))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return (tunedThreshold, eer, fpr, fnr)


def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(*sorted([(index, threshold) for index, threshold in enumerate(scores)], key = itemgetter(1)))

    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1-labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])

    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds

def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float('inf')
    min_c_det_threshold = thresholds[0]

    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold




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
        num_eval=Config.NUM_EVAL,
        eval_frames=Config.EVAL_FRAMES
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
            ref_feat = encoder(inp1.unsqueeze(1), 'target')
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
    trainer = Trainer(exp_name='eval')
    
    if PRE_TRAINED is not None:
        trainer.model.load_state_dict(torch.load(PRE_TRAINED))
        print('pre-trained weight loaded!')
    
    eer, mindcf = evaluate(trainer.model)
    print(f'eer = {eer:.4f}, mindcf = {mindcf:.4f}')
    
