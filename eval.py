from glob import glob
from tqdm import tqdm
from data_loader import test_dataset_loader
from utils import Config
from trainer import Trainer
from sklearn import metrics
from operator import itemgetter
from data_loader import load_wav
from argparse import ArgumentParser

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
import re

sys.path.append("..")

TEST_LIST = Config.TEST_LIST
TEST_PATH = Config.TEST_PATH
MODEL_NAME = 'ECAPA_TDNN'
PRE_TRAINED = './save/ECAPA_TDNN_test/2022-11-07 04.24.51/model-10.model'
# PRE_TRAINED = './pre_trained/ECAPA_TDNN.model'
NUM_WORKERS = 1

parser = ArgumentParser()
parser.add_argument('--cohorts', action = 'store_true', dest = 'cohorts')
parser.add_argument('--eval', action = 'store_true', dest = 'eval')
args = parser.parse_args()

def produce_cohorts(model):
    used_speakers = []
    files = []
    feats = []
    
    with open(TEST_LIST) as f:
        for line in tqdm(f):
            if (not line):
                break
            data = line.split()

            data_1_class = re.findall(r'(id\d+)', data[1])[0]
            data_2_class = re.findall(r'(id\d+)', data[2])[0]

            if data_1_class not in used_speakers:
                used_speakers.append(data_1_class)
                files.append(data[1])
            if data_2_class not in used_speakers:
                used_speakers.append(data_2_class)
                files.append(data[2])
                
    setfiles = list(set(files))
    setfiles.sort()
    
    model.eval()
    for _, f in enumerate(tqdm(setfiles)):
        inp1 = torch.FloatTensor(
            load_wav('./data/cnceleb/eval/' + f, Config.EVAL_FRAMES, evalmode=True,
                    num_eval=Config.NUM_EVAL)).to(Config.DEVICE)
        
        feat = model(inp1, domain = 'target')
        feat = F.normalize(feat, p=2, dim=1).detach().cpu().numpy().squeeze()
        feats.append(feat)

        np.save('./cohorts.npy', np.array(feats))

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


def score_normalization(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """
    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)


def evaluateFromList(encoder, test_list, test_path, num_eval=10, cohorts_path = None):

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
            if torch.isnan(torch.mean(ref_feat)):
                raise ValueError('feat NaN')

        feats[data[1][0]] = torch.mean(ref_feat, dim = 0, keepdim = True)
        

    ########## compute the scores ##########
    all_scores = []
    all_labels = []
    all_trails = []

    for line in tqdm(lines):
        data = line.split()

        # Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = feats[data[1]]
        com_feat = feats[data[2]]

        # normalize feature
        # ref_feat = F.normalize(ref_feat, p=2, dim=1)
        # com_feat = F.normalize(com_feat, p=2, dim=1)

        # calculate scores
        if cohorts_path is None:
            dist = F.cosine_similarity(ref_feat, com_feat).cpu().numpy()[0]
            score = 1 * dist
        else:
            cohorts = np.load(cohorts_path)
            score = score_normalization(ref_feat, com_feat,cohorts,top = 200)

        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trails.append(data[1] + " " + data[2])
        
    # df = pd.DataFrame({'all_scores': all_scores, 'all_labels': all_labels, 'all_trails': all_trails})
    # df.to_csv(f'./scores.csv', index = False)
    
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
        
    if args.cohorts:
        produce_cohorts(trainer.model)
    
    if args.eval:
        eer, mindcf = evaluate(trainer.model)
        print(f'eer = {eer:.4f}, mindcf = {mindcf:.4f}')
    
