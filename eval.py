from tqdm import tqdm
from loader import test_dataset_loader
from joblib import Parallel, delayed
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from utils import Config
from models import ECAPA_TDNN_WITH_FBANK

import torch.nn.functional as F
import random
import torch
import itertools
import sys

import numpy as np

sys.path.append("..")

TEST_LIST = Config.TEST_LIST
TEST_PATH = Config.TEST_PATH
MODEL_NAME = 'ECAPA_TDNN'
# PRE_TRAINED = './save/ECAPA_TDNN_simCLR_Voxceleb/encoder-60.model'
PRE_TRAINED = './pre_trained/ECAPA_TDNN.model'
NUM_WORKERS = 1


def compute_one_score(all_scores, all_labels, all_trials, line, feats, num_eval):

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
    all_trials.append(data[1] + " " + data[2])



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
            ref_feat = encoder(inp1.unsqueeze(1)).detach().cpu()
        feats[data[1][0]] = ref_feat

    ########## compute the scores ##########
    all_scores = []
    all_labels = []
    all_trials = []

    Parallel(n_jobs=4, backend="threading")(
        delayed(compute_one_score)
        (all_scores, all_labels, all_trials, line, feats, num_eval)
        for line in tqdm(lines))

    return (all_scores, all_labels, all_trials)


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
    encoder = ECAPA_TDNN_WITH_FBANK(C = Config.C, embed_size=Config.EMBED_SIZE)
    
    if PRE_TRAINED is not None:
        encoder.load_state_dict(torch.load(PRE_TRAINED))
        print('pre-trained weight loaded!')
    
    eer, mindcf = evaluate(encoder)
    print(f'eer = {eer:.4f}, mindcf = {mindcf:.4f}')
    
