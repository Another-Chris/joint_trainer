from models.ResNetSE34L import MainModel
from loader import test_dataset_loader
from tqdm import tqdm
from tuneThreshold import ComputeErrorRates, tuneThresholdfromScore, ComputeMinDcf
from utils import get_args
from joblib import Parallel, delayed

import torch
import itertools
import re
import random
import importlib
import json

import numpy as np
import torch.nn.functional as F

torch.cuda.empty_cache()

args = get_args()


def get_features(lines, test_path, num_eval, model, **kwargs):
    # Get a list of unique file names
    files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
    setfiles = list(set(files))
    setfiles.sort()

    # Define test data loader
    test_dataset = test_dataset_loader(
        setfiles, test_path, num_eval=num_eval, eval_frames=300)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        drop_last=False,
        sampler=None
    )

    EncoderFn = importlib.import_module(
        'models.' + model).__getattribute__('MainModel')
    encoder = EncoderFn(**kwargs)
    # try:
    #     encoder_path = f'{args.save_path}/model/model-{args.epoch}.model'
    #     encoder.load_state_dict(torch.load(encoder_path))
    # except:
    #     encoder_path = args.initial_model
    #     encoder.load_state_dict(torch.load(encoder_path))
    # print(f'{encoder_path} loaded!')
    encoder.cuda()

    print('--- extract features ---')
    pbar = tqdm(test_loader, total=len(test_loader))

    features = []
    labels = []
    feats = {}

    for data in pbar:

        inp1 = data[0][0].cuda()

        with torch.no_grad():
            ref_feat = encoder(inp1).detach().cpu()

            # label = re.findall(r'(id\d+)', data[1][0])[0]
            # mean_feat = torch.mean(ref_feat, dim=0)
            # features.append(mean_feat.numpy())
            # labels.append(label)

        feats[data[1][0]] = ref_feat

    return feats

def compute_one_feat(data, encoder):
    inp1 = data[0][0].cuda()
    with torch.no_grad():
        return (data[1][0], encoder(inp1).detach().cpu())


def compute_one_score(all_scores, all_labels, all_trials, line, feats, num_eval):
    
    data = line.split()

    # Append random label if missing
    if len(data) == 2:
        data = [random.randint(0, 1)] + data

    ref_feat = feats[data[1]].cuda()
    com_feat = feats[data[2]].cuda()

    with torch.no_grad():
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

# def compute_scores(all_scores, all_labels, all_trials, lines, feats, num_eval, pid):

#     pbar = tqdm(lines, position=pid,  desc=f"{pid}")

#     for line in pbar:

#         data = line.split()

#         # Append random label if missing
#         if len(data) == 2:
#             data = [random.randint(0, 1)] + data

#         ref_feat = feats[data[1]].cuda()
#         com_feat = feats[data[2]].cuda()

#         # normalize feature
#         ref_feat = F.normalize(ref_feat, p=2, dim=1)
#         com_feat = F.normalize(com_feat, p=2, dim=1)

#         # euclidean dis
#         dist = torch.cdist(ref_feat.reshape(
#             num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()
#         score = -1 * np.mean(dist)

#         all_scores.append(score)
#         all_labels.append(int(data[0]))
#         all_trials.append(data[1] + " " + data[2])


if __name__ == '__main__':

    args.num_eval = int(args.num_eval)

    with open(args.test_list) as f:
        lines = f.readlines()
                
    feats =  get_features(lines = lines, **vars(args))

    all_scores = list()
    all_labels = list()
    all_trials = list()
    njobs = 8

    # then use another pair to access feats
    works = Parallel(n_jobs = njobs, backend="threading")(delayed(compute_one_score)(all_scores, all_labels, all_trials, line, feats, args.num_eval) for line in tqdm(lines))
    print(len(all_scores))

    _, eer, _, _ = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss,
                              args.dcf_c_fa)

    save_path = args.save_path
    with open(f"{save_path}/result/epoch_{args.epoch}.json", 'w') as f:
        eval = {"EER": eer, "minDCF": mindcf}

        f.write(json.dumps(eval))

    print("test finish! ")
    print(f"{eer = }")

    # create the score file
    # df = pd.DataFrame({"label": all_labels, "score": all_scores})
    # split = pd.Series(all_trials).str.split(" ", expand = True)
    # split.columns = ["enroll", "test"]
    # df = pd.concat([df, split], axis = 1)
    # df.to_csv(f"./scores-{eer :.3f}.csv", index = False)
