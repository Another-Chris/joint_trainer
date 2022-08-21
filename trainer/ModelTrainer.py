from tqdm import tqdm
import numpy as np
from loader import test_dataset_loader
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed

import torch.nn.functional as F
import random
import torch
import itertools
import sys
import time
sys.path.append("..")

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
    

class ModelTrainer(object):
    def __init__(self, model):

        self.writer = SummaryWriter(log_dir=f"./logs/{model}/{time.time()}")

    def evaluateFromList(self, test_list, test_path, num_eval=10, **kwargs):

        self.encoder.eval()
        self.encoder.to(torch.device('cuda'))

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
                    eval_frames = 400
                )

        test_loader = torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers = 2, 
                    drop_last=False, 
                    sampler=None
                    )

        ########## extract features ##########
        print('--- extract features ---')
        pbar = tqdm(test_loader, total=len(test_loader))

        for data in pbar:
            inp1 = data[0][0].cuda()

            with torch.no_grad():
                ref_feat = self.encoder(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat

        ########## compute the scores ##########
        all_scores = []
        all_labels = []
        all_trials = []
        
        Parallel(n_jobs = 4, backend="threading")(
            delayed(compute_one_score)
            (all_scores, all_labels, all_trials, line, feats, num_eval)
            for line in tqdm(lines))

        return (all_scores, all_labels, all_trials)

    def saveParameters(self, path):
        torch.save(self.encoder.state_dict(), path)

    def loadParameters(self, path):
        self.encoder.load_state_dict(torch.load(path))
