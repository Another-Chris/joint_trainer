from models.ResNetSE34L import MainModel
from loader import test_dataset_loader
from tqdm import tqdm 
from tuneThreshold import ComputeErrorRates, tuneThresholdfromScore

import torch
import itertools
import re
import random 

import numpy as np 
import torch.nn.functional as F

#####################
##### load model  
#####################
model_args = dict(
    nOut=512,
    encoder_type='SAP',
    n_mels=40,
    log_input=True
)


model = MainModel(**vars(model_args))

loaded_dict = torch.load_dict('./ResNetSE34L.model')
model.load_state_dict(loaded_dict)

model.eval()
model.to(torch.device('cuda'))

#####################
##### start test  
#####################
test_list = "./data/test_list.txt"
test_path = "./data/cnceleb"
nDataLoaderThread = 6
num_eval = 10


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
test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=nDataLoaderThread, 
    drop_last=False, 
    sampler=None
)

########## extract features ##########
print('--- extract features ---')
pbar = tqdm(test_loader, total=len(test_loader))

features = []
labels = []

for data in pbar:
    # data[0]: size(1,10,48240)
    # data[1]: tuple(fdir, )

    inp1 = data[0][0].cuda()

    with torch.no_grad():
        ref_feat = model(inp1).detach().cpu()

        mean_feat = torch.mean(ref_feat, dim=0)
        label = re.findall(r'(id\d+)', data[1][0])[0]

        features.append(mean_feat.numpy())
        labels.append(label)

    feats[data[1][0]] = ref_feat


########## compute the scores ##########
all_scores = []
all_labels = []
all_trials = []

pbar = tqdm(lines)
for line in pbar:

    data = line.split()

    # Append random label if missing
    if len(data) == 2:
        data = [random.randint(0, 1)] + data

    ref_feat = feats[data[1]].cuda()
    com_feat = feats[data[2]].cuda()

    # normalize feature 
    ref_feat = F.normalize(ref_feat, p=2, dim=1)
    com_feat = F.normalize(com_feat, p=2, dim=1)

    dist = torch.cdist(ref_feat.reshape(
        num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

    score = -1 * np.mean(dist)

    all_scores.append(score)
    all_labels.append(int(data[0]))
    all_trials.append(data[1] + " " + data[2])


_, eer, _, _ = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])
fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)

print("test finish! ")
print(f"{eer = }")