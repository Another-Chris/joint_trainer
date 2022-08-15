from tqdm import tqdm
import numpy as np
from loader import test_dataset_loader
from .utils import save_features
from .ModelWithHead import ModelWithHead
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import importlib
import random
import torch
import re
import itertools
import sys
import time
sys.path.append("..")


class ModelTrainer(object):
    def __init__(self, nPerSpeaker, model, scheduler, optimizer,  **kwargs):

        # model_fn = importlib.import_module('models.' + model).__getattribute__('MainModel')

        # self.encoder = model_fn(**kwargs) # embeddings
        # self.encoder_with_head = ModelWithHead(self.encoder, dim_in=kwargs['nOut'], head='mlp') # projections with head

        # Optimizer = importlib.import_module('optimizer.' + optimizer).__getattribute__('Optimizer')
        # self.__optimizer__ = Optimizer(self.encoder_with_head.parameters(), **kwargs)

        self.nPerSpeaker = nPerSpeaker
        self.model = model 
        self.scheduler = scheduler
        self.optimizer = optimizer

        self.writer = SummaryWriter(log_dir=f"./logs/{self.model}/{time.time()}")


    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, num_eval=10, feat_save_path='.', **kwargs):

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
            setfiles, test_path, num_eval=num_eval, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=None)

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
                ref_feat = self.encoder(inp1).detach().cpu()

                mean_feat = torch.mean(ref_feat, dim=0)
                label = re.findall(r'(id\d+)', data[1][0])[0]

                features.append(mean_feat.numpy())
                labels.append(label)

            feats[data[1][0]] = ref_feat

        save_features(feat_save_path, features, labels)

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

            dist = torch.cdist(ref_feat.reshape(
                num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

            score = -1 * np.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

        return (all_scores, all_labels, all_trials)

    def saveParameters(self, path):
        torch.save(self.encoder.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.encoder.state_dict()

        loaded_state = torch.load(path, map_location="cuda:0")

        if 'model' in loaded_state:
            loaded_state = loaded_state['model']

        if '__S__' in list(loaded_state.keys())[0]:
            newdict = {}
            delete_list = []

            for name, param in loaded_state.items():
                new_name = name.replace('__S__.', '')
                newdict[new_name] = param
                delete_list.append(name)

            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            # this is how you load the params
            self_state[name].copy_(param)
