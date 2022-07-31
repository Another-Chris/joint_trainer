import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import importlib
import torch
import itertools
import random

from torch.cuda.amp import autocast, GradScaler
from DatasetLoader import test_dataset_loader
from tqdm import tqdm

# model: string
# this is the core class
class SpeakerNet(nn.Module):
    def __init__(
        self,
        model: str,
        trainfunc: str,
        nPerSpeaker,
        **kwargs
    ):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module(
            'models.' + model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module(
            'loss.' + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):
        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp
        else:

            outp = outp.reshape(self.nPerSpeaker, -1,
                                outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)
            return nloss, prec1



# driver class, define the optimizer and scheduler
# this states how the model should be trained
class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, mixedprec, **kwargs):
        self.__model__ = speaker_model

        Optimizer = importlib.import_module(
            'optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module(
            'scheduler.' + scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.mixedprec = mixedprec
                
        assert self.lr_step in ['epoch', 'iteration']

    def train_network(self, loader):
        self.__model__.train()
        self.__model__.to(torch.device('cuda'))

        counter = 0
        loss = 0
        top1 = 0

        pbar = tqdm(enumerate(loader))
        for i, (data, data_label) in pbar:
            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            
            pbar.set_description(f'loss: {loss / counter :.3f}, TEER/TAcc: {top1 / counter :.3f}')
            pbar.total = len(loader)
            
            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return (loss / (counter + 1e-6), top1 / (counter + 1e-6))
    

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, print_interval=100, num_eval=10, **kwargs):

        self.__model__.eval()
        self.__model__.to(torch.device('cuda'))

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
        for data in pbar:
            # data[0][0]: signal; data[1][0]: fdir of signal

            inp1 = data[0][0].cuda()

            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()

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

            if self.__model__.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = torch.cdist(ref_feat.reshape(
                num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

            score = -1 * np.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

        return (all_scores, all_labels, all_trials)
    

    def saveParameters(self, path):
        torch.save(self.__model__.state_dict(), path)


    def loadParameters(self, path):

        # device = torch.device('cuda')
        # loaded_state = torch.load(path, map_location=device)
        # self.__model__.load_state_dict(loaded_state)
        # loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        
        self_state = self.__model__.state_dict()
        loaded_state = torch.load(path, map_location="cuda:0")
        
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
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
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)