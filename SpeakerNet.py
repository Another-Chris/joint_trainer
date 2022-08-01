import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import importlib
import torch
import itertools
import random
import re
import pickle
import os 

from DatasetLoader import test_dataset_loader
from tqdm import tqdm


def save_features(feat_save_path, features, labels):
    
    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)
    
    with open(f'{feat_save_path}/features.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    with open(f'{feat_save_path}/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)


class ModelWithHead(nn.Module):
    """backbone + projection head"""

    def __init__(self, encoder, dim_in, head='mlp', feat_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
            
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1)
        return feat


# driver class, define the optimizer and scheduler
# this states how the model should be trained
class ModelTrainer(object):
    def __init__(self, model, optimizer, scheduler, trainfunc, nPerSpeaker, **kwargs):
        
        
        model_fn = importlib.import_module(
            'models.' + model).__getattribute__('MainModel')
        
        encoder = model_fn(**kwargs)                 # embeddings
        self.__model__ = ModelWithHead(encoder, 512) # training
        

        Optimizer = importlib.import_module(
            'optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        

        Scheduler = importlib.import_module(
            'scheduler.' + scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **kwargs)


        LossFunction = importlib.import_module(
            'loss.' + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)
        
        self.nPerSpeaker = nPerSpeaker

        assert self.lr_step in ['epoch', 'iteration']

    def train_network(self, loader):
        self.__model__.train()
        self.__model__.to(torch.device('cuda'))

        counter = 0
        loss = 0

        pbar = tqdm(loader)
        for data in pbar:
            self.__model__.zero_grad()

            # forward pass
            data = torch.cat([data[0], data[1]], dim=0)
            
            # batch, 1, len
            # 1, batch len
            # batch * 1, len
            data = data.transpose(1, 0)
            data = data.reshape(-1, data.size()[-1]).cuda()
            outp = self.__model__.forward(data)

            bsz = outp.size()[0] // 2
            f1, f2 = torch.split(outp, [bsz, bsz], dim=0)
            outp = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            nloss = self.__L__.forward(outp)

            # backward pass
            nloss.backward()
            self.__optimizer__.step()

            # record
            loss += nloss.detach().cpu().item()
            counter += 1

            pbar.set_description(f'loss: {loss / counter :.3f}')
            pbar.total = len(loader)

            if self.lr_step == 'iteration':
                self.__scheduler__.step()
                
        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return loss / (counter + 1e-6)

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, num_eval=10, feat_save_path = '.', **kwargs):

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
        
        features = []
        labels = []
        
        for data in pbar:
            # data[0]: size(1,10,48240)
            # data[1]: tuple(fdir, )

            inp1 = data[0][0].cuda()

            with torch.no_grad():
                ref_feat = self.__model__.encoder(inp1).detach().cpu()
                
                mean_feat = torch.mean(ref_feat, dim = 0)
                label = re.findall(r'(id\d+)',data[1][0])[0]
                
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

            if self.__L__.test_normalize:
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

        self_state = self.__model__.encoder.state_dict()
        
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
