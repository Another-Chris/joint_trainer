from models.ResNet34_DSBN import ResNet34_DSBN
from models.ECAPA_TDNN_WITH_DSBN import ECAPA_TDNN_WITH_DSBN
from utils import Config
from trainer import Trainer

import torch 
import re


from data_loader import DsLoader
from trainer import Trainer
from pathlib import Path
from utils import Config, inf_train_gen
from eval import evaluate
from tqdm import tqdm

import numpy as np

import torch
import torch.cuda
import multiprocessing
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.cuda.empty_cache()

MODEL_NAME = "ECAPA_TDNN"
# MODEL_NAME = "ResNet34"
EXP_NAME = f"{MODEL_NAME}_DSBN_simCLR"
# EXP_NAME = f'{MODEL_NAME}_test'
MODEL_SAVE_PATH = f"./save/{EXP_NAME}"
SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'
TARGET_PATH = './data/cnceleb/data/'
TARGET_LIST = './data/cnceleb_train_concat.txt'
# PRE_TRAINED = './pre_trained/ECAPA_TDNN_BN.model'
PRE_TRAINED = './save/ECAPA_TDNN_DSBN_simCLR/model-10.model'


# d = torch.normal(mean = 0, std = 1, size = (32, 16000))
# model = ResNet34_DSBN(nOut = 512, encoder_type = 'ASP').to('cpu')
# for name, param in model.named_parameters():
#     print(name, torch.mean(param))
#     break



    

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    ds = DsLoader(
        source_list=SOURCE_LIST,
        source_path=SOURCE_PATH,
        target_list=TARGET_LIST,
        target_path=TARGET_PATH,
        augment=True,
        musan_path=Config.MUSAN_PATH,
        rir_path=Config.RIR_PATH,
        max_frames=Config.MAX_FRAMES
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    ds_gen = inf_train_gen(loader)
    trainer = Trainer(exp_name='test')

    if PRE_TRAINED is not None:
        trainer.model.load_state_dict(torch.load(PRE_TRAINED))
        print('pre-trained weight loaded!')
        
    while True:
        pbar = tqdm(loader)
        for data, label in pbar:
            target_anchor, target_pos = data['target_data']['anchor'],data['target_data']['pos']
            with torch.cuda.amp.autocast():
                f1,f2 = trainer.model(target_anchor.to(Config.DEVICE), 'target'),trainer.model(target_pos.to(Config.DEVICE), 'target')
            
            pbar.set_description(f"{torch.mean(f1):.4f}, {torch.mean(f2):.4f}")
            if torch.isnan(torch.mean(f1)):
                print(torch.mean(f1))
                exit()
        
        
    
# model.load_state_dict(torch.load('./pre_trained/resnet34_DSBN.model'))
# print(model(d, 'source').shape)

# state_dict = torch.load('./pre_trained/resnet34.model', map_location='cpu')
# new_dict = {}
# for key, val in state_dict.items():
    
#     if 'softmax' in key or 'angleproto' in key:
#         continue
    
#     if '__S__' in key:
#         key = key.replace("__S__.", '')
#     if '__L__' in key: 
#         key = key.replace('__L__.', '')
        
#     if 'bn' in key:
#         bn = re.findall(r'(bn\d+)', key)
#         if bn:
#             bn = bn[0]
#             new_dict[key.replace(bn, f'{bn}.bn_source')] = val
#             new_dict[key.replace(bn, f'{bn}.bn_target')] = val
#         else:
#             new_dict[key.replace('bn_last', f'bn_last.bn_source')] = val
#             new_dict[key.replace('bn_last', f'bn_last.bn_target')] = val
            
#     elif 'downsample.0' in key:
#             new_dict[key.replace('downsample.0', f'downsample.conv2d')] = val
#     elif 'downsample.1' in key:
#             new_dict[key.replace('downsample.1', f'downsample.bn.bn_source')] = val
#             new_dict[key.replace('downsample.1', f'downsample.bn.bn_target')] = val
#     elif 'attention.0' in key:
#             new_dict[key.replace('attention.0', f'attention.conv1')] = val
#     elif 'attention.2' in key:
#             new_dict[key.replace('attention.2', f'attention.bn.bn_source')] = val
#             new_dict[key.replace('attention.2', f'attention.bn.bn_target')] = val
#     elif 'attention.3' in key:
#             new_dict[key.replace('attention.3', f'attention.conv2')] = val
#     else:
#         new_dict[key] = val
    
        
# torch.save(new_dict, './pre_trained/resnet34_DSBN.model')
# model.load_state_dict(new_dict)
 