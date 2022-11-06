from handle_cn import concat_cn
from data_loader import DsLoader

import glob 
from data_loader import DsLoader
from trainer import Trainer
from pathlib import Path
from utils import Config, inf_train_gen
from eval import evaluate

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

EXP_NAME = f"{MODEL_NAME}_DSBN_newConcat"
# EXP_NAME = f'{MODEL_NAME}_test'
MODEL_SAVE_PATH = f"./save/{EXP_NAME}"

SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'
TARGET_PATH = './data/cnceleb/data/'
TARGET_LIST = './data/cnceleb_train.txt'

# PRE_TRAINED = './save/ECAPA_TDNN_fineTune_aamsoftmax/model-10.model'
PRE_TRAINED = './pre_trained/ECAPA_TDNN_BN.model'
# PRE_TRAINED = './pre_trained/resnet34_DSBN.model'
# PRE_TRAINED = None 

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
    num_workers=1,
    drop_last=True,
)

if __name__ == '__main__':
    for data, label in loader:
        target_anchor, target_pos = data['target_data']['anchor'],data['target_data']['pos']
        print(target_anchor.shape)
        break