#! /usr/bin/python
# -*- encoding: utf-8 -*-

import yaml
import sys
import argparse
import glob 
import datetime
import zipfile 
import json
import torch

import matplotlib.pyplot as plt
import librosa.display


class Config():
    GIM_SEGS = 3
    MAX_EPOCH = 300
    TEST_INTERVAL = 10
    NUM_WORKERS = 4
    BATCH_SIZE = 64
    NUM_CLASSES = 5994
    MAX_FRAMES = 200
    LEARNING_RATE = 1e-4
    EMBED_SIZE = 192
    C = 1024
    MUSAN_PATH = "./data/musan_split"
    RIR_PATH = "./data/RIRS_NOISES/simulated_rirs"    
    # TEST_PATH = "./data/cnceleb/eval/"
    # TEST_LIST = "./data/cnceleb_test.txt"
    TEST_PATH = "./data/voxceleb1_test/"
    TEST_LIST = "./data/voxceleb_test.txt"
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def plot_batch(batch):
    batch = batch.detach().cpu().numpy()
    librosa.display.specshow(batch[0])
    plt.show()
    

def inf_train_gen(loader):
    while True:
        for data, label in loader:
            yield data, label


def get_grad(params):
    grads = [0]
    for p in params:
        if p.grad is None: continue
        grads.append(torch.mean(p.grad).item())
    return max(grads)


def save_scripts(result_save_path, configs):
    # save training code and params
    pyfiles = glob.glob('./*.py') + glob.glob('./*/*.py')
    strtime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    zipf = zipfile.ZipFile(
        f'{result_save_path}/run{strtime}.zip',
        'w',
        zipfile.ZIP_DEFLATED)

    for file in pyfiles:
        zipf.write(file)
    zipf.close()

    with open(result_save_path + f'/run_{strtime}.cmd', 'w') as f:
        js = json.dumps(configs)
        f.write(js)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def get_args():

    parser = argparse.ArgumentParser(description='SpeakerNet')

    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML file')

    # DataLoader
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Input length to the network for training')
    parser.add_argument('--eval_frames', type=int, default=300,
                        help='Input length to the network for testing 0 uses the whole files')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk', type=int, default=500,
                        help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread', type=int,
                        default=5, help='Number of loader threads')
    parser.add_argument('--augment', type=bool, default=10,
                        help='Seed for the random number generator')
    parser.add_argument('--seed', type=int, default=10,
                        help='Seed for the random number generator')

    # Training details
    parser.add_argument('--test_interval', type=int, default=10,
                        help='Test and save every [test_interval] epochs')
    parser.add_argument('--max-epoch', type=int, default=500,
                        help='Maximum number of epochs')
    parser.add_argument('--epoch', type=int, default=10,
                        help='eval only, num of the current epoch')
    # parser.add_argument('--trainfunc', type=str,
    #                     default='', help='loss function')

    parser.add_argument('--sup_loss', type=str, default='',
                        help='supervised loss, used only for joint training')
    parser.add_argument('--ssl_loss', type=str, default='',
                        help='ssl loss, used only for joint training')
    
    parser.add_argument('--ssl_path', default='./data/cnceleb/data', type=str,
                        help='only for joint training. data for ssl')
    parser.add_argument('--sup_path', default='./data/voxceleb2', type=str,
                        help='only for joint training. data for sup')
    parser.add_argument('--ssl_list', default='./data/train_list_cnceleb.txt', type=str,
                        help='only for joint training. train list for ssl')
    parser.add_argument('--sup_list', default='./data/train_list.txt', type=str,
                        help='only for joint training. train list for sup')
    

    # optimizer
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='sgd or adam')
    parser.add_argument('--scheduler', type=str, default='steplr',
                        help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='learning rate decay every [test_interval] epochs')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay in the optimizer')

    # loss function
    parser.add_argument('--hard_prob', type=float, default=0.5,
                        help='Hard negative mining prob, otherwise random, only for some loss functions')
    parser.add_argument('--hard_rank', type=int, default=10,
                        help='Hard negative mining rank in the batch, only for some loss functions')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Loss margin, only for some loss functions')
    parser.add_argument('--scale', type=float, default=30,
                        help='loss scale, only for some loss functions')
    parser.add_argument('--nPerSpeaker', type=int, default=1,
                        help='Number of utterances per speaker per batch, only for metric learning based losses')
    parser.add_argument('--nClasses', type=int, default=5994,
                        help='Number of speakers in the softmax layer, only for softmax-based losses')

    # evaluation parameters
    parser.add_argument('--dcf_p_target', type=float, default=0.05,
                        help='A priori prob of the specified target speaker')
    parser.add_argument('--dcf_c_miss', type=float, default=1,
                        help='Cost of a missed detection')
    parser.add_argument('--dcf_c_fa', type=float, default=1,
                        help='Cost of a spurious detection')

    # load and save
    parser.add_argument('--initial_model', type=str,
                        default='', help='initial model weights')
    parser.add_argument('--save_path', type=str,
                        default='exps/exp1', help='Path for model and logs')

    # training and test data
    parser.add_argument('--train_list', type=str,
                        default='data/train_list.txt', help='Train list')
    parser.add_argument('--test_list', type=str,
                        default='data/test_list.txt', help='Evaluation list')
    parser.add_argument('--train_path', type=str,
                        default='data/voxceleb2', help='Absolute path to the train set')
    parser.add_argument('--test_path', type=str,
                        default='data/voxceleb1', help='Absolute path to the test set')
    parser.add_argument('--musan_path', type=str, default='data/musan_split',
                        help='Absolute path to the musan data')
    parser.add_argument('--rir_path', type=str, default='data/RIRS_NOISES/simulated_rirs',
                        help='Absolute path to the rir data')

    # model definition
    parser.add_argument('--n_mels', type=int, default=40,
                        help='Number of mel filterbanks')
    parser.add_argument('--log_input', type=bool,
                        default=False, help='log input features')
    parser.add_argument('--model', type=str, default='ResNetSE34L',
                        help='Name of model definition')
    parser.add_argument('--encoder_type', type=str,
                        default='SAP', help='Type of encoder')
    parser.add_argument('--nOut', type=int, default=512,
                        help='Embedding size in the last FC layer')
    parser.add_argument('--sinc_stride', type=int, default=10,
                        help='Stride size of the first analytic filterbank layer of RawNet3')

    # for test only
    parser.add_argument('--eval', dest='eval',
                        action='store_true', help='Eval only')
    
    parser.add_argument('--num_eval', dest='num_eval',
                        default = 10, help='number of eval segments')

    parser.add_argument('--experiment_name', type=str,
                        help='experiment name', dest='experiment_name')

    args, _ = parser.parse_known_args()

    # parse YAML
    def find_option_type(key, parser):
        for opt in parser._get_optional_actions():
            if ('--' + key) in opt.option_strings:
                return opt.type
        raise ValueError

    if args.config is not None:
        with open(args.config, 'r') as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write(
                    "Ignored unknown parameter {} in yaml. \n".format(k))
    return args
