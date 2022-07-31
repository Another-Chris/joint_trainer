import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn
import argparse
from models.ResNetSE34V2 import MainModel
import torch
from loss.amsoftmax import LossFunction



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
parser.add_argument('--trainfunc', type=str, default='', help='loss function')

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
                    default='../data/train_list.txt', help='Train list')
parser.add_argument('--test_list', type=str,
                    default='../data/test_list.txt', help='Evaluation list')
parser.add_argument('--train_path', type=str,
                    default='../data/voxceleb2', help='Absolute path to the train set')
parser.add_argument('--test_path', type=str,
                    default='../data/voxceleb1', help='Absolute path to the test set')
parser.add_argument('--musan_path', type=str, default='../data/musan_split',
                    help='Absolute path to the musan data')
parser.add_argument('--rir_path', type=str, default='../data/RIRS_NOISES/simulated_rirs',
                    help='Absolute path to the rir data')

# model definition
parser.add_argument('--n_mels', type=int, default=40,
                    help='Number of mel filterbanks')
parser.add_argument('--log_input', type=bool,
                    default=False, help='log input features')
parser.add_argument('--model', type=str, default='',
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

# distributed and mixed precision training
parser.add_argument('--port', type=str, default='8888',
                    help='port for distributed training, input as text')
parser.add_argument('--distributed', dest='distributed',
                    action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec', dest='mixedprec',
                    action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

train_dataset = train_dataset_loader(**vars(args))

train_sampler = train_dataset_sampler(train_dataset, **vars(args))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=args.nDataLoaderThread,
    sampler=train_sampler,
    pin_memory=False,
    worker_init_fn=worker_init_fn,
    drop_last=False
)


if __name__ == "__main__":
    print("initialize model and loss")
    
    device = torch.device('cuda')
    
    model = MainModel(**vars(args))
    model.to(device)
    model.zero_grad()

    loss_fn = LossFunction(**vars(args))
    loss_fn.to(device)
    
    for data, label in train_loader:

        data = data.transpose(1, 0)
        data = data.reshape(-1, data.size()[-1]).to(device)
        label = torch.LongTensor(label).to(device)
        
        print(f"{data.size() = }")
        print(f'{label.size() = }')
        
        outp = model.forward(data)
        loss = loss_fn(outp, label)

        print(f"{loss = }")

        break
