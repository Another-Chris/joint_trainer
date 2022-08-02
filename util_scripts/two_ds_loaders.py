import torch
import sys 
sys.path.append('..')

from loader.train import ssl_dataset_loader, train_dataset_loader
from loader.sampler import train_dataset_sampler
from loader.utils import worker_init_fn
from utils import get_args

args = get_args()
kargs = vars(args)
del kargs['train_list']
del kargs['train_path']
del kargs['musan_path']
del kargs['rir_path']


train_dataset = train_dataset_loader(
    train_list = '../data/train_list.txt',
    train_path = '../data/voxceleb2',
    musan_path = '../data/musan_split',
    rir_path = '../data/RIRS_NOISES/simulated_rirs',
    **kargs)
train_sampler = train_dataset_sampler(train_dataset, **kargs)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread // 2,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

def inf_train_gen():
    while True:
        for data, label in train_loader:
            yield data, label

train_gen = inf_train_gen()

ssl_dataset = ssl_dataset_loader(    
    train_list = '../data/train_list_cnceleb.txt',
    train_path = '../data/cnceleb/data',
    musan_path = '../data/musan_split',
    rir_path = '../data/RIRS_NOISES/simulated_rirs',
    **kargs)

ssl_loader = torch.utils.data.DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread // 2,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        shuffle=True
    )

if __name__ == '__main__':
    for data in ssl_loader:
        print('ssl data shape:', data[0].shape)
        
        vox, label = next(train_gen)

        
        print('vox shape:', vox.shape)
        
        break

