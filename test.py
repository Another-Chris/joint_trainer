# from utils import get_args
# from DatasetLoader import ssl_dataset_loader
# import torch

# args = get_args()
# ssl_dataset = ssl_dataset_loader(**vars(args))
# train_loader = torch.utils.data.DataLoader(
#         ssl_dataset,
#         batch_size=args.batch_size,
#         num_workers=args.nDataLoaderThread,
#         pin_memory=False,
#         drop_last=False,
#         shuffle = True
#     )


# if __name__ == '__main__':
#     for data in train_loader:
#         print(data.size())
#         break


from torch.utils.tensorboard import SummaryWriter
import random
import shutil
import os

logdir = 'logs/test'
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = SummaryWriter(logdir)

def worker():

    for it in range(10):
        writer.add_scalar('train/loss', random.random(), it)
        
def main():
    worker()        

    
if __name__ == '__main__':
    main()

