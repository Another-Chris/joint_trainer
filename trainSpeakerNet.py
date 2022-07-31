import os.path
import sys
import zipfile

import torch.cuda
import torch
import datetime
import time
import shutil
import os

from SpeakerNet import *
from DatasetLoader import *
from tuneThreshold import *
from utils import get_args
from torch.utils.tensorboard import SummaryWriter


args = get_args()
logdir = f'./logs/{args.experiment_name}'
if os.path.exists(logdir):
    shutil.rmtree(logdir)

writer = SummaryWriter(logdir)


def evaluate(trainer):
    sc, lab, _ = trainer.evaluateFromList(**vars(args))

    _, eer, _, _ = tuneThresholdfromScore(sc, lab, [1, 0.1])

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss,
                              args.dcf_c_fa)

    return eer, mindcf


def save_scripts():
    # save training code and params
    pyfiles = glob.glob('./*.py')
    strtime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    zipf = zipfile.ZipFile(args.result_save_path + '/run%s.zip' %
                           strtime, 'w', zipfile.ZIP_DEFLATED)
    for file in pyfiles:
        zipf.write(file)
    zipf.close()

    with open(args.result_save_path + '/run%s.cmd' % strtime, 'w') as f:
        f.write('%s' % args)


def main_worker(args):

    # load models
    s = SpeakerNet(**vars(args))
    trainer = ModelTrainer(s, **vars(args))

    it = 1

    # initialize trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))
    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=False
    )

    # load model weights
    modelfiles = glob.glob(f'{args.model_save_path}/model0*.model')
    modelfiles.sort()

    if (args.initial_model != ''):
        trainer.loadParameters(args.initial_model)
        print('model {} loaded!'.format(args.initial_model))

    # restart training
    elif len(modelfiles) > 1:
        trainer.loadParameters(modelfiles[-1])
        print('model {} loaded from previous state!'.format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for _ in range(1, it):
        trainer.__scheduler__.step()

    # evaluation code
    # this is a separate command, not during training.
    if args.eval == True:
        pytorch_total_params = sum(p.numel() for p in s.__S__.parameters())

        print('total params: ', pytorch_total_params)
        print('Test list: ', args.test_list)

        eer, mindcf = evaluate(trainer)

        print(
            f'\n {time.strftime("%Y-%m-%d %H:%M:%S")}, eer: {eer:.4f}, minDCF: {mindcf:4f}')

        return

    # core training script
    for it in range(it, args.max_epoch + 1):
        print(f'epoch {it}')
        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        # train_network: iterate through all the data        
        loss, traineer = trainer.train_network(train_loader)
        writer.add_scalar('Loss/Train', loss, it)
        print('Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}'.format(it,
              traineer, loss, max(clr)))

        if it % args.test_interval == 0:
            eer, mindcf = evaluate(trainer)

            print('\n', 'Epoch {:d}, VEER {:2.4f}, MinDCF: {:2.5f}'.format(
                it, eer, mindcf))

            trainer.saveParameters(
                args.model_save_path + "/model%09d.model" % it)

            with open(args.model_save_path + '/model%09d.eer' % it, 'w') as eerfile:
                eerfile.write('{:2.4f}'.format(eer))

            writer.add_scalar('Loss/Train', eer, it)


def main():
    args.model_save_path = args.save_path + "/model"
    args.result_save_path = args.save_path + "/result"
    args.feat_save_path = ""

    # exps/modelname/model
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    print(f"Python version: {sys.version}")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Save path: {args.save_path}")

    main_worker(args)


if __name__ == "__main__":
    main()
