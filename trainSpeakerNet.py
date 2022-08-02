import os.path
import sys
import zipfile

import torch.cuda
import torch
import datetime
import time
import shutil
import os
import glob

from trainer import *
from loader import *
from trainer.JointTrainer import JointTrainer
from tuneThreshold import *
from utils import get_args
from torch.utils.tensorboard import SummaryWriter

args = get_args()
logdir = f'./logs/{args.experiment_name}'


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


def get_ssl_loader():
    train_dataset = ssl_dataset_loader(**vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        shuffle=True
    )
    return train_loader


def get_sup_loader():
    sup_dataset = train_dataset_loader(**vars(args))
    sup_sampler = train_dataset_sampler(sup_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        sup_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        sampler=sup_sampler
    )
    return train_loader


def inf_train_gen(loader):
    while True:
        for data, label in loader:
            yield data, label


def main_worker(args):
    writer = SummaryWriter(logdir)

    ############### load models ###############

    if args.training_mode == 'ssl':
        train_loader = get_ssl_loader()
        trainer = SSLTrainer(**vars(args))

    elif args.training_mode == 'joint':
        train_loader = get_ssl_loader()
        sup_loader = get_sup_loader()
        sup_gen = inf_train_gen(sup_loader)
        trainer = JointTrainer(supervised_gen=sup_gen, **vars(args))

    elif args.training_mode == 'supervised':
        train_loader = get_sup_loader()
        trainer = SupervisedTrainer(**vars(args))

    # either load the initial_model or read the previous model files
    it = 1
    if (args.initial_model != ''):
        trainer.loadParameters(args.initial_model)
        print('model {} loaded!'.format(args.initial_model))

    # restart training
    else:
        modelfiles = glob.glob(f'{args.model_save_path}/model0*.model')
        if len(modelfiles) > 1:
            modelfiles.sort()
            trainer.loadParameters(modelfiles[-1])
            print('model {} loaded from previous state!'.format(
                modelfiles[-1]))
            it = int(os.path.splitext(
                os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for _ in range(1, it):
        trainer.__scheduler__.step()

    # evaluation code
    # this is a separate command, not during training.
    if args.eval == True:
        pytorch_total_params = sum(p.numel()
                                   for p in trainer.__model__.parameters())

        print('total params: ', pytorch_total_params)
        print('Test list: ', args.test_list)

        eer, mindcf = evaluate(trainer)

        print(
            f'\n {time.strftime("%Y-%m-%d %H:%M:%S")}, eer: {eer:.4f}, minDCF: {mindcf:4f}')

        return

    # core training script
    print(f'training model: {args.training_mode}')
    for it in range(it, args.max_epoch + 1):
        print(f'epoch {it}')

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        # train_network: iterate through all the data
        loss = trainer.train_network(train_loader)
        writer.add_scalar('Loss/Train', loss, it)
        print(f'Epoch {it}, TLOSS {loss :.2f}, LR {max(clr):.8f}')

        if it % args.test_interval == 0:
            eer, mindcf = evaluate(trainer)

            print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')

            mpath = f'{args.model_save_path}/model-{it}.model'
            trainer.saveParameters(mpath)

            save_scripts()
            writer.add_scalar('EER/Eval', eer, it)


def main():
    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    args.model_save_path = args.save_path + "/model"
    args.result_save_path = args.save_path + "/result"
    args.feat_save_path = args.save_path + '/feature'

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
