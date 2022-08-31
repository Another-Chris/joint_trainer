from utils import get_args
from tuneThreshold import *
from trainer.JointTrainer import JointTrainer
from loader import *
from trainer import *

import glob
import os
import datetime
import os.path
import zipfile
import torch.cuda
import torch
torch.cuda.empty_cache()


args = get_args()

def evaluate(trainer):
    sc, lab, _ = trainer.evaluateFromList(**vars(args))

    _, eer, _, _ = tuneThresholdfromScore(sc, lab, [1, 0.1])

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss,
                              args.dcf_c_fa)

    return eer, mindcf


def inf_train_gen(loader):
    while True:
        for data, label in loader:
            yield data, label


def main_worker(args):

    ssl_loader = get_ssl_loader(
        train_list=args.ssl_list, train_path=args.ssl_path
    )
    ssl_gen = inf_train_gen(ssl_loader)

    sup_loader = get_sup_loader(
        train_list=args.sup_list, train_path=args.sup_path
    )
    sup_gen = inf_train_gen(sup_loader)
    trainer = JointTrainer(supervised_gen=sup_gen,
                           ssl_gen=ssl_gen, **vars(args))

    # either load the initial_model or read the previous model files
    it = 1
    if (args.initial_model != ''):
        trainer.encoder.load_state_dict(torch.load(args.initial_model))
        print('model {} loaded!'.format(args.initial_model))


    # core training script
    for it in range(it, args.max_epoch + 1):
        print(f'epoch {it}, exp: {args.experiment_name}')
        # train_network: iterate through all the data
        loss = trainer.train_network('', it - 1)
        clr = trainer.scheduler.get_last_lr()[0]

        loss_total, loss_ssl, loss_sup = loss
        trainer.writer.add_scalar('epoch/loss_total', loss_total, it)
        trainer.writer.add_scalar('epoch/loss_ssl', loss_ssl, it)
        trainer.writer.add_scalar('epoch/loss_sup', loss_sup, it)
        print(
            f'Epoch {it}, {loss_total = :.2f}, {loss_ssl = :.2f} {loss_sup = :.2f} {clr = :.8f}')

        if it % args.test_interval == 0:
            eer, mindcf = evaluate(trainer)

            print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')

            mpath = f'{args.model_save_path}/model-{it}.model'
            trainer.saveParameters(mpath)
            trainer.writer.add_scalar('Eval/EER', eer, it)
            trainer.writer.add_scalar('Eval/MinDCF', mindcf, it)


def main():

    args.model_save_path = args.save_path + "/model"
    args.result_save_path = args.save_path + "/result"
    args.feat_save_path = args.save_path + '/feature'

    # exps/modelname/model
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    print(f"{args.batch_size = }")

    main_worker(args)


if __name__ == "__main__":
    main()
