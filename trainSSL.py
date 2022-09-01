from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from loader import TrainDatasetLoader
from trainer import SSLTrainer
from pathlib import Path

import torch.cuda
import torch
torch.cuda.empty_cache()


MAX_EPOCH = 500
TEST_INTERVAL = 10
MODEL_SAVE_PATH = "./save/ECAPA_TDNN"
BATCH_SIZE = 32

TRAIN_LIST = './data/cnceleb_train.txt'
TRAIN_PATH = './data/cnceleb/data'
TEST_LIST = './data/cnceleb_test.txt'
TEST_PATH = './data/cnceleb/eval'
MUSAN_PATH = "./data/musan_split"
RIR_PATH = "./data/RIRS_NOISES/simulated_rirs"


Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)


def evaluate(trainer):
    sc, lab, _ = trainer.evaluateFromList(
        test_list=TEST_LIST, test_path=TEST_PATH, num_eval=10
    )
    _, eer, _, _ = tuneThresholdfromScore(sc, lab, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    return eer, mindcf


if __name__ == "__main__":
    ds = TrainDatasetLoader(
        train_list=TRAIN_LIST,
        train_path=TRAIN_PATH,
        augment=True,
        musan_path=MUSAN_PATH,
        rir_path=RIR_PATH,
        max_frames=200,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        drop_last=True,
    )
    trainer = SSLTrainer()

    # either load the initial_model or read the previous model files
    it = 1

    trainer.encoder.load_state_dict(
        torch.load("./pre_trained/ECAPA_TDNN.model"))
    print('pretrained ECAPA_TDNN loaded!')

    # core training script
    for it in range(it, MAX_EPOCH + 1):
        print(f'epoch {it}')
        # train_network: iterate through all the data
        loss = trainer.train_network(loader, it - 1)
        clr = trainer.scheduler.get_last_lr()[0]

        loss_total, loss_ssl, loss_sup = loss
        trainer.writer.add_scalar('epoch/loss_total', loss_total, it)
        trainer.writer.add_scalar('epoch/loss_ssl', loss_ssl, it)
        trainer.writer.add_scalar('epoch/loss_sup', loss_sup, it)
        print(
            f'Epoch {it}, {loss_total = :.2f}, {loss_ssl = :.2f} {loss_sup = :.2f} {clr = :.8f}')

        if it % TEST_INTERVAL == 0:
            eer, mindcf = evaluate(trainer)

            print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')

            mpath = f'{MODEL_SAVE_PATH}/model-{it}.model'
            trainer.saveParameters(mpath)
            trainer.writer.add_scalar('Eval/EER', eer, it)
            trainer.writer.add_scalar('Eval/MinDCF', mindcf, it)
