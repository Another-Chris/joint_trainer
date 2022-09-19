from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from loader import TrainDatasetLoader,DummyLoader
from trainer import SupTrainer
from pathlib import Path
from utils import Config

import torch.cuda
import torch
torch.cuda.empty_cache()

MODEL_SAVE_PATH = "./save/PASE_sup"
TRAIN_LIST = './data/voxceleb_train.txt'
TRAIN_PATH = './data/voxceleb2'
TEST_LIST = './data/voxceleb_test.txt'
TEST_PATH = './data/voxceleb1_test'
PRE_TRAINED = "./pre_trained/FE_e199.ckpt"

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
        musan_path=Config.MUSAN_PATH,
        rir_path=Config.RIR_PATH,
        max_frames=Config.MAX_FRAMES,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        drop_last=True,
    )
    trainer = SupTrainer()
    
    # trainer.encoder.load_pretrained(PRE_TRAINED)
    # print('pretrained PASE loaded!')

    # core training script
    it = 1
    for it in range(it, Config.MAX_EPOCH + 1):
        print(f'epoch {it}')
        # train_network: iterate through all the data
        loss = trainer.train_network(loader, it - 1)
        
        # save params for every epoch
        mpath = f'{MODEL_SAVE_PATH}/model-{it}.model'
        trainer.saveParameters(mpath)

        if it % Config.TEST_INTERVAL == 0:
                eer, mindcf = evaluate(trainer)

                print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')

                trainer.writer.add_scalar('Eval/EER', eer, it)
                trainer.writer.add_scalar('Eval/MinDCF', mindcf, it)