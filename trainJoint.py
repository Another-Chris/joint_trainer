import torch
from loader.train import GIMDatasetLoader
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from loader import TrainDatasetLoader
from trainer import JointTrainer
from pathlib import Path
from utils import Config

import torch.cuda
torch.cuda.empty_cache()

MODEL_NAME = "ECAPA_TDNN"
EXP_NAME = f"{MODEL_NAME}_infoMax"
MODEL_SAVE_PATH = f"./save/{EXP_NAME}"
SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'
TARGET_LIST = './data/cnceleb_train.txt'
TARGET_PATH = './data/cnceleb/data'
TEST_LIST = './data/cnceleb_test.txt'
TEST_PATH = './data/cnceleb/eval'

Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

def evaluate(trainer):
    sc, lab, _ = trainer.evaluateFromList(
        test_list=TEST_LIST, test_path=TEST_PATH, num_eval=10
    )
    _, eer, _, _ = tuneThresholdfromScore(sc, lab, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    return eer, mindcf


def inf_train_gen(loader):
    while True:
        for data, label in loader:
            yield data, label


if __name__ == "__main__":
    source_ds = TrainDatasetLoader(
        train_list=SOURCE_LIST,
        train_path=SOURCE_PATH,
        augment=True,
        musan_path=Config.MUSAN_PATH,
        rir_path=Config.RIR_PATH,
        max_frames=Config.MAX_FRAMES
    )
    source_loader = torch.utils.data.DataLoader(
        source_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    target_ds = TrainDatasetLoader(
        train_list=TARGET_LIST,
        train_path=TARGET_PATH,
        augment=True,
        musan_path=Config.MUSAN_PATH,
        rir_path=Config.RIR_PATH,
        max_frames=Config.MAX_FRAMES
    )
    target_loader = torch.utils.data.DataLoader(
        target_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    gim_ds = GIMDatasetLoader(
        train_list=TARGET_LIST,
        train_path=TARGET_PATH,
        augment=True,
        musan_path=Config.MUSAN_PATH,
        rir_path=Config.RIR_PATH,
        max_frames=Config.MAX_FRAMES
    )
    gim_loader = torch.utils.data.DataLoader(
        source_ds,
        batch_size = Config.BATCH_SIZE // 5, # 5 segments 
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    
    
    ds_gens = {
        "source_gen": inf_train_gen(source_loader),
        "target_gen": inf_train_gen(target_loader),
        "gim_gen": inf_train_gen(gim_loader)
    }

    
    trainer = JointTrainer(
        model_name=MODEL_NAME, 
        ds_gens = ds_gens
        )
    
    trainer.encoder.load_state_dict(
        torch.load(f"./pre_trained/{MODEL_NAME}.model"))
    print(f'pretrained {MODEL_NAME} loaded!')

    # core training script
    for it in range(1, Config.MAX_EPOCH + 1):
        print(f'epoch {it}')
        loss = trainer.train_network()
        
        loss_val_dict = loss
        desc = f"EPOCH {it}: "
        for key, val in loss_val_dict.items():
            trainer.writer.add_scalar(f'epoch/{key}', val, it)
            desc += f" {key} = {val :.4f}"

        mpath = f'{MODEL_SAVE_PATH}/model-{it}.model'
        trainer.saveParameters(mpath)
        print(desc)

        if it % Config.TEST_INTERVAL == 0:
            eer, mindcf = evaluate(trainer)
            print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')
            trainer.writer.add_scalar('Eval/EER', eer, it)
            trainer.writer.add_scalar('Eval/MinDCF', mindcf, it)
