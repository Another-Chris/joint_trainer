from loader import SimpleDataLoader
from trainer import SSLTrainer
from pathlib import Path
from utils import Config
from eval import evaluate

import torch
import torch.cuda
torch.cuda.empty_cache()

MODEL_NAME = "ECAPA_TDNN"
EXP_NAME = f"{MODEL_NAME}_SSL_supCon"
MODEL_SAVE_PATH = f"./save/{EXP_NAME}"
SOURCE_LIST = './data/voxceleb_train.txt'
SOURCE_PATH = './data/voxceleb2/'
TARGET_LIST = './data/cnceleb_train.txt'
TARGET_PATH = './data/cnceleb/data'
TEST_LIST = './data/voxceleb_test.txt'
TEST_PATH = './data/voxceleb1/'

Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ds = SimpleDataLoader(
        source_list=SOURCE_LIST,
        source_path=SOURCE_PATH,
        target_list=TARGET_LIST,
        target_path=TARGET_PATH,
        augment=True,
        musan_path=Config.MUSAN_PATH,
        rir_path=Config.RIR_PATH,
        max_frames=Config.MAX_FRAMES
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
      
    trainer = SSLTrainer(
        exp_name = EXP_NAME,
        model_name=MODEL_NAME, 
        )

    # core training script
    for it in range(1, Config.MAX_EPOCH + 1):
        print(f'epoch {it}')
        loss = trainer.train_network(loader, epoch = it - 1)
        
        loss_val_dict = loss
        desc = f"EPOCH {it}: "
        for key, val in loss_val_dict.items():
            trainer.writer.add_scalar(f'epoch/{key}', val, it)
            desc += f" {key} = {val :.4f}"

        torch.save(trainer.encoder.state_dict(), f'{MODEL_SAVE_PATH}')
        print(desc)

        if it % Config.TEST_INTERVAL == 0:
            eer, mindcf = evaluate(trainer)
            print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')
            trainer.writer.add_scalar('Eval/EER', eer, it)
            trainer.writer.add_scalar('Eval/MinDCF', mindcf, it)
