from loader import SimpleDataLoader
from trainer import SSLTrainer
from pathlib import Path
from utils import Config,inf_train_gen
from eval import evaluate

import torch
import torch.cuda
torch.cuda.empty_cache()


MODEL_NAME = "ECAPA_TDNN"
EXP_NAME = f"{MODEL_NAME}_simCLR_CNCeleb_gt5_timeShift"
# EXP_NAME = 'test'
MODEL_SAVE_PATH = f"./save/{EXP_NAME}"
SOURCE_LIST = './data/cnceleb_train_gt5.txt'
SOURCE_PATH = './data/cnceleb/data/'
# PRE_TRAINED = './pre_trained/ECAPA_TDNN.model'
PRE_TRAINED = None

Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ds = SimpleDataLoader(
        source_list=SOURCE_LIST,
        source_path=SOURCE_PATH,
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
      
    ds_gen = inf_train_gen(loader)
    trainer = SSLTrainer(exp_name = EXP_NAME)
    
    if PRE_TRAINED is not None:
        trainer.encoder.load_state_dict(torch.load(PRE_TRAINED))
        print('pre-trained weight loaded!')

    # core training script
    for it in range(1, Config.MAX_EPOCH + 1):
        print(f'epoch {it}')
        loss = trainer.train_network(ds_gen, epoch = it - 1)
        
        loss_val_dict = loss
        desc = f"EPOCH {it}: "
        for key, val in loss_val_dict.items():
            trainer.writer.add_scalar(f'epoch/{key}', val, it)
            desc += f" {key} = {val :.4f}"

        lr = trainer.scheduler.get_last_lr()[0]
        desc += f' lr = {lr:12f}'
        
        print(desc)
        
        torch.save(trainer.encoder.state_dict(), f'{MODEL_SAVE_PATH}/encoder-{it}.model')
        if lr > 1e-7:
            trainer.scheduler.step()
        
        if it % Config.TEST_INTERVAL == 0:
            eer, mindcf = evaluate(trainer.model)
            print(f'\n Epoch {it}, VEER {eer:.4f}, MinDCF: {mindcf:.5f}')
            trainer.writer.add_scalar('Eval/EER', eer, it)
            trainer.writer.add_scalar('Eval/MinDCF', mindcf, it)

