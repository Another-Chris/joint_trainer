import subprocess
import sys
import glob
import itertools
import re
import torch 


# to joint train that 3 model for 20 epochs and test them

BASE_ARGS = [
    ("--ssl_list", "data/train_list_cnceleb.txt"),
    ("--ssl_path", "data/cnceleb/data"),
    ("--sup_list", "data/train_list.txt"),
    ("--sup_path", "data/voxceleb2"),
    ("--test_path", "data/cnceleb/eval"),
    ("--test_list", "data/test_list_cnceleb.txt"),
    ("--nDataLoaderThread", "2"),
    ("--max-epoch", "20"),
    ("--test_interval", "20")
    # ("--eval", "")
]


def create_command(config, ssl_loss):
    exp_name = re.findall(r'\\(\w+)\.yaml', config)[0] + '_' + ssl_loss
    print(exp_name)
    CONFIGS = [
        ("--config", config),
        ("--ssl_loss", ssl_loss),
        ("--experiment_name", exp_name),
        ("--save_path", f"./save/{exp_name}")
    ]
    
    return " ".join([" ".join(arg) for arg in BASE_ARGS + CONFIGS])

configs = glob.glob("./configs/*")
ssl_losses = ["subConLoss", "angleproto"]

ssl_losses = ['softmaxproto', 'angleproto', 'aamsoftmax']

commands = []
print('=== will ran the following experiments ===')
for config in configs:
    for ssl_loss in ssl_losses:
        commands.append(create_command(config, ssl_loss))
print('======')

for command in commands:
    torch.cuda.empty_cache()
    subprocess.call(['powershell.exe', f"python trainSpeakerNet.py {command}"])