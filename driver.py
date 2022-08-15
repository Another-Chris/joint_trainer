import subprocess
import sys


# to joint train that 3 model for 20 epochs and test them

BASE_ARGS = [
    ("--ssl_list", "data/train_list_cnceleb.txt"),
    ("--ssl_path", "data/cnceleb/data"),
    ("--sup_list", "data/train_list.txt"),
    ("--sup_path", "data/voxceleb2"),
    ("--test_path", "data/cnceleb/eval"),
    ("--test_list", "data/test_list_cnceleb.txt"),
    
    ("--nDataLoaderThread", "2"),
]

def create_command(config):
    CONFIGS = [
        ("--config", config),
        ("--ssl_loss", ssl_loss),
        ("--sup_loss", sup_loss),
        
        
    ]
