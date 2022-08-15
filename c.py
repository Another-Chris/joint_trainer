import glob
import torch


modelfiles = glob.glob("./pre_trained/*")

new_dict = {}
for mf in modelfiles:
    
    state_dict = torch.load(mf)
    print(list(state_dict.keys())[:2], mf)
    
    if state_dict.get('model') is not None:
        state_dict = state_dict['model'] 
    
    for key,val in state_dict.items():
        new_key = key.replace('__L__.', "")
        new_dict[new_key] = val
        
    torch.save(new_dict, mf)