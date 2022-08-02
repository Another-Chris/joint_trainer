import os 
import pickle

def save_features(feat_save_path, features, labels):
    
    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)
    
    with open(f'{feat_save_path}/features.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    with open(f'{feat_save_path}/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
