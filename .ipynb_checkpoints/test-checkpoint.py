import soundfile as sf

from tqdm import tqdm

train_path = './data/voxceleb2'
void_file = []
with open('./data/voxceleb_train.txt') as f:
    train_list = []
    for line in tqdm(f):
        fname = train_path + "/" + line.strip().split(" ")[1]
        
        try:
            signal, _ = sf.read(fname)
        except:
            print(fname)
            void_file.append(fname)

            
with open('./data/void_file.txt', 'w') as f:
    for line in void_file:
        f.write(line + '/n')
            
        

