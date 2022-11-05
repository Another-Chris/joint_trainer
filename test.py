from handle_cn import concat_cn
import glob 

files = glob.glob('./data/cnceleb/data/*/*.flac')
cat_dict = concat_cn(files, target_path='./data/cnceleb/data', trange = (0, 4 * 16000))
