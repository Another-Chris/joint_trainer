import glob 


files = glob.glob("./data/voxceleb1_test/*/*/*.flac")
print(f'{len(files) = }')
print(f'{len(files) * len(files) = }')