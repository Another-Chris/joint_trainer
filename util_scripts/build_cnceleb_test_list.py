import glob 

lines = []

with open('../data/cnceleb/eval/lists/trials.lst', 'r') as f:
    for line in f:
        lines.append(line.strip())
        
        
with open('../data/test_list_cnceleb.txt', 'w') as f:
    for line in lines:
        
        enroll, test, same = line.split()   
        enroll = 'enroll/' + enroll + '.flac'
        test = test.replace('.wav', '.flac')
        
        f.write(" ".join([same, enroll, test]) + '\n')
        