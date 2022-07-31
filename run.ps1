# python ./trainSpeakerNet.py `
# --config ./configs/ResNetSE34L_AM.yaml

python ./trainSpeakerNet.py `
--config ./configs/ResNetSE34L_AM.yaml `
--train_list ./data/train_list_cnceleb.txt `
--train_path ./data/cnceleb/data `
--experiment_name ResNetSE34L_AM_cnceleb `
--test_path data/cnceleb/eval `
--test_list data/test_list_cnceleb.txt


