python ./trainSpeakerNet.py `
--config ./configs/ResNetSE34L_SimSLR.yaml `
--initial_model ./pre_trained/ResNetSE34L.model `
--train_list data/train_list_cnceleb.txt `
--test_path data/cnceleb/eval `
--test_list data/test_list_cnceleb.txt `
--train_path data/cnceleb/data `
--save_path ./save/ResNetSE34L_SSL_aug `
--experiment_name ResNetSE34L_SSL_aug


# --experiment_name ResNetSE34L_SSLpython ./trainSpeakerNet.py `
# --config ./configs/ResNetSE34L_SimSLR.yaml `
# --save_path ./save/ResNetSE34L_SSL `
# --initial_model ./pre_trained/ResNetSE34L.model `
# --train_list data/train_list_cnceleb.txt `
# --test_path data/cnceleb/eval `
# --test_list data/test_list_cnceleb.txt `
# --train_path data/cnceleb/data `
# --experiment_name ResNetSE34L_SSL
