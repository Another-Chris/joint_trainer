python ./trainSpeakerNet.py `
--config ./configs/ResNetSE34L_ap.yaml `
--experiment_name ResNetSE34L_sup_cnceleb `
--save_path ./save/ResNetSE34L_sup_cnceleb `
--train_list data/train_list_cnceleb.txt `
--train_path data/cnceleb/data `
--test_path data/cnceleb/eval `
--test_list data/test_list_cnceleb.txt `
--training_mode supervised `
--initial_model ./pre_trained/ResNetSE34L.model `
# --scheduler cos `
# --optimizer sgd `
# --lr 1e-2 `
