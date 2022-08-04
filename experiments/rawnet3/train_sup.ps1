python ./trainSpeakerNet.py `
--config ./configs/RawNet3_AAM.yaml `
--test_list data/test_list_cnceleb.txt `
--test_path data/cnceleb/eval `
--train_path data/voxceleb2 `
--train_list data/train_list.txt `
--initial_model ./pre_trained/RawNet3.pt `
--experiment_name RawNet3_sup `
--save_path ./save/RawNet3_sup `
--training_mode supervised