python ./trainSpeakerNet.py `
--eval `
--eval_frames 400 `
--config ./configs/ResNetSE34L_SimSLR.yaml `
--save_path ./save/ResNetSE34L_SSL `
--test_path data/cnceleb/eval `
--test_list data/test_list_cnceleb.txt `
--initial_model ./pre_trained/ResNetSE34L.model ` #pre-trained weights