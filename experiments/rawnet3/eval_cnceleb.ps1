python ./trainSpeakerNet.py `
--eval `
--config ./configs/RawNet3_SimSLR.yaml `
--save_path ./save/RawNet3_SSL `
--test_path data/cnceleb/eval `
--test_list data/test_list_cnceleb.txt `
--initial_model ./pre_trained/RawNet3.pt ` #pre-trained weights