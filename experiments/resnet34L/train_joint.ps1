python ./trainSpeakerNet.py `
--save_path ./save/ResNetSE34L_joint_voxceleb `
--experiment_name ResNetSE34L_joint_voxceleb `
--config ./configs/ResNetSE34L_SimSLR.yaml `
--test_list data/test_list_cnceleb.txt `
--test_path data/cnceleb/eval `
--sup_list data/train_list.txt `
--sup_path data/voxceleb2 `
--ssl_list data/train_list_cnceleb.txt `
--ssl_path data/cnceleb/data `
--initial_model ./pre_trained/ResNetSE34L.model `
--batch_size 128 `
--supervised_loss angleproto `
--ssl_loss subConLoss `
--training_mode joint `
--nDataLoaderThread 3 `
--nPerSpeaker 2
# --scheduler cos `
# --optimizer sgd `
# --lr 1e-3 `