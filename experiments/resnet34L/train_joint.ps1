python ./trainSpeakerNet.py `
--save_path ./save/ResNetSE34L_joint_cnceleb `
--experiment_name ResNetSE34L_joint_cnceleb `
--config ./configs/ResNetSE34L_SimSLR.yaml `
--test_list data/test_list_cnceleb.txt `
--test_path data/cnceleb/eval `
--sup_list data/train_list_cnceleb.txt `
--sup_path data/cnceleb/data `
--initial_model ./pre_trained/ResNetSE34L.model `
--batch_size 128 `
--supervised_loss angleproto `
--ssl_loss subConLoss `
--training_mode joint `
--nDataLoaderThread 3 `
--test_interval 5 `
--nPerSpeaker 2
# --scheduler cos `
# --optimizer sgd `
# --lr 1e-3 `