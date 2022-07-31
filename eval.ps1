# python ./trainSpeakerNet.py `
# --eval `
# --model ResNetSE34V2 `
# --log_input True `
# --encoder_type ASP `
# --n_mels 64 `
# --trainfunc softmaxproto `
# --save_path exps/test `
# --eval_frames 400 `
# --initial_model baseline_v2_ap.model `
# --test_path data/cnceleb/eval `
# --test_list data/test_list_cnceleb.txt

python ./trainSpeakerNet.py `
--eval `
--model ResNetSE34L `
--log_input True `
--trainfunc angleproto `
--save_path exps/test `
--eval_frames 400 `
--initial_model baseline_lite_ap.model `
--test_path data/cnceleb/eval `
--test_list data/test_list_cnceleb.txt