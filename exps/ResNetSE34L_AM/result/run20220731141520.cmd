Namespace(config='./configs/ResNetSE34L_AM.yaml', max_frames=200, eval_frames=300, batch_size=200, max_seg_per_spk=500, nDataLoaderThread=5, augment=10, seed=10, test_interval=10, max_epoch=500, trainfunc='amsoftmax', optimizer='adam', scheduler='steplr', lr=0.001, lr_decay=0.95, weight_decay=0, hard_prob=0.5, hard_rank=10, margin=0.3, scale=30.0, nPerSpeaker=1, nClasses=5994, dcf_p_target=0.05, dcf_c_miss=1, dcf_c_fa=1, initial_model='', save_path='exps/ResNetSE34L_AM', train_list='data/train_list.txt', test_list='data/test_list.txt', train_path='data/voxceleb2', test_path='data/voxceleb1', musan_path='data/musan_split', rir_path='data/RIRS_NOISES/simulated_rirs', n_mels=40, log_input=True, model='ResNetSE34L', encoder_type='SAP', nOut=512, sinc_stride=10, eval=False, mixedprec=False, model_save_path='exps/ResNetSE34L_AM/model', result_save_path='exps/ResNetSE34L_AM/result', feat_save_path='')