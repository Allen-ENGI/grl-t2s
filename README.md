run_1_collect - build a dataset with state-space information

run_2_t2s - train various t2s models
    t2s_train
    t2s_model: the ML structure
    t2s_target: the loss functions
run_3_eval - evaluate the performance of the t2s models
    compute MSE graphs and store videos
    
    make_videos.py / manual_drive - live evaluation
    generate live video with selected reward and t2s model from target trained checkpoitns or expert policy

    compare_rewards: compare the reward shaping between t2s and metaworld (Require more work)

run_4_policy - downstream RL training
    reward_fn: the scripte store all rewards
    policy_train
    