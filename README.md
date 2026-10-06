## Pipeline Structure

### `run_1_collect`
Build a dataset with state-space information.

### `run_2_t2s`
Train various T2S models.

- `t2s_train`: T2S model training
- `t2s_model`: ML model structure
- `t2s_target`: Loss functions and training targets

### `run_3_eval`
Evaluate the performance of the trained T2S models.

- Compute MSE graphs and store evaluation videos.
- `make_videos.py` / `manual_drive`: Live evaluation
  - Generate live videos using a selected reward and T2S model.
  - Run from trained checkpoints or an expert policy.
- `compare_rewards`: Compare reward shaping between T2S and MetaWorld.  
  - Requires more work.

### `run_4_policy`
Downstream RL training using the trained T2S models.

- `reward_fn`: Stores the reward functions.
- `policy_train`: Downstream policy training.
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
    


New added scriptes for adjust peg initial positions:

peg_range - check the range of expert policy to succed with different peg location
check_pegrandom - visualize the random inital position for pegs in the dataset

scene_edit - funciton that modify the peg location
t2s_attribution - analyze how different element in the dataset matrix impact the t2s predicition