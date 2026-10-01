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