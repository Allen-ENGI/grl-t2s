from stable_baselines3 import SAC
m = SAC.load(r"train_res\policy\diff_time_v2_tdlambdaboot_all_seed0\policy_50000", device="cpu")
print("buffer_size:", m.buffer_size)
print("gradient_steps:", m.gradient_steps, "  learning_starts:", m.learning_starts)