# check_metaworld_reward.py

import inspect
from env_utils import make_fixed_scene_env

# Create the exact same environment used by your project
env = make_fixed_scene_env()
obs, info = env.reset(seed=0)

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print("Wrapped:  ", type(env))
print("Unwrapped:", type(env.unwrapped))


# ---------------------------------------------------------
# 1. Check what MetaWorld returns during an actual step
# ---------------------------------------------------------
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("\n" + "=" * 60)
print("ONE STEP OUTPUT")
print("=" * 60)

print("Total reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)

print("\nReward/info components:")
for key, value in info.items():
    print(f"  {key:25s}: {value}")


# ---------------------------------------------------------
# 2. Print the exact reward implementation
# ---------------------------------------------------------
base_env = env.unwrapped

print("\n" + "=" * 60)
print("compute_reward() SOURCE")
print("=" * 60)

if hasattr(base_env, "compute_reward"):
    print(inspect.getsource(base_env.compute_reward))
else:
    print("No compute_reward() found.")


print("\n" + "=" * 60)
print("evaluate_state() SOURCE")
print("=" * 60)

if hasattr(base_env, "evaluate_state"):
    print(inspect.getsource(base_env.evaluate_state))
else:
    print("No evaluate_state() found.")

env.close()