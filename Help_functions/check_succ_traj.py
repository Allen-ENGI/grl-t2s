# check how different successful trajectories are in terms of hand position at a third of the way through

import numpy as np, core.results as results
d = np.load(results.get_run_dir("data_collection", "v3") + "/dataset.npz")
X, y, ep = d["X"], d["y_steps"], d["episode_ids"]
succ = np.unique(ep[y == 0])
t_succ = np.array([y[ep == e].max() for e in succ])
mid = [X[ep == e][len(X[ep == e]) // 3, 0:3] for e in succ]     # hand position a third of the way in
print(f"{len(succ)} successes; time to success: median {np.median(t_succ):.0f}, "
      f"10th-90th percentile {np.percentile(t_succ, 10):.0f}-{np.percentile(t_succ, 90):.0f}")
print(f"spread of hand position a third of the way in (cm): {100 * np.std(mid, axis=0).round(4)}")