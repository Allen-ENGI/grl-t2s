import numpy as np, results

h = np.load(results.get_run_dir("t2s_model", "v6") + "/tdlambdaboot_all_hist.npy")
at = int(np.argmin(h[:, 2]))
print(f"{'epoch':>6}{'succ MSE':>10}{'mean failed pred':>18}")
for e, _, succ, fail in h[::10]:
    print(f"{int(e):>6}{succ:>10.1f}{fail:>18.1f}")
print(f"selected: epoch {at}, succ MSE {h[at, 2]:.1f}, mean failed pred {h[at, 3]:.1f}")