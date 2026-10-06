import json, numpy as np, core.results as results
run = results.get_run_dir("data_collection", "v4")
d = np.load(run + "/dataset.npz")
peg = json.load(open(run + "/dataset_summary.json"))["peg_starts"]

first = d["frame_idxs"] == 0
start = d["X"][first][:, 4:6]                       # observed peg x, y at each episode's start
pid, split = d["position_ids"][first], d["split"][first].astype(str)
pool = np.array([[p["x"], p["y"]] for p in peg["positions"]])

shift = start[pid < 0].mean(axis=0) - np.array(peg["fixed_start"][:2])   # observation vs placement
expected = pool[pid[pid >= 0]] + shift
gap = 100 * np.linalg.norm(start[pid >= 0] - expected, axis=1)

print(f"episodes: {len(start)}, distinct observed starts (1 mm): {len(np.unique(start.round(3), axis=0))}")
print(f"at the fixed start: {np.mean(pid < 0):.0%}   at random positions: {np.mean(pid >= 0):.0%}")
print(f"observed start x {start[:, 0].min():.3f} to {start[:, 0].max():.3f}, "
      f"y {start[:, 1].min():.3f} to {start[:, 1].max():.3f}")
print(f"actual vs intended start: median {np.median(gap):.2f} cm, largest {gap.max():.2f} cm")
tr, va = set(pid[(pid >= 0) & (split == "train")]), set(pid[(pid >= 0) & (split == "val")])
print(f"positions used in training: {len(tr)}, in validation: {len(va)}, in both: {len(tr & va)}")
print(f"spread of fixed-start episodes (should be ~0 cm): {100 * start[pid < 0].std(axis=0).round(4)}")

import matplotlib.pyplot as plt
for name, mask, c in (("fixed start", pid < 0, "black"),
                      ("train positions", (pid >= 0) & (split == "train"), "tab:blue"),
                      ("validation positions", (pid >= 0) & (split == "val"), "tab:orange")):
    plt.scatter(*start[mask].T, s=8, c=c, label=name)
plt.xlabel("peg x at start (observation)"); plt.ylabel("peg y at start (observation)")
plt.legend(); plt.axis("equal"); plt.savefig(run + "/peg_starts.png", dpi=130)