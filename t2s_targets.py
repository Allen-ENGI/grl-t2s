"""
Target construction for the three T2S formulations: MC, TD(0), TD(lambda).

Pulled out of the training loop so it can be unit-tested on its own (see
tests/test_t2s_targets.py) without needing a GPU, a trained network, or the
sim — you can hand it synthetic episodes and check the target arithmetic
directly.
"""
import numpy as np


def build_targets(method, bootstrap_fn, rows, y_all, next_obs_n_all, is_terminal_all,
                   terminal_value_all, episode_ids_all, frame_idxs_all,
                   gamma=1.0, lam=0.9, clip_max=None):
    """
    method: "mc" | "td0" | "tdlambda"
    bootstrap_fn: callable(obs_n) -> np.ndarray of V(next_obs) for MC this is unused.
    rows: index array selecting which dataset rows to build targets for.
    All *_all arrays are full-dataset arrays indexed by `rows`.
    """
    y = y_all[rows]
    if method == "mc":
        return y.astype(np.float32)

    nxt = next_obs_n_all[rows]
    term = is_terminal_all[rows]
    term_val = terminal_value_all[rows]
    v_next = bootstrap_fn(nxt)
    if clip_max is not None:
        v_next = np.clip(v_next, 0.0, clip_max)

    if method == "td0":
        target = np.where(term, term_val, 1.0 + gamma * v_next)

    elif method == "tdlambda":
        target = np.zeros(len(rows), dtype=np.float32)
        eps = episode_ids_all[rows]
        frames = frame_idxs_all[rows]
        for ep in np.unique(eps):
            idx_local = np.flatnonzero(eps == ep)
            idx_local = idx_local[np.argsort(frames[idx_local])]
            G_next = None
            for pos in range(len(idx_local) - 1, -1, -1):
                i = idx_local[pos]
                if term[i]:
                    target[i] = term_val[i]
                else:
                    boot = 1.0 + gamma * v_next[i]
                    full = boot if G_next is None else 1.0 + gamma * G_next
                    target[i] = (1 - lam) * boot + lam * full
                G_next = target[i]
    else:
        raise ValueError(f"unknown method {method!r}, expected 'mc' | 'td0' | 'tdlambda'")

    if clip_max is not None:
        target = np.clip(target, 0.0, clip_max)
    return target.astype(np.float32)


def build_bookkeeping(X, y, episode_ids, frame_idxs, censor_label):
    """
    Per-row next_obs / is_terminal / terminal_value needed for TD-style targets.
    A row is terminal if:
      - y == 0 (success frame itself; terminal value 0), or
      - it's the last row of its episode (censored failure; terminal value = censor_label)
    """
    n = len(X)
    next_obs = np.zeros_like(X)
    is_terminal = np.zeros(n, dtype=bool)
    terminal_value = np.zeros(n, dtype=np.float32)
    for i in range(n):
        same_ep_next = (i + 1 < n) and (episode_ids[i + 1] == episode_ids[i])
        if y[i] == 0.0:
            is_terminal[i] = True
            terminal_value[i] = 0.0
        elif not same_ep_next:
            is_terminal[i] = True
            terminal_value[i] = censor_label
        else:
            next_obs[i] = X[i + 1]
    return next_obs, is_terminal, terminal_value
