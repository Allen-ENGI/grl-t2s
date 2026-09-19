"""
Target construction for the T2S formulations: MC, TD(0), TD(lambda).

Pulled out of the training loop so it can be unit-tested on its own (see
tests/test_t2s_targets.py) without needing a GPU, a trained network, or the
sim — you can hand it synthetic episodes and check the target arithmetic
directly.

TWO WAYS TO HANDLE A FAILED (TRUNCATED) EPISODE
-----------------------------------------------
censor_bootstrap=False (the original scheme)
    The truncated frame is supervised with a fabricated ceiling,
    terminal_value = censor_label = 300. The true time-to-success there is
    unknown and possibly infinite, so this trains the model against a number
    that is simply not true. Because MSE is quadratic, those large invented
    targets also dominate the gradient over the genuine small targets near
    success. Observed effect: models trained WITH failures ("_all") predicted
    ~172-200 on failure trajectories, visibly pulled toward the 300 fiction.

censor_bootstrap=True (value bootstrap)
    The truncated frame is NOT supervised at all — it is masked out of the
    loss — but the network's own prediction of it still bootstraps the frame
    before it:

        T(s_t) = dt + gamma * V(s_{t+1})        for t < K
        T(s_K)   unsupervised                    (K = truncation index)

    So failed trajectories contribute only genuine transition relationships
    and no invented numbers. This also matches how gymnasium/SB3 separate
    `terminated` (episode genuinely ended -> value 0) from `truncated`
    (episode was cut off -> bootstrap from the last state); success frames
    are terminated, failures are truncated, and the censor label treated
    truncation as a third thing that is neither.

    REQUIRES gamma < 1. With gamma == 1 the recursion V = dt + V has no
    finite fixed point, so values on trajectories that never succeed drift
    upward every target-network refresh until clip_max catches them — which
    quietly reinstates an arbitrary ceiling by a noisier route. With a
    discount the fixed point is dt/(1-gamma): 100 at gamma=0.99, 20 at 0.95.

    One constraint comes free: T2SModel.t2s_head ends in Softplus, so V >= 0
    always, which makes T(s_t) >= dt*(K-t) — the correct censored-regression
    lower bound ("it had not succeeded yet after this many more steps") with
    no hinge loss needed.
"""
import numpy as np


def build_targets(method, bootstrap_fn, rows, y_all, next_obs_n_all, is_terminal_all,
                   terminal_value_all, episode_ids_all, frame_idxs_all,
                   gamma=1.0, lam=0.9, clip_max=None, is_truncated_all=None,
                   dt=1.0):
    """
    method: "mc" | "td0" | "tdlambda"
    bootstrap_fn: callable(obs_n) -> np.ndarray of V(next_obs); unused by MC.
    rows: index array selecting which dataset rows to build targets for.
    All *_all arrays are full-dataset arrays indexed by `rows`.

    is_truncated_all: optional bool array marking frames whose true
        time-to-success is UNKNOWN (the last frame of a failed episode).
        When given, build_targets returns (targets, supervise_mask) and those
        frames are masked out — the caller must exclude them from the loss.
        Their target value is meaningless and set to 0 only so the array has
        a defined shape. When None, behaviour is unchanged and only targets
        are returned, so existing callers keep working.
    """
    y = y_all[rows]
    truncated = (is_truncated_all[rows] if is_truncated_all is not None
                 else np.zeros(len(rows), dtype=bool))

    if method == "mc":
        target = y.astype(np.float32)
        if is_truncated_all is None:
            return target
        # a truncated frame has no Monte-Carlo return to regress onto
        target = np.where(truncated, 0.0, target).astype(np.float32)
        return target, ~truncated

    nxt = next_obs_n_all[rows]
    term = is_terminal_all[rows]
    term_val = terminal_value_all[rows]
    v_next = bootstrap_fn(nxt)
    if clip_max is not None:
        v_next = np.clip(v_next, 0.0, clip_max)

    if method == "td0":
        target = np.where(term, term_val, dt + gamma * v_next)

    elif method == "tdlambda":
        target = np.zeros(len(rows), dtype=np.float32)
        eps = episode_ids_all[rows]
        frames = frame_idxs_all[rows]
        # Group episodes by sorting once instead of scanning the whole array
        # per episode. The previous version called np.flatnonzero(eps == ep)
        # inside a loop over every unique episode, which is O(n_episodes * n_rows)
        # and became the dominant cost once the dataset grew past ~100k rows.
        order = np.lexsort((frames, eps))
        ep_sorted = eps[order]
        boundaries = np.flatnonzero(np.diff(ep_sorted)) + 1
        for grp in np.split(order, boundaries):
            G_next = None
            for i in grp[::-1]:            # backwards through the episode
                if term[i]:
                    target[i] = term_val[i]
                else:
                    boot = dt + gamma * v_next[i]
                    full = boot if G_next is None else dt + gamma * G_next
                    target[i] = (1 - lam) * boot + lam * full
                G_next = target[i]
    else:
        raise ValueError(f"unknown method {method!r}, expected 'mc' | 'td0' | 'tdlambda'")

    if clip_max is not None:
        target = np.clip(target, 0.0, clip_max)
    target = target.astype(np.float32)

    if is_truncated_all is None:
        return target
    # truncated frames: value unknown -> excluded from supervision. Their
    # PREDICTION still bootstraps the frame before them (that happened above,
    # via v_next), which is the whole point of the value-bootstrap scheme.
    target = np.where(truncated, 0.0, target).astype(np.float32)
    return target, ~truncated


def build_bookkeeping(X, y, episode_ids, frame_idxs, censor_label, censor_bootstrap=False):
    """
    Per-row next_obs / is_terminal / terminal_value needed for TD-style targets.

    A row is TERMINATED if y == 0 — the success frame, whose value is genuinely
    known to be 0.

    The last row of a FAILED episode is TRUNCATED, not terminated: the episode
    was cut off by the step limit, so its true time-to-success is unknown.

    censor_bootstrap=False (original): treats truncation as termination with
        terminal_value = censor_label, i.e. asserts the unknown value is 300.
    censor_bootstrap=True (value bootstrap): marks it truncated instead. It is
        not terminal, so build_targets bootstraps through it, and it is masked
        out of the loss so no invented number is ever regressed onto.

    Returns (next_obs, is_terminal, terminal_value, is_truncated).
    """
    n = len(X)
    next_obs = np.zeros_like(X)
    is_terminal = np.zeros(n, dtype=bool)
    is_truncated = np.zeros(n, dtype=bool)
    terminal_value = np.zeros(n, dtype=np.float32)
    for i in range(n):
        same_ep_next = (i + 1 < n) and (episode_ids[i + 1] == episode_ids[i])
        if y[i] == 0.0:
            is_terminal[i] = True          # genuine success: value known to be 0
            terminal_value[i] = 0.0
        elif not same_ep_next:
            if censor_bootstrap:
                # unknown value: not terminal, no label. next_obs stays zeros
                # because there is no next state — build_targets masks the row
                # out, so its bootstrap is never used for its own target.
                is_truncated[i] = True
            else:
                is_terminal[i] = True
                terminal_value[i] = censor_label
        else:
            next_obs[i] = X[i + 1]
    return next_obs, is_terminal, terminal_value, is_truncated
