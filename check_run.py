#!/usr/bin/env python3
"""
Read an existing policy run's eval_history.json and say whether MORE TRAINING
would plausibly help, or whether the run was stuck.

    python check_run.py sweep_v6_tdlambdaboot_succ_seed0

This costs nothing — the history is already on disk from the run you did. The
question "was 400k too few steps?" is answerable without spending another 400k.

WHAT IT LOOKS FOR
  peg_moved_rate trend   the decisive one. If it is 0 at every eval, the policy
                         NEVER touched the peg in 400k steps. That is an
                         exploration failure: more steps sample the same
                         useless region. If it is climbing, the budget is the
                         constraint and extending is reasonable.
  best_min_hand_peg      is the arm at least getting closer to the peg?
  success_rate trend     rising, flat, or peaked-then-regressed.
  false_optimism         (new runs only) lowest prediction on episodes that did
                         NOT succeed. Low means the reward was paying for
                         states that never solve anything.
"""
import json
import os
import sys


def timing(hist):
    """Wall-clock throughput from the recorded timestamps, so 'how long will a
    longer run take' is measured on YOUR machine rather than guessed."""
    ts = [h.get("timestamp") for h in hist if h.get("timestamp")]
    steps = [h.get("step") for h in hist if h.get("step")]
    if len(ts) < 2 or len(steps) < 2:
        return None
    dt = ts[-1] - ts[0]
    dsteps = steps[-1] - steps[0]
    if dt <= 0 or dsteps <= 0:
        return None
    sps = dsteps / dt
    return dict(elapsed_h=dt / 3600.0, steps=dsteps, steps_per_sec=sps,
                hours_per_100k=100_000 / sps / 3600.0)


def verdict(hist):
    def series(k):
        return [h[k] for h in hist if h.get(k) is not None]

    moved = series("peg_moved_rate")
    succ = series("success_rate")
    hand = series("best_min_hand_peg_distance")
    out = []

    # peg_moved_rate saturates at 1.00 on runs recorded with the old 1e-6
    # threshold, where physics jitter counted as movement. When it is pinned at
    # 1.00 the column says nothing, so fall through to peg->goal distance,
    # which measures whether the peg was actually taken anywhere.
    if moved and min(moved) == 1.0:
        out.append(("UNRELIABLE", "peg_moved_rate is 1.00 at EVERY eval. On runs "
                    "recorded before the threshold fix it fires on physics jitter "
                    "(1e-6), so a random policy scores 1.00 too. Ignore it; read "
                    "peg->goal below."))
        moved = []

    if moved and max(moved) == 0:
        out.append(("EXPLORATION", "peg_moved_rate is 0 at every eval — the policy "
                    "never touched the peg. More steps sample the same region; "
                    "extending the budget will not fix this."))
    elif moved and max(moved) > 0:
        late = moved[len(moved) // 2:]
        out.append(("BUDGET" if max(late) > max(moved[:len(moved) // 2] or [0])
                    else "PLATEAU",
                    f"peg_moved_rate reached {max(moved):.2f} "
                    f"(first half max {max(moved[:len(moved)//2] or [0]):.2f}, "
                    f"second half max {max(late):.2f})"))

    if hand and len(hand) > 2:
        improved = hand[-1] < hand[0] - 1e-6
        out.append(("REACH", f"best hand->peg {hand[0]:.4f} -> {hand[-1]:.4f}"
                    + ("  (improving)" if improved else "  (NOT improving)")))
    if succ:
        peak = max(succ)
        out.append(("SUCCESS", f"best {peak:.2f}, final {succ[-1]:.2f}"
                    + ("  — peaked then regressed" if peak > succ[-1] + 1e-9 else "")))
    peg = series("best_min_peg_goal_distance")
    if peg:
        # the real manipulation signal: did the peg get closer to the hole?
        # success is NOT at distance 0 — the peg centre sits ~0.192 from the
        # hole at insertion, versus ~0.4135 at reset
        span = 0.4135 - 0.192
        frac = max(0.0, (0.4135 - min(peg)) / span)
        out.append(("MANIPULATION",
                    f"best peg->goal {peg[0]:.4f} -> {min(peg):.4f}  "
                    f"({frac:.0%} of the way from reset to insertion)"
                    + ("  — the peg never went anywhere" if frac < 0.05 else "")))
    lift = series("mean_peg_lift_max")
    lifted = series("peg_lifted_rate")
    ctrl = series("control_rate")
    if lifted or lift:
        best_lift = max(lift) if lift else None
        rate = max(lifted) if lifted else None
        msg = (f"peg_lifted_rate {rate:.2f}" if rate is not None else "")
        if best_lift is not None:
            msg += f"  max height above reset {best_lift:+.4f} m"
        out.append(("PICKED UP?", msg + ("  — the peg was never LIFTED; the policy "
                                         "is pushing it, not picking it up"
                                         if (rate == 0 if rate is not None else
                                             (best_lift or 0) < 0.01) else "")))
    elif ctrl:
        out.append(("GRASP?", f"control_rate best {max(ctrl):.2f} — but this fires on "
                    "PUSHING too. This run predates peg_lifted; re-run to get it, or "
                    "check the video for whether the peg leaves the table."))

    fo = series("mean_false_optimism")
    if fo:
        out.append(("FALSE OPTIMISM",
                    f"lowest prediction on failing episodes: {min(fo):.1f}"
                    + ("  — the reward paid for states that never solved anything"
                       if min(fo) < 30 else "")))
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    import results
    try:
        run_dir = results.get_run_dir("policy", argv[0])
    except KeyError:
        print(f"no policy run named {argv[0]!r}. Registered: "
              f"{results.list_runs('policy')}", file=sys.stderr)
        return 1
    path = os.path.join(run_dir, "eval_history.json")
    if not os.path.exists(path):
        print(f"no eval_history.json in {run_dir}", file=sys.stderr)
        return 1
    with open(path) as f:
        hist = json.load(f)
    print(f"=== {argv[0]}: {len(hist)} evals, "
          f"{hist[-1]['step']:,} steps ===\n")
    t = timing(hist)
    if t:
        print(f"  [TIMING] {t['steps']:,} steps in {t['elapsed_h']:.2f} h "
              f"= {t['steps_per_sec']:.0f} steps/s, {t['hours_per_100k']:.2f} h per 100k")
        for n in (400_000, 800_000, 1_500_000):
            print(f"           {n:>10,} steps -> {n / t['steps_per_sec'] / 3600:5.1f} h "
                  f"per seed, {3 * n / t['steps_per_sec'] / 3600:5.1f} h for 3 seeds")
        print()
    for tag, msg in verdict(hist):
        print(f"  [{tag}] {msg}")
    print("\nEXPLORATION  -> more steps will NOT help; change exploration or the task")
    print("BUDGET       -> it was still improving; extending is reasonable")
    print("PLATEAU      -> it moved the peg early then stopped improving")
    return 0


if __name__ == "__main__":
    sys.exit(main())