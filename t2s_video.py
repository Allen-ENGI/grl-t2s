"""
Live T2S prediction video — the visual companion to Stage 4's numbers.

WHY THIS EXISTS
---------------
Every Stage-4 metric is an aggregate. `mae` says predictions are off by N
frames; `pred_min` on a failure rollout says the model dipped to 12. Neither
tells you WHAT THE ARM WAS DOING at the moment the model claimed success was
imminent, and that is the question that actually decides whether a T2S model
is usable as a reward. The project's central failure — a policy hovering the
empty gripper near the hole while a success-only model's prediction fell
toward zero, because in every demonstration "hand near hole" and "peg in
hole" always co-occurred — is obvious in one second of video and invisible in
a table.

WHAT IT RENDERS
---------------
`render_t2s_video` composes, per frame:
  - the rendered scene
  - a live readout panel: predicted steps-to-success, ground truth when it
    exists, signed error, hand->peg and peg->goal distance, and the live
    per-step reward the shaping function would emit for that transition
  - a countdown bar that turns red inside the false-optimism band
  - a scrolling prediction trace with the ground-truth line overlaid, so the
    divergence is visible as it happens rather than only in a static plot

`render_model_comparison_video` puts every model's prediction on the SAME
stored frames, so any difference between the traces is the model and not the
rollout.

`evaluate_with_video` is the Stage-4 entry point: it runs the usual numeric
evaluation and writes the videos from the same rollouts, so the numbers and
the footage always describe the same trajectories.

The prediction overlay uses the same t2s_predict predictor and the same
reward_fn arithmetic as training, so what the video shows is what the policy
was paid.

Dependencies: PIL (already used by visualize.annotate_video) and imageio for
writing. Both are imported lazily so importing this module is cheap.
"""
import os

import numpy as np

from config import (CENSOR_LABEL, RL_GAMMA, TIME_PENALTY, SUCCESS_KEY,
                    steps_remaining_curve)
from progress import hand_peg_distance_curve, peg_goal_distance_curve

DANGER_THRESHOLD = 10.0     # predictions below this claim success is imminent
# MuJoCo's OpenGL framebuffer is bottom-up. gymnasium's MujocoRenderer usually
# flips it back before returning, but not on every version/render path — when
# it doesn't, every frame comes out vertically mirrored. Flip here rather than
# at each of the three capture sites, and set False if your stack already does
# it (the giveaway is the arm appearing to hang from the ceiling).
FLIP_FRAMES = True

# Playback speed. MetaWorld steps at 500 per episode and the readout panel
# carries numbers the viewer has to actually read — at 20 fps a 90-step
# success is over in 4.5 seconds and the prediction is a blur. 12 fps stretches
# the same episode to 7.5s without making it feel like a slideshow. This is the
# ONE place it is set; it used to be a hardcoded 20 in four separate defaults,
# and evaluate_with_video had no way to pass it at all.
FPS = 12
PANEL_W = 340               # width of the readout panel, px
TRACE_H = 150               # height of the scrolling trace, px


# ---- rollout with frames ------------------------------------------------

def rollout_with_frames(policy, predict_t2s=None, seed=0, max_steps=500,
                        camera_name="corner2", stop_after_success=20,
                        random_policy=False):
    """
    Roll out on the fixed scene recording observations, RGB frames, per-step
    T2S predictions and the real success flag.

    success_step follows the package-wide contract: the index of the first
    state that IS successful.
    """
    import torch

    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env(render_mode="rgb_array", camera_name=camera_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    obs_list, frames, preds = [], [], []
    success_step = None
    for t in range(max_steps):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        frames.append(render_frame(env))
        if predict_t2s is not None:
            preds.append(float(predict_t2s(obs)))
        if random_policy or policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=False)
        obs, _r, term, trunc, info = env.step(action)
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        if success_step is not None and t >= success_step + stop_after_success:
            break
        if term or trunc:
            break
    env.close()

    obs_arr = np.array(obs_list, dtype=np.float32)
    return dict(
        obs=obs_arr, frames=frames, seed=seed, success_step=success_step,
        t2s_preds=np.array(preds, dtype=np.float64) if preds else None,
        truth=(steps_remaining_curve(len(obs_arr), success_step)
               if success_step is not None else None),
    )


# ---- frame composition --------------------------------------------------

def render_frame(env):
    """One RGB frame, corrected for MuJoCo's bottom-up framebuffer."""
    frame = env.render()
    return frame[::-1] if (FLIP_FRAMES and frame is not None) else frame


def _font(size=14):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_trace(draw, preds, truth, i, box, danger=DANGER_THRESHOLD):
    """Scrolling prediction trace inside `box` = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    n = max(len(preds), 2)
    hi = max(float(np.nanmax(preds)) if len(preds) else 1.0,
             float(np.nanmax(truth)) if truth is not None and len(truth) else 1.0, 1.0)

    draw.rectangle(box, fill=(18, 18, 22))
    # danger band
    yb = y1 - int(h * min(danger / hi, 1.0))
    draw.rectangle((x0, yb, x1, y1), fill=(60, 20, 20))

    def pts(series, upto):
        return [(x0 + int(w * k / (n - 1)), y1 - int(h * min(max(series[k], 0) / hi, 1.0)))
                for k in range(min(upto + 1, len(series)))]

    if truth is not None and len(truth):
        p = pts(truth, i)
        if len(p) > 1:
            draw.line(p, fill=(120, 120, 130), width=1)
    p = pts(preds, i)
    if len(p) > 1:
        draw.line(p, fill=(90, 200, 255), width=2)
    if p:
        cx, cy = p[-1]
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255))
    draw.text((x0 + 8, y0 + 10), f"T2S trace   max {hi:.0f}", fill=(150, 150, 162),
              font=_font(10))
    # label the live value at the cursor, so the trace is readable on its own
    if len(preds) and i < len(preds):
        draw.text((x0 + 8, y1 - 16), f"now {float(preds[i]):,.1f}",
                  fill=(120, 200, 245), font=_font(10))


def _bar(d, x, y, w, h, frac, color, ticks=5, hi=None):
    """Countdown bar with tick marks and an end label."""
    d.rectangle((x, y, x + w, y + h), fill=(26, 26, 32), outline=(70, 70, 82))
    if frac > 0:
        d.rectangle((x, y, x + int(w * min(frac, 1.0)), y + h), fill=color)
    for k in range(1, ticks):
        tx = x + int(w * k / ticks)
        d.line((tx, y, tx, y + h), fill=(70, 70, 82), width=1)
    if hi is not None:
        d.text((x + w + 4, y - 1), f"{hi:.0f}", fill=(120, 120, 132), font=_font(10))


def compose_frame(frame, i, preds, truth=None, obs=None, rewards=None,
                  success_step=None, danger_threshold=DANGER_THRESHOLD,
                  label=None, censor_label=CENSOR_LABEL, reward_mode=None):
    """
    One annotated frame: scene on the left, readout on the right, trace beneath.

    The prediction is the reason this video exists, so it gets the visual
    hierarchy: a 52px figure with a "STEPS LEFT" caption, the truth and error
    directly beneath it in a fixed-width comparison, and a draining countdown
    bar with ticks. Everything else is secondary and drawn smaller. The first
    version gave the prediction the same weight as the distance readouts, so
    the number the viewer is meant to be reading did not stand out from five
    other numbers.

    Colour encodes MEANING, not just magnitude:
      red     claims imminent success on a state that is not successful
      amber   off the truth by more than 25%
      green   within 25% of the truth, or genuinely successful
      grey    no truth exists (a trajectory that never succeeds)

    The canvas height is constant within a video — every optional line has a
    reserved slot — because a changing height makes the encoder reject the
    file outright.
    """
    from PIL import Image, ImageDraw

    scene = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    sw, sh = scene.size
    f_huge, f_big, f_med, f_sm, f_xs = _font(52), _font(20), _font(15), _font(12), _font(10)

    p = float(preds[i]) if i < len(preds) else float("nan")
    t = float(truth[i]) if (truth is not None and i < len(truth)) else None
    succeeded_yet = success_step is not None and i >= success_step
    optimistic = p < danger_threshold

    if succeeded_yet:
        col = (120, 230, 140)
    elif optimistic and t is None:
        col = (255, 80, 80)                       # imminent success that never comes
    elif t is None:
        col = (215, 215, 225)
    else:
        rel = abs(p - t) / max(t, 1.0)
        col = ((120, 230, 140) if rel <= 0.25 else
               (235, 190, 90) if rel <= 0.75 else (255, 110, 90))

    pad, gap = 12, 8
    x = sw + pad
    inner = PANEL_W - 2 * pad

    # ---- fixed-height layout, measured top-down --------------------------
    head_h = 16 if label else 0
    hero_h = 14 + 54 + 14                          # caption + figure + unit line
    cmp_h = 2 * 18                                 # truth + error (or placeholders)
    bar_h = 26
    bar_caption_h = 12                             # the "true" tick caption below the bar
    sec_h = 4 * 18                                 # step r, return, hand, peg
    status_h = 2 * 15
    panel_h = (pad + head_h + hero_h + cmp_h + gap + bar_h + bar_caption_h + gap
               + sec_h + status_h + gap + TRACE_H + pad)
    H = max(sh, panel_h)

    out = Image.new("RGB", (sw + PANEL_W, H), (12, 12, 15))
    out.paste(scene, (0, 0))
    d = ImageDraw.Draw(out)
    y = pad

    if label:
        d.text((x, y), label[:44], fill=(140, 140, 152), font=f_sm)
        y += head_h

    # ---- the hero number -------------------------------------------------
    d.text((x, y), "PREDICTED STEPS LEFT", fill=(150, 150, 164), font=f_xs)
    y += 14
    txt = f"{p:,.1f}" if np.isfinite(p) else "--"
    d.text((x, y), txt, fill=col, font=f_huge)
    # unit tucked against the figure
    tw = d.textlength(txt, font=f_huge)
    d.text((x + tw + 8, y + 30), "steps", fill=(120, 120, 134), font=f_sm)
    y += 54 + 14

    # ---- truth / error, fixed two lines ----------------------------------
    if t is not None:
        err = p - t
        d.text((x, y), f"{'truth':<7}{t:>8,.1f}", fill=(185, 185, 196), font=f_med)
        y += 18
        ec = (150, 230, 160) if abs(err) <= max(0.1 * t, 2) else (235, 165, 120)
        d.text((x, y), f"{'error':<7}{err:>+8,.1f}", fill=ec, font=f_med)
        y += 18
    else:
        d.text((x, y), f"{'truth':<7}{'n/a':>8}", fill=(120, 120, 134), font=f_med)
        y += 18
        d.text((x, y), f"never succeeds (censor {censor_label:.0f})",
               fill=(120, 120, 134), font=f_xs)
        y += 18

    # ---- countdown bar ---------------------------------------------------
    y += gap
    hi = max(float(np.nanmax(preds)), 1.0)
    _bar(d, x, y, inner - 26, bar_h, (max(p, 0.0) / hi) if np.isfinite(p) else 0.0,
         col, hi=hi)
    if t is not None:
        # a tick where the TRUTH sits, so over/under-prediction is visible
        tx = x + int((inner - 26) * min(max(t, 0.0) / hi, 1.0))
        d.line((tx, y - 3, tx, y + bar_h + 3), fill=(255, 255, 255), width=2)
        d.text((tx - 6, y + bar_h + 4), "true", fill=(200, 200, 210), font=f_xs)
    y += bar_h + gap + 12

    # ---- secondary readouts ----------------------------------------------
    tag = f"  ({reward_mode})" if reward_mode else ""
    if rewards is not None:
        if 0 < i <= len(rewards):
            r = float(rewards[i - 1])
            rc = (120, 220, 140) if r > 0 else (225, 160, 120) if r < 0 else (140, 140, 152)
            d.text((x, y), f"{'step r':<8}{r:>+8.2f}{tag}", fill=rc, font=f_sm)
        else:
            d.text((x, y), f"{'step r':<8}{'--':>8}{tag}", fill=(110, 110, 122), font=f_sm)
        y += 18
        cum = float(np.sum(rewards[:max(i, 0)]))
        d.text((x, y), f"{'return':<8}{cum:>+8.1f}", fill=(170, 170, 182), font=f_sm)
        y += 18
    else:
        y += 36
    if obs is not None and i < len(obs):
        hp = hand_peg_distance_curve(obs[:i + 1])[-1]
        pg = peg_goal_distance_curve(obs[:i + 1])[-1]
        d.text((x, y), f"{'hand>peg':<9}{hp:>7.3f}", fill=(160, 210, 165), font=f_sm)
        y += 18
        d.text((x, y), f"{'peg>goal':<9}{pg:>7.3f}", fill=(220, 185, 140), font=f_sm)
        y += 18
    else:
        y += 36

    # ---- status, two reserved lines --------------------------------------
    status = ([("SUCCESS", (120, 230, 140))] if succeeded_yet
              else [("claims imminent success", (255, 90, 90)),
                    ("- but it never comes", (255, 90, 90))] if optimistic
              else [(f"step {i}", (120, 120, 134))])
    for txt_, c_ in (status + [("", (0, 0, 0))] * 2)[:2]:
        d.text((x, y), txt_, fill=c_, font=f_sm)
        y += 15

    _draw_trace(d, preds, truth, i, (x, H - TRACE_H - pad, sw + PANEL_W - pad, H - pad),
                danger=danger_threshold)
    return np.array(out)


# ---- public rendering entry points --------------------------------------

def render_t2s_video(rollout, predict_t2s=None, save_path=None, reward_mode="difference_timed",
                     gamma=RL_GAMMA, time_penalty=TIME_PENALTY, fps=FPS,
                     danger_threshold=DANGER_THRESHOLD, label=None):
    """
    Annotate a rollout (from rollout_with_frames, or anything with "frames",
    "obs", "success_step") with the live T2S prediction and the live shaped
    reward, and optionally write it to `save_path`.

    predict_t2s: needed only if the rollout has no "t2s_preds" yet.

    Returns (frames, info) where info carries the prediction range and the
    minimum prediction reached on a trajectory that never succeeded — the
    false-optimism number.
    """
    frames = rollout.get("frames")
    if not frames:
        raise ValueError("rollout has no frames; call rollout_with_frames(...) "
                         "or visualize.rollout(..., render=True)")
    obs = rollout["obs"]
    preds = rollout.get("t2s_preds")
    if preds is None:
        if predict_t2s is None:
            raise ValueError("rollout has no t2s_preds and no predict_t2s was given")
        preds = np.array([predict_t2s(o) for o in obs], dtype=np.float64)

    success_step = rollout.get("success_step")
    truth = rollout.get("truth")
    if truth is None and success_step is not None:
        truth = steps_remaining_curve(len(obs), success_step)

    # the live reward uses the SAME arithmetic training does
    from reward_preview import compute_step_rewards
    rewards = compute_step_rewards(preds, reward_mode, time_penalty=time_penalty, gamma=gamma)
    if success_step is not None and len(rewards):
        rewards = rewards.copy()
        rewards[success_step:] = 0.0      # match the wrapper's success latch

    n = min(len(frames), len(preds), len(obs))
    out = [compose_frame(frames[i], i, preds[:n], truth=truth, obs=obs,
                         rewards=rewards, success_step=success_step,
                         danger_threshold=danger_threshold, label=label,
                         reward_mode=reward_mode)
           for i in range(n)]

    info = dict(
        n_frames=n, seed=rollout.get("seed"), success_step=success_step,
        pred_min=float(preds[:n].min()), pred_max=float(preds[:n].max()),
        # only meaningful when the episode never succeeded
        false_optimism=(float(preds[:n].min()) if success_step is None else None),
        entered_danger_band=bool(success_step is None and preds[:n].min() < danger_threshold),
        reward_mode=reward_mode, gamma=gamma,
        total_return=float(np.sum(rewards)) if len(rewards) else 0.0,
    )
    if save_path:
        save_video(out, save_path, fps=fps)
        info["path"] = save_path
    return out, info


def render_model_comparison_video(rollout, predictors, save_path=None, fps=FPS,
                                  danger_threshold=DANGER_THRESHOLD):
    """
    One video, every model's prediction on the SAME stored frames, stacked as
    a labelled column of readouts plus a shared trace panel.

    This is the comparison Stage 4's bar chart cannot make: all models score
    identical states, so a divergence between their numbers is entirely the
    model. On a rollout that never succeeds, whichever readout turns red is
    the model that would have paid the policy to keep doing that.
    """
    from PIL import Image, ImageDraw

    frames = rollout.get("frames")
    if not frames:
        raise ValueError("rollout has no frames")
    obs = rollout["obs"]
    success_step = rollout.get("success_step")
    truth = rollout.get("truth")
    if truth is None and success_step is not None:
        truth = steps_remaining_curve(len(obs), success_step)

    curves = {name: np.array([fn(o) for o in obs], dtype=np.float64)
              for name, fn in predictors.items()}
    n = min([len(frames), len(obs)] + [len(c) for c in curves.values()])
    hi = max(1.0, max(float(c[:n].max()) for c in curves.values()))
    colors = [(90, 200, 255), (255, 170, 90), (150, 230, 150), (230, 140, 230),
              (240, 230, 120), (180, 180, 255)]

    panel_w = 340
    out_frames = []
    for i in range(n):
        scene = Image.fromarray(np.asarray(frames[i], dtype=np.uint8)).convert("RGB")
        sw, sh = scene.size
        canvas = Image.new("RGB", (sw + panel_w, max(sh, 60 + 26 * len(curves) + TRACE_H)),
                           (12, 12, 15))
        canvas.paste(scene, (0, 0))
        d = ImageDraw.Draw(canvas)
        x, y = sw + 12, 10
        succeeded_yet = success_step is not None and i >= success_step
        d.text((x, y), f"step {i}" + ("   SUCCESS" if succeeded_yet else ""),
               fill=(150, 150, 160), font=_font(13)); y += 20
        if truth is not None and i < len(truth):
            d.text((x, y), f"truth {float(truth[i]):7.1f}", fill=(200, 200, 210),
                   font=_font(14)); y += 22
        else:
            d.text((x, y), "truth     n/a (never succeeds)", fill=(150, 150, 160),
                   font=_font(12)); y += 22

        for k, (name, c) in enumerate(curves.items()):
            p = float(c[i])
            bad = p < danger_threshold and not succeeded_yet
            col = (255, 80, 80) if bad else colors[k % len(colors)]
            d.text((x, y), f"{name[:20]:<20}{p:7.1f}", fill=col, font=_font(13))
            bx = x + 245
            d.rectangle((bx, y + 3, bx + 80, y + 12), outline=(60, 60, 70))
            d.rectangle((bx, y + 3, bx + int(80 * min(max(p, 0) / hi, 1.0)), y + 12), fill=col)
            y += 24

        # shared trace
        tx0, ty0 = sw + 12, canvas.size[1] - TRACE_H - 10
        tx1, ty1 = sw + panel_w - 12, canvas.size[1] - 10
        d.rectangle((tx0, ty0, tx1, ty1), fill=(18, 18, 22))
        yb = ty1 - int((ty1 - ty0) * min(danger_threshold / hi, 1.0))
        d.rectangle((tx0, yb, tx1, ty1), fill=(60, 20, 20))
        if truth is not None and len(truth) > 1:
            pts = [(tx0 + int((tx1 - tx0) * k / (n - 1)),
                    ty1 - int((ty1 - ty0) * min(max(truth[k], 0) / hi, 1.0)))
                   for k in range(min(i + 1, len(truth)))]
            if len(pts) > 1:
                d.line(pts, fill=(120, 120, 130), width=1)
        for k, (_name, c) in enumerate(curves.items()):
            pts = [(tx0 + int((tx1 - tx0) * j / (n - 1)),
                    ty1 - int((ty1 - ty0) * min(max(c[j], 0) / hi, 1.0)))
                   for j in range(i + 1)]
            if len(pts) > 1:
                d.line(pts, fill=colors[k % len(colors)], width=2)
        out_frames.append(np.array(canvas))

    info = {name: dict(pred_min=float(c[:n].min()), pred_max=float(c[:n].max()),
                       falsely_optimistic=bool(success_step is None
                                                and c[:n].min() < danger_threshold))
            for name, c in curves.items()}
    if save_path:
        save_video(out_frames, save_path, fps=fps)
    return out_frames, info


def _pad_to_macroblock(frames, block=16):
    """
    Pad every frame to one common size, rounded up to a multiple of `block`.

    Two reasons. H.264 encoders want macroblock-aligned dimensions, and
    imageio otherwise silently RESIZES, resampling the text in the readout
    panel. And frames must all be identical in size or the encoder refuses
    outright — padding to the per-batch MAXIMUM rather than to the first
    frame's size makes that impossible to get wrong from the caller's side.
    """
    arrs = [np.asarray(f) for f in frames]
    H = max(a.shape[0] for a in arrs)
    W = max(a.shape[1] for a in arrs)
    H += (-H) % block
    W += (-W) % block
    if all(a.shape[0] == H and a.shape[1] == W for a in arrs):
        return arrs
    return [np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1]), (0, 0)), mode="constant")
            for a in arrs]


def save_video(frames, path, fps=FPS):
    """Write frames to mp4 (or gif, by extension). Needs imageio."""
    import imageio

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    if not path.lower().endswith(".gif"):
        frames = _pad_to_macroblock(frames)
    imageio.mimsave(path, frames, fps=fps)
    return path


# ---- Stage-4 integration ------------------------------------------------

def evaluate_with_video(predict_fn, eval_success_policy, eval_failure_policy,
                        out_dir, label="model", seeds=(500,), n_metric_seeds=8,
                        gamma=RL_GAMMA, fps=FPS, reward_mode="difference_timed",
                        time_penalty=TIME_PENALTY):
    """
    Run the Stage-4 numeric evaluation AND render live-prediction videos from
    rollouts of the same held-out policies, into out_dir.

    n_metric_seeds=0 skips the numeric evaluation, for when Stage 2 has
    already run it and only the footage is wanted.

    Returns (report, videos) where `report` is what
    t2s_eval.run_full_evaluation returns plus a "videos" key, and `videos`
    maps "<scenario>_seed<N>" to the per-video info dict. The failure-scenario
    entries carry `false_optimism` (how low the prediction fell on a
    trajectory that never succeeded) and `entered_danger_band`, which is the
    numeric form of the thing the video shows.
    """
    import t2s_eval

    os.makedirs(out_dir, exist_ok=True)
    report = (t2s_eval.run_full_evaluation(
                  predict_fn, eval_success_policy, eval_failure_policy,
                  n_seeds=n_metric_seeds)
              if n_metric_seeds else {})

    videos = {}
    for scenario, policy in (("success", eval_success_policy),
                             ("failure", eval_failure_policy)):
        for sd in seeds:
            roll = rollout_with_frames(policy, predict_t2s=predict_fn, seed=sd)
            path = os.path.join(out_dir, f"t2s_{label}_{scenario}_seed{sd}.mp4")
            _frames, info = render_t2s_video(
                roll, save_path=path, gamma=gamma, fps=fps, reward_mode=reward_mode,
                time_penalty=time_penalty, label=f"{label} | {scenario}")
            videos[f"{scenario}_seed{sd}"] = info
            print(f"  wrote {os.path.basename(path)}  "
                  f"pred {info['pred_min']:.1f}-{info['pred_max']:.1f}"
                  + ("  >>> FALSELY OPTIMISTIC" if info["entered_danger_band"] else ""))
    report["videos"] = videos
    return report, videos


def display_in_notebook(path, width=640):
    """Inline a written video in Jupyter."""
    from IPython.display import Video, display
    return display(Video(path, embed=True, width=width))