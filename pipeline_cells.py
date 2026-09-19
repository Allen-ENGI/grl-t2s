"""
Replacement pipeline.ipynb cells, as copy-pasteable blocks.

Only the cells whose API changed are here. Each block is delimited by a
`# %% [n] ...` marker naming the section it replaces. Everything that decides
a parameter now reads from config, so a value appears in exactly one place.
"""

# %% [5+6] Section 2 — data collection -------------------------------------
SECTION_2 = r'''
import data_collection, results, config
from env_utils import ensure_fixed_task
ensure_fixed_task()

DATA_RUN_NAME = 'v2'                 # bump: v1 predates the stored train/val split

# Checkpoints RESERVED for Stage 4 and therefore NOT collected from. These are
# the two Section 4 loads. "collect from every checkpoint" and "evaluate on
# policies that generated none of the training data" cannot both hold for the
# same file, so the reservation is explicit and gets recorded in
# dataset_summary.json for t2s_eval.assert_policies_held_out to check.
HOLDOUT_CKPTS = ['peg_insert_side_v3_700000.zip',    # eval_success_pol
                 'peg_insert_side_v3_200000.zip']    # eval_failure_pol

COLLECTION_SEEDS = [0, 1, 2]         # each re-runs the whole plan independently
VAL_FRACTION     = 0.2               # stratified by (source, succeeded), duplicate-safe
EPISODES_PER_CKPT = 40   # per checkpoint per seed

data_run_dir = results.new_run_dir(
    'data_collection', DATA_RUN_NAME,
    meta=dict(seeds=COLLECTION_SEEDS, val_fraction=VAL_FRACTION,
              holdout=HOLDOUT_CKPTS))

# every discovered checkpoint except the holdout; `deterministic` is gone —
# the policy is always sampled, which is the only source of trajectory
# variation on a pinned scene
summary = data_collection.collect(
    data_run_dir, holdout=HOLDOUT_CKPTS, episodes=EPISODES_PER_CKPT,
    seeds=COLLECTION_SEEDS, val_fraction=VAL_FRACTION)
'''

# %% [11] Section 2b — audit ------------------------------------------------
SECTION_2B = r'''
import dataset_audit, os

audit = dataset_audit.audit_dataset(
    os.path.join(data_run_dir, 'dataset.npz'),
    summary_path=os.path.join(data_run_dir, 'dataset_summary.json'))

CHECKS = ('success_boundary', 'failure_coverage', 'exploration_coverage',
          'object_movement', 'label_sanity', 'split_integrity',
          'trajectory_uniqueness')
flagged = [k for k in CHECKS if audit[k]['verdict'] != 'OK']
print(f">>> {len(flagged)} check(s) flagged: {flagged}" if flagged else '>>> all checks passed')
'''

# %% [16+17] Section 3 — T2S training --------------------------------------
SECTION_3 = r'''
import t2s_train, results, config, torch, os

T2S_RUN_NAME = 'v3'
# 'bootstrap' is the default now: the truncated frame is left unsupervised and
# bootstrapped through, instead of being regressed onto a fabricated 300.
# Pass both to rank them side by side.
CENSOR_SCHEMES = ('bootstrap',)
METHODS    = ('td0', 'tdlambda')
CONDITIONS = ('succ', 'all')
T2S_SEEDS_TRAIN = (0, 1, 2)          # 3 seeds here too, so val MSE gets a std

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

t2s_run_dir = results.new_run_dir(
    't2s_model', T2S_RUN_NAME,
    meta={'data_run': DATA_RUN_NAME, 'censor_schemes': list(CENSOR_SCHEMES),
          'bootstrap_gamma': config.T2S_BOOTSTRAP_GAMMA})

# epochs / patience / batch_size are constants at the top of t2s_train.py now
models, histories, summary_rows = t2s_train.run_all_combos(
    t2s_run_dir, os.path.join(data_run_dir, 'dataset.npz'),
    methods=METHODS, conditions=CONDITIONS, seeds=T2S_SEEDS_TRAIN,
    censor_schemes=CENSOR_SCHEMES,
    # NOTE: this is T2S_BOOTSTRAP_GAMMA (fixed point 1/(1-g) = 100 for the
    # never-succeeding recursion), NOT config.RL_GAMMA. Different jobs — see
    # config. Do not "unify" them.
    bootstrap_gamma=config.T2S_BOOTSTRAP_GAMMA, device=DEVICE)

for row in sorted(summary_rows, key=lambda r: r['mean_val_mse']):
    print(f"{row['combo']:<20}{row['censor_scheme']:<11}"
          f"gamma={row['gamma']:<6.2f}val MSE {row['mean_val_mse']:8.2f} "
          f"± {row['std_val_mse']:.2f}")
'''

# %% [19-23] Section 4 — held-out evaluation -------------------------------
SECTION_4 = r'''
import json, os
import t2s_eval, t2s_io, results
from stable_baselines3 import SAC

SUCC_CKPT = os.path.join(config.EXPERT_POLICY_DIR, 'peg_insert_side_v3_700000.zip')
FAIL_CKPT = os.path.join(config.EXPERT_POLICY_DIR, 'peg_insert_side_v3_200000.zip')

# verify the held-out claim instead of trusting it. Raises if either policy was
# collected from, or was not declared as holdout during collection.
print(t2s_eval.assert_policies_held_out(
    os.path.join(data_run_dir, 'dataset_summary.json'), [SUCC_CKPT, FAIL_CKPT]))

eval_success_pol, eval_failure_pol = SAC.load(SUCC_CKPT), SAC.load(FAIL_CKPT)

EVAL_RUN_NAME = 'v3'
eval_run_dir = results.new_run_dir('t2s_eval', EVAL_RUN_NAME,
                                   meta={'t2s_run': T2S_RUN_NAME})

def rows_from_manifest(run_dir):
    """Rebuild summary_rows from a run's manifest.json — survives a restart."""
    man = t2s_io.load_manifest(run_dir)
    return [dict(combo=c, best_seed=i['best_seed'], mean_val_mse=i['mean_val_mse'],
                 std_val_mse=i['std_val_mse'], seeds_val_mse=i['per_seed_val_mse'],
                 mean_val_mse_succ_only=i['mean_val_mse_succ_only'],
                 censor_scheme=i.get('censor_scheme'), gamma=i.get('gamma'))
            for c, i in man['combos'].items()]

# dataset_path adds the held-out-ROWS evaluation on the stored val split —
# thousands of states, versus 8 rollouts, and its correlation is pooled across
# many episodes by construction
reports = t2s_eval.evaluate_runs(
    t2s_run_dir, summary_rows, eval_success_pol, eval_failure_pol,
    n_seeds=8, dataset_path=os.path.join(data_run_dir, 'dataset.npz'))

with open(os.path.join(eval_run_dir, 'eval_report.json'), 'w') as f:
    json.dump(reports, f, indent=2)

print()
print(t2s_eval.format_report_table(reports, sort_by='mae'))
_ = t2s_eval.plot_combo_comparison(
    reports, sort_by='mae', save_path=os.path.join(eval_run_dir, 'combo_comparison.png'))
'''

# %% [NEW] Section 4d — live T2S prediction videos -------------------------
SECTION_4D = r'''
import os
import t2s_video, t2s_predict

VIDEO_DIR = os.path.join(eval_run_dir, 'videos')
VIDEO_SEEDS = (500,)

# one video per (model, scenario): the scene with the live prediction, the
# ground-truth countdown, the live shaped reward, and hand->peg / peg->goal
# side by side — so "the model said 6 steps left" can be read against "the
# gripper was empty and the peg never moved".
video_info = {}
for row in summary_rows:
    method, cond = row['combo'].rsplit('_', 1)
    predict_fn = t2s_predict.load_t2s_predictor(t2s_run_dir, method, cond,
                                                seed=row['best_seed'])
    print(f"{row['combo']}:")
    _rep, vids = t2s_video.evaluate_with_video(
        predict_fn, eval_success_pol, eval_failure_pol, VIDEO_DIR,
        label=row['combo'], seeds=VIDEO_SEEDS, n_metric_seeds=0,
        reward_mode=REWARD_MODE, gamma=config.RL_GAMMA)
    video_info[row['combo']] = vids

# every model on the SAME frames, so a divergence is the model not the rollout
roll = t2s_video.rollout_with_frames(eval_failure_pol, seed=VIDEO_SEEDS[0])
_frames, cmp_info = t2s_video.render_model_comparison_video(
    roll, PREDICTORS, save_path=os.path.join(VIDEO_DIR, 'all_models_failure.mp4'))
for name, i in sorted(cmp_info.items(), key=lambda kv: kv[1]['pred_min']):
    print(f"  {name:<24} min pred {i['pred_min']:7.1f}"
          f"{'   FALSELY OPTIMISTIC' if i['falsely_optimistic'] else ''}")

t2s_video.display_in_notebook(os.path.join(VIDEO_DIR, 'all_models_failure.mp4'))
'''

# %% [33] Section 5 — downstream RL config ---------------------------------
SECTION_5 = r'''
import t2s_predict, t2s_io, results, config

MODELS = [('v3', 'td0boot_all'), ('v3', 'tdlambdaboot_all')]
SEEDS       = list(config.RL_SEEDS)         # (0, 1, 2)
REWARD_MODE = 'difference_timed'
TIMESTEPS   = 400_000
SWEEP_NAME  = 'sweep_v6'

MODEL_SPECS, PREDICTORS = {}, {}
for run_name, combo in MODELS:
    run_dir = results.get_run_dir('t2s_model', run_name)
    info = t2s_io.load_manifest(run_dir)['combos'][combo]      # KeyError if absent
    label = f'{combo}_{run_name}'
    method, condition = combo.rsplit('_', 1)
    PREDICTORS[label] = t2s_predict.load_t2s_predictor(
        run_dir, method, condition, seed=info['best_seed'])
    MODEL_SPECS[label] = (run_dir, combo, info['best_seed'])
    print(f"  {label:26s} seed {info['best_seed']}  t2s gamma={info.get('gamma')}")

print(f"\n{len(MODELS)} model(s) x {len(SEEDS)} seed(s) = {len(MODELS)*len(SEEDS)} runs "
      f"@ {TIMESTEPS:,} steps, reward_mode={REWARD_MODE!r}, RL gamma={config.RL_GAMMA}")
'''

# %% [35+36] Section 6 — reward preview ------------------------------------
SECTION_6 = r'''
import os
import reward_preview, t2s_eval, config

# SEVERAL seeds per scenario: return_gap used to be one success/failure pair,
# and the table sorted every model by it
trajectories = t2s_eval.collect_eval_trajectories(
    eval_success_pol, eval_failure_pol, seeds=(500, 501, 502))
for scenario, trajs in trajectories.items():
    print(f"{scenario}: " + ", ".join(
        f"seed{t['seed']} {len(t['obs'])} steps succ@{t['success_step']}" for t in trajs))
print()

preview = reward_preview.preview_reward(
    trajectories, PREDICTORS,
    reward_modes=('absolute', 'difference', 'difference_timed'),
    gamma=config.RL_GAMMA)
print(reward_preview.format_preview_table(preview))

# go/no-go, checked rather than eyeballed. `hover r` is the one to watch: with
# the (correct) gamma in the shaping term, difference_timed pays
# (1-gamma)*pred - time_penalty for standing still, so at gamma=0.99 a model
# predicting 200 is paid +1.0/step to freeze. config.RL_GAMMA=0.999 is chosen
# to keep this negative.
print(f"\n--- go/no-go for {REWARD_MODE} at gamma={config.RL_GAMMA} ---")
verdicts = reward_preview.check_preview(preview, REWARD_MODE)
PRED_MAXES = {}
for label, v in verdicts.items():
    PRED_MAXES[label] = v['pred_max']
    print(f"  {label:26s} {'PASS' if v['ok'] else 'FAIL'}  "
          f"gap={v['return_gap']:,.0f}  hover={v['hover_reward']:+.2f}")
    for r in v['reasons']:
        print(f"      - {r}")

reward_preview.plot_reward_preview(
    trajectories, PREDICTORS, reward_mode=REWARD_MODE, gamma=config.RL_GAMMA,
    save_path=os.path.join(eval_run_dir, f'reward_preview_{REWARD_MODE}.png'))
'''

# %% [38] Section 7 — train ------------------------------------------------
SECTION_7 = r'''
import json, os
import policy_train, results, config

# one call, not a loop: run_sweep takes the MODEL_SPECS dict directly, so run
# dirs are {sweep}_{label}_seed{n} instead of the doubled-up combo names the
# per-model loop produced
sweep_results = policy_train.run_sweep(
    MODEL_SPECS, results, seeds=SEEDS, sweep_name=SWEEP_NAME,
    reward_mode=REWARD_MODE, total_timesteps=TIMESTEPS, gamma=config.RL_GAMMA,
    pred_maxes=PRED_MAXES,          # real prediction range -> real hovering check
    skip_existing=True, ckpt_freq=10_000, eval_freq=10_000, n_eval_episodes=5)

sweep_rows = policy_train.summarize_sweep(sweep_results)
print()
print(policy_train.format_sweep_table(sweep_rows))

sweep_dir = results.new_run_dir('policy', SWEEP_NAME,
                                meta={'kind': 'sweep_summary', 'models': MODELS,
                                      'seeds': SEEDS, 'reward_mode': REWARD_MODE,
                                      'gamma': config.RL_GAMMA, 'timesteps': TIMESTEPS})
with open(os.path.join(sweep_dir, 'sweep_summary.json'), 'w') as f:
    json.dump({'rows': sweep_rows,
               'raw': {f'{c}|{s}': h for (c, s), h in sweep_results.items()}}, f, indent=2)
print(f'\nsaved to {sweep_dir}')
'''

# %% [49+50] Section 8 — inspect a trained policy, with live T2S video -----
SECTION_8 = r'''
import os
import visualize, progress, t2s_video, results, config

CHECK_LABEL = sweep_rows[0]['label']
CHECK_RUN   = f'{SWEEP_NAME}_{CHECK_LABEL}_seed{SEEDS[0]}'
CHECK_CKPT  = 'policy_final.zip'

check_dir = results.get_run_dir('policy', CHECK_RUN)
policy, resolved = visualize.load_policy(CHECK_CKPT, check_dir)
predict_fn = PREDICTORS[CHECK_LABEL]        # the exact reward it trained against
print('loaded:', resolved)

roll = t2s_video.rollout_with_frames(policy, predict_t2s=predict_fn, seed=0)
path = os.path.join(check_dir, 'check_t2s_live.mp4')
_frames, info = t2s_video.render_t2s_video(
    roll, save_path=path, reward_mode=REWARD_MODE, gamma=config.RL_GAMMA,
    label=f'{CHECK_LABEL} | policy')
m = progress.compute_progress_metrics(roll['obs'], roll['success_step'])
print(f"  succeeded={m['succeeded']} at {roll['success_step']}  "
      f"hand->peg={m['min_hand_peg_distance']:.4f}  peg_moved={m['peg_moved']}")
print(f"  pred range {info['pred_min']:.1f}-{info['pred_max']:.1f}"
      + ('   >>> FALSELY OPTIMISTIC' if info['entered_danger_band'] else ''))

visualize.plot_rollouts([roll], labels=[CHECK_LABEL],
                        save_path=os.path.join(check_dir, 'check_t2s_curves.png'))
t2s_video.display_in_notebook(path)
'''

ALL = {k: v for k, v in sorted(globals().items())
       if k.startswith("SECTION_") and isinstance(v, str)}

if __name__ == "__main__":
    for name, src in ALL.items():
        print("=" * 72)
        print(f"# {name}")
        print("=" * 72)
        print(src.strip())
        print()