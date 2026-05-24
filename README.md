# Franka EE Tracking — Residual PPO

7-DoF Franka Panda end-effector tracking in MuJoCo.  
A damped-least-squares IK baseline handles reactive tracking; a PPO residual policy learns to correct what IK can't model (obs noise, action delay, near-singular geometry).

## Repository layout

```
ee_tracking/
  env/
    franka_tracking_env.py   # Gym env — obs / action / reward / architecture
    ik_controller.py         # DLS inverse-kinematics baseline
    trajectories.py          # circle, figure-8, moving_target, unreachable
    disturbances.py          # obs noise, joint noise, action delay
  configs/
    default.yaml             # tuned env + PPO hyperparameters
assets/
  mujoco_menagerie/franka_emika_panda/   # Franka XML + meshes

train.py                     # train residual PPO from a config YAML
evaluate.py                  # IK-vs-residual ablation + rollout plots
sweep.py                     # overnight hyperparameter sweep (sequential)
eval_ema.py                  # post-hoc EMA smoothing sweep on trained models
eval_posthoc.py              # post-hoc filter comparison (Butterworth, deadzone, …)
```

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install mujoco gymnasium "stable-baselines3>=2.3" torch numpy \
               matplotlib pyyaml scipy tqdm tensorboard
```

Franka assets (if not already present):
```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
```

## Architecture

```
policy output (7-D, [-1,1])
    ↓  2nd-order Butterworth @ 2 Hz  (baked into env during training)
    ↓  × residual_scale (0.05 rad/s)
    ↓  action-delay buffer (1 step = 20 ms)
    ↓
q_setpoint(t) = q_ik(t) + filtered_residual(t) × dt
                └─ integrates freely, never contaminated by residual
```

The residual is a **non-accumulating position offset**: a wrong correction at step `t` is fully self-correcting at `t+1` without IK needing to fight drift.  Setting action → 0 instantly degrades to pure IK.

## Performance (moving_target, settled RMSE)

| System | mm |
|---|---|
| IK only | 15.6 |
| Residual PPO (raw) | ~15.5 |
| + post-hoc Butterworth 2 Hz | ~14.0 |
| Residual PPO (baked Butterworth, retrained) | TBD |

`moving_target` is the primary benchmark — the target follows a band-limited random walk (0.05–0.15 Hz), so IK lags significantly without lookahead or delay compensation.

## Quick start

**Train** (saves model + VecNormalize stats + TensorBoard logs):
```bash
python train.py --out results/run1
# or override timesteps:
python train.py --timesteps 3_000_000 --out results/run1
```

**Evaluate** (IK vs residual table + plots):
```bash
python evaluate.py ablation --model results/run1/final_model.zip
python evaluate.py rollout  --model results/run1/final_model.zip --trajectory moving_target
```

**Hyperparameter sweep** (~10 h on CPU, resumable):
```bash
python sweep.py --out results/sweep
python sweep.py --timesteps 500_000   # quick smoke-test
```

**Post-hoc filter experiments** (no retraining needed):
```bash
# EMA alpha grid across all sweep models
python eval_ema.py --sweep-dir results/sweep

# Full filter comparison (Butterworth, deadzone, error-gain, obs-smooth)
python eval_posthoc.py --models rs012_wr05 rs008_wr01
```

## Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `residual_scale` | 0.05 rad/s | Max per-joint residual; keep small so IK dominates |
| `w_residual` | 0.5 | Quadratic penalty on residual magnitude |
| `action_filter_hz` | 2.0 | Butterworth cutoff; trains and evals with same filter |
| `residual_scale × w_residual` | — | Key interaction: higher scale needs higher weight |
| `lookahead_horizon` | 5 | Future target positions in obs (× lookahead_dt = 0.5 s ahead) |
| `act_delay` | 1 step (20 ms) | Action delay modelling actuator lag |

See `ee_tracking/configs/default.yaml` for all values.
