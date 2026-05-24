# Franka EE Tracking — Residual PPO

7-DoF Franka Panda end-effector tracking in MuJoCo.  
A damped-least-squares IK baseline handles reactive tracking; a PPO residual policy learns to compensate for what IK cannot model: a whole-pipeline command delay that makes reactive control systematically lag behind the target.

## Repository layout

```
ee_tracking/
  env/
    franka_tracking_env.py   # Gym env — obs / action / reward / architecture
    ik_controller.py         # DLS inverse-kinematics baseline
    trajectories.py          # circle, figure-8, moving_target, unreachable
    disturbances.py          # obs noise, joint noise, command delay
  configs/
    default.yaml             # tuned env + PPO hyperparameters
assets/
  mujoco_menagerie/franka_emika_panda/   # Franka XML + meshes

train.py                     # train residual PPO from a config YAML
evaluate.py                  # IK-vs-residual ablation + rollout plots
sweep.py                     # overnight hyperparameter sweep (sequential)
probe.py                     # short diagnostic runs to compare configs quickly
eval_ema.py                  # post-hoc EMA smoothing sweep (legacy, pre-delay arch)
eval_posthoc.py              # post-hoc filter comparison (legacy, pre-delay arch)
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
    ↓
q_set(t) = clip(q_ik(t) + filtered_residual(t) × dt, limits)
    ↓
    [  cmd_delay-step FIFO  ]   ← IK + residual travel through the same buffer
    ↓
ctrl(t) = q_set(t − cmd_delay)          sent to MuJoCo actuators
```

**Why the delay is the key disturbance:**  
Both IK and the residual pass through the same `cmd_delay`-step FIFO (default 5 steps = 100 ms), modelling the full sensor-to-actuator round-trip (network latency, controller loop). IK, being purely reactive, commands based on the *current* measured position — which will be stale by the time the command executes. A pure IK controller with 100 ms delay degrades from ~18 mm to ~44 mm on `moving_target`.

The residual policy observes `cmd_delta_history` (the setpoints currently in the FIFO) and a 0.5 s lookahead of future target positions. The lookahead window exactly covers the delay (`5 steps × 0.1 s = 0.5 s`), so the policy can read exactly where the target will be when each queued command executes and add the right predictive correction. An untrained policy (action ≈ 0) degrades gracefully to IK-with-delay.

**Non-accumulating residual:**  
The IK setpoint integrates freely every step; the residual is applied as a one-step position offset, not accumulated. A wrong correction at step `t` is fully self-correcting at `t+1` without the IK path being contaminated.

```
q_ik(t)  = clip(q_ik(t-1) + ik_qdot(t) × dt, limits)   # IK integrates freely
q_set(t) = clip(q_ik(t)   + residual(t) × dt, limits)   # one-step offset
ctrl(t)  = q_set(t − cmd_delay)                          # whole pipeline delayed
```

## Observation space

```
dim   field
 7    q                — joint positions (+ obs_jnt_noise)
 7    qdot             — joint velocities
 3    ee_pos           — measured EE position (+ obs_pos_noise)
 3    ee_pos_error     — target − ee_pos
 3    target_vel       — desired EE velocity (analytic derivative)
 7    ik_qdot          — current IK joint-velocity command
3×H   lookahead_pos    — future target positions at lookahead_dt intervals (H=5)
7×D   cmd_delta_hist   — pending setpoints minus current q, oldest→newest (D=5)
|P|   traj_onehot      — trajectory type one-hot (reduces critic variance)
```

`cmd_delta_hist` is the Markov-restoring element: with a D-step delay the robot
is executing q_set(t−D), so the policy must know q_set(t−D+1…t) to avoid
redundant corrections.

## Performance (moving_target)

| System | RMSE (mm) | Notes |
|---|---|---|
| IK only, no delay | ~18 | baseline without pipeline delay |
| IK only, 100 ms delay | ~44 | realistic delay; this is what residual must beat |
| Residual PPO, 100 ms delay | TBD | training in progress (`results/run_delay5`) |

`moving_target` uses a band-limited random walk (0.05–0.15 Hz, 8–14 cm extent).
IK's lag grows with target speed; at 0.15 Hz and 14 cm the delay-induced error
is ~40 mm — a large, learnable signal for the residual policy.

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

**Diagnostic probes** (short runs to compare configs, ~3 min/probe):
```bash
python probe.py                          # run all probes at 300k steps
python probe.py --only wr020 rs010       # specific probes
python probe.py --timesteps 150000       # quick smoke-test
```

**Hyperparameter sweep** (~10 h on CPU, resumable):
```bash
python sweep.py --out results/sweep
python sweep.py --timesteps 500_000     # quick smoke-test
```

## Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `cmd_delay` | 5 steps (100 ms) | Whole-pipeline delay; both IK and residual equally delayed |
| `residual_scale` | 0.05 rad/s | Max per-joint residual; keep small so IK dominates at zero action |
| `w_residual` | 0.5 | Quadratic penalty on residual magnitude |
| `action_filter_hz` | 2.0 Hz | 2nd-order Butterworth baked into env (trains and evals together) |
| `lookahead_horizon` | 5 | Future target positions in obs; covers the full delay window |
| `lookahead_dt` | 0.10 s | Spacing between lookahead samples (5 × 0.1 s = 0.5 s = delay window) |
| `obs_pos_noise` | 5 mm | Gaussian noise on EE measurement (affects IK and policy obs equally) |

See `ee_tracking/configs/default.yaml` for all values.

## Design decisions and dead ends

**Why total command delay, not residual-only delay?**  
An earlier version delayed only the residual correction, leaving IK undelayed. This gave IK an unfair advantage (it acted instantly; residual arrived 20 ms late), meaning the residual had to fight its own delay just to break even. Probe experiments confirmed the policy made zero progress (std ≈ 1.08 across all hyperparameter variants at 300k steps) — a task problem, not a tuning problem. Delaying the whole pipeline levels the playing field and creates a clear, learnable advantage for the predictive policy.

**Why not velocity-additive residual?**  
An earlier formulation accumulated IK and residual velocities together:
`q_setpoint += (ik_qdot + residual) × dt`. A wrong residual at step `t` drifted
into subsequent steps; 60 steps of max residual → 0.144 rad of accumulated error.
The current non-accumulating offset bounds the worst-case correction to
`residual_scale × dt = 0.001 rad` per step.

**Why `moving_target` only in the training pool?**  
Circle and figure-8 are too easy for IK (smooth, predictable, near-default config).
Training on them provides weak gradient signal. `moving_target` is the hard case
where delay matters most; other trajectories are used for eval only.

**Why lookahead and not learned prediction?**  
The lookahead gives the policy oracle future knowledge, which is unrealistic for
real hardware. The goal here is to verify the residual architecture works before
replacing the oracle with a learned predictor or Kalman filter.
