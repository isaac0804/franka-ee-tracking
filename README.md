# Franka EE Tracking — Residual PPO with Delay-Aware Transformer

7-DoF Franka Panda end-effector tracking in MuJoCo, trained with residual PPO on top of a damped-least-squares IK baseline. The core challenge is a **5-step (100 ms) whole-pipeline command delay** that causes reactive controllers to systematically lag the target. A transformer policy with delay-aware observations learns predictive corrections the IK cannot make.

![Tracking animation](results/figures/tracking_3d_circle.gif)

---

## Results

![RMSE comparison](results/figures/comparison_5M_bars.png)

Settled RMSE (mm) — lower is better. Shaded regions mark out-of-distribution trajectories never seen during training. Error bars: ±SEM for stochastic tasks (MT 10 seeds, Step 30 seeds); inter-seed range for deterministic. †=1 seed.

| Model | Steps | Moving Target† | Circle | Figure-8 |
|---|---|---|---|---|
| IK baseline (no delay) | — | ~18 mm | ~8 mm | ~4 mm |
| **IK baseline (100 ms delay)** | — | **38.1 mm** | **12.1 mm** | **7.7 mm** |
| MLP | 5M | 21.0 mm | 7.6 mm | 7.0 mm |
| MLP | 10M | 16.0 mm | 5.3 mm | 4.7 mm |
| **Transformer** | **5M** | **19.7 mm** | **5.3 mm** | **6.0 mm** |

**Key result:** At just 300k steps, the transformer already matches MLP at 10M on periodic trajectories — a **33× step efficiency advantage** (see step-efficiency figure below).

![Step efficiency](results/figures/efficiency_curve.png)

---

### Rigorous 5M comparison (multi-seed)

For the final 5M models, moving-target RMSE is averaged over 10 random-walk seeds; circle and figure-8 are deterministic. Smoothness metrics are computed over the settled portion of each episode.

| Model | Moving Target | Circle | Figure-8 |
|---|---|---|---|
| IK (100 ms delay) | 48.6 ± 8.0 mm | 11.5 mm | 7.7 mm |
| MLP 5M (mean, 2 seeds) | 19.6 mm | 7.9 mm | 6.7 mm |
| **Transformer 5M (mean, 2 seeds)** | **20.6 ± 3.3 mm** | **4.8 mm** | **4.8 mm** |

| Model | Action roughness¹ | Saturation rate² |
|---|---|---|
| MLP 5M (mean, 2 seeds) | 0.796 / 0.472 / 0.520 | 44.8% / 56.5% / 58.0% |
| **Transformer 5M (mean, 2 seeds)** | **0.624 / 0.242 / 0.233** | **28.0% / 64.3% / 55.8%** |

¹ Mean \|a_t − a_{t−1}\| per joint per step (MT / CI / F8). Lower = smoother commands.  
² Fraction of (timestep × joint) pairs where \|action\| > 0.9 (MT / CI / F8). Lower = less bang-bang.

On periodic trajectories (circle, figure-8) the transformer is **39% and 28% more accurate** than the MLP at the same training budget. On the random-walk trajectory both are within noise (20.6 vs 19.6 mm). The transformer also produces smoother joint commands: 20–50% lower roughness across all trajectories.

---

### Out-of-distribution generalization

Traj-type one-hot is all-zeros at eval time (unknown trajectory type). Four OOD scenarios across two categories:

**OOD shapes** — same task, unseen geometry:

| Trajectory | IK | MLP 5M (mean) | **Transformer 5M** |
|---|---|---|---|
| Square (hard corners) | 10.4 mm | 7.5 mm | **4.2 mm** |
| Rectangle (asymmetric) | 10.9 mm | 8.6 mm | **5.1 mm** |

Transformer is **44% better on square, 41% on rectangle**. Hard corners are where the delay is most damaging — the fine lookahead sees the upcoming corner 100ms ahead and pre-steers.

**OOD task conditions** — same geometry, different regime:

| Trajectory | IK | MLP 5M (mean, 2 seeds) | **Transformer 5M (seed=42)** |
|---|---|---|---|
| Fast circle (2× speed) | 31.1 mm | 11.1 mm | **8.9 mm** |
| Step target (pick-and-place) | 61.2 ± 12.5 mm | 40.3 ± 10.7 mm | **41.0 ± 11.8 mm** |

Fast circle: transformer edges out MLP (−20%). Step target: **essentially tied** (41.0 vs 40.3 mm, σ≈11 mm — within seed-to-seed noise). Step target is a stochastic trajectory (random waypoint sequences); both models improve substantially over IK (~33%).

---

## Design Note

### State and action
The 95-D observation is structured around the delay. The key blocks are the **fine lookahead** (target positions at t+20ms … t+100ms, covering the exact 5-step delay window) and the **command history** (the 5 setpoints already queued in the FIFO). The command history is essential for the Markov property — without it the policy cannot tell whether IK is already compensating and stacks redundant corrections.

Actions are 7-D per-joint residuals in [−1, 1], scaled by `residual_scale = 0.12 rad` and added to the IK setpoint. A zero action always falls back to IK, so an untrained policy degrades gracefully.

### Reward
```
r = w_pos × (‖err_prev‖ − ‖err_now‖)   # reward progress, not absolute error
  − w_vel × ‖ee_vel‖                    # penalise unnecessary motion
  − w_residual × ‖action‖²              # small L2 regularisation on residual
```
No jerk or smoothness penalty. With a fixed 100 ms delay, the optimal strategy is a *predictive impulse* — inherently discontinuous. Penalising action changes would directly penalise the delay-compensation mechanism.

### Trajectory representation
Three types trained simultaneously in a mixed pool: **moving target** (band-limited random walk), **circle** (constant-speed orbit), **figure-8** (Lissajous curve). Single-trajectory training overfits — mixed training was the largest single improvement in the MLP phase and stabilises the critic across all architectures.

Each trajectory provides oracle future positions as a lookahead. Orientation is not tracked (position only).

### How tracking performance is evaluated
`residual_settled_rmse_mm` — RMSE of the 3-D EE position error over the *settled* portion of each episode (after 0.5 s warmup), averaged per trajectory type. Settled RMSE separates steady-state tracking from transient startup and is the primary reported metric throughout.

For stochastic trajectories (moving target, step target) results are averaged over 10 random seeds with ± std reported. Smoothness is quantified via **action roughness** (mean |a_t − a_{t−1}| per joint) and **saturation rate** (fraction of timesteps with |action| > 0.9) — both hardware-relevant proxies for motor stress.

---

## Quickstart

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Clone Franka assets
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
```

**Option A — run with pre-trained weights (no training required):**
```bash
# Evaluate transformer vs IK baseline, all trajectory types
python evaluate.py ablation --model results/canonical/transformer_5M.zip

# Record a tracking video
python record_video.py --model results/canonical/transformer_5M.zip --trajectory circle
```

**Option B — train from scratch (~55 min, 20 parallel envs):**
```bash
python train.py \
    --config ee_tracking/configs/transformer/tfm_no_xattn_5M.yaml \
    --out results/my_run

python evaluate.py ablation --model results/my_run/final_model.zip

tensorboard --logdir results/my_run/tb
```

---

## Approach

### The delay problem

A standard IK controller commands joint positions based on the *current* measured target. With a 5-step FIFO delay (100 ms round-trip), that command only executes when the target has already moved. On a fast random-walk trajectory this causes ~38 mm lag — more than double the no-delay IK error.

The key insight: the delay window is known and fixed. If the policy can see both **where the target will be** when each queued command executes, and **what commands are already queued**, it can add a predictive correction that pre-compensates for the lag.

### Residual control

The policy outputs a **residual** on top of IK, not a full joint position command:

```
q_set(t) = clip(q_ik(t) + residual(t) × residual_scale, joint_limits)
ctrl(t)  = q_set(t − 5)          ← whole pipeline delayed 5 steps
```

An untrained policy (residual ≈ 0) degrades gracefully to IK — the baseline is always available. The IK handles gross positioning; the residual only needs to learn the predictive delay-compensation correction.

### Observation design

The 95-D observation is structured around the delay:

| Block | Dims | Content |
|---|---|---|
| Robot state | 30 | Joint positions/velocities, EE position, position error, IK command |
| Fine lookahead | 15 | Target position at t+20ms … t+100ms — covers exact delay window |
| Coarse lookahead | 12 | Target position at t+100ms … t+400ms — trajectory trend |
| Command history | 35 | 5 queued setpoints minus current q — Markov restoring element |
| Trajectory ID | 3 | One-hot: moving target / circle / figure-8 |

The fine lookahead window covers exactly the 5-step delay. The command history reveals what corrections are already queued, preventing the policy from stacking redundant commands.

---

## Architecture

### Transformer with paired slot tokens

The transformer processes the delay queue as a sequence of **slot tokens**, where each token pairs the queued command with the fine lookahead target it will execute against:

```
slot[i] = Linear(concat(fine_lookahead[i], cmd_history[i]))  →  d_model
```

This pairing is the key structural prior. `cmd[i]` will execute when the target is at `fine[i]` — wiring this temporal alignment into the token representation gives the encoder a structure the MLP must discover from scratch.

The slot sequence is processed by a pre-LN TransformerEncoder (2 layers, 4 heads, d_model=64), mean-pooled, and concatenated with the encoded robot state:

```
Observation
    ├── robot_state (30D) ─┐
    ├── coarse_look (12D)  ├─ concat (45D) ──► Linear → state_enc (64D)
    ├── traj_onehot  (3D) ─┘
    └── [fine[i] ‖ cmd[i]] × 5 ──► TransformerEncoder ──► mean pool → slots_enc (64D)
                                                                         │
                                                     concat(state_enc, slots_enc) (128D)
                                                                         │
                                          ┌──────────────────────────────┴──────────────────────┐
                                       Actor MLP                                         Critic MLP
                                    [256, 256] → 7D                              [256, 256, 256] → 1
```

### Ablation study

![Ablation chart](results/figures/ablation_bar.png)

Ablations at 300k steps, each removing or adding one component from the canonical architecture (self-attention only, paired tokens, positional embedding):

| Ablation | Moving Target | Circle | Figure-8 | Conclusion |
|---|---|---|---|---|
| **Canonical (self-attn, no xattn)** | **24.4 mm†** | **5.7 mm†** | **5.2 mm†** | **— baseline** |
| A: + cross-attention layers | 27.0 mm | 5.0 mm | 6.5 mm | xattn adds no benefit — regresses MT and F8 |
| B: − positional embedding | 26.8 mm | 11.1 mm | 10.7 mm | PE critical for periodic trajectories |
| C: unpaired tokens | 26.6 mm‡ | 6.8 mm‡ | 6.9 mm‡ | pairing helps periodic trajectories |

† mean over 2 seeds (seed=42: MT=23.6 CI=4.9 F8=4.8; seed=1: MT=25.2 CI=6.5 F8=5.5)
‡ mean over 2 seeds (seed=42: MT=25.9 CI=5.9 F8=5.9; seed=1: MT=27.2 CI=7.7 F8=7.8)

**Finding A:** Adding cross-attention layers *degrades* performance on moving target and figure-8 while offering negligible benefit on circle. The paired token design already encodes the temporal alignment — `cmd[i]` paired with `fine[i]` gives the encoder the same structural prior cross-attention was meant to discover, making extra attention heads redundant.

**Finding B:** Removing positional embeddings is catastrophic for periodic trajectories (+5.4 mm on circle, +5.5 mm on figure-8) but barely affects the random walk. The transformer needs PE to encode the execution-order of queued commands — without it, slot[0] and slot[4] look identical.

**Finding C:** Unpairing the tokens degrades CI by +1.1 mm and F8 by +0.7 mm on average. The `cmd[i]↔fine[i]` pairing is the key structural prior — it wires temporal alignment into the encoder input rather than requiring the model to discover it from scratch.

---

## Design choices

### State and action space

**State (95D):** Concatenates current robot state, oracle future target positions (fine + coarse lookahead), and the pending command queue. The command history is essential for the Markov property: without it the policy cannot distinguish "IK is already compensating" from "nothing is queued" and stacks redundant corrections.

**Action (7D):** Per-joint position residuals in [−1, 1], scaled by `residual_scale = 0.12 rad`. A zero action always falls back to IK.

### Reward

```
r = w_pos × (‖err_prev‖ − ‖err_now‖)   # reward progress toward target
  − w_vel × ‖ee_vel‖                    # penalise unnecessary motion
  − w_residual × ‖action‖²              # small regularisation on residual
```

No jerk or smoothness penalty. With a 5-step delay the optimal strategy is a *predictive impulse* — inherently discontinuous. Penalising action changes directly penalises the delay-compensation mechanism.

### Trajectory representation

Three trajectory types trained simultaneously:
- **Moving target** — band-limited random walk (0.05–0.15 Hz, 8–14 cm amplitude)
- **Circle** — constant-speed circular orbit
- **Figure-8** — Lissajous curve with direction reversals

Training with a mixed pool was essential: single-trajectory training overfits and degrades on held-out trajectories.

### Evaluation metric

`residual_settled_rmse_mm`: RMSE of the EE position error over the settled portion of each episode (after 0.5 s), per trajectory type. Separates steady-state tracking from transient startup.

### Uncertainty sources

| Source | Implementation |
|---|---|
| Observation noise | Gaussian on EE position (σ=5 mm) and joint positions (σ=2 mm) |
| Command delay | 5-step FIFO applied to the full IK+residual setpoint (100 ms) |
| Unreachable positions | Episode terminates at 0.30 m tracking error; included as an eval scenario |

---

## Repository structure

```
ee_tracking/
  env/
    franka_tracking_env.py   # Gymnasium env — obs, reward, delay FIFO
    ik_controller.py         # Damped least-squares IK baseline
    trajectories.py          # circle, figure-8, moving_target generators
    disturbances.py          # noise + command delay
  policies/
    transformer_policy.py    # Paired-slot transformer (SB3-compatible)
    gelu_policy.py           # MLP with GELU + LayerNorm
    delay_buffer.py          # FIFO delay buffer
  configs/
    mlp/                     # MLP configs (mlp_best_10M.yaml = champion MLP)
    transformer/             # Transformer configs + ablations

train.py                     # Train from YAML config
evaluate.py                  # IK vs policy evaluation + plots
sweep.py                     # Hyperparameter sweep runner
record_video.py              # Record tracking video

docs/architecture.md         # Full architecture diagrams (Mermaid)
scripts/make_figures.py      # Generate result figures
scripts/show_results.py      # Terminal results viewer
```

---

## Acknowledgements

Franka Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Training via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
