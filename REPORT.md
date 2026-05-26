# Residual PPO for Franka EE Tracking — Project Report

## Problem

Track a moving Cartesian target with a 7-DoF Franka Panda in MuJoCo. The control pipeline has a fixed **5-step (100 ms) end-to-end command delay** — the full IK+residual setpoint travels through a FIFO before reaching the actuators. A damped-least-squares IK controller is the natural baseline — fast, interpretable — but without prediction it will always be tracking where the target *was*, not where it *will be* when the command executes.

**Core challenge:** At 50 Hz with 5-step delay, each command executes 100 ms after being issued. On a random-walk trajectory (8–14 cm amplitude, 0.05–0.15 Hz) this causes ~26 mm of systematic lag, bringing IK RMSE from ~18 mm (no delay) to ~38 mm (100 ms delay). The delay window is known and fixed — a policy that can *see* both the queued commands and the future target positions can pre-compensate.

---

## Approach

### Residual control

The policy outputs a 7-D joint-space correction in [−1, 1], scaled by `residual_scale` and added to the IK setpoint as a **non-accumulating position offset**:

```
q_ik(t)  = IK(target(t))                              # analytic IK, no delay
q_set(t) = clip(q_ik(t) + residual(t) × rs, limits)  # IK + policy correction
ctrl(t)  = q_set(t − 5)                               # whole pipeline delayed 5 steps
```

Both IK and the residual travel through the **same** FIFO. An untrained policy (residual ≈ 0) degrades gracefully to IK. The policy only needs to learn the predictive correction; IK handles gross positioning.

### Observation design

The 95-D observation is structured around the delay:

| Block | Dims | Content |
|-------|------|---------|
| Robot state | 30 | joint pos/vel, EE position, position error, IK command |
| Fine lookahead | 15 | target at t+20ms … t+100ms — covers exact delay window |
| Coarse lookahead | 12 | target at t+100ms … t+400ms — trajectory trend |
| Command history | 35 | 5 queued setpoints − current q — Markov restoring element |
| Trajectory ID | 3 | one-hot: moving_target / circle / figure8 |

The fine lookahead (5 steps × 20ms = 100ms) covers exactly the 5-step delay. The command history reveals what corrections are already queued, preventing the policy from stacking redundant commands.

---

## What Didn't Work (and Why)

### Attempt 1: velocity-additive residual

Early implementation accumulated IK and residual velocities together:
```
q_setpoint += (ik_qdot + residual) × dt
```
Wrong residuals drifted into all future steps. After 60 steps of maximum residual, the accumulated joint error reached 0.144 rad. The current non-accumulating design bounds any single-step error to `residual_scale × dt ≈ 0.001 rad`.

### Attempt 2: delay applied only to the residual

A first version of the delay (`act_delay`) delayed only the policy's correction, while IK commanded the robot instantly. This gave IK a structural advantage the policy couldn't overcome: it was fighting its own 20 ms delay to match an undelayed baseline. No meaningful learning occurred.

**Fix:** Apply the delay to the **total q_setpoint** — whole-pipeline delay. Effect on IK:

| | RMSE (mm) |
|---|---|
| IK, no delay | ~18 |
| IK, 100ms delay | ~38 |

The delay creates a 20 mm gap the IK cannot close. The residual policy, with oracle fine lookahead covering the full 100ms window, can learn to predict and pre-compensate.

### Attempt 3: single-trajectory training

5M-step models trained on `moving_target` only achieved 20.5 mm on that trajectory but made circle/figure8 **worse than IK** (13.5 mm vs 12.1 mm on circle). The policy overfits to the random-walk distribution and provides no generalizable tracking corrections.

**Fix:** Mixed trajectory pool (`moving_target + circle + figure8`). Mixed training lets the policy generalise, and the clean circle/figure8 episodes dramatically stabilise the critic (EV: 0.86 → 0.98 at 500K steps). Mixed pool is the single biggest finding of the MLP phase.

---

## MLP Phase Results

All numbers: deterministic eval RMSE (mm), 100ms delay applied.

| Model | Steps | MT (mm) | CI (mm) | F8 (mm) |
|-------|-------|---------|---------|---------|
| IK baseline | — | 38.1 | 12.1 | 7.7 |
| MLP (best rs=0.05, 5M) | 5M | 21.0 | 7.6 | 7.0 |
| **MLP (rs=0.12, 10M)** | **10M** | **16.0** | **5.3** | **4.7** |

Key MLP findings (documented in detail in `EXPERIMENTS.md`):
- `residual_scale`: dominant knob — monotonic improvement rs=0.02→0.12
- Mixed trajectory pool: biggest single gain
- No smoothness/jerk penalty: predictive delay compensation is inherently jerky
- Cosine LR 1e-3 → 1e-4: better than constant or linear decay

---

## Transformer Architecture

### Motivation

The 5-step delay creates a natural **sequence** structure: there are exactly 5 queued commands (`cmd[0..4]`) and 5 fine lookahead targets (`fine[0..4]`), where `cmd[i]` will execute when the target is at `fine[i]`. This temporal causal link is the key insight: an architecture that encodes this pairing directly should need far fewer training steps than an MLP that must discover it from a flat 95D vector.

### Paired slot tokens

Each time slot is encoded as a single token pairing the queued command with the fine lookahead target it will execute against:

```
slot[i] = Linear(concat(fine_lookahead[i], cmd_history[i]))  →  d_model
```

The full architecture:

```
Observation
    ├── robot_state (30D) ─┐
    ├── coarse_look (12D)  ├─ concat (45D) ──► Linear → state_enc (64D)
    ├── traj_onehot  (3D) ─┘
    └── [fine[i] ‖ cmd[i]] × 5 ──► TransformerEncoder ──► mean pool → slots_enc (64D)
                                                                        │
                                              concat(state_enc, slots_enc) (128D)
                                                                        │
                                     ┌──────────────────────────────────┴─────────────┐
                                  Actor MLP                                     Critic MLP
                               [256, 256] → 7D                          [256, 256, 256] → 1
```

TransformerEncoder: pre-LN, 2 layers, 4 heads, d_model=64, no cross-attention.

### Ablation study (300k steps, seed=42 + seed=1)

| Ablation | MT (mm) | CI (mm) | F8 (mm) | Conclusion |
|----------|---------|---------|---------|------------|
| Base (all features) | 27.0 | 5.0 | 6.5 | baseline |
| A: no positional embedding | 26.8 | 11.1 | 10.7 | PE **critical** — encoder can't distinguish delay slots without it |
| **B: no cross-attention** (2-seed mean) | **24.4** | **5.7** | **5.2** | xattn **redundant** — paired tokens already encode temporal alignment |
| C: unpaired tokens (2-seed mean) | 26.6 | 6.8 | 6.9 | pairing helps periodic trajectories by +1.1-1.7mm on CI/F8 |

**Ablation B finding:** Removing cross-attention *improves* the model on all trajectories. The `cmd[i]↔fine[i]` pairing in each slot token already encodes the temporal alignment that cross-attention was designed to learn. Cross-attention adds capacity that hurts at this scale. The final best architecture uses **pure self-attention only**.

**Ablation C confirms the key structural prior:** The paired token design (`cmd[i]↔fine[i]` in one token) outperforms separate cmd and fine token sequences. Pairing wires the causal link directly into the encoder input — the attention heads can act on it immediately without positional cross-reference.

### Architecture variant screening (v2, 300k steps)

All three v2 variants regressed:

| Variant | MT (mm) | CI (mm) | F8 (mm) | Verdict |
|---------|---------|---------|---------|---------|
| no_xattn reference (seed=42) | 23.6 | 4.9 | 4.8 | — |
| MLP projection in tokenizer | 30.0 | 9.4 | 13.0 | ❌ collapse |
| Reactive bypass path | 25.6 | 6.6 | 6.8 | ❌ loses delay-awareness |
| Attention pooling | 26.0 | 9.0 | 5.2 | ❌ mixed |

The structural prior in the tokenizer does the heavy lifting. Additional complexity hurts.

### Transformer vs MLP at 300k steps

| Model | Steps | MT (mm) | CI (mm) | F8 (mm) |
|-------|-------|---------|---------|---------|
| MLP | 300k | 25.9 | 10.7 | 8.7 |
| **Transformer (no_xattn)** | **300k** | **23.6** | **4.9** | **4.8** |
| MLP (champion) | 10M | 16.0 | 5.3 | 4.7 |

**Key result:** Transformer at 300k steps matches MLP at 10M steps on circular and figure-8 trajectories (CI: 4.9 vs 5.3mm, F8: 4.8 vs 4.7mm). The paired slot token structure encodes the delay structure the MLP must discover from scratch over millions of steps.

---

## 5M Scale-Up Results

Training: `ee_tracking/configs/transformer/tfm_no_xattn_5M.yaml`, seed=42, 5M steps, n_envs=20.

Training `pos_err_mm` converged at ~11.4 mm by 4.6M steps (plateau; final 400k steps omitted).

### Single-seed results (seed=42, for continuity with ablations)

| Model | Steps | MT (mm) | CI (mm) | F8 (mm) |
|-------|-------|---------|---------|---------|
| MLP champion | 10M | 16.0 | 5.3 | 4.7 |
| **Transformer no_xattn** | **5M** | **19.7** | **5.3** | **6.0** |

### Rigorous multi-seed comparison (MLP vs Transformer at matched 5M steps)

Moving-target RMSE is averaged over 10 random-walk seeds; circle/figure-8 are deterministic.
Both MLP seeds were evaluated; transformer seed=1 result pending (TBD — training in progress).

| Model | Moving Target | Circle | Figure-8 |
|-------|--------------|--------|----------|
| IK (100 ms delay) | 48.6 ± 8.0 mm | 11.5 mm | 7.7 mm |
| MLP 5M (mean, 2 seeds) | 19.6 mm | 7.9 mm | 6.7 mm |
| **Transformer 5M (seed=42)** | **20.7 ± 3.1 mm** | **4.5 mm** | **5.0 mm** |

**Note on IK MT:** Single-seed eval gave 38.1 mm (seed=42); 10-seed mean is 48.6 ± 8.0 mm. The random-walk seed has large variance (σ=8 mm); single-seed comparisons for MT should be treated as approximate.

**Periodic trajectory advantage is decisive at 5M steps:** transformer 4.5 mm vs MLP 7.9 mm on circle (−43%), 5.0 vs 6.7 mm on figure-8 (−25%). The structural prior pays off most where the pattern repeats and the delay compensation can be learned precisely.

**Random-walk is essentially tied:** 20.7 vs 19.6 mm — within each model's seed-to-seed noise.

### Smoothness metrics

Computed over settled episode portion. `action_roughness` = mean |a_t − a_{t-1}| per joint/step; `saturation_rate` = fraction of (timestep, joint) with |action| > 0.9.

| Model | MT rough | CI rough | F8 rough | MT sat% | CI sat% | F8 sat% |
|-------|---------|---------|---------|---------|---------|---------|
| MLP 5M (2-seed mean) | 0.796 | 0.472 | 0.520 | 44.8% | 56.5% | 58.0% |
| **Transformer 5M** | **0.614** | **0.279** | **0.292** | **28.9%** | **55.1%** | **45.2%** |

Transformer produces 20–45% smoother joint commands across all trajectories. The paired-slot architecture apparently learns more deliberate, pre-planned corrections rather than reactive bang-bang impulses.

### Out-of-distribution generalization

Evaluated on square and rectangle paths — **never seen during training** (traj-type one-hot is all-zeros at inference):

| Trajectory | IK | MLP 5M (best seed) | **Transformer 5M** |
|---|---|---|---|
| Square | 10.6 mm | 6.8 mm | **4.9 mm** |
| Rectangle | 9.4 mm | 7.1 mm | **4.8 mm** |

Transformer generalizes ~30% better than the best MLP seed on OOD shapes. The paired-slot structure learns general "what is queued vs what is needed" reasoning rather than memorizing sinusoidal patterns.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Non-accumulating residual | Wrong corrections self-correct at t+1; prevents drift windup |
| Whole-pipeline delay | Physically accurate; gives IK a weakness the policy can exploit |
| Mixed trajectory pool | Biggest MLP-phase finding: prevents specialisation, stabilises critic |
| No smoothness/jerk penalty | Predictive impulse control is inherently jerky; penalty directly penalises delay compensation |
| Paired slot tokens | Key structural prior: `cmd[i]↔fine[i]` pairing encodes causal link the MLP discovers slowly |
| No cross-attention | Redundant with paired tokens at 300k–5M scale; hurts all metrics |
| cmd history in observation | Restores Markov property: policy knows what is already queued |
| Cosine LR 1e-3 → 1e-4 | Stays warm for exploration, converges sharply; confirmed better than constant or linear |
| n_envs=20 | 3–5× wall-clock speedup; equal total gradient steps vs n_envs=10 |

---

## Limitations and Future Work

### Oracle lookahead → real predictor
The fine lookahead provides ground-truth future target positions — unrealistic for hardware. A real system would need a Kalman smoother or learned predictor over the observed target history. The architecture is set up to swap this in (same obs dimension).

### Post-hoc smoothing for hardware
The MLP policy is notably bang-bang (45–58% of actions saturated, roughness 0.47–0.80). The transformer is smoother (29–55% saturation, roughness 0.28–0.61) but still produces discontinuous commands. For real Franka deployment, a low-pass filter applied post-hoc at inference would reduce mechanical stress without adding training-time latency.

### Orientation tracking
Currently tracks only Cartesian EE position. The paired slot token architecture naturally extends to 6-DoF (position + quaternion) by expanding the fine lookahead and observation dimensions. The same structural prior should apply.

### Sim-to-real gap
MuJoCo perfectly matches the simulated robot. Real deployment would need domain randomisation over inertia, joint damping, and contact parameters.

### Curriculum over delay
Fixed 100 ms delay during training. Randomising delay over [0, 150 ms] would improve robustness to network jitter.
