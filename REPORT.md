# Residual PPO for Franka EE Tracking — Project Report

## Problem

Track a moving Cartesian target with a 7-DoF Franka Panda in MuJoCo. The practical constraint: the control pipeline has **end-to-end latency** (network round-trip + controller loop) that makes purely reactive control systematically late. A damped-least-squares IK controller is the natural baseline — fast, interpretable, reliable — but it cannot anticipate where the target will be by the time its command executes.

**Goal:** train a PPO residual policy on top of IK that learns to compensate for the delay, using a short lookahead of future target positions.

---

## Approach

The policy outputs a 7-D joint-space correction in `[-1, 1]`. This is filtered, scaled, and added to the IK setpoint as a **non-accumulating position offset** — not an integrated velocity. The combined command (IK + residual) then travels through a shared delay buffer before reaching the actuators.

```
q_ik(t)  = clip(q_ik(t-1) + ik_qdot(t) × dt, limits)   # IK integrates freely
q_set(t) = clip(q_ik(t)   + residual(t) × dt, limits)   # one-step offset
ctrl(t)  = q_set(t − D)                                  # whole pipeline delayed D steps
```

Both IK and the residual travel through the same FIFO. The policy observes:
- Current joint state + EE position (noisy)
- IK's current velocity command
- **5-step lookahead** of future target positions (covers the full delay window)
- **cmd\_delta\_history**: the D setpoints currently in the delay buffer, so the policy knows what movement is already queued

---

## What Didn't Work (and Why)

### Attempt 1: velocity-additive residual
Early implementation accumulated IK and residual velocities together:
```
q_setpoint += (ik_qdot + residual) × dt
```
A wrong residual at step `t` drifted into all subsequent steps. After 60 steps of maximum residual, the accumulated error reached **0.144 rad** in joint space. IK had to fight this drift constantly and often failed. The current non-accumulating design bounds any single-step error to `residual_scale × dt = 0.001 rad`.

### Attempt 2: fixing three evaluation bugs
After fixing the architecture, training metrics looked promising but eval crashed or gave misleading numbers:
1. **Obs-space mismatch**: `evaluate.py` rebuilt the env from defaults, not the saved config. With different `lookahead_horizon` or `act_delay`, the obs dimension didn't match the saved VecNormalize stats → crash.
2. **Train/eval disturbance skew**: a hardcoded eval disturbance applied `act_delay=1` to models trained with `act_delay=0`, evaluating them out-of-distribution.
3. **IK baseline inflated**: the old code routed the IK command through the same delay buffer as the residual, so "IK-only" was actually IK-with-delay. The corrected baseline (undelayed IK) dropped from 19.8 mm to 15.6 mm.

After all three fixes: correct eval, correct baseline.

### Attempt 3: baking a Butterworth filter + full training run
Post-hoc experiments showed a 2 Hz Butterworth on the policy output improved results by ~2% over EMA smoothing. Baked this into the training env (so the policy trains against its own filtered output).

Full 1.5 M-step training run: **16.1 mm** — marginally *worse* than the 15.6 mm IK baseline.

### Diagnosis: the task formulation was wrong

Probe experiments ran 7 hyperparameter variants at 300k steps each. Every variant showed `std ≈ 1.08` and `clip_fraction ≈ 0.163` — identical to a random policy. No learning was happening regardless of tuning.

Root cause analysis:
- **IK was near the noise floor.** The dominant IK error (15.6 mm) was already close to the theoretical minimum given the DLS damping and obs noise. The maximum EE correction the residual could apply was ~0.8 mm/step — barely enough to matter.
- **Delay was applied only to the residual.** The old `act_delay` parameter delayed only the residual correction, while IK acted instantly. This gave IK an unfair structural advantage: the residual policy had to fight its own 20 ms delay to break even with an undelayed IK.
- **No clear learnable signal.** The policy's corrections were small relative to the noise and the task didn't require prediction — IK already received the analytic target velocity as feedforward.

---

## The Fix: whole-pipeline delay

Apply the delay to the **total q\_setpoint** — both IK and residual travel through the same buffer. This is the physically accurate model: the full command takes D steps to reach the actuators.

Effect on IK:

| | RMSE (mm) |
|---|---|
| IK, no delay | ~18 |
| IK, D = 5 steps (100 ms) | ~44 |

The delay creates a **26 mm gap** the IK cannot close without prediction. The residual policy, which has a 0.5 s lookahead that exactly covers the 0.1 s delay window, can learn to pre-position the EE by predicting where the target will be when each command executes.

The lookahead alignment is intentional: `lookahead_dt × lookahead_horizon = 0.1 × 5 = 0.5 s` covers the full `cmd_delay × dt = 5 × 0.02 = 0.1 s` delay window, with additional horizon for anticipation.

---

## Current Results

All numbers are deterministic post-training eval RMSE (mm). IK baseline uses 100 ms delay.

### Best confirmed models

| Model | moving_target | circle | figure8 | Notes |
|---|---|---|---|---|
| IK + 100ms delay | 38.1 mm | 12.1 mm | 7.7 mm | baseline to beat |
| PPO 5M, single-pool, linear LR | 20.5 mm | 13.5 mm ❌ | 13.0 mm ❌ | hurts unseen trajectories |
| **PPO 500K, mixed pool, cosine LR** | **26.2 mm** | **7.9 mm ✓** | **5.9 mm ✓** | beats IK on all 3 at 500K |
| **PPO 500K, rs=0.12, mixed pool** | **22.9 mm** | 12.0 mm | 10.1 mm | best moving_target; under-converged |

### Key metric progression (moving_target eval)

```
IK + delay:              38.1 mm   (fixed reference)
5M single-pool:          20.5 mm   (+46% over IK)
500K mixed-pool:         26.2 mm   (+31% — beats IK on circle/fig8 too)
500K rs=0.12 mixed:      22.9 mm   (+40% — best 500K result; needs more steps)
Overnight target:       <16.0 mm   (rs=0.12 at 5–10M steps)
```

### Training dynamics (confirmed patterns)

- **Explained Variance (EV)**: rises from ~0.86 (5M single-pool) to 0.982 (500K mixed pool) — mixed pool dramatically stabilises the critic via clean circle/figure8 episodes
- **Bang-bang policy**: 47–87% of post-Tanh actions have |a| > 0.9; the policy always saturates because the action penalty is negligible at rs=0.05
- **LR decay is critical**: constant LR 1e-3 → EV=0.86; cosine 1e-3→1e-4 → EV=0.98 at 500K

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Non-accumulating residual offset | Wrong corrections at step t are fully self-correcting at t+1; prevents drift windup |
| Whole-pipeline delay (both IK + residual) | Physically accurate; gives IK a real weakness the policy can exploit; residual-only delay gave zero learning |
| Lookahead covers delay window exactly | Fine lookahead (5×0.02s=0.1s) gives oracle knowledge of target position at each command's execution time |
| **Mixed trajectory pool** | Biggest single finding: EV 0.888→0.982, beats IK on circle/figure8 at 500K; single-pool models hurt unseen trajectories |
| `cmd_delta_history` in observation | Restores Markov property: policy knows what commands are in-flight; `delay_aware_gae` must stay OFF with this |
| No smoothness/jerk penalty | Predictive impulse control is inherently jerky; penalising jerk penalises delay compensation directly |
| cosine LR 1e-3→1e-4 | Probe-confirmed better than linear; stays warm for exploration then converges sharply |
| n_envs=20 | 3–5× wall-clock speedup; equal total gradient steps vs n_envs=10; allows 9 runs per 8-hour budget |
| Action filter disabled | 2 Hz Butterworth adds ~5.6 steps of group delay on top of 5-step FIFO; doubles effective latency |

---

## Limitations and Next Steps

### Immediate (overnight sweep in progress)
- **Optimal residual_scale at 5M**: rs=0.12 is the hypothesis; sweep tests rs ∈ {0.05, 0.08, 0.10, 0.12, 0.15} with the full confirmed recipe. Results in `results/sweep/` by morning.
- **Step depth**: rs=0.12 at 10M (vs 5M) tested overnight. Target: <16mm moving_target, matching or beating IK floor on circle/figure8.

### Medium-term
- **Oracle lookahead → real predictor**: the fine lookahead provides ground-truth future positions — unrealistic for hardware. Replace with a Kalman smoother or learned predictor over observed target history. The architecture is already set up to swap this in (same obs dimension).
- **Scale-invariant w_residual**: current penalty is `−wr × rs² × ‖action‖²`; the rs² factor makes tuning inconsistent across residual_scale values. Fix: `−wr × ‖action‖²` makes wr interpretation scale-independent.
- **Post-hoc smoothing for hardware**: the bang-bang policy produces jerky joint commands (47–87% saturated). For real Franka deployment, a 2 Hz Butterworth applied post-hoc at inference (not baked into training) would smooth commands without adding training-time latency.

### Long-term
- **Sim-to-real gap**: MuJoCo perfectly matches the simulated robot. Real deployment needs domain randomisation over inertia, joint damping, and contact parameters.
- **Curriculum over delay**: training and evaluating at a fixed 100 ms delay. Randomising delay over [0, 150 ms] during training would improve robustness to network jitter.
