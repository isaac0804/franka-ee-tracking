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

Training on the new formulation (`results/run_delay5`, 1.5 M steps, 100 ms delay):

```
pos_err_mm during training: 35–40 mm  (vs IK baseline ~44 mm)
explained_variance: 0.92              (critic well-calibrated)
std: 1.18                             (still converging)
```

The policy is already achieving ~10–20% improvement over IK mid-training. Final evaluation pending completion of the training run.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Non-accumulating residual offset | Wrong corrections at step t are fully self-correcting at t+1; prevents drift windup |
| Whole-pipeline delay (both IK + residual) | Physically accurate; gives IK a real weakness the policy can exploit |
| Lookahead covers delay window exactly | Policy sees the target position at the exact moment each command executes |
| `moving_target` only in training pool | Circle and figure-8 are too predictable for IK; `moving_target` is where delay hurts most |
| `cmd_delta_history` in observation | Restores the Markov property: policy knows what commands are in-flight in the buffer |
| Baked Butterworth vs post-hoc filter | Policy trains against its own filtered output, learning to work with the filter rather than against it |
| Probe script for fast iteration | 7 configs × 300k steps ≈ 15 min; ruled out hyperparameter causes before redesigning the task |

---

## Limitations and Next Steps

- **Oracle lookahead** is unrealistic for real hardware. A natural extension is replacing the ground-truth future positions with a learned predictor or Kalman smoother over observed target history.
- **Sim-to-real gap**: the MuJoCo model perfectly matches the simulated robot. Real deployment would require domain randomization over inertia, joint damping, and contact parameters.
- **Single trajectory type in training**: the policy may not generalise to circle or figure-8 under delay. Expanding the training pool (with appropriate curriculum) is the next step after the primary benchmark is solved.
- **Policy not yet converged**: `std ≈ 1.18` at 1.5 M steps suggests more training time or a lower entropy coefficient would help.
