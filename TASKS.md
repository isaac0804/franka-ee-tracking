# Orientation Tracking — Task Design

Branch: `feat/orientation-tracking`

This document defines the five orientation-tracking tasks that extend the
position-only residual PPO baseline to full 6-DoF EE tracking.

---

## Background

The current system tracks 3-D Cartesian position with a 5-step (100 ms)
whole-pipeline delay. The key insight — pairing queued commands with fine
lookahead targets as slot tokens — transfers directly to 6-DoF:

```
slot[i] = Linear(concat(fine_pos[i](3), fine_quat[i](4), cmd[i](7)))  →  d_model
```

The slot is 14-D instead of 10-D. Everything else stays the same: paired
tokens, pre-LN 2-layer transformer, no cross-attention.

### Observation extension (95D → 141D)

| Block | Dims | Change |
|-------|------|--------|
| Robot state | 40 | +10: ee_quat(4), ori_err_axisangle(3), ee_angvel(3) |
| Fine lookahead | 35 | +20: target quat(4) × 5 steps, paired with fine_pos |
| Coarse lookahead | 28 | +16: target quat(4) × 4 steps |
| Command history | 35 | unchanged — joint space |
| Trajectory ID | 3 | unchanged |
| **Total** | **141** | +46 from position-only |

**Quaternion convention:** MuJoCo `(w, x, y, z)`. Lookahead quaternions are
expressed in the world frame (not relative to current orientation). Orientation
error in the state block uses the 3-D axis-angle log-map so it has no
double-cover ambiguity:

```python
e_ori = log_map(q_des ⊗ q_cur^{-1})   # 3D axis-angle, magnitude = angle in rad
```

### Reward extension

```
r = w_pos  × (‖e_pos_prev‖ − ‖e_pos_now‖)     # progress on position
  + w_ori  × (‖e_ori_prev‖ − ‖e_ori_now‖)     # progress on orientation
  − w_vel  × ‖ee_vel‖                          # unnecessary translation
  − w_angvel × ‖ee_angvel‖                     # unnecessary rotation
  − w_residual × ‖action‖²                     # regularise residual
```

Default weight suggestion: `w_ori = 2.0` (orientation error in radians, while
position error is in metres; scaling keeps both terms comparable in magnitude).

### IK extension

Full 6-DoF DLS:
```
J    = stack(J_pos(3×7), J_rot(3×7))         # 6×7 Jacobian
v_des = [kp_pos * e_pos + v_pos_ff,
         kp_ori * e_ori + v_ori_ff]           # 6-D task-space velocity
q_dot = J^T (J J^T + λ²I)^{-1} v_des
```

Separate gains `kp_pos` and `kp_ori` because position and orientation errors
have different units and convergence rates.

---

## Task 1 — Upright Constraint

**What:** track an existing position trajectory while keeping the end-effector
"upright" — the EE z-axis within ±20° of the world +z direction.

**Model:** carrying a full cup or tray while reaching. Position must follow the
target; spilling is the failure mode.

**Orientation target:** constant — identity quaternion (z-axis ≡ world +z).
Angular velocity target: zeros. The trajectory wraps any position trajectory.

**Why it's interesting:**
- Simplest orientation task: the target does not move, only position changes.
- Reveals whether the policy can decouple: "move IK to follow position" while
  "add residual correction to keep orientation steady."
- Baseline: pure IK (position-only) ignores orientation entirely. Any RL signal
  from orientation requires the policy to actively maintain upright pose.

**Termination:** same as position — 0.30 m position error. Add a soft
orientation penalty (not hard termination) so the agent learns rather than
gives up.

**Config name:** `upright_constraint`

---

## Task 2 — Tilted Circle

**What:** follow a circle in the y-z plane (same as training), but orientation
must rotate continuously so the EE z-axis always points inward toward the
center of the circle.

**Think:** a drill tip moving around the inside of a cylinder, always
perpendicular to the surface.

**Orientation target at angle θ:**
```
# Circle is in y-z plane; EE at angle θ
# Desired EE z-axis points from EE toward center:
z_desired = -[0, cos θ, sin θ]    (in world frame)
# Build rotation that takes world-z to z_desired
R = rotation_from_z_to(z_desired)
q = mat_to_quat(R)
```

Angular velocity:
```
ω = dθ/dt × [1, 0, 0]    # rotation around x-axis at the circle's angular rate
```

**Why it's interesting:**
- Orientation is **coupled** to position: knowing where you are on the circle
  exactly determines the required orientation.
- The fine lookahead gives the policy both the future position **and** future
  orientation; the paired slot tokens encode both.
- The delay makes this hard: without lookahead, IK commands joint angles for
  the current orientation. When the command executes 100 ms later, the target
  has rotated by `ω × 0.1 s`. At a 6-s circle period that's 10.5° of
  misalignment per step — measurable as EE-axis angular error.

**Config name:** `tilted_circle`

---

## Task 3 — Look-At Tracking

**What:** position follows a band-limited random walk (same as existing
`moving_target`), but orientation must always point the EE z-axis at a **fixed
beacon** in space.

**Think:** a camera on a robotic gimbal always keeping a POI in frame while
the body moves.

**Beacon:** a fixed world-frame point, e.g. `[0.5, 0.2, 0.7]` (offset from
the workspace centre so the "look direction" changes meaningfully as EE moves).

**Orientation target at EE position p:**
```
d = beacon - p                      # look direction vector
d_hat = d / ‖d‖
# Build rotation so EE z-axis aligns with d_hat
R = rotation_from_z_to(d_hat)
q = mat_to_quat(R)
```

Angular velocity (implicit differentiation):
```
ω = (d_hat × ṗ) / ‖d‖             # cross-product gives angular rate of look vector
```

**Why it's interesting:**
- Orientation is **derived from position** non-linearly: the same position
  lookahead that predicts where the arm will be also determines the required
  orientation 100 ms ahead.
- The transformer's delay structure is fully exploited: slot[i] pairs the
  queued command with both `fine_pos[i]` and the derived `fine_quat[i]`.
- A flat MLP would need to discover the `p → q` relationship from scratch;
  the transformer processes it as a causal sequence.

**Config name:** `look_at`

---

## Task 4 — Rotating Grasp

**What:** position converges to a fixed grasp point and stays there;
orientation rotates around the EE approach axis at a slow, constant rate.

**Think:** unscrewing a bolt — the hand must stay at the bolt's position while
rotating the wrist by 360°.

**Position target:** fixed point `p_grasp`, zero velocity.

**Orientation target:**
```
q(t) = axisangle_to_quat([0, 0, 1], ω × t)   # rotate around z-axis
ω = 0.5 rad/s (≈ one full turn in 12.6 s)
```
Angular velocity: `[0, 0, ω]` constant.

**Why it's interesting:**
- Pure orientation challenge: the position task is trivial (converge and hold),
  so the reward gradient is entirely from orientation.
- The arm has 7 DoF against a 6-DoF task — the null-space of the Jacobian
  allows wrist rotation without moving the EE. The policy must learn to use
  distal joints while proximal joints hold position.
- The delay is still present: an uncompensated ω × 0.1 s = 0.05 rad of
  orientation lag per step. The fine orientation lookahead tells the policy
  exactly what angle to command 100 ms ahead.

**Termination:** only on position error (soft rotation penalty, not
termination, so the full rotation is always completed).

**Config name:** `rotating_grasp`

---

## Task 5 — 6-DoF Random Walk

**What:** independent band-limited random walks in R³ (position) and SO(3)
(orientation). The hardest and most general case.

**Position:** existing `MovingTarget` random walk, 8–14 cm amplitude,
0.05–0.15 Hz.

**Orientation:** SO(3) random walk via filtered axis-angle integration:
```
# Generate filtered Gaussian noise in so(3)
noise_so3 = rng.standard_normal((n, 3))
filtered_so3 = lowpass_filter(noise_so3, cutoff_hz=0.08)
# Normalise to max_angle amplitude
filtered_so3 *= max_angle / max_abs   # max_angle ~ 45° = π/4 rad

# Integrate: each step is a small rotation
quats[0] = identity
for i in range(1, n):
    dq = axisangle_to_quat(filtered_so3[i] * dt)
    quats[i] = quat_mul(dq, quats[i-1])
    quats[i] /= ‖quats[i]‖   # renormalise

# Angular velocity: the instantaneous axis-angle rate
ang_vel[i] = filtered_so3[i]
```

**Why it's interesting:**
- No structural coupling between position and orientation: they are independent
  random signals. The transformer cannot rely on a geometric relationship
  between pos and ori fine lookaheads.
- This is the direct analogue of `moving_target` for 6-DoF — the true stress
  test of the architecture under full task uncertainty.
- Expected result: orientation generalisation may be harder than position
  generalisation, since the SO(3) manifold is curved and the random walk can
  accumulate large rotations.

**Config name:** `random_walk_6dof`

---

## Training Pool

Suggested mixed training pool for the orientation branch:
```
trajectory_pool: [tilted_circle, look_at, random_walk_6dof]
```

Reasoning:
- `tilted_circle` — periodic, coupled: stabilises the critic (same role as `circle` in the position phase)
- `look_at` — random walk position + derived orientation: mixes stochastic and coupled
- `random_walk_6dof` — fully independent: forces generalisation beyond geometric coupling

`upright_constraint` and `rotating_grasp` are evaluation-only OOD tasks (not
in the training pool), analogous to `square`/`rectangle`/`step_target` in the
position-only branch.

---

## Implementation Checklist

- [x] `trajectories.py`: `Trajectory.sample_ori()` base + 5 new classes
- [x] `ik_controller.py`: `DLS6DoFController` with 6×7 Jacobian
- [ ] `franka_tracking_6dof_env.py`: subclass of `FrankaTrackingEnv` with extended obs/reward
- [ ] `configs/orientation/`: YAML configs for each task
- [ ] `evaluate.py`: add orientation RMSE metrics
- [ ] `train.py`: no changes needed (reads config)
