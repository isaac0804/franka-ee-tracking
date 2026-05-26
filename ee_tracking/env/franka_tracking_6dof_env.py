"""6-DoF Franka EE tracking environment — position + orientation.

Extends `FrankaTrackingEnv` to full 6-DoF tracking (position + orientation).
The core RL structure is unchanged: residual policy on top of a DLS-IK
baseline, whole-pipeline 5-step delay, paired-slot transformer observation.

What changes vs the position-only env
--------------------------------------
1.  **IK baseline**: `DLS6DoFController` solves a 6×7 system (translational +
    rotational Jacobian rows).  Separate gains `kp_pos` and `kp_ori`.

2.  **Observation** (141-D, up from 95-D):
    The state block gains 10 extra dimensions; the fine/coarse lookahead blocks
    each gain orientation quaternion columns.  The cmd_delta_history and
    traj_onehot blocks are unchanged.

    | Block             | Dims | Notes                                      |
    |-------------------|------|--------------------------------------------|
    | q                 |  7   | joint positions (noisy)                    |
    | qd                |  7   | joint velocities                           |
    | ee_pos (noisy)    |  3   | EE Cartesian position                      |
    | pos_error         |  3   | target_pos − ee_pos                        |
    | target_vel        |  3   | desired EE translational velocity          |
    | ik_qdot           |  7   | 6-DoF IK joint-velocity command            |
    | ee_quat           |  4   | current EE orientation (w,x,y,z)           |
    | ori_error         |  3   | axis-angle error (target − current)        |
    | ee_angvel         |  3   | estimated EE angular velocity              |
    | fine_lookahead    | 5×7  | [pos(3) ‖ quat(4)] at t+1..t+5 steps (35D)|
    | coarse_lookahead  | 4×7  | [pos(3) ‖ quat(4)] beyond FIFO (28D)       |
    | cmd_delta_hist    | 5×7  | queued setpoints − q (35D, unchanged)      |
    | traj_onehot       |  N   | one-hot pool index (unchanged)             |
    | **Total**         | 141  | (with N=3 and D=5)                         |

    Transformer slot tokens for the orientation case:
        slot[i] = Linear(concat(fine_pos[i](3), fine_quat[i](4), cmd[i](7))) → d_model
    The token width grows from 10-D to 14-D.  Everything else (2 layers, 4 heads,
    d_model=64, no cross-attention) stays the same.

3.  **Reward** — adds an orientation progress term:
        r_ori = w_ori × (‖e_ori_prev‖ − ‖e_ori_now‖)
    Default w_ori=2.0.  (Orientation error in radians, position in metres;
    w_ori=2 keeps them at a similar scale for typical workspace motions.)

4.  **Training pool** — uses orientation-aware trajectories.  Default:
        ["tilted_circle", "look_at", "random_walk_6dof"]

Usage
-----
    from ee_tracking.env.franka_tracking_6dof_env import (
        FrankaTracking6DoFEnv, Env6DoFConfig
    )
    env = FrankaTracking6DoFEnv(Env6DoFConfig())
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

The environment is SB3-compatible and can be wrapped in `VecNormalize` exactly
like the position-only env.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .franka_tracking_env import FrankaTrackingEnv, EnvConfig
from .ik_controller import DLS6DoFController
from .disturbances import DisturbanceConfig
from . import trajectories as traj_module
from .trajectories import quat_error, _IDENTITY_QUAT


@dataclass
class Env6DoFConfig(EnvConfig):
    """Configuration for the 6-DoF tracking environment.

    Inherits all position-only fields from `EnvConfig` and adds orientation-
    specific parameters.

    Key default override: `disturbance.cmd_delay = 5` (5-step / 100ms delay).
    The position-only `EnvConfig` defaults to cmd_delay=0 and relies on the
    YAML to set it; here we bake in the delay so the default instance is
    immediately usable for testing without a YAML file.
    """

    # ── orientation IK gain ───────────────────────────────────────────────
    kp_ori: float = 3.0            # proportional gain on orientation error (rad/s / rad)

    # ── orientation reward ────────────────────────────────────────────────
    w_ori: float = 2.0             # weight on orientation progress reward

    # ── trajectory pool ───────────────────────────────────────────────────
    # Override the default position-only pool.
    trajectory_pool: tuple = ("tilted_circle", "look_at", "random_walk_6dof")

    # ── episode length ────────────────────────────────────────────────────
    episode_seconds: float = 8.0   # slightly longer to let orientation converge

    # ── disturbance: bake in the 5-step delay ────────────────────────────
    # Override the base-class default (cmd_delay=0) with the canonical 5-step
    # whole-pipeline delay so the env works correctly without a YAML file.
    disturbance: DisturbanceConfig = field(
        default_factory=lambda: DisturbanceConfig(cmd_delay=5)
    )


class FrankaTracking6DoFEnv(FrankaTrackingEnv):
    """6-DoF end-effector tracking: position + orientation.

    Subclasses `FrankaTrackingEnv`; overrides only the methods that need to
    change for orientation.  The MuJoCo simulation, delay buffer, action space,
    and VecEnv interface are all inherited unchanged.
    """

    def __init__(self, cfg: Env6DoFConfig | None = None, render_mode: str | None = None):
        cfg = cfg or Env6DoFConfig()

        # `super().__init__` calls `_compute_observation()` to infer obs shape,
        # but `ik6` hasn't been created yet.  Set a sentinel first so that
        # `_compute_observation` knows to use fallback values on that first call.
        self.ik6 = None                 # sentinel — replaced right after super()
        self._prev_ee_angvel = np.zeros(3)
        super().__init__(cfg=cfg, render_mode=render_mode)

        # Upgrade the IK controller to 6-DoF.
        # DLS6DoFController has the same constructor signature; just add kp_ori.
        self.ik6 = DLS6DoFController(
            self.model,
            damping=self.ik.damping,
            kp=self.ik.kp,
            kp_ori=cfg.kp_ori,
            max_jntvel=self.ik.max_jntvel,
        )

        # Orientation tracking state
        self._prev_ori_err: float = 0.0
        self._prev_ee_quat: np.ndarray = _IDENTITY_QUAT.copy()
        # _prev_ee_angvel already set before super() call (sentinel needed there)

        # Re-infer observation space with the 6-DoF obs vector
        # (super().__init__ already called _compute_observation() but with
        # the position-only version; recompute now that ik6 exists)
        from gymnasium import spaces
        obs = self._compute_observation()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    # ── gym API overrides ──────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_ori_err = 0.0
        self._prev_ee_quat = self.ik6.ee_quat(self.data)
        self._prev_ee_angvel = np.zeros(3)
        # Recompute obs with 6-DoF method
        return self._compute_observation(), info

    def step(self, action):
        # Run the parent step (uses position-only IK internally)
        obs, reward, terminated, truncated, info = super().step(action)

        # --- Add orientation reward (post-hoc on current state) ---
        target_pos, target_vel = self._desired(self._t - self.control_dt)
        tgt_quat, _ = self._desired_ori(self._t - self.control_dt)
        ee_quat = self.ik6.ee_quat(self.data)
        ori_err_vec = quat_error(tgt_quat, ee_quat)
        ori_err = float(np.linalg.norm(ori_err_vec))

        r_ori = self.cfg.w_ori * (self._prev_ori_err - ori_err)
        reward += r_ori

        # Update orientation tracking state
        ee_quat_new = self.ik6.ee_quat(self.data)
        self._prev_ori_err = ori_err
        self._prev_ee_angvel = (
            quat_error(ee_quat_new, self._prev_ee_quat) / self.control_dt
        )
        self._prev_ee_quat = ee_quat_new

        # Extend info with orientation metrics
        info["ori_err"] = ori_err
        info["ori_err_deg"] = np.degrees(ori_err)
        info.setdefault("reward_breakdown", {})["ori"] = r_ori

        return self._compute_observation(), reward, terminated, truncated, info

    # ── 6-DoF IK command ──────────────────────────────────────────────────

    def _ik_command(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """6-DoF IK: position + orientation targets (if trajectory has orientation)."""
        ee_meas = self.disturb.perturb_ee_pos(self.data.xpos[self.hand_id].copy())
        e_pos = target_pos - ee_meas
        v_pos = target_vel + self.ik.kp * e_pos

        if self.trajectory is not None and self.trajectory.has_orientation:
            tgt_quat, tgt_angvel = self._desired_ori(self._t)
            return self.ik6.compute6d(
                self.data, target_pos, target_vel, tgt_quat, tgt_angvel
            )
        else:
            # Fall back to position-only IK (trajectory has no orientation)
            J = self.ik.jacobian(self.data)
            lam2 = self.ik.damping ** 2
            A = J @ J.T + lam2 * np.eye(3)
            q_dot = J.T @ np.linalg.solve(A, v_pos)
            return np.clip(q_dot, -self.ik.max_jntvel, self.ik.max_jntvel)

    # ── target helpers ────────────────────────────────────────────────────

    def _desired_ori(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Target orientation at time t, with smoothstep blend-in (same as _desired)."""
        if self.trajectory is None or not self.trajectory.has_orientation:
            return _IDENTITY_QUAT.copy(), np.zeros(3)

        quat, ang_vel = self.trajectory.sample_ori(t)
        T = self.cfg.ramp_in_seconds
        if T <= 0 or t >= T:
            return quat, ang_vel

        # Blend from identity to target orientation over the ramp window
        u = t / T
        s = u * u * (3 - 2 * u)
        # Slerp from identity to target quat
        dot = float(np.dot(_IDENTITY_QUAT, quat))
        if dot < 0:
            quat = -quat; dot = -dot
        dot = min(dot, 1.0)
        if dot > 0.9995:
            blended_quat = _IDENTITY_QUAT + s * (quat - _IDENTITY_QUAT)
        else:
            theta = np.arccos(dot)
            blended_quat = (
                np.sin((1 - s) * theta) * _IDENTITY_QUAT
                + np.sin(s * theta) * quat
            ) / np.sin(theta)
        blended_quat /= np.linalg.norm(blended_quat)
        blended_angvel = s * ang_vel
        return blended_quat, blended_angvel

    # ── 6-DoF observation ─────────────────────────────────────────────────

    def _compute_observation(self) -> np.ndarray:
        """141-D observation with orientation blocks."""
        from .disturbances import Disturbance

        q   = self.disturb.perturb_joints(self.data.qpos[:7].copy())
        qd  = self.data.qvel[:7].copy()
        ee_meas = self.disturb.perturb_ee_pos(self.data.xpos[self.hand_id].copy())

        if self.trajectory is None:
            target_pos  = ee_meas.copy()
            target_vel  = np.zeros(3)
            target_quat = _IDENTITY_QUAT.copy()
            fine_pos_la  = np.tile(ee_meas, self.cfg.lookahead_horizon)
            fine_quat_la = np.tile(_IDENTITY_QUAT, self.cfg.lookahead_horizon)
            coarse_pos_la  = np.tile(ee_meas, self.cfg.lookahead_coarse_horizon)
            coarse_quat_la = np.tile(_IDENTITY_QUAT, self.cfg.lookahead_coarse_horizon)
        else:
            target_pos, target_vel = self._desired(self._t)
            target_quat, _ = self._desired_ori(self._t)

            # Fine lookahead: [pos(3) ‖ quat(4)] for each step in the FIFO window
            fine_pos_la = np.concatenate([
                self._desired(self._t + i * self.cfg.lookahead_dt)[0]
                for i in range(1, self.cfg.lookahead_horizon + 1)
            ])
            fine_quat_la = np.concatenate([
                self._desired_ori(self._t + i * self.cfg.lookahead_dt)[0]
                for i in range(1, self.cfg.lookahead_horizon + 1)
            ])

            # Coarse lookahead
            fine_end = self.cfg.lookahead_horizon * self.cfg.lookahead_dt
            if self.cfg.lookahead_coarse_horizon > 0:
                coarse_pos_la = np.concatenate([
                    self._desired(self._t + fine_end + i * self.cfg.lookahead_coarse_dt)[0]
                    for i in range(1, self.cfg.lookahead_coarse_horizon + 1)
                ])
                coarse_quat_la = np.concatenate([
                    self._desired_ori(self._t + fine_end + i * self.cfg.lookahead_coarse_dt)[0]
                    for i in range(1, self.cfg.lookahead_coarse_horizon + 1)
                ])
            else:
                coarse_pos_la  = np.empty(0)
                coarse_quat_la = np.empty(0)

        # Position error
        pos_err = target_pos - ee_meas

        # Orientation error (axis-angle, 3D)
        # Guard for bootstrap: ik6 is None during super().__init__ obs-shape probe.
        ee_quat = self.ik6.ee_quat(self.data) if self.ik6 is not None else _IDENTITY_QUAT.copy()
        ori_err = quat_error(target_quat, ee_quat)   # 3D axis-angle

        # Estimated EE angular velocity
        ee_angvel = self._prev_ee_angvel

        # IK joint-velocity command (6-DoF)
        # Skip during super().__init__ bootstrap (ik6 not ready yet).
        if self.trajectory is not None and self.ik6 is not None:
            ik_qdot = self._ik_command(target_pos, target_vel)
        else:
            ik_qdot = np.zeros(7)

        # Trajectory one-hot
        traj_names = list(self.cfg.trajectory_pool)
        traj_onehot = np.zeros(len(traj_names), dtype=np.float32)
        if self.trajectory is not None:
            active_cls = type(self.trajectory).__name__
            _pool_to_cls = {
                "tilted_circle":     "TiltedCircle",
                "look_at":           "LookAt",
                "lookat":            "LookAt",
                "rotating_grasp":    "RotatingGrasp",
                "random_walk_6dof":  "RandomWalk6DoF",
                "rw6dof":            "RandomWalk6DoF",
                # position-only still supported in the 6-DoF env
                "circle":            "Circle",
                "figure8":           "FigureEight",
                "moving_target":     "MovingTarget",
            }
            for i, name in enumerate(traj_names):
                if _pool_to_cls.get(name) == active_cls:
                    traj_onehot[i] = 1.0
                    break

        # Command delta history (unchanged — joint space)
        q_true = self.data.qpos[:7]
        cmd_deltas = [cmd - q_true for cmd in self._cmd_history]

        # Paired fine lookahead block: interleave pos(3) ‖ quat(4) per step → 7×5=35D
        fine_block = np.concatenate([
            np.concatenate([fine_pos_la[3*i:3*i+3], fine_quat_la[4*i:4*i+4]])
            for i in range(self.cfg.lookahead_horizon)
        ])

        # Coarse lookahead block: interleave pos(3) ‖ quat(4) per step → 7×4=28D
        if self.cfg.lookahead_coarse_horizon > 0:
            coarse_block = np.concatenate([
                np.concatenate([coarse_pos_la[3*i:3*i+3], coarse_quat_la[4*i:4*i+4]])
                for i in range(self.cfg.lookahead_coarse_horizon)
            ])
        else:
            coarse_block = np.empty(0)

        # Assemble 141-D observation
        parts = (
            [q, qd, ee_meas, pos_err, target_vel, ik_qdot,   # 30D
             ee_quat, ori_err, ee_angvel,                     # 10D
             fine_block,                                       # 35D
             coarse_block]                                     # 28D
            + list(cmd_deltas)                                 # 35D
            + [traj_onehot]                                    # 3D
        )
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    # ── trajectory kwargs ─────────────────────────────────────────────────

    def _sample_traj_kwargs(self, name: str) -> dict:
        """Randomise orientation trajectory params at reset."""
        # First try parent class (handles all position-only trajectories)
        parent_known = {
            "circle", "figure8", "figure_eight", "fig8",
            "moving_target", "moving", "unreachable",
            "square", "rectangle", "rect", "step_target", "step",
            "fast_circle", "circle_fast",
        }
        if name.lower() in parent_known:
            return super()._sample_traj_kwargs(name)

        home_ee = self.data.xpos[self.hand_id].copy()
        rng = self._rng

        if name == "tilted_circle":
            return {
                "center": home_ee,
                "radius": float(rng.uniform(0.08, 0.14)),
                "period": float(rng.uniform(4.0, 8.0)),
            }
        if name in ("look_at", "lookat"):
            # Beacon: fixed point offset from workspace centre
            beacon = home_ee + np.array([0.0,
                                         float(rng.uniform(0.15, 0.25)),
                                         float(rng.uniform(0.10, 0.20))])
            return {
                "beacon": beacon,
                "position_traj": traj_module.MovingTarget(
                    center=home_ee,
                    extent=float(rng.uniform(0.08, 0.14)),
                    duration=self.cfg.episode_seconds + 1.0,
                    cutoff_hz=float(rng.uniform(0.05, 0.12)),
                    seed=int(rng.integers(0, 1 << 30)),
                ),
            }
        if name == "rotating_grasp":
            return {
                "position": home_ee,
                "omega": float(rng.uniform(0.3, 0.7)),
                "duration": self.cfg.episode_seconds + 1.0,
            }
        if name in ("random_walk_6dof", "rw6dof"):
            return {
                "center":    home_ee,
                "extent":    float(rng.uniform(0.08, 0.14)),
                "max_angle": float(rng.uniform(np.pi / 6, np.pi / 3)),
                "duration":  self.cfg.episode_seconds + 1.0,
                "cutoff_hz": float(rng.uniform(0.05, 0.12)),
                "seed":      int(rng.integers(0, 1 << 30)),
            }

        return {}
