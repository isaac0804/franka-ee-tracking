"""Franka end-effector tracking environment with a residual-RL action space.

Architecture
------------
The policy outputs a 7-D normalised action in [-1, 1].  An optional 2nd-order
Butterworth low-pass filters the raw action to suppress step-to-step noise;
the filtered signal is scaled by `residual_scale` (rad/s) and added as a
*non-accumulating* position offset on top of the IK setpoint.  The combined
command is then passed through a whole-pipeline delay buffer:

    IK setpoint (integrates freely every step, never touched by residual):
        q_ik(t)  = q_ik(t-1) + ik_qdot(t) * dt

    Desired setpoint (IK + residual offset, clipped to joint limits):
        q_set(t) = clip(q_ik(t) + filter(action(t)) * residual_scale * dt, limits)

    Command actually sent to the robot (delayed by cmd_delay steps):
        ctrl(t)  = q_set(t - cmd_delay)

The delay models the full sensor-to-actuator round-trip (network + controller
loop).  Both IK and residual travel through the same delay, so pure IK has no
advantage — the only way for the policy to beat IK is to use the lookahead in
its observation to *predict* where the target will be in cmd_delay steps and
pre-position accordingly.  An untrained policy (action ≈ 0) degrades to IK
with delay, which is the hardest IK baseline.

Observation (per step, total dim = 7+7+3+3+3+7 + 3*H + 7*D + |pool|)
    q                   (7)   — joint positions (noisy)
    qdot                (7)   — joint velocities
    ee_pos              (3)   — measured EE position (noisy)
    ee_pos_error        (3)   — target - measured EE
    target_vel          (3)   — desired EE velocity
    ik_qdot             (7)   — IK joint-velocity command (current step)
    lookahead_pos   (3*H)     — future target positions at lookahead_dt intervals
    cmd_delta_hist  (7*D)     — pending setpoints minus current q (oldest→newest)
                               tells the policy what movement is already queued
    traj_onehot    (|pool|)   — one-hot trajectory type (reduces critic variance)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .disturbances import Disturbance, DisturbanceConfig
from .ik_controller import DLSController
from . import trajectories


ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "mujoco_menagerie" / "franka_emika_panda"
DEFAULT_SCENE = str(ASSETS_DIR / "scene.xml")


@dataclass
class EnvConfig:
    """Environment configuration.

    The field defaults below are bare fallbacks used only when no YAML
    config is provided (e.g. in unit tests).  For actual training use
    ee_tracking/configs/default.yaml, which overrides the tuned values.
    Significant divergences from the YAML are flagged inline.
    """
    scene_xml: str = DEFAULT_SCENE
    control_hz: float = 50.0           # policy decision rate (Hz)
    episode_seconds: float = 6.0       # one full trajectory pass
    trajectory: str = "circle"
    trajectory_kwargs: dict = field(default_factory=dict)
    randomize_trajectory: bool = True  # sample from pool at each reset
    # yaml default: ["moving_target"] — pool kept narrow to maximise gradient
    trajectory_pool: tuple = ("circle", "figure8", "moving_target")

    # ── action shaping ────────────────────────────────────────────────────
    # yaml default: 0.05 rad/s — residual trims IK, doesn't compete with it
    residual_scale: float = 0.4        # rad/s — max per-joint residual velocity
    use_residual: bool = True          # False → IK-only ablation

    # ── reward weights ────────────────────────────────────────────────────
    # yaml defaults differ significantly; see default.yaml for tuned values
    w_pos: float = 5.0                 # L1 position error weight (dominant term)
    w_vel: float = 0.1                 # EE velocity error
    w_residual: float = 0.5            # penalty on ||residual||² (keeps corrections small)
    w_jerk: float = 0.001              # EE acceleration² (regulariser only)
    w_smooth: float = 0.05             # penalty on step-to-step residual change
    # delta-pos reward: positive when error decreases vs previous step.
    # Eliminates trajectory-driven baseline noise; set w_pos=0 when using.
    # Off by default — too noisy on the slow random-walk moving_target.
    w_delta_pos: float = 0.0
    # proximity bonus: w_bonus * exp(-sharpness * ||pos_err||²)
    # Off by default — was 55× larger than pos term in early runs, killed gradient.
    w_bonus: float = 0.0
    bonus_sharpness: float = 0.0

    # ── termination ───────────────────────────────────────────────────────
    fail_pos_err: float = 0.30         # m — bail if EE error exceeds this

    # ── disturbances ──────────────────────────────────────────────────────
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)

    # ── rng ───────────────────────────────────────────────────────────────
    seed: int = 0

    # ── observation ───────────────────────────────────────────────────────
    lookahead_horizon: int = 5         # number of future target positions in obs
    lookahead_dt: float = 0.10         # time between lookahead samples (s)

    # ── trajectory blend-in ───────────────────────────────────────────────
    # Smoothstep blend from home pose to trajectory over this many seconds.
    # Prevents a position step at t=0 that IK would have to chase.
    ramp_in_seconds: float = 0.5

    # ── action smoothing ──────────────────────────────────────────────────
    # 2nd-order Butterworth applied to raw policy output before execution.
    # Baked into the env so the policy trains against its own filtered output
    # and _residual_history reflects what the robot actually executed.
    # 0.0 = off.  2.0 Hz: +10% vs IK in post-hoc eval (vs +8% for EMA).
    # yaml default: 2.0
    action_filter_hz: float = 0.0


class FrankaTrackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene_xml)
        self.data = mujoco.MjData(self.model)
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        self.ik = DLSController(self.model)
        self.disturb = Disturbance(self.cfg.disturbance, action_dim=7)

        self.control_dt = 1.0 / self.cfg.control_hz
        self.sim_steps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self._rng = np.random.default_rng(self.cfg.seed)
        self.trajectory: trajectories.Trajectory | None = None
        self._t = 0.0
        self._t_steps = 0
        self._q_setpoint = np.zeros(7)
        # IK-only setpoint tracked independently of the residual so that
        # residual errors cannot accumulate into the IK path.  At every step:
        #   _ik_q_setpoint += ik_qdot * dt          (IK integrates freely)
        #   _q_setpoint     = _ik_q_setpoint + residual * dt  (offset, not integral)
        # A wrong residual at step t only displaces q_setpoint by one dt worth of
        # correction; it self-corrects at t+1 without IK having to fight it.
        self._ik_q_setpoint = np.zeros(7)
        # 2nd-order Butterworth filter state: (n_sections, 2, 7).
        # Built once in __init__, zeroed on each reset.
        self._action_filter_sos: np.ndarray | None = None
        self._action_filter_zi: np.ndarray | None = None
        if self.cfg.action_filter_hz > 0.0:
            from scipy.signal import butter, sosfilt_zi
            nyq = self.cfg.control_hz / 2.0
            self._action_filter_sos = butter(
                2, self.cfg.action_filter_hz / nyq, btype="low", output="sos"
            )
            # zi template shape: (n_sections, 2, 7) — zeroed at reset
            zi_1d = sosfilt_zi(self._action_filter_sos)  # (n_sections, 2)
            self._action_filter_zi = np.zeros((*zi_1d.shape, 7))
        self._prev_ee_pos = np.zeros(3)
        self._prev_ee_vel = np.zeros(3)
        self._prev_pos_err: float = 0.0
        self._prev_residual = np.zeros(7)
        # History of desired setpoints in the command delay buffer.
        # We need act_delay entries so the policy can observe every pending
        # residual that has been sent but not yet executed — restoring the
        # Markov property under delayed execution.
        # cmd_delta_history: the D most recent desired setpoints (q_set) minus
        # the current joint positions.  Exposes what movement is "in flight"
        # inside the delay buffer so the policy can reason about queued commands.
        # Initialised to zeros at construction; reset() fills with home-q deltas.
        self._cmd_history_len = max(1, self.cfg.disturbance.cmd_delay)
        self._cmd_history: deque[np.ndarray] = deque(
            [np.zeros(7)] * self._cmd_history_len,
            maxlen=self._cmd_history_len,
        )
        self._ee_initial = np.zeros(3)

        # rendering
        self._renderer: mujoco.Renderer | None = None

        # spaces
        obs = self._compute_observation()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    # -- gym API -----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.disturb.cfg.seed = seed
            self.disturb = Disturbance(self.disturb.cfg, action_dim=7)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        home_q = self.data.qpos[:7].copy()
        self._q_setpoint = home_q.copy()
        self._ik_q_setpoint = home_q.copy()
        if self._action_filter_zi is not None:
            self._action_filter_zi[:] = 0.0   # reset filter to rest each episode
        self._prev_ee_pos = self.data.xpos[self.hand_id].copy()
        self._prev_ee_vel = np.zeros(3)
        self._prev_pos_err: float = 0.0
        self._prev_residual = np.zeros(7)
        # Fill delay buffer with home_q so the robot doesn't lurch from a
        # zero-command transient on the first cmd_delay steps of each episode.
        self.disturb.reset(fill_value=home_q)
        self._cmd_history.clear()
        self._cmd_history.extend([home_q.copy()] * self._cmd_history_len)
        self._t = 0.0
        self._t_steps = 0

        if self.cfg.randomize_trajectory:
            name = self._rng.choice(self.cfg.trajectory_pool)
        else:
            name = self.cfg.trajectory
        kwargs = self._sample_traj_kwargs(name)
        self.trajectory = trajectories.make(name, **kwargs)
        self._ee_initial = self.data.xpos[self.hand_id].copy()

        return self._compute_observation(), {"trajectory": name}

    def _desired(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Trajectory blended in from the home EE position over `ramp_in_seconds`.

        Removes the position step at t=0 (the bare `circle` starts a radius
        away from the home pose) so the IK doesn't have to chase a transient.
        """
        pos, vel = self.trajectory.sample(t)
        T = self.cfg.ramp_in_seconds
        if T <= 0 or t >= T:
            return pos, vel
        # smoothstep blend: s in [0,1], s(0)=0, s(T)=1, s'(0)=s'(T)=0
        u = t / T
        s = u * u * (3 - 2 * u)
        s_dot = 6 * u * (1 - u) / T
        blended_pos = (1 - s) * self._ee_initial + s * pos
        blended_vel = (-s_dot) * self._ee_initial + s_dot * pos + s * vel
        return blended_pos, blended_vel

    def step(self, action: np.ndarray):
        # Apply 2nd-order Butterworth to raw policy output (if configured).
        # The filtered action is what the robot actually executes; the policy
        # trains against its own smoothed behaviour so it learns to work with
        # the filter.  r_residual and r_smooth are computed on the filtered
        # action so rewards reflect true actuator effort, not noisy output.
        raw = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        if self._action_filter_sos is not None:
            from scipy.signal import sosfilt
            filtered, self._action_filter_zi = sosfilt(
                self._action_filter_sos,
                raw.reshape(1, 7),          # (1 sample, 7 channels)
                axis=0,
                zi=self._action_filter_zi,  # (n_sections, 2, 7)
            )
            residual = filtered.reshape(7) * self.cfg.residual_scale
        else:
            residual = raw * self.cfg.residual_scale

        # IK command — uses the *measured* (noisy) EE position so observation
        # noise actually affects it, like it would on real hardware. The
        # Jacobian still comes from the simulated joint state because that
        # part is proprioceptive (joint encoders are accurate).
        target_pos, target_vel = self._desired(self._t)
        ik_qdot = self._ik_command(target_pos, target_vel)

        jnt_lo = self.model.jnt_range[:7, 0]
        jnt_hi = self.model.jnt_range[:7, 1]

        # ── IK setpoint: integrates independently, never touched by residual ──
        # This is the joint trajectory that pure IK would produce.  Keeping it
        # separate ensures that any residual error at step t is fully absorbed
        # by step t+1 — the IK path is never contaminated.
        self._ik_q_setpoint = np.clip(
            self._ik_q_setpoint + ik_qdot * self.control_dt, jnt_lo, jnt_hi
        )

        # ── Residual: non-accumulating position offset (no individual delay) ──
        correction = residual * self.control_dt if self.cfg.use_residual else np.zeros(7)
        self._q_setpoint = np.clip(self._ik_q_setpoint + correction, jnt_lo, jnt_hi)

        # ── Whole-pipeline delay: IK + residual travel through the same FIFO ──
        # This is the realistic model: the full command takes cmd_delay steps to
        # reach the actuators.  IK cannot compensate for this without prediction;
        # the residual policy can, via the lookahead in its observation.
        # With cmd_delay=0 this is a passthrough — identical to no delay.
        actual_cmd = self.disturb.delay_command(self._q_setpoint)
        self.data.ctrl[:7] = actual_cmd

        # Track what setpoints are in flight (for cmd_delta_hist in obs).
        self._cmd_history.append(self._q_setpoint.copy())
        self.data.ctrl[7] = 0.0  # gripper closed-ish, doesn't matter

        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        self._t += self.control_dt
        self._t_steps += 1

        ee_pos = self.data.xpos[self.hand_id].copy()
        ee_vel = (ee_pos - self._prev_ee_pos) / self.control_dt
        ee_acc = (ee_vel - self._prev_ee_vel) / self.control_dt

        pos_err = target_pos - ee_pos
        vel_err = target_vel - ee_vel

        # reward shaping --------------------------------------------------
        pos_err_norm = float(np.linalg.norm(pos_err))
        r_pos = -self.cfg.w_pos * pos_err_norm
        r_vel = -self.cfg.w_vel * float(np.linalg.norm(vel_err))
        r_residual = -self.cfg.w_residual * float(np.dot(residual, residual))
        r_jerk = -self.cfg.w_jerk * float(np.dot(ee_acc, ee_acc))
        r_smooth = -self.cfg.w_smooth * float(np.sum((residual - self._prev_residual) ** 2))
        # small shaped bonus for being close — keeps gradient strong near zero error
        r_bonus = (self.cfg.w_bonus * float(np.exp(-self.cfg.bonus_sharpness * np.dot(pos_err, pos_err)))
                   if self.cfg.bonus_sharpness > 0.0 else 0.0)
        # delta-pos reward: positive when error decreases vs previous step.
        # Eliminates trajectory-driven baseline noise in r_pos (SNR ~1 with absolute
        # reward) so only the policy's contribution to error reduction is rewarded.
        r_delta_pos = self.cfg.w_delta_pos * (self._prev_pos_err - pos_err_norm)
        reward = r_pos + r_vel + r_residual + r_jerk + r_smooth + r_bonus + r_delta_pos

        # termination -----------------------------------------------------
        truncated = self._t >= self.cfg.episode_seconds
        terminated = bool(np.linalg.norm(pos_err) > self.cfg.fail_pos_err)
        if terminated:
            reward -= 5.0

        info = {
            "ee_pos": ee_pos,
            "target_pos": target_pos,
            "pos_err": float(np.linalg.norm(pos_err)),
            "vel_err_norm": float(np.linalg.norm(vel_err)),
            "ik_qdot_norm": float(np.linalg.norm(ik_qdot)),
            "residual_norm": float(np.linalg.norm(residual)),
            "ee_acc_norm": float(np.linalg.norm(ee_acc)),
            "ee_acc_sq": float(np.dot(ee_acc, ee_acc)),
            "reward_breakdown": {
                "pos": r_pos, "vel": r_vel, "residual": r_residual,
                "jerk": r_jerk, "smooth": r_smooth, "bonus": r_bonus,
                "delta_pos": r_delta_pos,
            },
        }

        self._prev_ee_pos = ee_pos
        self._prev_ee_vel = ee_vel
        self._prev_pos_err = pos_err_norm
        self._prev_residual = residual.copy()

        return self._compute_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
            # Set a reasonable third-person camera pointing at the robot base
            self._cam.lookat = np.array([0.3, 0.0, 0.5])
            self._cam.distance = 1.8
            self._cam.azimuth = 135.0
            self._cam.elevation = -20.0
        # Always show the target as a red sphere; the actual EE as a blue trail
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _ik_command(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """DLS-IK against the *measured* EE position.

        Substitutes the noisy measurement for the true EE position the
        controller would otherwise use, so observation noise has a real
        effect on tracking — this is what gives the residual policy
        room to add value via implicit filtering / lookahead.
        """
        ee_meas = self.disturb.perturb_ee_pos(self.data.xpos[self.hand_id].copy())
        e = target_pos - ee_meas
        v_des = target_vel + self.ik.kp * e
        J = self.ik.jacobian(self.data)
        lam2 = self.ik.damping ** 2
        A = J @ J.T + lam2 * np.eye(3)
        q_dot = J.T @ np.linalg.solve(A, v_des)
        return np.clip(q_dot, -self.ik.max_jntvel, self.ik.max_jntvel)

    # -- helpers -----------------------------------------------------------

    def _camera_names(self):
        out = []
        for i in range(self.model.ncam):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if n:
                out.append(n)
        return out

    def _compute_observation(self) -> np.ndarray:
        q = self.disturb.perturb_joints(self.data.qpos[:7].copy())
        qd = self.data.qvel[:7].copy()
        ee_meas = self.disturb.perturb_ee_pos(self.data.xpos[self.hand_id].copy())

        if self.trajectory is None:
            target_pos = ee_meas.copy()
            target_vel = np.zeros(3)
            lookahead = np.tile(ee_meas, self.cfg.lookahead_horizon)
        else:
            target_pos, target_vel = self._desired(self._t)
            lookahead = np.concatenate([
                self._desired(self._t + i * self.cfg.lookahead_dt)[0]
                for i in range(self.cfg.lookahead_horizon)
            ])

        pos_err = target_pos - ee_meas
        ik_qdot = (
            self._ik_command(target_pos, target_vel)
            if self.trajectory is not None
            else np.zeros(7)
        )

        # Trajectory type one-hot — lets the critic condition on which trajectory
        # is active, eliminating between-trajectory return variance that would
        # otherwise make explained_variance < 0 and corrupt advantage estimates.
        traj_names = list(self.cfg.trajectory_pool)
        traj_onehot = np.zeros(len(traj_names), dtype=np.float32)
        if self.trajectory is not None:
            active_cls = type(self.trajectory).__name__  # e.g. "Circle", "FigureEight", "MovingTarget"
            # Map pool name → expected class name
            _pool_to_cls = {
                "circle": "Circle",
                "figure8": "FigureEight", "figure_eight": "FigureEight", "fig8": "FigureEight",
                "moving_target": "MovingTarget", "moving": "MovingTarget",
                "unreachable": "Unreachable",
            }
            for i, name in enumerate(traj_names):
                if _pool_to_cls.get(name) == active_cls:
                    traj_onehot[i] = 1.0
                    break

        # cmd_delta_hist: pending desired setpoints minus current joint positions.
        # Tells the policy what movement is already queued in the delay buffer
        # so it can avoid double-commanding the same correction.
        q_true = self.data.qpos[:7]
        cmd_deltas = [cmd - q_true for cmd in self._cmd_history]  # oldest→newest

        obs = np.concatenate(
            [q, qd, ee_meas, pos_err, target_vel, ik_qdot, lookahead,
             *cmd_deltas,       # (7 * cmd_delay) — pending command deltas
             traj_onehot]
        ).astype(np.float32)
        return obs

    def _sample_traj_kwargs(self, name: str) -> dict:
        """Lightly randomise trajectory params at reset for generalisation."""
        # All trajectories are centred so the home pose is on the curve
        # (we use the home EE position as the centre, with small jitter).
        # We use the current ee position as the centre.
        home_ee = self.data.xpos[self.hand_id].copy()
        if name == "circle":
            return {
                "center": home_ee,
                "radius": float(self._rng.uniform(0.08, 0.14)),
                "period": float(self._rng.uniform(4.0, 8.0)),
            }
        if name in ("figure8", "figure_eight", "fig8"):
            return {
                "center": home_ee,
                "size": float(self._rng.uniform(0.10, 0.16)),
                "period": float(self._rng.uniform(6.0, 10.0)),
            }
        if name in ("moving_target", "moving"):
            return {
                "center": home_ee,
                "extent": float(self._rng.uniform(0.08, 0.14)),
                "duration": self.cfg.episode_seconds + 1.0,
                # 0.05–0.15 Hz → EE speed ~0.03–0.10 m/s, hard enough that IK
                # lags (~25–40 mm) but not so fast it's geometrically infeasible.
                "cutoff_hz": float(self._rng.uniform(0.05, 0.15)),
                "seed": int(self._rng.integers(0, 1 << 30)),
            }
        if name == "unreachable":
            return {"center": home_ee}
        return {}
