"""Time-varying Cartesian trajectories for end-effector tracking.

Each generator returns (position, velocity) at time t. Velocity is the
analytic derivative — used by the IK baseline and exposed as part of the
observation lookahead so the policy doesn't have to estimate it numerically.

Orientation extension
---------------------
Trajectories that also specify a target orientation implement `sample_ori(t)`
and set `has_orientation = True`.  The return convention is:

    pos, vel       = trajectory.sample(t)
    quat, ang_vel  = trajectory.sample_ori(t)   # quat: (w,x,y,z), ang_vel: rad/s

The default `sample_ori` returns (identity, zeros) so position-only
trajectories can be used in a 6-DoF env without modification.

Quaternion utilities (`quat_mul`, `axisangle_to_quat`, `quat_error`,
`rotation_from_z_to`, `mat_to_quat`) are module-level helpers available to all
trajectory subclasses.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion utilities  (MuJoCo convention: w, x, y, z)
# ---------------------------------------------------------------------------

def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p ⊗ q. Convention: (w, x, y, z)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def axisangle_to_quat(v: np.ndarray) -> np.ndarray:
    """Axis-angle vector → unit quaternion (w, x, y, z).

    `v` encodes both the rotation axis and angle: `v = angle * axis`.
    """
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0]*s, axis[1]*s, axis[2]*s])


def quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion → axis-angle vector (3D, magnitude = angle in rad).

    Ensures the returned angle is in [0, π] by flipping to positive hemisphere.
    """
    if q[0] < 0:
        q = -q  # positive hemisphere
    # clamp w for numerical safety
    w = float(np.clip(q[0], -1.0, 1.0))
    vec = q[1:]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vec_norm, w)
    return angle * vec / vec_norm


def quat_error(q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Orientation error as a 3-D axis-angle vector.

    Returns the rotation that takes `q_cur` to `q_des`.
    Magnitude = angle in radians; direction = rotation axis.
    """
    # q_err = q_des ⊗ q_cur^{-1}    (unit quat inverse = conjugate)
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    q_err = quat_mul(q_des, q_cur_inv)
    return quat_to_axisangle(q_err)


def rotation_from_z_to(z_desired: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix R such that R @ [0,0,1] = z_desired (normalised).

    Uses Rodrigues' formula. When z_desired ≈ -z the rotation is 180° around
    a stable perpendicular axis.
    """
    z_d = z_desired / (np.linalg.norm(z_desired) + 1e-12)
    z_w = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_w, z_d)
    sin_a = np.linalg.norm(cross)
    cos_a = float(np.dot(z_w, z_d))

    if sin_a < 1e-10:
        # z_d ≈ z_w (identity) or z_d ≈ -z_w (180° around x)
        if cos_a > 0:
            return np.eye(3)
        else:
            return np.diag([-1.0, 1.0, -1.0])

    axis = cross / sin_a
    K = np.array([
        [    0, -axis[2],  axis[1]],
        [ axis[2],    0,  -axis[0]],
        [-axis[1], axis[0],     0],
    ])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion (w, x, y, z). Shepperd's method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class Trajectory:
    """Base class. Subclasses implement `sample(t) -> (pos, vel)`.

    Orientation extension
    ---------------------
    Subclasses that also specify a target orientation should:
      1. Set `has_orientation = True` as a class attribute.
      2. Override `sample_ori(t)` to return (quat, ang_vel).
         quat: (w,x,y,z) unit quaternion in the world frame.
         ang_vel: 3-D angular velocity in the world frame (rad/s).
    """

    duration: float = 10.0
    has_orientation: bool = False

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample_ori(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Target orientation at time t.  Returns (quat, ang_vel).

        Default: identity quaternion, zero angular velocity.
        Override in subclasses that require orientation tracking.
        """
        return _IDENTITY_QUAT.copy(), np.zeros(3)

    def lookahead(self, t: float, dt: float, horizon: int) -> np.ndarray:
        """Return positions at t, t+dt, ..., t+(horizon-1)*dt as a flat vector."""
        pts = [self.sample(t + i * dt)[0] for i in range(horizon)]
        return np.concatenate(pts)

    def lookahead_ori(self, t: float, dt: float, horizon: int) -> np.ndarray:
        """Return quaternions at t, t+dt, ..., t+(horizon-1)*dt as a flat (4*horizon,) vector."""
        quats = [self.sample_ori(t + i * dt)[0] for i in range(horizon)]
        return np.concatenate(quats)


class Circle(Trajectory):
    """Circle in the y-z plane, centered at `center`, radius `r`, period `T`."""

    def __init__(self, center=(0.5, 0.0, 0.5), radius=0.12, period=6.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.r = float(radius)
        self.T = float(period)
        self.duration = 3 * self.T

    def sample(self, t):
        w = 2 * np.pi / self.T
        c, s = np.cos(w * t), np.sin(w * t)
        pos = self.center + np.array([0.0, self.r * c, self.r * s])
        vel = np.array([0.0, -self.r * w * s, self.r * w * c])
        return pos, vel


class FigureEight(Trajectory):
    """Lissajous figure-eight in the y-z plane (a:b = 1:2)."""

    def __init__(self, center=(0.5, 0.0, 0.5), size=0.14, period=8.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.a = float(size)
        self.T = float(period)
        self.duration = 2 * self.T

    def sample(self, t):
        w = 2 * np.pi / self.T
        pos = self.center + np.array([0.0, self.a * np.sin(w * t), self.a * np.sin(2 * w * t) / 2])
        vel = np.array([0.0, self.a * w * np.cos(w * t), self.a * w * np.cos(2 * w * t)])
        return pos, vel


class MovingTarget(Trajectory):
    """Band-limited random walk in 3D. Smooth via filtered Gaussian noise.

    The same `seed` reproduces the same trajectory, which keeps evaluation
    deterministic across the IK/residual ablation.
    """

    def __init__(self, center=(0.5, 0.0, 0.5), extent=0.12, duration=12.0,
                 cutoff_hz=0.4, seed=0, dt=0.02):
        self.center = np.asarray(center, dtype=np.float64)
        self.extent = float(extent)
        self.duration = float(duration)
        self.dt = float(dt)
        rng = np.random.default_rng(seed)
        n = int(self.duration / dt) + 16
        noise = rng.standard_normal((n, 3))
        alpha = float(np.exp(-2 * np.pi * cutoff_hz * dt))
        filtered = np.zeros_like(noise)
        for i in range(1, n):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * noise[i]
        # normalise so |filtered| <= 1
        max_abs = np.max(np.abs(filtered)) + 1e-9
        self._traj = filtered / max_abs
        self._n = n
        # Precompute velocity at each sample using dt-spaced finite differences.
        # Using eps << dt (old code: eps=1e-3 vs dt=0.02) amplified interpolation
        # artefacts into unrealistically large velocities. Central differences at
        # dt give the true band-limited velocity of the filtered signal.
        vel = np.zeros_like(self._traj)
        vel[1:-1] = (self._traj[2:] - self._traj[:-2]) / (2 * dt)
        vel[0] = (self._traj[1] - self._traj[0]) / dt
        vel[-1] = (self._traj[-1] - self._traj[-2]) / dt
        self._vel = vel  # shape (n, 3), in normalised units/s

    def _at(self, t):
        idx = t / self.dt
        i0 = int(np.floor(idx))
        i1 = i0 + 1
        if i0 < 0:
            return self._traj[0], self._vel[0]
        if i1 >= self._n:
            return self._traj[-1], self._vel[-1]
        a = idx - i0
        pos = (1 - a) * self._traj[i0] + a * self._traj[i1]
        vel = (1 - a) * self._vel[i0] + a * self._vel[i1]
        return pos, vel

    def sample(self, t):
        p_norm, v_norm = self._at(t)
        return self.center + self.extent * p_norm, self.extent * v_norm


class Unreachable(Trajectory):
    """Circle that periodically stretches outside the reachable workspace.

    Useful to test how the policy handles infeasible targets — the IK
    saturates, so this is exactly where a learned residual can shine
    (e.g. by slowing down rather than chattering).
    """

    def __init__(self, center=(0.5, 0.0, 0.5), radius_min=0.10, radius_max=0.55,
                 period=6.0, stretch_period=18.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.r0 = float(radius_min)
        self.r1 = float(radius_max)
        self.T = float(period)
        self.S = float(stretch_period)
        self.duration = self.S

    def sample(self, t):
        # radius oscillates between r0 and r1
        r = self.r0 + 0.5 * (self.r1 - self.r0) * (1 - np.cos(2 * np.pi * t / self.S))
        dr = 0.5 * (self.r1 - self.r0) * (2 * np.pi / self.S) * np.sin(2 * np.pi * t / self.S)
        w = 2 * np.pi / self.T
        c, s = np.cos(w * t), np.sin(w * t)
        pos = self.center + np.array([0.0, r * c, r * s])
        vel = np.array([0.0, dr * c - r * w * s, dr * s + r * w * c])
        return pos, vel


class StepTarget(Trajectory):
    """Sequence of random waypoints held for `dwell_seconds` each — OOD.

    Models pick-and-place style targets: discrete position jumps with zero
    velocity during each hold period.  Velocity is always zero (stationary
    target between jumps).

    Why this is an interesting OOD test
    ------------------------------------
    At dwell time T_k the target jumps instantly.  The fine lookahead sees the
    upcoming jump *before* the delay window closes:

        t = T_k − 0.10 s  →  fine[4] already shows the new waypoint
        t = T_k − 0.08 s  →  fine[3..4] show new waypoint  …

    A policy with the delay-aware structural prior can pre-queue commands
    pointing to the new waypoint 100ms before the step executes.  The IK and
    a flat MLP both react only at t = T_k and then wait 100ms for the command
    to arrive — guaranteed overshoot on the old side, then undershoot on the new.
    """

    def __init__(self, center=(0.5, 0.0, 0.5), reach=0.12, dwell_seconds=1.0,
                 n_waypoints=8, seed=0):
        self.center        = np.asarray(center, dtype=np.float64)
        self.reach         = float(reach)
        self.dwell         = float(dwell_seconds)
        self.duration      = n_waypoints * dwell_seconds

        # Uniform random waypoints inside a sphere of radius `reach`
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < n_waypoints:
            p = rng.uniform(-1.0, 1.0, 3)
            if np.linalg.norm(p) <= 1.0:
                pts.append(self.center + self.reach * p)
        self._waypoints = np.array(pts)   # (n_waypoints, 3)

    def sample(self, t: float):
        idx = min(int(t / self.dwell), len(self._waypoints) - 1)
        return self._waypoints[idx], np.zeros(3)


class FastCircle(Circle):
    """Circle at 2× training speed — OOD.

    Training circle period: 4–8 s.  FastCircle period: 2–4 s (half range).
    At 2× speed the 100ms delay causes twice the spatial lag compared to the
    training distribution, testing whether the learned compensation scales.

    Inherits all Circle geometry; only the default period changes.
    """

    def __init__(self, center=(0.5, 0.0, 0.5), radius=0.12, period=3.0):
        super().__init__(center=center, radius=radius, period=period)
        self.duration = 3 * self.T   # three full laps


class Square(Trajectory):
    """Square path in the y-z plane — OOD trajectory (not in training pool).

    Travels counterclockwise: right→top→left→bottom→right.
    Velocity is piecewise-constant (magnitude `side/T_edge`) with instantaneous
    direction changes at each corner.  The policy's fine lookahead can *see*
    the upcoming corner 100ms ahead and pre-steer; the delayed IK cannot.
    The one-hot in the observation will be all-zeros (unknown trajectory type).
    """

    def __init__(self, center=(0.5, 0.0, 0.5), side=0.16, period=8.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.side   = float(side)
        self.T      = float(period)
        self.duration = 3 * self.T

    def sample(self, t: float):
        s = self.side / 2.0
        # Corners (y, z) going counterclockwise: BR → TR → TL → BL
        corners = np.array([
            [0.0,  s, -s],
            [0.0,  s,  s],
            [0.0, -s,  s],
            [0.0, -s, -s],
        ])
        phase     = (t % self.T) / self.T       # [0, 1)
        seg       = int(phase * 4) % 4          # which edge 0-3
        seg_phase = (phase * 4) % 1.0           # how far along this edge

        start = corners[seg]
        end   = corners[(seg + 1) % 4]
        pos = self.center + start + seg_phase * (end - start)
        vel = (end - start) * (4.0 / self.T)   # constant velocity vector on this edge
        return pos, vel


class Rectangle(Trajectory):
    """Rectangular path (2:1 aspect ratio) in the y-z plane — OOD trajectory.

    Wider than tall, tests asymmetric tracking.  Otherwise same structure as
    Square (piecewise-constant velocity, hard corners).
    """

    def __init__(self, center=(0.5, 0.0, 0.5), width=0.20, height=0.10, period=9.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.w = float(width)   # y-extent
        self.h = float(height)  # z-extent
        self.T = float(period)
        self.duration = 3 * self.T

    def sample(self, t: float):
        yw, zh = self.w / 2.0, self.h / 2.0
        perimeter = 2 * (self.w + self.h)

        # Corners (y, z) going counterclockwise: BR → TR → TL → BL
        corners = np.array([
            [0.0,  yw, -zh],
            [0.0,  yw,  zh],
            [0.0, -yw,  zh],
            [0.0, -yw, -zh],
        ])
        # Edge lengths and durations
        edge_lens = np.array([self.h, self.w, self.h, self.w])
        edge_durs = edge_lens / perimeter * self.T   # time allocated per edge

        phase = t % self.T
        cum = 0.0
        for seg in range(4):
            if phase < cum + edge_durs[seg] or seg == 3:
                seg_phase = (phase - cum) / edge_durs[seg]
                seg_phase = np.clip(seg_phase, 0.0, 1.0)
                start = corners[seg]
                end   = corners[(seg + 1) % 4]
                pos = self.center + start + seg_phase * (end - start)
                vel = (end - start) / edge_durs[seg]
                return pos, vel
            cum += edge_durs[seg]
        # fallback
        return self.center + corners[0], np.zeros(3)


# ===========================================================================
# Orientation trajectories (Task 1–5 from TASKS.md)
# ===========================================================================

class UprightConstraint(Trajectory):
    """Task 1 — Upright Constraint.

    Wraps any position trajectory and overlays a constant "keep upright"
    orientation target: the EE z-axis must stay aligned with world +z
    (identity quaternion).

    Models carrying a full cup/tray while reaching.  The position challenge
    is identical to the wrapped trajectory; the orientation task is purely a
    constraint — the IK baseline ignores it, giving the RL policy room to add
    value through orientation-aware corrections.

    Args:
        position_traj:  any `Trajectory` instance supplying pos/vel.
    """

    has_orientation: bool = True

    def __init__(self, position_traj: Trajectory):
        self._pos = position_traj
        self.duration = position_traj.duration

    def sample(self, t: float):
        return self._pos.sample(t)

    def sample_ori(self, t: float):
        # Target: EE z-axis ≡ world +z, no angular motion required.
        return _IDENTITY_QUAT.copy(), np.zeros(3)


class TiltedCircle(Trajectory):
    """Task 2 — Tilted Circle.

    Circle in the y-z plane (same geometry as `Circle`) with a continuously
    rotating orientation target: at angle θ(t) the EE z-axis must point
    **inward** toward the circle centre — like a drill tip perpendicular to
    the inside wall of a cylinder.

    The coupled pos→ori relationship is analytically exact, so the transformer's
    paired slot tokens (fine_pos[i] ‖ fine_quat[i] ‖ cmd[i]) encode the exact
    causal link without any approximation.

    Orientation derivation:
        At angle θ the EE is at [0, r cosθ, r sinθ] relative to centre.
        The inward radial direction is -[0, cosθ, sinθ].
        We want the EE z-axis aligned with this direction.
        R = rotation_from_z_to(−[0, cosθ, sinθ])
        q = mat_to_quat(R)

    Angular velocity (exact derivative):
        ω(t) = dθ/dt × x̂  (rotation around world x-axis as EE moves around circle)
    """

    has_orientation: bool = True

    def __init__(self, center=(0.5, 0.0, 0.5), radius=0.12, period=6.0):
        self._circle = Circle(center=center, radius=radius, period=period)
        self.duration = self._circle.duration
        self._omega = 2 * np.pi / period   # angular rate rad/s

    def sample(self, t: float):
        return self._circle.sample(t)

    def sample_ori(self, t: float):
        theta = self._omega * t
        # Inward-pointing unit vector in the y-z plane
        z_desired = np.array([0.0, -np.cos(theta), -np.sin(theta)])
        R = rotation_from_z_to(z_desired)
        q = mat_to_quat(R)
        # Angular velocity: circle rotates at ω around world x-axis
        ang_vel = np.array([self._omega, 0.0, 0.0])
        return q, ang_vel


class LookAt(Trajectory):
    """Task 3 — Look-At Tracking.

    Position follows a band-limited random walk (`MovingTarget`); orientation
    must always point the EE z-axis toward a fixed **beacon** point in space.

    Models a camera on a robotic gimbal that must keep a point of interest
    in frame while the body moves freely.

    Orientation derivation:
        d(t) = beacon − pos(t)            look direction
        z_desired = d(t) / ‖d(t)‖
        R = rotation_from_z_to(z_desired)
        q = mat_to_quat(R)

    Angular velocity (first-order approximation via finite difference):
        We can't differentiate pos analytically through the look-at mapping,
        so ang_vel is estimated numerically with eps = 1e-4 s.
        This is accurate to O(eps²) for smooth trajectories.

    Args:
        beacon:          fixed world-frame point the EE must look toward.
        position_traj:   `MovingTarget` (or any position trajectory) to wrap.
    """

    has_orientation: bool = True

    def __init__(self, position_traj: Trajectory | None = None,
                 beacon=(0.5, 0.2, 0.7), **kwargs):
        if position_traj is None:
            position_traj = MovingTarget(**kwargs)
        self._pos = position_traj
        self.beacon = np.asarray(beacon, dtype=np.float64)
        self.duration = position_traj.duration
        self._eps = 1e-4   # finite-difference step for angular velocity

    def sample(self, t: float):
        return self._pos.sample(t)

    def _look_quat(self, t: float) -> np.ndarray:
        pos, _ = self._pos.sample(t)
        d = self.beacon - pos
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return _IDENTITY_QUAT.copy()
        z_desired = d / norm
        R = rotation_from_z_to(z_desired)
        return mat_to_quat(R)

    def sample_ori(self, t: float):
        q = self._look_quat(t)
        # Numerical angular velocity: ω ≈ 2 * log(q_{t+eps} ⊗ q_t^{-1}) / eps
        q_next = self._look_quat(t + self._eps)
        err = quat_error(q_next, q)   # axis-angle from q to q_next
        ang_vel = err / self._eps
        return q, ang_vel


class RotatingGrasp(Trajectory):
    """Task 4 — Rotating Grasp.

    Position converges to a fixed grasp point; orientation rotates steadily
    around the EE z-axis (the approach axis) at constant angular rate ω.

    Models unscrewing a bolt: the hand stays at the bolt's position while
    the wrist executes a full rotation.  The Franka's 7th joint (wrist
    rotation) has a null-space contribution — the policy must learn to use
    it without disturbing the proximal position joints.

    With a 5-step delay and ω = 0.5 rad/s:
        orientation lag per step = 0.5 × 0.1 s = 0.05 rad ≈ 2.9°
    The fine orientation lookahead shows the target angle 100 ms ahead,
    allowing exact pre-compensation.

    Args:
        position:  fixed grasp point (3D world coordinates).
        omega:     angular rate around the z-axis (rad/s).  Default: 0.5 rad/s.
        duration:  episode length in seconds.
    """

    has_orientation: bool = True

    def __init__(self, position=(0.5, 0.0, 0.5), omega=0.5, duration=12.0):
        self._position = np.asarray(position, dtype=np.float64)
        self._omega = float(omega)
        self.duration = float(duration)

    def sample(self, t: float):
        return self._position.copy(), np.zeros(3)

    def sample_ori(self, t: float):
        # Rotate around world z-axis at constant rate
        q = axisangle_to_quat(np.array([0.0, 0.0, self._omega * t]))
        ang_vel = np.array([0.0, 0.0, self._omega])
        return q, ang_vel


class RandomWalk6DoF(Trajectory):
    """Task 5 — 6-DoF Random Walk.

    Independent band-limited random walks in R³ (position) and SO(3)
    (orientation).  This is the hardest and most general orientation task —
    the direct 6-DoF analogue of `MovingTarget`.

    Position: identical to `MovingTarget` (filtered Gaussian noise, scaled
    to `extent` metres amplitude).

    Orientation: SO(3) random walk via filtered axis-angle integration.
        1.  Generate 3D Gaussian noise in so(3) (tangent space at identity).
        2.  Apply the same 1st-order low-pass filter as position.
        3.  Scale to `max_angle` radians amplitude.
        4.  Integrate as incremental rotations:
                q[0] = identity
                dq[i] = axisangle_to_quat(filtered_so3[i] * dt)
                q[i]  = dq[i] ⊗ q[i-1]          (left-multiply: world frame)
                q[i]  /= ‖q[i]‖                  (renormalise)
        5.  Angular velocity: the instantaneous axis-angle rate filtered_so3[i].

    Args:
        center:     workspace centre (3D).
        extent:     position amplitude (metres), default 0.12.
        max_angle:  orientation amplitude (radians), default π/4 ≈ 45°.
        duration:   trajectory length in seconds.
        cutoff_hz:  low-pass cutoff for both position and orientation noise.
        seed:       RNG seed for reproducibility.
        dt:         discretisation step (should match `control_dt`).
    """

    has_orientation: bool = True

    def __init__(self, center=(0.5, 0.0, 0.5), extent=0.12, max_angle=np.pi / 4,
                 duration=12.0, cutoff_hz=0.08, seed=0, dt=0.02):
        self.center   = np.asarray(center, dtype=np.float64)
        self.extent   = float(extent)
        self.max_angle = float(max_angle)
        self.duration = float(duration)
        self.dt       = float(dt)

        rng = np.random.default_rng(seed)
        n   = int(duration / dt) + 16
        alpha = float(np.exp(-2 * np.pi * cutoff_hz * dt))

        # --- Position random walk (same as MovingTarget) ---
        noise_pos = rng.standard_normal((n, 3))
        filt_pos  = np.zeros_like(noise_pos)
        for i in range(1, n):
            filt_pos[i] = alpha * filt_pos[i - 1] + (1 - alpha) * noise_pos[i]
        max_abs_pos = np.max(np.abs(filt_pos)) + 1e-9
        self._traj_pos = filt_pos / max_abs_pos   # in [-1, 1] per axis

        # Position velocity via central differences
        vel_pos = np.zeros_like(self._traj_pos)
        vel_pos[1:-1] = (self._traj_pos[2:] - self._traj_pos[:-2]) / (2 * dt)
        vel_pos[0]    = (self._traj_pos[1]  - self._traj_pos[0])   / dt
        vel_pos[-1]   = (self._traj_pos[-1] - self._traj_pos[-2])  / dt
        self._vel_pos = vel_pos

        # --- Orientation random walk in SO(3) ---
        noise_ori = rng.standard_normal((n, 3))
        filt_ori  = np.zeros_like(noise_ori)
        for i in range(1, n):
            filt_ori[i] = alpha * filt_ori[i - 1] + (1 - alpha) * noise_ori[i]
        # Scale to max_angle amplitude
        max_abs_ori = np.max(np.linalg.norm(filt_ori, axis=1)) + 1e-9
        filt_ori = filt_ori / max_abs_ori   # each row is a unit-scale axis-angle velocity

        # Integrate incremental rotations
        quats = np.zeros((n, 4))
        quats[0] = _IDENTITY_QUAT
        for i in range(1, n):
            dq = axisangle_to_quat(filt_ori[i] * dt)
            quats[i] = quat_mul(dq, quats[i - 1])
            nrm = np.linalg.norm(quats[i])
            if nrm > 1e-10:
                quats[i] /= nrm

        self._traj_ori = quats              # (n, 4) quaternion table
        self._vel_ori  = filt_ori           # (n, 3) axis-angle rates (world frame)
        self._n = n

    def _at(self, t: float):
        """Interpolate position + orientation at time t."""
        idx = t / self.dt
        i0 = int(np.floor(idx))
        i1 = i0 + 1
        if i0 < 0:
            return self._traj_pos[0], self._vel_pos[0], self._traj_ori[0], self._vel_ori[0]
        if i1 >= self._n:
            return self._traj_pos[-1], self._vel_pos[-1], self._traj_ori[-1], self._vel_ori[-1]
        a = idx - i0

        pos  = (1 - a) * self._traj_pos[i0] + a * self._traj_pos[i1]
        vpos = (1 - a) * self._vel_pos[i0]  + a * self._vel_pos[i1]

        # SLERP between neighbouring quaternions
        q0, q1 = self._traj_ori[i0], self._traj_ori[i1]
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1 = -q1; dot = -dot
        dot = min(dot, 1.0)
        if dot > 0.9995:
            q = q0 + a * (q1 - q0)
        else:
            theta = np.arccos(dot)
            q = (np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
        q = q / (np.linalg.norm(q) + 1e-12)

        vori = (1 - a) * self._vel_ori[i0] + a * self._vel_ori[i1]
        return pos, vpos, q, vori

    def sample(self, t: float):
        pos, vpos, _, _ = self._at(t)
        return self.center + self.extent * pos, self.extent * vpos

    def sample_ori(self, t: float):
        _, _, q, vori = self._at(t)
        return q.copy(), self.max_angle * vori


def make(name: str, **kwargs) -> Trajectory:
    name = name.lower()
    if name == "circle":
        return Circle(**kwargs)
    if name in ("figure8", "figure_eight", "fig8"):
        return FigureEight(**kwargs)
    if name in ("moving_target", "moving"):
        return MovingTarget(**kwargs)
    if name == "unreachable":
        return Unreachable(**kwargs)
    if name == "square":
        return Square(**kwargs)
    if name in ("rectangle", "rect"):
        return Rectangle(**kwargs)
    if name in ("step_target", "step"):
        return StepTarget(**kwargs)
    if name in ("fast_circle", "circle_fast"):
        return FastCircle(**kwargs)
    # ── Orientation trajectories ──────────────────────────────────────────
    if name in ("tilted_circle",):
        return TiltedCircle(**kwargs)
    if name in ("look_at", "lookat"):
        return LookAt(**kwargs)
    if name in ("rotating_grasp",):
        return RotatingGrasp(**kwargs)
    if name in ("random_walk_6dof", "rw6dof"):
        return RandomWalk6DoF(**kwargs)
    if name in ("upright_constraint", "upright"):
        # kwargs are forwarded to the inner position trajectory.
        # Pass `inner="circle"` (default) to choose which trajectory to wrap.
        inner_name = kwargs.pop("inner", "circle")
        return UprightConstraint(make(inner_name, **kwargs))
    raise ValueError(f"unknown trajectory: {name}")


def make_upright(position_traj_name: str, **kwargs) -> UprightConstraint:
    """Convenience factory for Task 1: wrap a position trajectory with upright constraint."""
    return UprightConstraint(make(position_traj_name, **kwargs))
