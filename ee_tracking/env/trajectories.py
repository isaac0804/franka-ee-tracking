"""Time-varying Cartesian trajectories for end-effector tracking.

Each generator returns (position, velocity) at time t. Velocity is the
analytic derivative — used by the IK baseline and exposed as part of the
observation lookahead so the policy doesn't have to estimate it numerically.
"""

from __future__ import annotations

import numpy as np


class Trajectory:
    """Base class. Subclasses implement `sample(t) -> (pos, vel)`."""

    duration: float = 10.0

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def lookahead(self, t: float, dt: float, horizon: int) -> np.ndarray:
        """Return positions at t, t+dt, ..., t+(horizon-1)*dt as a flat vector."""
        pts = [self.sample(t + i * dt)[0] for i in range(horizon)]
        return np.concatenate(pts)


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
    raise ValueError(f"unknown trajectory: {name}")
