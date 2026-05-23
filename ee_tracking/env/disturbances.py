"""Configurable disturbances applied during simulation.

Three sources of uncertainty are supported and can be combined:

  - obs_pos_noise:   Gaussian noise added to the *measured* EE position.
  - obs_jnt_noise:   Gaussian noise added to the observed joint positions.
  - act_delay:       integer step delay on the joint-velocity command
                     (a simple FIFO).  Models actuator/communication lag.

Disturbances are intentionally crude — the point is to expose the policy
to mismatch during training, not to model any particular real robot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DisturbanceConfig:
    obs_pos_noise: float = 0.0      # std (metres)
    obs_jnt_noise: float = 0.0      # std (radians)
    act_delay: int = 0              # control steps
    seed: int = 0


class Disturbance:
    def __init__(self, cfg: DisturbanceConfig, action_dim: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._delay_buf: deque[np.ndarray] = deque(maxlen=max(cfg.act_delay + 1, 1))
        self._action_dim = action_dim
        self.reset()

    def reset(self):
        self._delay_buf.clear()
        for _ in range(self.cfg.act_delay):
            self._delay_buf.append(np.zeros(self._action_dim))

    def perturb_ee_pos(self, p: np.ndarray) -> np.ndarray:
        if self.cfg.obs_pos_noise <= 0:
            return p
        return p + self.rng.normal(0.0, self.cfg.obs_pos_noise, size=p.shape)

    def perturb_joints(self, q: np.ndarray) -> np.ndarray:
        if self.cfg.obs_jnt_noise <= 0:
            return q
        return q + self.rng.normal(0.0, self.cfg.obs_jnt_noise, size=q.shape)

    def delay_action(self, a: np.ndarray) -> np.ndarray:
        if self.cfg.act_delay <= 0:
            return a
        self._delay_buf.append(a.copy())
        return self._delay_buf[0]
