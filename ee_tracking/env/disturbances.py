"""Configurable disturbances applied during simulation.

Three sources of uncertainty are supported and can be combined:

  - obs_pos_noise:   Gaussian noise added to the *measured* EE position.
  - obs_jnt_noise:   Gaussian noise added to the observed joint positions.
  - cmd_delay:       Integer step delay on the total joint-setpoint command
                     (a simple FIFO).  Models the full sensor-to-actuator
                     round-trip latency (network, controller loop, etc.).
                     IK and residual are both delayed equally, so the only
                     way to beat IK is to predict the future — which the
                     policy can do via the lookahead in its observation.

Disturbances are intentionally crude — the point is to expose the policy
to mismatch during training, not to model any particular real robot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DisturbanceConfig:
    obs_pos_noise: float = 0.0      # std (metres)
    obs_jnt_noise: float = 0.0      # std (radians)
    cmd_delay: int = 0              # control steps — applied to total q_setpoint
    seed: int = 0

    # ---------------------------------------------------------------------------
    # Back-compat alias: old code used act_delay; new code uses cmd_delay.
    # Reading act_delay returns cmd_delay; setting act_delay sets cmd_delay.
    # ---------------------------------------------------------------------------
    @property
    def act_delay(self) -> int:
        return self.cmd_delay

    @act_delay.setter
    def act_delay(self, v: int) -> None:
        self.cmd_delay = v


class Disturbance:
    def __init__(self, cfg: DisturbanceConfig, action_dim: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._action_dim = action_dim
        # Buffer capacity: cmd_delay slots.  An empty delay (cmd_delay=0) is
        # handled by the passthrough branch in delay_command().
        self._delay_buf: deque[np.ndarray] = deque(maxlen=max(cfg.cmd_delay + 1, 1))
        self.reset()

    def reset(self, fill_value: Optional[np.ndarray] = None) -> None:
        """Clear the delay buffer and fill with `fill_value` (default: zeros).

        Pass the robot's home joint positions so the buffer doesn't produce a
        spurious zero-command transient at the start of each episode.
        """
        fill = np.zeros(self._action_dim) if fill_value is None else fill_value.copy()
        self._delay_buf.clear()
        for _ in range(self.cfg.cmd_delay):
            self._delay_buf.append(fill.copy())

    def perturb_ee_pos(self, p: np.ndarray) -> np.ndarray:
        if self.cfg.obs_pos_noise <= 0:
            return p
        return p + self.rng.normal(0.0, self.cfg.obs_pos_noise, size=p.shape)

    def perturb_joints(self, q: np.ndarray) -> np.ndarray:
        if self.cfg.obs_jnt_noise <= 0:
            return q
        return q + self.rng.normal(0.0, self.cfg.obs_jnt_noise, size=q.shape)

    def delay_command(self, cmd: np.ndarray) -> np.ndarray:
        """Push `cmd` into the FIFO and return the command from `cmd_delay` steps ago.

        With cmd_delay=0 this is a no-op passthrough.
        With cmd_delay=D, the robot executes the command issued D steps ago,
        giving a predictive policy a genuine advantage over reactive IK.
        """
        if self.cfg.cmd_delay <= 0:
            return cmd
        self._delay_buf.append(cmd.copy())
        return self._delay_buf[0].copy()

    # ------------------------------------------------------------------
    # Back-compat: old code called delay_action() on the residual alone.
    # Route it through delay_command() so existing call sites still work
    # (e.g. evaluate.py post-hoc paths).
    # ------------------------------------------------------------------
    def delay_action(self, a: np.ndarray) -> np.ndarray:
        return self.delay_command(a)

    @property
    def pending_commands(self) -> list[np.ndarray]:
        """Return the commands currently in the delay buffer (oldest first).

        Used by the environment to build the cmd_delta_history observation.
        """
        return list(self._delay_buf)
