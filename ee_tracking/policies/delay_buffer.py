"""Delay-aware rollout buffer for PPO under actuator command delay.

Problem
-------
Standard GAE computes the advantage of action(t) using reward(t), but with a
D-step command delay, action(t) only begins affecting the environment at t+D.
Rewards from t to t+D-1 are caused by older actions already in the FIFO, not
by action(t). This creates a systematic credit mis-assignment: the policy
gradient tells action(t) to repeat whatever pattern correlated with rewards
that those older actions actually caused.

Concretely, with D=5 and γ=0.97:

    Standard GAE:  A(t) accounts for r(t), r(t+1), ..., r(t+4), r(t+5), ...
                                       ↑ caused by a(t-5)..a(t-1), not a(t)
    Delay-aware:   A(t) accounts for              r(t+5), r(t+6), ...
                                                  ↑ caused by a(t) ✓

Fix
---
Before calling the standard GAE backward pass, swap in "causal" rewards where
position t holds the reward from t+D (the first reward a(t) can affect).
Shifts that cross episode boundaries are zeroed — the causal reward for an
action near the end of an episode lies outside the episode and isn't observed.

Implementation note
-------------------
The augmented state (with cmd_delta_history) IS Markovian, so the value
function V(s) already implicitly accounts for the queued commands.  The shift
here corrects the TD-error signal δ_t = r_t + γV(s_{t+1}) - V(s_t) to use
the reward that a(t) actually causes rather than the reward caused by a(t-D).
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer


class DelayAwareRolloutBuffer(RolloutBuffer):
    """RolloutBuffer that credits action(t) with reward(t+D).

    Drop-in replacement for the standard SB3 RolloutBuffer.
    Pass as::

        PPO(...,
            rollout_buffer_class=DelayAwareRolloutBuffer,
            rollout_buffer_kwargs={"cmd_delay": 5})

    All other PPO machinery (VecNormalize, clipping, log_std) is unchanged.
    When cmd_delay=0 this is identical to the standard RolloutBuffer.
    """

    def __init__(self, *args, cmd_delay: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cmd_delay = cmd_delay

    # ------------------------------------------------------------------
    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        if self.cmd_delay <= 0:
            return super().compute_returns_and_advantage(last_values, dones)

        # Swap in causal rewards, run standard GAE, then restore.
        orig_rewards = self.rewards.copy()
        self.rewards = self._causal_rewards(orig_rewards, self.cmd_delay)
        super().compute_returns_and_advantage(last_values, dones)
        self.rewards = orig_rewards

    # ------------------------------------------------------------------
    def _causal_rewards(self, rewards: np.ndarray, D: int) -> np.ndarray:
        """Build causal reward array: causal[t] = rewards[t+D] if same episode.

        Args:
            rewards:        (buffer_size, n_envs)  raw rewards from the rollout
            D:              command delay in steps

        Returns:
            causal:         (buffer_size, n_envs)  shifted rewards; positions
                            that would cross an episode boundary are set to 0.
        """
        T, n = rewards.shape
        causal = np.zeros_like(rewards)

        if D >= T:
            # Entire rollout fits inside one delay window; no causal signal.
            return causal

        # Step 1: plain shift — causal[t] = rewards[t+D]
        causal[: T - D] = rewards[D:]

        # Step 2: zero out positions where the shift crosses an episode boundary.
        # episode_starts[s] == 1 iff step s is the first step of a new episode.
        # If any of episode_starts[t+1 .. t+D] == 1, the shift at t is invalid.
        ep = self.episode_starts  # (T, n_envs), dtype float32
        # Accumulate: for each offset d in 1..D, if episode_starts[t+d]==1 then
        # causal[t] must be zeroed.
        for d in range(1, D + 1):
            end = T - d
            if end <= 0:
                break
            # ep[d : T] has shape (T-d, n); broadcast against causal[:T-d]
            boundary = ep[d:T].astype(bool)          # (T-d, n)
            causal[:end] *= ~boundary                 # zero where boundary

        return causal
