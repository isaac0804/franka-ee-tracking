#!/usr/bin/env python3
"""Train a residual PPO policy for Franka EE tracking.

Usage:
    python train.py                                      # default config
    python train.py --config ee_tracking/configs/default.yaml --out results/run1
    python train.py --timesteps 3000000 --out results/run2

Key fixes vs prior run:
    residual_scale  0.05  (was 0.4) — residual trims IK, doesn't overpower it
    w_residual      0.5   (was 0.1) — strong deterrent against gratuitous corrections
    w_delta_pos     0.3   (was 0.0) — reward step-over-step improvement, not just abs error
    trajectory_pool moving_target only — maximise gradient signal on the hard case
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from ee_tracking.env.franka_tracking_env import EnvConfig, FrankaTrackingEnv
from ee_tracking.env.disturbances import DisturbanceConfig


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def env_config_from_dict(d: dict) -> EnvConfig:
    dist = d.get("disturbance", {})
    return EnvConfig(
        control_hz=d.get("control_hz", 50.0),
        episode_seconds=d.get("episode_seconds", 6.0),
        randomize_trajectory=d.get("randomize_trajectory", True),
        trajectory_pool=tuple(d.get("trajectory_pool", ["moving_target"])),
        residual_scale=d.get("residual_scale", 0.05),
        use_residual=d.get("use_residual", True),
        w_pos=d.get("w_pos", 0.7),
        w_vel=d.get("w_vel", 0.05),
        w_residual=d.get("w_residual", 0.5),
        w_jerk=d.get("w_jerk", 0.01),
        w_smooth=d.get("w_smooth", 0.05),
        w_delta_pos=d.get("w_delta_pos", 0.3),
        w_bonus=d.get("w_bonus", 0.5),
        bonus_sharpness=d.get("bonus_sharpness", 50.0),
        fail_pos_err=d.get("fail_pos_err", 0.30),
        lookahead_horizon=d.get("lookahead_horizon", 5),
        lookahead_dt=d.get("lookahead_dt", 0.10),
        action_ema=d.get("action_ema", 1.0),
        disturbance=DisturbanceConfig(
            obs_pos_noise=dist.get("obs_pos_noise", 0.005),
            obs_jnt_noise=dist.get("obs_jnt_noise", 0.002),
            act_delay=dist.get("act_delay", 1),
        ),
        seed=d.get("seed", 0),
    )


# ---------------------------------------------------------------------------
# Callback: log tracking metrics to TensorBoard
# ---------------------------------------------------------------------------

class TrackingCallback(BaseCallback):
    """Records mean EE error (mm) and residual norm every log_freq steps."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._pos_errs: list[float] = []
        self._residual_norms: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "pos_err" in info:
                self._pos_errs.append(float(info["pos_err"]) * 1000.0)
            if "residual_norm" in info:
                self._residual_norms.append(float(info["residual_norm"]))

        if self.num_timesteps % self.log_freq < self.training_env.num_envs:
            if self._pos_errs:
                self.logger.record("tracking/pos_err_mm", np.mean(self._pos_errs))
                self._pos_errs.clear()
            if self._residual_norms:
                self.logger.record("tracking/residual_norm", np.mean(self._residual_norms))
                self._residual_norms.clear()
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train residual PPO on Franka EE tracking")
    parser.add_argument("--config", default="ee_tracking/configs/default.yaml",
                        help="YAML config file")
    parser.add_argument("--out", default="results/run",
                        help="Output directory for model + logs")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override train.total_timesteps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_d = cfg.get("env", {})
    train_d = cfg.get("train", {})

    if args.timesteps is not None:
        train_d["total_timesteps"] = args.timesteps
    if args.seed is not None:
        train_d["seed"] = args.seed
        env_d["seed"] = args.seed

    env_config = env_config_from_dict(env_d)
    n_envs = train_d.get("n_envs", 32)
    total_timesteps = train_d.get("total_timesteps", 1_500_000)
    seed = train_d.get("seed", 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used so runs are reproducible
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"\n{'='*60}")
    print(f"  residual_scale = {env_config.residual_scale}")
    print(f"  w_residual     = {env_config.w_residual}")
    print(f"  w_delta_pos    = {env_config.w_delta_pos}")
    print(f"  trajectory     = {list(env_config.trajectory_pool)}")
    print(f"  n_envs         = {n_envs}   |   timesteps = {total_timesteps:,}")
    print(f"  out            = {out_dir}")
    print(f"{'='*60}\n")

    # Vectorised env
    def make_env():
        return FrankaTrackingEnv(env_config)

    vec_env = SubprocVecEnv([make_env] * n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO
    policy_kwargs = dict(
        net_arch=train_d.get("policy_kwargs", {}).get("net_arch", [256, 256])
    )

    model = PPO(
        policy=train_d.get("policy", "MlpPolicy"),
        env=vec_env,
        learning_rate=train_d.get("learning_rate", 3e-4),
        n_steps=train_d.get("n_steps", 2048),
        batch_size=train_d.get("batch_size", 512),
        n_epochs=train_d.get("n_epochs", 10),
        gamma=train_d.get("gamma", 0.97),
        gae_lambda=train_d.get("gae_lambda", 0.95),
        clip_range=train_d.get("clip_range", 0.2),
        ent_coef=train_d.get("ent_coef", 0.01),
        vf_coef=train_d.get("vf_coef", 0.5),
        max_grad_norm=train_d.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(out_dir / "tb"),
        verbose=1,
        seed=seed,
        device="cpu",
    )

    checkpoint_freq = max(200_000 // n_envs, 1)
    callbacks = [
        TrackingCallback(log_freq=10_000),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
            verbose=0,
        ),
    ]

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    model.save(out_dir / "final_model")
    vec_env.save(out_dir / "vecnormalize.pkl")

    print(f"\nDone in {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
    print(f"Model   → {out_dir}/final_model.zip")
    print(f"VecNorm → {out_dir}/vecnormalize.pkl")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py ablation --model {out_dir}/final_model.zip")
    print(f"  python evaluate.py rollout  --model {out_dir}/final_model.zip --trajectory moving_target")


if __name__ == "__main__":
    main()
