#!/usr/bin/env python3
"""Evaluate a trained residual PPO policy against the IK baseline.

Modes:
    rollout   Single deterministic episode — 3D path + error-over-time plot.
    ablation  IK-only vs IK+residual across trajectories — comparison table + plot.

Usage:
    python evaluate.py rollout  --model results/run1/final_model.zip
    python evaluate.py rollout  --model results/run1/final_model.zip --trajectory circle
    python evaluate.py ablation --model results/run1/final_model.zip
    python evaluate.py ablation --model results/run1/final_model.zip --trajectories moving_target,circle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ee_tracking.env.franka_tracking_env import EnvConfig, FrankaTrackingEnv
from ee_tracking.env.disturbances import DisturbanceConfig

ALL_TRAJECTORIES = ["moving_target", "circle", "figure8", "unreachable"]

# Fallback disturbance when no saved config is available.
_DEFAULT_DISTURBANCE = DisturbanceConfig(obs_pos_noise=0.005, obs_jnt_noise=0.002, cmd_delay=5)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_kwargs_from_cfg(saved_cfg: dict) -> dict:
    """Extract ALL observation-space-affecting env params from a saved config dict.

    These must match the training env exactly or VecNormalize will have the
    wrong input dimension and model.predict() will crash.

    Affected obs dimensions:
      - trajectory_pool  → one-hot length
      - lookahead_horizon → 3 * horizon elements
      - cmd_delay        → 7 * max(1, delay) cmd-delta-history elements
    """
    env = saved_cfg.get("env", {})
    dist = env.get("disturbance", {})
    return dict(
        trajectory_pool=tuple(env.get("trajectory_pool", ["moving_target"])),
        lookahead_horizon=int(env.get("lookahead_horizon", 5)),
        lookahead_dt=float(env.get("lookahead_dt", 0.10)),
        action_filter_hz=float(env.get("action_filter_hz", 0.0)),
        disturbance=DisturbanceConfig(
            obs_pos_noise=float(dist.get("obs_pos_noise", 0.005)),
            obs_jnt_noise=float(dist.get("obs_jnt_noise", 0.002)),
            # cmd_delay is new name; fall back to act_delay for old saved configs
            cmd_delay=int(dist.get("cmd_delay", dist.get("act_delay", 5))),
        ),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load PPO model + freeze VecNormalize stats (if available).

    Returns (model, vn_ref, saved_cfg).

    Reads config.yaml saved alongside the model so the temp env used to load
    VecNormalize has the EXACT same observation space as training.  All three
    axes that change obs dim are restored: trajectory_pool, lookahead_horizon,
    and act_delay.
    """
    import yaml
    model_p = Path(model_path)
    vn_path = model_p.parent / "vecnormalize.pkl"
    cfg_path = model_p.parent / "config.yaml"

    model = PPO.load(str(model_p), device="cpu")

    saved_cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved_cfg = yaml.safe_load(f) or {}

    if vn_path.exists():
        kwargs = _env_kwargs_from_cfg(saved_cfg)
        tmp_cfg = EnvConfig(**kwargs)
        tmp = DummyVecEnv([lambda: FrankaTrackingEnv(tmp_cfg)])
        vn_ref = VecNormalize.load(str(vn_path), tmp)
        vn_ref.training = False
        vn_ref.norm_reward = False
        tmp.close()
    else:
        vn_ref = None

    return model, vn_ref, saved_cfg


def wrap_eval_env(env: FrankaTrackingEnv, vn_ref) -> DummyVecEnv | VecNormalize:
    """Wrap a single env for evaluation, applying frozen normalisation stats."""
    venv = DummyVecEnv([lambda: env])  # noqa: B023
    if vn_ref is not None:
        venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
        venv.obs_rms = vn_ref.obs_rms
        venv.ret_rms = vn_ref.ret_rms
        venv.clip_obs = vn_ref.clip_obs
    return venv


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _eval_config(
    trajectory: str,
    use_residual: bool,
    seed: int,
    trajectory_pool: tuple = ("moving_target",),
    lookahead_horizon: int = 5,
    lookahead_dt: float = 0.10,
    action_filter_hz: float = 0.0,
    disturbance: DisturbanceConfig | None = None,
) -> EnvConfig:
    return EnvConfig(
        trajectory=trajectory,
        randomize_trajectory=False,
        use_residual=use_residual,
        disturbance=disturbance if disturbance is not None else _DEFAULT_DISTURBANCE,
        trajectory_pool=trajectory_pool,
        seed=seed,
        lookahead_horizon=lookahead_horizon,
        lookahead_dt=lookahead_dt,
        action_filter_hz=action_filter_hz,
    )


def run_residual(
    model,
    vn_ref,
    trajectory: str,
    seed: int = 42,
    trajectory_pool: tuple = ("moving_target",),
    lookahead_horizon: int = 5,
    lookahead_dt: float = 0.10,
    action_filter_hz: float = 0.0,
    disturbance: DisturbanceConfig | None = None,
    action_ema_posthoc: float = 1.0,
) -> dict:
    """Run one episode with the trained policy.

    All env kwargs must match the training config so the obs dim aligns with
    the loaded model / VecNormalize statistics.

    action_filter_hz: restored from training config — the baked-in Butterworth
        cutoff the env applies internally.  Do not set manually.

    action_ema_posthoc: additional EMA applied at inference time before the env
        sees the action.  For post-hoc experiments on models trained without a
        baked filter (action_filter_hz=0).  Default 1.0 = off.
    """
    cfg = _eval_config(
        trajectory, use_residual=True, seed=seed,
        trajectory_pool=trajectory_pool,
        lookahead_horizon=lookahead_horizon,
        lookahead_dt=lookahead_dt,
        action_filter_hz=action_filter_hz,
        disturbance=disturbance,
    )
    env = FrankaTrackingEnv(cfg)
    venv = wrap_eval_env(env, vn_ref)

    obs = venv.reset()
    ee_pos, tgt_pos, err_mm, res_norms = [], [], [], []
    smoothed_action: np.ndarray | None = None   # lazily initialised to match action shape

    while True:
        raw_action, _ = model.predict(obs, deterministic=True)
        if action_ema_posthoc < 1.0:
            if smoothed_action is None:
                smoothed_action = raw_action.copy()
            else:
                smoothed_action = (1.0 - action_ema_posthoc) * smoothed_action + action_ema_posthoc * raw_action
            action = smoothed_action
        else:
            action = raw_action

        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        ee_pos.append(info["ee_pos"].copy())
        tgt_pos.append(info["target_pos"].copy())
        err_mm.append(float(info["pos_err"]) * 1000.0)
        res_norms.append(float(info.get("residual_norm", 0.0)))
        if dones[0]:
            break

    venv.close()
    return _metrics(np.array(ee_pos), np.array(tgt_pos), np.array(err_mm), np.array(res_norms))


def run_ik(
    trajectory: str,
    seed: int = 42,
    disturbance: DisturbanceConfig | None = None,
) -> dict:
    """Run one episode with pure IK (zero residual action).

    Uses the same disturbance as the residual policy so the comparison is
    fair within each sweep config (e.g. delay_0 compares both under delay=0).
    """
    cfg = _eval_config(trajectory, use_residual=False, seed=seed, disturbance=disturbance)
    env = FrankaTrackingEnv(cfg)
    obs, _ = env.reset(seed=seed)

    ee_pos, tgt_pos, err_mm = [], [], []
    while True:
        obs, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        ee_pos.append(info["ee_pos"].copy())
        tgt_pos.append(info["target_pos"].copy())
        err_mm.append(float(info["pos_err"]) * 1000.0)
        if terminated or truncated:
            break

    env.close()
    return _metrics(np.array(ee_pos), np.array(tgt_pos), np.array(err_mm), np.zeros(len(err_mm)))


def _metrics(ee_pos, tgt_pos, err_mm, res_norms) -> dict:
    settle = max(1, int(len(err_mm) / 6))  # discard first ~1 s (50 steps at 50 Hz)
    return {
        "rmse_mm": float(np.sqrt(np.mean(err_mm ** 2))),
        "settled_rmse_mm": float(np.sqrt(np.mean(err_mm[settle:] ** 2))),
        "max_mm": float(np.max(err_mm)),
        "mean_residual_norm": float(np.mean(res_norms)),
        # arrays kept for plotting — stripped before JSON serialisation
        "_ee_pos": ee_pos,
        "_tgt_pos": tgt_pos,
        "_err_mm": err_mm,
    }


def _strip_arrays(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rollout(result: dict, traj_name: str, save_path: Path):
    ee = result["_ee_pos"]
    tgt = result["_tgt_pos"]
    err = result["_err_mm"]
    t = np.arange(len(err)) / 50.0

    fig = plt.figure(figsize=(12, 4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], "k--", lw=1, alpha=0.5, label="desired")
    ax3.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="steelblue", lw=1.2, label="actual")
    ax3.set_title(f"{traj_name}   RMSE {result['rmse_mm']:.1f} mm")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.legend(fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(t, err, color="steelblue", lw=1)
    ax.axhline(result["rmse_mm"], color="red", ls="--", lw=1,
               label=f"RMSE {result['rmse_mm']:.1f} mm")
    ax.axhline(result["settled_rmse_mm"], color="orange", ls=":", lw=1,
               label=f"settled {result['settled_rmse_mm']:.1f} mm")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("EE error (mm)")
    ax.set_title("Tracking error vs time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {save_path}")


def plot_ablation(results: dict, save_path: Path):
    trajs = list(results.keys())
    ncols = min(2, len(trajs))
    nrows = (len(trajs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for i, traj in enumerate(trajs):
        ax = axes[i // ncols][i % ncols]
        r = results[traj]
        t = np.arange(len(r["ik"]["_err_mm"])) / 50.0
        improv = r["improvement_pct"]

        ax.plot(t, r["ik"]["_err_mm"], color="red", lw=1, alpha=0.8,
                label=f"IK  {r['ik']['settled_rmse_mm']:.1f} mm")
        ax.plot(t, r["residual"]["_err_mm"], color="steelblue", lw=1, alpha=0.8,
                label=f"IK + residual  {r['residual']['settled_rmse_mm']:.1f} mm")
        ax.set_title(f"{traj}   Δ = {improv:+.1f} %")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("EE error (mm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(len(trajs), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    plt.suptitle("IK-only  vs  IK + Residual PPO", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {save_path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_rollout(args):
    model, vn_ref, saved_cfg = load_model(args.model)
    env_kwargs = _env_kwargs_from_cfg(saved_cfg)
    traj = args.trajectory
    posthoc_ema = args.action_ema
    print(f"rollout: {traj} (residual policy, baked_hz={env_kwargs['action_filter_hz']}, posthoc_ema={posthoc_ema}) ...")
    result = run_residual(model, vn_ref, traj, action_ema_posthoc=posthoc_ema, **env_kwargs)

    print(json.dumps(_strip_arrays(result), indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_rollout(result, traj, out / f"rollout_{traj}.png")
    with open(out / f"rollout_{traj}.json", "w") as f:
        json.dump(_strip_arrays(result), f, indent=2)


def cmd_ablation(args):
    model, vn_ref, saved_cfg = load_model(args.model)
    env_kwargs = _env_kwargs_from_cfg(saved_cfg)
    trajs = args.trajectories.split(",") if args.trajectories else ALL_TRAJECTORIES
    posthoc_ema = args.action_ema

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    baked = env_kwargs["action_filter_hz"]
    ema_tag = f"  [baked={baked}" + (f", posthoc={posthoc_ema}]" if posthoc_ema < 1.0 else "]")
    results = {}
    print(f"\n{'trajectory':<16} {'IK (mm)':>10} {'residual (mm)':>14} {'Δ':>8}{ema_tag}")
    print("-" * 52)

    for traj in trajs:
        # Both IK and residual use the training disturbance so the comparison
        # is fair within each config (e.g. delay_0 compares both under delay=0).
        ik = run_ik(traj, disturbance=env_kwargs["disturbance"])
        res = run_residual(model, vn_ref, traj, action_ema_posthoc=posthoc_ema, **env_kwargs)
        improv = (ik["settled_rmse_mm"] - res["settled_rmse_mm"]) / ik["settled_rmse_mm"] * 100
        results[traj] = {"ik": ik, "residual": res, "improvement_pct": float(improv)}
        marker = " ✓" if improv > 0 else ""
        print(f"{traj:<16} {ik['settled_rmse_mm']:>10.1f} {res['settled_rmse_mm']:>14.1f} "
              f"{improv:>+7.1f}%{marker}")

    print()
    plot_ablation(results, out / "ablation.png")

    save = {t: {"ik_settled_rmse_mm": r["ik"]["settled_rmse_mm"],
                "residual_settled_rmse_mm": r["residual"]["settled_rmse_mm"],
                "improvement_pct": r["improvement_pct"]}
            for t, r in results.items()}
    with open(out / "ablation.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"  json  → {out}/ablation.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate residual PPO policy")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("rollout", help="Single deterministic rollout + plot")
    p.add_argument("--model", required=True, help="Path to final_model.zip")
    p.add_argument("--trajectory", default="moving_target", choices=ALL_TRAJECTORIES)
    p.add_argument("--out", default="results/eval")
    p.add_argument("--action-ema", type=float, default=1.0, dest="action_ema",
                   help="EMA coefficient on policy actions (1.0=off, 0.3=~38ms half-life)")

    p = sub.add_parser("ablation", help="IK vs residual comparison table + plot")
    p.add_argument("--model", required=True, help="Path to final_model.zip")
    p.add_argument("--trajectories", default=None,
                   help="Comma-separated subset, e.g. moving_target,circle (default: all)")
    p.add_argument("--out", default="results/eval")
    p.add_argument("--action-ema", type=float, default=1.0, dest="action_ema",
                   help="EMA coefficient on policy actions (1.0=off, 0.3=~38ms half-life)")

    args = parser.parse_args()
    {"rollout": cmd_rollout, "ablation": cmd_ablation}[args.mode](args)


if __name__ == "__main__":
    main()
