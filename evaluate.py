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
from ee_tracking.policies.gelu_policy import POLICY_REGISTRY as _POLICY_REGISTRY
import ee_tracking.policies.transformer_policy  # noqa: F401 — registers transformer classes for cloudpickle on PPO.load

ALL_TRAJECTORIES = ["moving_target", "circle", "figure8", "unreachable"]
# OOD trajectories: never seen during training; one-hot is all-zeros at eval time.
OOD_TRAJECTORIES = ["square", "rectangle"]

# Fallback disturbance when no saved config is available.
_DEFAULT_DISTURBANCE = DisturbanceConfig(obs_pos_noise=0.005, obs_jnt_noise=0.002, cmd_delay=5)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_kwargs_from_cfg(saved_cfg: dict) -> dict:
    """Extract ALL env params that must match training from a saved config dict.

    Obs-space-affecting (mismatch → crash):
      - trajectory_pool       → one-hot length
      - lookahead_horizon     → 3 * horizon elements
      - lookahead_coarse_horizon → 3 * horizon elements
      - cmd_delay             → 7 * max(1, delay) cmd-delta-history elements

    Behaviour-affecting (mismatch → catastrophic performance, no crash):
      - residual_scale        → action multiplier; MUST match training or policy
                                overshoots by (default/trained) ratio (e.g. 8×)
      - lookahead_dt / lookahead_coarse_dt → which future steps are observed
    """
    env = saved_cfg.get("env", {})
    dist = env.get("disturbance", {})
    return dict(
        trajectory_pool=tuple(env.get("trajectory_pool", ["moving_target"])),
        lookahead_horizon=int(env.get("lookahead_horizon", 5)),
        lookahead_dt=float(env.get("lookahead_dt", 0.02)),
        lookahead_coarse_horizon=int(env.get("lookahead_coarse_horizon", 4)),
        lookahead_coarse_dt=float(env.get("lookahead_coarse_dt", 0.10)),
        residual_scale=float(env.get("residual_scale", 0.05)),
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
    lookahead_dt: float = 0.02,
    lookahead_coarse_horizon: int = 4,
    lookahead_coarse_dt: float = 0.10,
    residual_scale: float = 0.05,
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
        lookahead_coarse_horizon=lookahead_coarse_horizon,
        lookahead_coarse_dt=lookahead_coarse_dt,
        residual_scale=residual_scale,
        action_filter_hz=action_filter_hz,
    )


def run_residual(
    model,
    vn_ref,
    trajectory: str,
    seed: int = 42,
    trajectory_pool: tuple = ("moving_target",),
    lookahead_horizon: int = 5,
    lookahead_dt: float = 0.02,
    lookahead_coarse_horizon: int = 4,
    lookahead_coarse_dt: float = 0.10,
    residual_scale: float = 0.05,
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
        lookahead_coarse_horizon=lookahead_coarse_horizon,
        lookahead_coarse_dt=lookahead_coarse_dt,
        residual_scale=residual_scale,
        action_filter_hz=action_filter_hz,
        disturbance=disturbance,
    )
    env = FrankaTrackingEnv(cfg)
    venv = wrap_eval_env(env, vn_ref)

    obs = venv.reset()
    ee_pos, tgt_pos, err_mm, res_norms, actions = [], [], [], [], []
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
        actions.append(raw_action[0].copy())   # shape (7,) — pre-scale policy output
        if dones[0]:
            break

    venv.close()
    return _metrics(np.array(ee_pos), np.array(tgt_pos), np.array(err_mm),
                    np.array(res_norms), np.array(actions))


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


def _metrics(ee_pos, tgt_pos, err_mm, res_norms, actions=None) -> dict:
    settle = max(1, int(len(err_mm) / 6))  # discard first ~1 s (50 steps at 50 Hz)
    out = {
        "rmse_mm": float(np.sqrt(np.mean(err_mm ** 2))),
        "settled_rmse_mm": float(np.sqrt(np.mean(err_mm[settle:] ** 2))),
        "max_mm": float(np.max(err_mm)),
        "mean_residual_norm": float(np.mean(res_norms)),
        # arrays kept for plotting — stripped before JSON serialisation
        "_ee_pos": ee_pos,
        "_tgt_pos": tgt_pos,
        "_err_mm": err_mm,
    }
    if actions is not None and len(actions) > settle + 1:
        a = actions[settle:]                              # (T, 7), in [-1, 1]
        diffs = np.abs(np.diff(a, axis=0))               # (T-1, 7)
        out["action_roughness"] = float(np.mean(diffs))  # mean |a_t - a_{t-1}| per joint per step
        out["saturation_rate"]  = float(np.mean(np.abs(a) > 0.9))  # fraction of (t,joint) near ±1
        out["action_std"]       = float(np.mean(np.std(a, axis=0))) # per-joint std, then mean
        out["_actions"] = actions
    return out


def _strip_arrays(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# Trajectories that vary with seed — need multi-run averaging
_STOCHASTIC_TRAJECTORIES = {"moving_target", "moving", "unreachable"}


def run_multi(run_fn, trajectory: str, n_seeds: int = 10, base_seed: int = 0, **kwargs) -> dict:
    """Run `run_fn` over `n_seeds` consecutive seeds and aggregate scalar metrics.

    Returns the same dict shape as a single run, but scalar metrics become
    mean values; adds `*_std` and `*_seeds` keys for stochastic trajectories.
    Array fields (_ee_pos, _tgt_pos, _err_mm, _actions) are taken from seed 0.

    For deterministic trajectories (circle, figure8, square, rectangle) n_seeds
    is silently clamped to 1 — multiple seeds give identical episodes.
    """
    if trajectory not in _STOCHASTIC_TRAJECTORIES:
        n_seeds = 1

    results = []
    for i in range(n_seeds):
        r = run_fn(trajectory=trajectory, seed=base_seed + i, **kwargs)
        results.append(r)

    if n_seeds == 1:
        return results[0]

    # Aggregate scalar metrics across seeds
    scalar_keys = [k for k in results[0] if not k.startswith("_")]
    aggregated = {}
    for k in scalar_keys:
        vals = [r[k] for r in results]
        aggregated[k]          = float(np.mean(vals))
        aggregated[k + "_std"] = float(np.std(vals))
        aggregated[k + "_n"]   = n_seeds
    # Preserve arrays from first seed for plotting
    for k in results[0]:
        if k.startswith("_"):
            aggregated[k] = results[0][k]
    return aggregated


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
    n_seeds = args.n_seeds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    baked = env_kwargs["action_filter_hz"]
    ema_tag = f"  [baked={baked}" + (f", posthoc={posthoc_ema}]" if posthoc_ema < 1.0 else "]")
    results = {}
    print(f"\n{'trajectory':<16} {'IK (mm)':>10} {'residual (mm)':>14} {'Δ':>8}  "
          f"{'roughness':>10} {'sat%':>6}{ema_tag}")
    print("-" * 72)

    for traj in trajs:
        ik  = run_multi(run_ik,  traj, n_seeds=n_seeds,
                        disturbance=env_kwargs["disturbance"])
        res = run_multi(run_residual, traj, n_seeds=n_seeds,
                        model=model, vn_ref=vn_ref,
                        action_ema_posthoc=posthoc_ema, **env_kwargs)

        ik_rmse  = ik["settled_rmse_mm"]
        res_rmse = res["settled_rmse_mm"]
        improv   = (ik_rmse - res_rmse) / ik_rmse * 100
        marker   = " ✓" if improv > 0 else ""

        # Uncertainty suffix — only for stochastic trajectories with n_seeds > 1
        ik_sfx  = f"±{ik.get('settled_rmse_mm_std', 0):.1f}"  if "settled_rmse_mm_std" in ik  else ""
        res_sfx = f"±{res.get('settled_rmse_mm_std', 0):.1f}" if "settled_rmse_mm_std" in res else ""
        roughness = res.get("action_roughness", float("nan"))
        sat_pct   = res.get("saturation_rate",  float("nan")) * 100

        print(f"{traj:<16} {ik_rmse:>7.1f}{ik_sfx:<4} {res_rmse:>11.1f}{res_sfx:<4} "
              f"{improv:>+7.1f}%{marker}  {roughness:>10.4f} {sat_pct:>5.1f}%")

        results[traj] = {"ik": ik, "residual": res, "improvement_pct": float(improv)}

    print()
    plot_ablation(results, out / "ablation.png")

    save = {}
    for t, r in results.items():
        entry = {
            "ik_settled_rmse_mm":       r["ik"]["settled_rmse_mm"],
            "residual_settled_rmse_mm": r["residual"]["settled_rmse_mm"],
            "improvement_pct":          r["improvement_pct"],
        }
        # carry through std and smoothness if present
        for k in ("settled_rmse_mm_std", "settled_rmse_mm_n"):
            if k in r["residual"]:
                entry["residual_" + k] = r["residual"][k]
            if k in r["ik"]:
                entry["ik_" + k] = r["ik"][k]
        for k in ("action_roughness", "saturation_rate", "action_std"):
            if k in r["residual"]:
                entry[k] = r["residual"][k]
        save[t] = entry

    with open(out / "ablation.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"  json  → {out}/ablation.json")


def cmd_ood(args):
    """Evaluate one or more models on OOD trajectories (square, rectangle).

    The training trajectory_pool is preserved so the obs dim matches.
    The traj_onehot will be all-zeros (trajectory type unknown to the policy).
    This tests generalisation: the policy never saw these shapes during training.
    """
    trajs = args.trajectories.split(",") if args.trajectories else OOD_TRAJECTORIES

    # Collect models: --model accepts multiple paths
    model_paths = args.models
    entries = []   # list of (label, model, vn_ref, env_kwargs)
    for path in model_paths:
        m, vn, cfg = load_model(path)
        label = Path(path).parent.name   # use run dir name as label
        entries.append((label, m, vn, _env_kwargs_from_cfg(cfg)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── print table ──────────────────────────────────────────────────────────
    col_w = 14
    header = f"{'trajectory':<16} {'IK (mm)':>{col_w}}"
    for label, *_ in entries:
        header += f" {label[:col_w]:>{col_w}}"
    print(f"\n{header}")
    print("-" * (16 + col_w + col_w * len(entries) + len(entries)))

    all_results = {}
    for traj in trajs:
        ik = run_ik(traj, disturbance=entries[0][3]["disturbance"])
        row = f"{traj:<16} {ik['settled_rmse_mm']:>{col_w}.1f}"
        traj_results = {"ik_settled_rmse_mm": ik["settled_rmse_mm"], "models": {}}
        for label, model, vn_ref, env_kwargs in entries:
            res = run_residual(model, vn_ref, traj, **env_kwargs)
            improv = (ik["settled_rmse_mm"] - res["settled_rmse_mm"]) / ik["settled_rmse_mm"] * 100
            row += f" {res['settled_rmse_mm']:>{col_w}.1f}"
            traj_results["models"][label] = {
                "residual_settled_rmse_mm": res["settled_rmse_mm"],
                "improvement_pct": float(improv),
            }
        print(row)
        all_results[traj] = traj_results

    print()

    # ── save json ────────────────────────────────────────────────────────────
    json_path = out / "ood.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  json  → {json_path}")

    # ── trajectory plots ──────────────────────────────────────────────────────
    for traj in trajs:
        fig, axes = plt.subplots(1, len(entries) + 1,
                                 figsize=(4 * (len(entries) + 1), 4),
                                 sharey=True)
        fig.suptitle(f"OOD: {traj}", fontsize=12)

        ik_res = run_ik(traj, disturbance=entries[0][3]["disturbance"])
        for ax, (label_or_ik, res_data) in zip(
            axes,
            [("IK baseline", ik_res)] + [(label, run_residual(m, vn, traj, **kw))
                                          for label, m, vn, kw in entries]
        ):
            err = np.array(res_data["_err_mm"])
            ax.plot(err, lw=1.0, color="#e05252" if label_or_ik != "IK baseline" else "#aaaaaa")
            ax.axhline(np.mean(err[len(err)//4:]), ls="--", lw=1.0, color="#333333")
            ax.set_title(f"{label_or_ik}\n{np.mean(err[len(err)//4:]):.1f} mm", fontsize=9)
            ax.set_xlabel("step")
            ax.set_ylim(0, None)
        axes[0].set_ylabel("tracking error (mm)")
        plt.tight_layout()
        fig_path = out / f"ood_{traj}.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  plot  → {fig_path}")


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
    p.add_argument("--n-seeds", type=int, default=10, dest="n_seeds",
                   help="Seeds to average for stochastic trajectories like moving_target (default: 10)")

    p = sub.add_parser("ood", help="OOD generalization eval (square, rectangle)")
    p.add_argument("--models", required=True, nargs="+",
                   help="One or more final_model.zip paths to compare side-by-side")
    p.add_argument("--trajectories", default=None,
                   help=f"Comma-separated OOD trajectories (default: {','.join(OOD_TRAJECTORIES)})")
    p.add_argument("--out", default="results/eval/ood")

    args = parser.parse_args()
    {"rollout": cmd_rollout, "ablation": cmd_ablation, "ood": cmd_ood}[args.mode](args)


if __name__ == "__main__":
    main()
