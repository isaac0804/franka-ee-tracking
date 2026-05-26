#!/usr/bin/env python3
"""Build a convergence (learning) curve: RMSE vs training steps for MLP and Transformer.

Evaluates saved checkpoints at selected step counts on circle and figure-8.
Results are cached to JSON so the script is resumable.

Usage:
    python scripts/make_learning_curve.py
    python scripts/make_learning_curve.py --out results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── model configs ─────────────────────────────────────────────────────────────

RUNS = {
    "Transformer": {
        "run_dir": "results/main_runs/tfm_no_xattn_5M_s42",
        "color":   "#e05252",
    },
    "MLP": {
        "run_dir": "results/sweep/rs012_5M",
        "color":   "#4e79a7",
    },
}

# Checkpoints to evaluate — every 400k for smooth curve, plus fine-grained
# early steps where the gap is widest
STEPS = [
    200_000, 400_000, 600_000, 800_000,
    1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000,
    2_000_000, 2_400_000, 2_800_000, 3_200_000, 3_600_000,
    4_000_000, 4_400_000, 4_800_000, 5_000_000,
]

TRAJS   = ["circle", "figure8"]
C_IK    = "#aaaaaa"


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(run_dir: str, steps: int):
    """Load a checkpoint PPO model + its VecNormalize stats."""
    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Register custom policies
    from ee_tracking.policies.transformer_policy import TRANSFORMER_POLICY_REGISTRY
    from ee_tracking.policies.gelu_policy import POLICY_REGISTRY
    _ = POLICY_REGISTRY, TRANSFORMER_POLICY_REGISTRY   # ensure registration

    run_p    = Path(run_dir)
    ckpt_dir = run_p / "checkpoints"
    cfg_path = run_p / "config.yaml"

    if steps == 5_000_000:
        model_path = run_p / "final_model.zip"
        vn_path    = run_p / "vecnormalize.pkl"
    else:
        model_path = ckpt_dir / f"ppo_{steps}_steps.zip"
        vn_path    = ckpt_dir / f"ppo_vecnormalize_{steps}_steps.pkl"

    if not model_path.exists():
        return None, None, {}

    saved_cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved_cfg = yaml.safe_load(f) or {}

    # We need evaluate.py helpers, but load_model looks for vecnorm in wrong place
    # for checkpoints — do the load manually here.
    from evaluate import _env_kwargs_from_cfg
    from ee_tracking.env.franka_tracking_env import FrankaTrackingEnv, EnvConfig

    model = PPO.load(str(model_path), device="cpu")

    vn_ref = None
    if vn_path.exists():
        kwargs  = _env_kwargs_from_cfg(saved_cfg)
        tmp_cfg = EnvConfig(**kwargs)
        tmp     = DummyVecEnv([lambda: FrankaTrackingEnv(tmp_cfg)])
        vn_ref  = VecNormalize.load(str(vn_path), tmp)
        vn_ref.training    = False
        vn_ref.norm_reward = False
        tmp.close()

    return model, vn_ref, saved_cfg


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_checkpoint(run_dir: str, steps: int, trajs: list[str]) -> dict | None:
    """Evaluate a single checkpoint on the given trajectories. Returns {traj: rmse}."""
    from evaluate import run_residual, _env_kwargs_from_cfg

    model, vn_ref, saved_cfg = load_checkpoint(run_dir, steps)
    if model is None:
        print(f"    [skip] checkpoint not found: {steps:,}")
        return None

    env_kwargs = _env_kwargs_from_cfg(saved_cfg)
    result = {}
    for traj in trajs:
        r = run_residual(model, vn_ref, traj, seed=42, **env_kwargs)
        result[traj] = float(r["settled_rmse_mm"])
    return result


# ── caching ───────────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache_path: Path, data: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2))


# ── IK baseline (deterministic, run once) ─────────────────────────────────────

IK_MM = {"circle": 12.1, "figure8": 7.7}   # from existing eval


# ── figure ────────────────────────────────────────────────────────────────────

TRAJ_TITLES = {"circle": "Circle", "figure8": "Figure-8"}


def smooth(ys: np.ndarray, w: int = 3) -> np.ndarray:
    """Centered rolling mean; edges handled with smaller windows."""
    out = np.empty_like(ys, dtype=float)
    for i in range(len(ys)):
        lo = max(0, i - w // 2)
        hi = min(len(ys), i + w // 2 + 1)
        out[i] = ys[lo:hi].mean()
    return out


def make_figure(all_results: dict, out: Path):
    """all_results: {model_name: {steps: {traj: rmse}}}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.patch.set_facecolor("white")

    fmt = matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}k")

    for ax, traj in zip(axes, TRAJS):
        # IK baseline
        ax.axhline(IK_MM[traj], color=C_IK, lw=1.4, ls="--", zorder=1,
                   label=f"IK baseline ({IK_MM[traj]:.1f} mm)")

        for name, cfg in RUNS.items():
            run_res = all_results.get(name, {})
            pts = sorted(
                [(s, v[traj]) for s, v in run_res.items() if traj in v],
                key=lambda x: x[0]
            )
            if not pts:
                continue
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            ys_smooth = smooth(ys, w=3)

            # raw as faint background scatter
            ax.scatter(xs, ys, color=cfg["color"], s=12, alpha=0.25, zorder=2)
            # smoothed as main line
            ax.plot(xs, ys_smooth, "-", color=cfg["color"], lw=2.2,
                    label=name, zorder=3)
            # final value annotation
            ax.annotate(f"{ys_smooth[-1]:.1f} mm",
                        xy=(xs[-1], ys_smooth[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=8.5, color=cfg["color"], va="center")

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel("Training steps", fontsize=11)
        ax.set_ylabel("Settled RMSE (mm)", fontsize=11)
        ax.set_title(TRAJ_TITLES[traj], fontsize=12, pad=8)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Sample efficiency — Transformer vs MLP  (18 checkpoints, seed 42)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   default="results/figures")
    ap.add_argument("--cache", default="results/eval/learning_curve_cache.json")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip eval, just regenerate figure from cache")
    args = ap.parse_args()

    cache_path  = Path(args.cache)
    all_results = load_cache(cache_path)   # {model_name: {steps_str: {traj: rmse}}}

    if not args.plot_only:
        for name, cfg in RUNS.items():
            if name not in all_results:
                all_results[name] = {}
            print(f"\n── {name} ({cfg['run_dir']}) ──")
            for steps in STEPS:
                key = str(steps)
                if key in all_results[name]:
                    cached = all_results[name][key]
                    print(f"  {steps/1e6:.1f}M  [cached]  "
                          + "  ".join(f"{t}={cached[t]:.1f}mm" for t in TRAJS if t in cached))
                    continue
                print(f"  {steps/1e6:.1f}M ...", end=" ", flush=True)
                res = eval_checkpoint(cfg["run_dir"], steps, TRAJS)
                if res is None:
                    print("skip")
                    continue
                all_results[name][key] = res
                save_cache(cache_path, all_results)
                print("  ".join(f"{t}={res[t]:.1f}mm" for t in TRAJS))

    # Convert str keys back to int for plotting
    plot_data = {
        name: {int(k): v for k, v in runs.items()}
        for name, runs in all_results.items()
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(plot_data, out_dir / "learning_curve.png")


if __name__ == "__main__":
    main()
