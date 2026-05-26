#!/usr/bin/env python3
"""Generate 3D matplotlib trajectory animations for README/submission.

Shows IK vs Transformer PPO tracking a moving target in 3D, with:
  - Full trajectory path shown faintly in the background
  - Fading trail of past positions (bright → dim)
  - Current position markers + error line to target
  - Rolling RMSE in the subtitle

Usage:
    python scripts/make_animation.py \\
        --model results/main_runs/tfm_no_xattn_5M_s42/final_model.zip \\
        --trajectory circle

    python scripts/make_animation.py \\
        --model results/main_runs/tfm_no_xattn_5M_s42/final_model.zip \\
        --all --out results/figures
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate import load_model, run_ik, run_residual, _env_kwargs_from_cfg
from ee_tracking.env.disturbances import DisturbanceConfig

# ── colours ───────────────────────────────────────────────────────────────────
C_TGT = "#cc3333"     # red — target
C_IK  = "#e8964a"     # orange — IK
C_PPO = "#3a9bd5"     # blue — PPO
TRAIL = 40            # frames to show in trail
FPS   = 15            # output GIF fps (every 3rd frame of 50 Hz sim)
STEP  = 3             # subsample: use every Nth sim step


def _run_episodes(model_path: str, trajectory: str, seed: int = 42):
    """Return (ik_result, ppo_result) dicts with _ee_pos, _tgt_pos, _err_mm."""
    model, vn, cfg = load_model(model_path)
    kw   = _env_kwargs_from_cfg(cfg)
    dist = kw.pop("disturbance", None) or DisturbanceConfig()

    print(f"  Running IK …", end=" ", flush=True)
    ik_r = run_ik(trajectory, seed=seed, disturbance=dist)
    print(f"{ik_r['settled_rmse_mm']:.1f} mm")

    print(f"  Running PPO …", end=" ", flush=True)
    ppo_r = run_residual(model, vn, trajectory, seed=seed,
                         disturbance=dist, **kw)
    print(f"{ppo_r['settled_rmse_mm']:.1f} mm")
    return ik_r, ppo_r


def _rolling_rmse(errs: np.ndarray, i: int, window: int = 25) -> float:
    lo = max(0, i - window)
    return float(np.sqrt(np.mean(errs[lo:i+1] ** 2)))


def make_animation(model_path: str, trajectory: str,
                   out_dir: Path, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n── {trajectory} ─────────────────────────────────────────")
    ik_r, ppo_r = _run_episodes(model_path, trajectory, seed)

    tgt = np.array(ik_r["_tgt_pos"])       # (T, 3)
    ik  = np.array(ik_r["_ee_pos"])        # (T, 3)
    ppo = np.array(ppo_r["_ee_pos"])       # (T, 3)
    ik_err  = np.array(ik_r["_err_mm"])
    ppo_err = np.array(ppo_r["_err_mm"])

    T = min(len(tgt), len(ik), len(ppo))
    frames_idx = list(range(0, T, STEP))

    # ── axis limits with a little padding ────────────────────────────────────
    all_pos = np.vstack([tgt, ik, ppo])
    lo = all_pos.min(axis=0) - 0.05
    hi = all_pos.max(axis=0) + 0.05

    # Choose a good view angle per trajectory
    elev, azim = (20, -60) if trajectory == "moving_target" else (15, -50)

    frames = []
    print(f"  Rendering {len(frames_idx)} frames …", end=" ", flush=True)

    for fi, t in enumerate(frames_idx):
        fig = plt.figure(figsize=(11, 5), facecolor="white")
        gs  = gridspec.GridSpec(1, 3, width_ratios=[5, 5, 3], wspace=0.35)
        ax_ik  = fig.add_subplot(gs[0], projection="3d")
        ax_ppo = fig.add_subplot(gs[1], projection="3d")
        ax_err = fig.add_subplot(gs[2])

        for ax, ee, ee_err, colour, title in [
            (ax_ik,  ik,  ik_err,  C_IK,  "IK + 100 ms delay"),
            (ax_ppo, ppo, ppo_err, C_PPO, "Transformer PPO"),
        ]:
            # ── faint full trajectory ──────────────────────────────────────
            ax.plot(tgt[:,0], tgt[:,1], tgt[:,2],
                    color=C_TGT, lw=0.8, alpha=0.15, zorder=1)
            ax.plot(ee[:,0],  ee[:,1],  ee[:,2],
                    color=colour, lw=0.8, alpha=0.12, zorder=1)

            # ── fading trail ───────────────────────────────────────────────
            t0 = max(0, t - TRAIL)
            n  = t - t0 + 1
            for j in range(t0, t + 1):
                frac = (j - t0) / max(n - 1, 1)   # 0=oldest, 1=newest
                alpha = 0.15 + 0.75 * frac ** 1.5
                lw    = 0.8  + 1.5  * frac
                s, e  = max(t0, j-1), j
                ax.plot(tgt[s:e+1, 0], tgt[s:e+1, 1], tgt[s:e+1, 2],
                        color=C_TGT, lw=lw*0.8, alpha=alpha*0.8, zorder=3)
                ax.plot(ee[s:e+1, 0],  ee[s:e+1, 1],  ee[s:e+1, 2],
                        color=colour, lw=lw, alpha=alpha, zorder=3)

            # ── current position markers ───────────────────────────────────
            ax.scatter(*tgt[t], color=C_TGT, s=80,  zorder=5, depthshade=False)
            ax.scatter(*ee[t],  color=colour, s=50,  zorder=5, depthshade=False)
            # error line
            ax.plot([ee[t,0], tgt[t,0]], [ee[t,1], tgt[t,1]],
                    [ee[t,2], tgt[t,2]],
                    color="#888", lw=1.2, ls="--", alpha=0.6, zorder=4)

            # ── formatting ─────────────────────────────────────────────────
            rmse = _rolling_rmse(ee_err, t)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_xlabel("X", fontsize=8, labelpad=-4)
            ax.set_ylabel("Y", fontsize=8, labelpad=-4)
            ax.set_zlabel("Z", fontsize=8, labelpad=-4)
            ax.tick_params(labelsize=7, pad=-2)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title}\nRMSE: {rmse:.1f} mm",
                         fontsize=10, pad=4)
            ax.grid(True, alpha=0.2)

        # ── error-over-time panel ─────────────────────────────────────────
        t_ax = np.arange(T) / 50.0
        settle = max(1, T // 6)
        ax_err.axvspan(0, settle/50, color="#f5f5f5", zorder=0)
        ax_err.plot(t_ax, ik_err,  color=C_IK,  lw=1.2, alpha=0.5)
        ax_err.plot(t_ax, ppo_err, color=C_PPO, lw=1.2, alpha=0.5)
        # highlight current position
        ax_err.axvline(t/50, color="#aaa", lw=1, ls=":", zorder=3)
        ax_err.scatter([t/50], [ik_err[t]],  color=C_IK,  s=30, zorder=4)
        ax_err.scatter([t/50], [ppo_err[t]], color=C_PPO, s=30, zorder=4)
        # settled RMSE annotations
        ik_final  = float(np.sqrt(np.mean(ik_err[settle:]**2)))
        ppo_final = float(np.sqrt(np.mean(ppo_err[settle:]**2)))
        ax_err.set_xlabel("Time (s)", fontsize=9)
        ax_err.set_ylabel("Error (mm)", fontsize=9)
        ax_err.set_title(
            f"IK: {ik_final:.1f} mm   PPO: {ppo_final:.1f} mm",
            fontsize=9, pad=4)
        ax_err.set_ylim(bottom=0)
        ax_err.grid(True, alpha=0.25, lw=0.7)
        ax_err.spines[["top","right"]].set_visible(False)
        ax_err.tick_params(labelsize=8)

        # ── legend ────────────────────────────────────────────────────────
        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Line2D([0],[0], color=C_TGT, lw=2, label="Target"),
            Line2D([0],[0], color=C_IK,  lw=2, label="IK"),
            Line2D([0],[0], color=C_PPO, lw=2, label="Transformer"),
        ], loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.38, -0.02), frameon=False)

        fig.suptitle(
            f"Franka EE Tracking — {trajectory.replace('_',' ').title()}",
            fontsize=12, fontweight="bold", y=1.01)

        # ── capture frame ─────────────────────────────────────────────────
        fig.canvas.draw()
        buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(buf)
        plt.close(fig)

        if fi % 20 == 0:
            print(".", end="", flush=True)

    print(f" done ({len(frames)} frames)")

    # ── save GIF ─────────────────────────────────────────────────────────────
    gif_path = out_dir / f"tracking_3d_{trajectory}.gif"
    from PIL import Image
    gif_pil = [Image.fromarray(f).quantize(colors=192,
               method=Image.Quantize.MEDIANCUT) for f in frames]
    gif_pil[0].save(
        str(gif_path), save_all=True, append_images=gif_pil[1:],
        loop=0, duration=int(1000/FPS), optimize=True,
    )
    print(f"  GIF → {gif_path}  ({gif_path.stat().st_size//1024} KB)")
    return gif_path


# ── CLI ────────────────────────────────────────────────────────────────────────
TRAJECTORIES = ["moving_target", "circle", "figure8"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--trajectory", choices=TRAJECTORIES, default="circle")
    p.add_argument("--all",        action="store_true", dest="all_trajs")
    p.add_argument("--out",        default="results/figures")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    trajs   = TRAJECTORIES if args.all_trajs else [args.trajectory]
    out_dir = Path(args.out)

    for traj in trajs:
        make_animation(args.model, traj, out_dir, seed=args.seed)

    print(f"\nDone — animations in {Path(args.out)}/")

if __name__ == "__main__":
    main()
