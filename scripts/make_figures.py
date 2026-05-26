#!/usr/bin/env python3
"""Generate all result figures for the README.

Produces four figures:
  1. rmse_comparison.png  — grouped bar chart: IK / MLP / Transformer
  2. efficiency_curve.png — RMSE vs training steps (33x advantage)
  3. ablation_bar.png     — ablation study bar chart
  4. tracking_{traj}.png  — trajectory tracking (error + path + histogram)

Usage:
    # Full suite (~2 min for rollouts):
    python scripts/make_figures.py \\
        --tfm results/main_runs/tfm_no_xattn_5M_s42/final_model.zip \\
        --mlp results/sweep/rs012_10M/final_model.zip

    # Static charts only (no rollouts):
    python scripts/make_figures.py --static-only
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

C_IK  = "#d94f4f"
C_MLP = "#e8964a"
C_TFM = "#3a9bd5"
C_TGT = "#888888"

TRAJ_LABELS = {"moving_target": "Moving Target",
               "circle": "Circle", "figure8": "Figure-8"}
TRAJS = ["moving_target", "circle", "figure8"]

# Pre-computed results (MT, CI, F8) in mm
KNOWN: dict[str, tuple] = {
    "ik":                (38.1, 12.1, 7.7),
    "mlp_300k":          (25.9, 10.7, 8.7),
    "mlp_5M":            (21.0,  7.6, 7.0),
    "mlp_10M":           (16.0,  5.3, 4.7),
    "tfm_base_300k":     (27.0,  5.0, 6.5),
    "tfm_noxattn_300k":  (24.4,  5.7, 5.2),   # mean of 2 seeds
    "tfm_nope_300k":     (26.8, 11.1, 10.7),
    "tfm_unpaired_300k": (26.6,  6.8, 6.9),   # mean of 2 seeds
    "tfm_noxattn_5M":    (None, None, None),   # filled after 5M run
}

def _refresh_known():
    import json
    checks = {
        "tfm_noxattn_5M": "results/eval/main_runs/tfm_no_xattn_5M_s42/ablation.json",
    }
    for key, path in checks.items():
        p = Path(path)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        mt = d.get("moving_target", {}).get("residual_settled_rmse_mm")
        ci = d.get("circle",        {}).get("residual_settled_rmse_mm")
        f8 = d.get("figure8",       {}).get("residual_settled_rmse_mm")
        if None not in (mt, ci, f8):
            KNOWN[key] = (mt, ci, f8)
            print(f"  Loaded {key}: MT={mt:.1f} CI={ci:.1f} F8={f8:.1f}")

def _settled_rmse(err):
    err = np.array(err)
    s = max(1, len(err) // 6)
    return float(np.sqrt(np.mean(err[s:] ** 2)))

def _proj(pos, traj):
    pos = np.array(pos)
    if traj in ("circle", "figure8"):
        return pos[:,0], pos[:,2], "X (m)", "Z (m)"
    return pos[:,0], pos[:,1], "X (m)", "Y (m)"


# ── Figure 1: RMSE bar chart ───────────────────────────────────────────────────
def make_rmse_bar_chart(out: Path):
    _refresh_known()
    methods = [
        ("IK baseline",        "ik",              C_IK,  "//"),
        ("MLP  300k",          "mlp_300k",        C_MLP, ""),
        ("MLP  10M",           "mlp_10M",         C_MLP, ""),
        ("Transformer  300k",  "tfm_noxattn_300k",C_TFM, ""),
        ("Transformer  5M",    "tfm_noxattn_5M",  C_TFM, ""),
    ]
    tidx = {t: i for i, t in enumerate(TRAJS)}
    x = np.arange(len(TRAJS)); w = 0.15
    offs = np.linspace(-(len(methods)-1)/2,(len(methods)-1)/2,len(methods))*w

    fig, ax = plt.subplots(figsize=(11,5))
    fig.patch.set_facecolor("white")
    for (lbl, key, col, hatch), off in zip(methods, offs):
        v = KNOWN[key]
        if v[0] is None: continue
        h = [v[tidx[t]] for t in TRAJS]
        bars = ax.bar(x+off, h, w, label=lbl, color=col,
                      hatch=hatch, alpha=0.85, edgecolor="white", lw=0.5)
        for b, hh in zip(bars, h):
            ax.text(b.get_x()+b.get_width()/2, hh+0.3, f"{hh:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([TRAJ_LABELS[t] for t in TRAJS], fontsize=11)
    ax.set_ylabel("Settled RMSE (mm)", fontsize=11)
    ax.set_title("Tracking accuracy — IK vs MLP vs Transformer", fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1]*1.15)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ── Figure 2: Step efficiency curve ───────────────────────────────────────────
def make_efficiency_curve(out: Path):
    _refresh_known()
    mlp_pts = [(300_000,  KNOWN["mlp_300k"]),
               (5_000_000, KNOWN["mlp_5M"]),
               (10_000_000,KNOWN["mlp_10M"])]
    tfm_pts = [(300_000,  KNOWN["tfm_noxattn_300k"])]
    if KNOWN["tfm_noxattn_5M"][0] is not None:
        tfm_pts.append((5_000_000, KNOWN["tfm_noxattn_5M"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    fmt = matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}M" if x>=1e6 else f"{int(x/1e3)}k")

    for ax, mi, ylabel in [(axes[0], 1, "Circle RMSE (mm)"),
                            (axes[1], 2, "Figure-8 RMSE (mm)")]:
        mx = [p[0] for p in mlp_pts]; my = [p[1][mi] for p in mlp_pts]
        tx = [p[0] for p in tfm_pts]; ty = [p[1][mi] for p in tfm_pts]
        ax.plot(mx, my, "o-", color=C_MLP, lw=2, ms=7, label="MLP", zorder=3)
        ax.plot(tx, ty, "s-", color=C_TFM, lw=2, ms=7,
                label="Transformer (no xattn)", zorder=4)
        for pts, col in [(mlp_pts, C_MLP),(tfm_pts, C_TFM)]:
            for steps, vals in pts:
                ax.annotate(f"{vals[mi]:.1f}mm", xy=(steps, vals[mi]),
                            xytext=(0,10), textcoords="offset points",
                            ha="center", fontsize=8, color=col)
        # reference line showing equivalence
        if mi == 1:
            ref = KNOWN["mlp_10M"][1]
            ax.axhline(ref, color="#bbb", lw=1, ls=":", zorder=1)
            ax.text(400_000, ref+0.3, f"MLP 10M target ({ref:.1f} mm)",
                    fontsize=8, color="#888")
        ax.set_xscale("log")
        ax.set_xlabel("Training steps", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel.split(" RMSE")[0], fontsize=11, pad=6)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, lw=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.set_major_formatter(fmt)

    fig.suptitle("Step efficiency — Transformer vs MLP",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ── Figure 3: Ablation bar chart ───────────────────────────────────────────────
def make_ablation_chart(out: Path):
    _refresh_known()
    ablations = [
        ("Best: no cross-attn",  "tfm_noxattn_300k",  C_TFM),
        ("Baseline (full)",      "tfm_base_300k",      "#7ec8e3"),
        ("No pos embedding",     "tfm_nope_300k",      "#f4a261"),
        ("Unpaired tokens",      "tfm_unpaired_300k",  "#e76f51"),
    ]
    tidx = {t: i for i, t in enumerate(TRAJS)}
    x = np.arange(len(TRAJS)); w = 0.20
    offs = np.linspace(-(len(ablations)-1)/2,(len(ablations)-1)/2,len(ablations))*w

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    for (lbl, key, col), off in zip(ablations, offs):
        v = KNOWN[key]
        if v[0] is None: continue
        h = [v[tidx[t]] for t in TRAJS]
        bars = ax.bar(x+off, h, w, label=lbl, color=col,
                      alpha=0.88, edgecolor="white", lw=0.5)
        for b, hh in zip(bars, h):
            ax.text(b.get_x()+b.get_width()/2, hh+0.2, f"{hh:.1f}",
                    ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([TRAJ_LABELS[t] for t in TRAJS], fontsize=11)
    ax.set_ylabel("Settled RMSE (mm)", fontsize=11)
    ax.set_title("Ablation study — 300k steps, mean over 2 seeds", fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1]*1.18)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ── Figure 4: Tracking plots (live rollout) ────────────────────────────────────
def make_tracking_figures(tfm_path: str, mlp_path: str | None, out_dir: Path):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluate import load_model, run_ik, run_residual, _env_kwargs_from_cfg
    from ee_tracking.env.disturbances import DisturbanceConfig

    tfm_model, tfm_vn, tfm_cfg = load_model(tfm_path)
    tfm_kw  = _env_kwargs_from_cfg(tfm_cfg)
    tfm_dist= tfm_kw.pop("disturbance", None) or DisturbanceConfig()

    mlp_model = mlp_vn = mlp_kw = mlp_dist = None
    if mlp_path:
        mlp_model, mlp_vn, mlp_cfg = load_model(mlp_path)
        mlp_kw  = _env_kwargs_from_cfg(mlp_cfg)
        mlp_dist= mlp_kw.pop("disturbance", None) or DisturbanceConfig()

    for traj in TRAJS:
        print(f"  Rolling out {traj}…", end=" ", flush=True)
        ik_r  = run_ik(traj, seed=42, disturbance=tfm_dist)
        tfm_r = run_residual(tfm_model, tfm_vn, traj, seed=42,
                             disturbance=tfm_dist, **tfm_kw)
        mlp_r = (run_residual(mlp_model, mlp_vn, traj, seed=42,
                              disturbance=mlp_dist, **mlp_kw)
                 if mlp_model else None)
        print("done")
        _plot_tracking(traj, ik_r, tfm_r, mlp_r,
                       out_dir / f"tracking_{traj}.png")


def _plot_tracking(traj, ik_r, tfm_r, mlp_r, out: Path):
    ik_err  = np.array(ik_r["_err_mm"])
    tfm_err = np.array(tfm_r["_err_mm"])
    mlp_err = np.array(mlp_r["_err_mm"]) if mlp_r else None
    t = np.arange(len(ik_err)) / 50.0

    ik_rmse  = _settled_rmse(ik_err)
    tfm_rmse = _settled_rmse(tfm_err)
    mlp_rmse = _settled_rmse(mlp_err) if mlp_err is not None else None

    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(16, 4.5))
    fig.patch.set_facecolor("white")

    # ── error over time ──────────────────────────────────────────────────────
    ax = axes[0]
    settle_t = t[max(1, len(t)//6)]
    ax.axvspan(0, settle_t, color="#f5f5f5", zorder=0, label="settling")
    ax.plot(t, ik_err,  color=C_IK,  lw=1.5, label=f"IK + delay   {ik_rmse:.1f} mm")
    if mlp_err is not None:
        ax.plot(t, mlp_err, color=C_MLP, lw=1.5, label=f"MLP (10M)   {mlp_rmse:.1f} mm")
    ax.fill_between(t, tfm_err, alpha=0.15, color=C_TFM)
    ax.plot(t, tfm_err, color=C_TFM, lw=2.0,
            label=f"Transformer  {tfm_rmse:.1f} mm")
    ax.set_xlabel("Time (s)", fontsize=11); ax.set_ylabel("EE error (mm)", fontsize=11)
    ax.set_title(f"{TRAJ_LABELS[traj]} — error over time", fontsize=11, pad=8)
    ax.legend(fontsize=9, loc="upper right"); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, lw=0.7); ax.spines[["top","right"]].set_visible(False)

    # ── EE path ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    tx, ty, xl, yl = _proj(ik_r["_tgt_pos"], traj)
    ix, iy, *_     = _proj(ik_r["_ee_pos"],  traj)
    px, py, *_     = _proj(tfm_r["_ee_pos"], traj)
    ax2.plot(tx, ty, color=C_TGT, lw=1.2, ls="--", alpha=0.6, label="Target")
    ax2.plot(ix, iy, color=C_IK,  lw=1.2, alpha=0.8, label=f"IK  {ik_rmse:.1f} mm")
    if mlp_r:
        mx, my, *_ = _proj(mlp_r["_ee_pos"], traj)
        ax2.plot(mx, my, color=C_MLP, lw=1.2, alpha=0.8,
                 label=f"MLP  {mlp_rmse:.1f} mm")
    ax2.plot(px, py, color=C_TFM, lw=1.8, alpha=0.9,
             label=f"Transformer  {tfm_rmse:.1f} mm")
    ax2.set_xlabel(xl, fontsize=11); ax2.set_ylabel(yl, fontsize=11)
    ax2.set_title("EE path", fontsize=11, pad=8); ax2.legend(fontsize=9)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.25, lw=0.7)
    ax2.spines[["top","right"]].set_visible(False)

    # ── error histogram ───────────────────────────────────────────────────────
    ax3 = axes[2]
    s = max(1, len(ik_err)//6)
    all_errs = [ik_err[s:]] + ([mlp_err[s:]] if mlp_err is not None else []) + [tfm_err[s:]]
    bins = np.linspace(0, max(e.max() for e in all_errs)*1.05, 35)
    ax3.hist(ik_err[s:], bins=bins, color=C_IK,  alpha=0.6, density=True,
             label=f"IK  {ik_rmse:.1f} mm")
    if mlp_err is not None:
        ax3.hist(mlp_err[s:], bins=bins, color=C_MLP, alpha=0.6, density=True,
                 label=f"MLP  {mlp_rmse:.1f} mm")
    ax3.hist(tfm_err[s:], bins=bins, color=C_TFM, alpha=0.7, density=True,
             label=f"Transformer  {tfm_rmse:.1f} mm")
    ax3.set_xlabel("Error (mm)", fontsize=11); ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Error distribution (settled)", fontsize=11, pad=8)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25, lw=0.7)
    ax3.spines[["top","right"]].set_visible(False)

    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfm", default=None)
    p.add_argument("--mlp", default=None)
    p.add_argument("--out", default="results/figures")
    p.add_argument("--static-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Static charts ────────────────────────────────────")
    make_rmse_bar_chart(out_dir / "rmse_comparison.png")
    make_efficiency_curve(out_dir / "efficiency_curve.png")
    make_ablation_chart(out_dir / "ablation_bar.png")

    if not args.static_only:
        if not args.tfm:
            print("\nSkipping trajectory plots — pass --tfm <model.zip>")
        else:
            print("\n── Trajectory tracking plots ────────────────────────")
            make_tracking_figures(args.tfm, args.mlp, out_dir)

    print(f"\nDone — figures in {out_dir}/")

if __name__ == "__main__":
    main()
