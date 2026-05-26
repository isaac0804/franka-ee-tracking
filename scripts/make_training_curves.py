"""Plot training curves (pos_err_mm vs steps) for MLP vs Transformer at 5M steps.

Shows 2-seed shaded bands for each architecture, plus IK baseline.

Usage:
    python scripts/make_training_curves.py
    python scripts/make_training_curves.py --out results/figures
"""
import argparse
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── config ────────────────────────────────────────────────────────────────────
RUNS = {
    "mlp": [
        "results/sweep/rs012_5M",
        "results/sweep/rs012_seed1_5M",
    ],
    "transformer": [
        "results/main_runs/tfm_no_xattn_5M_s42",
        "results/main_runs/tfm_no_xattn_5M_s1",
    ],
}

IK_DELAY_MM   = 38.1   # IK with 100ms delay — worst-case ref line
IK_NODELAY_MM = 18.0   # IK without delay — floor reference

COLORS = {
    "mlp":         "#4e79a7",   # blue
    "transformer": "#e05252",   # red
    "ik_delay":    "#aaaaaa",   # grey
    "ik_floor":    "#cccccc",   # light grey
}

LABELS = {
    "mlp":         "MLP",
    "transformer": "Transformer (no cross-attn)",
}

TAG = "tracking/pos_err_mm"


def load_curve(run_dir: str):
    """Return (steps, values) arrays from a TB log directory."""
    tb_dir = pathlib.Path(run_dir) / "tb"
    if not tb_dir.exists():
        return None, None
    subdirs = [d for d in tb_dir.iterdir() if d.is_dir()]
    path = str(subdirs[0]) if subdirs else str(tb_dir)
    ea = EventAccumulator(path)
    ea.Reload()
    try:
        events = ea.Scalars(TAG)
    except KeyError:
        return None, None
    steps = np.array([e.step for e in events], dtype=float)
    vals  = np.array([e.value for e in events], dtype=float)
    return steps, vals


def smooth(vals, window=5):
    """Simple moving average."""
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    padded = np.pad(vals, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(vals)]


def interpolate_to_common(all_steps, all_vals, n_points=300):
    """Interpolate all curves onto a common x-grid for band plotting."""
    x_min = max(s[0] for s in all_steps if s is not None)
    x_max = min(s[-1] for s in all_steps if s is not None)
    x_grid = np.linspace(x_min, x_max, n_points)
    interp_vals = []
    for steps, vals in zip(all_steps, all_vals):
        if steps is None:
            continue
        interp_vals.append(np.interp(x_grid, steps, vals))
    return x_grid, np.array(interp_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    for arch, run_dirs in RUNS.items():
        color = COLORS[arch]
        label = LABELS[arch]

        all_steps, all_vals = [], []
        for run_dir in run_dirs:
            steps, vals = load_curve(run_dir)
            if steps is None:
                print(f"  [skip] {run_dir} — no TB data found")
                continue
            if not args.no_smooth:
                vals = smooth(vals, window=7)
            all_steps.append(steps)
            all_vals.append(vals)
            # faint individual seed line
            ax.plot(steps / 1e6, vals, color=color, lw=0.8, alpha=0.35)

        if len(all_steps) < 2:
            # Only one seed: just plot it solidly
            if all_steps:
                ax.plot(all_steps[0] / 1e6, all_vals[0],
                        color=color, lw=2.0, label=label)
            continue

        # Shaded band over common x-grid
        x_grid, mat = interpolate_to_common(all_steps, all_vals)
        mean_curve = mat.mean(axis=0)
        lo = mat.min(axis=0)
        hi = mat.max(axis=0)

        ax.fill_between(x_grid / 1e6, lo, hi, color=color, alpha=0.15)
        ax.plot(x_grid / 1e6, mean_curve, color=color, lw=2.2,
                label=f"{label} (mean, 2 seeds)")

    # IK reference lines
    ax.axhline(IK_DELAY_MM, color=COLORS["ik_delay"], lw=1.2, ls="--",
               label=f"IK + 100ms delay ({IK_DELAY_MM:.0f} mm)")
    ax.axhline(IK_NODELAY_MM, color=COLORS["ik_floor"], lw=1.0, ls=":",
               label=f"IK no delay (~{IK_NODELAY_MM:.0f} mm)")

    ax.set_xlabel("Training steps (millions)", fontsize=11)
    ax.set_ylabel("Training pos_err_mm (lower = better)", fontsize=11)
    ax.set_title("MLP vs Transformer — training convergence at 5M steps\n"
                 "(mixed trajectory pool: moving_target + circle + figure‑8)",
                 fontsize=11, pad=10)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)

    plt.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
