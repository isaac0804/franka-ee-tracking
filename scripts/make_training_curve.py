#!/usr/bin/env python3
"""Training error curve: online pos_err_mm from TensorBoard, 2 seeds each.

Plots mean ± std band for MLP and Transformer across the full 5M training run.

Usage:
    python scripts/make_training_curve.py
    python scripts/make_training_curve.py --out results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# ── run config ────────────────────────────────────────────────────────────────

MODELS = {
    "Transformer": {
        "tb_dirs": [
            "results/main_runs/tfm_no_xattn_5M_s42/tb/PPO_1",
            "results/main_runs/tfm_no_xattn_5M_s1/tb/PPO_1",
        ],
        "color": "#e05252",
    },
    "MLP": {
        "tb_dirs": [
            "results/sweep/rs012_5M/tb/PPO_1",
            "results/sweep/rs012_seed1_5M/tb/PPO_1",
        ],
        "color": "#4e79a7",
    },
}

TAG = "tracking/pos_err_mm"

# IK baseline with 100ms delay, mixed-trajectory training rollouts
# (approximate — actual value depends on trajectory mix)
IK_BASELINE_MM = 25.0   # rough online env average (MT+CI+F8 mixed)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_tb(tb_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays from a TensorBoard event dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def align_to_common_steps(all_steps, all_values):
    """Interpolate each series onto a shared step grid (finest common grid)."""
    # Use the step grid of the first seed as reference
    ref_steps = all_steps[0]
    aligned = [all_values[0]]
    for steps, vals in zip(all_steps[1:], all_values[1:]):
        interp = np.interp(ref_steps, steps, vals)
        aligned.append(interp)
    return ref_steps, np.array(aligned)


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")

    fmt = matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}k")

    for name, cfg in MODELS.items():
        all_steps, all_vals = [], []
        for tb_dir in cfg["tb_dirs"]:
            s, v = load_tb(tb_dir, TAG)
            all_steps.append(s)
            all_vals.append(v)

        steps, vals_matrix = align_to_common_steps(all_steps, all_vals)
        mean = vals_matrix.mean(axis=0)
        std  = vals_matrix.std(axis=0)

        ax.fill_between(steps, mean - std, mean + std,
                        color=cfg["color"], alpha=0.15, zorder=2)
        ax.plot(steps, mean, color=cfg["color"], lw=2.2,
                label=name, zorder=3)

    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Tracking error (mm)  — lower is better", fontsize=11)
    ax.set_title("Training convergence — Transformer vs MLP  (mean ± std, 2 seeds)",
                 fontsize=12, pad=10)

    fmt2 = matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}k")
    ax.xaxis.set_major_formatter(fmt2)

    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/training_curve.png")
    args = ap.parse_args()
    make_figure(Path(args.out))


if __name__ == "__main__":
    main()
