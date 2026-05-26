"""Per-trajectory RMSE bar chart: IK vs MLP@5M vs Transformer@5M (2-seed means ± range).

Includes both in-distribution trajectories (MT, CI, F8) and OOD (square, rectangle).
OOD trajectories have a shaded background to distinguish them visually.

Usage:
    python scripts/make_comparison_bars.py
    python scripts/make_comparison_bars.py --out results/figures
"""
import argparse
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── model registry ────────────────────────────────────────────────────────────
MODELS = {
    "mlp": {
        "label": "MLP",
        "color": "#4e79a7",
        # multi-seed ablation.json dirs (10-seed MT averaging)
        "indist_eval_dirs": [
            "results/eval/mlp_5M_s42_multi",
            "results/eval/mlp_5M_s1_multi",
        ],
        # per-model OOD ablation.json dirs (single seed, deterministic)
        "ood_eval_dirs": [
            "results/sweep/rs012_5M/eval_ood",
            "results/sweep/rs012_seed1_5M/eval_ood",
        ],
    },
    "transformer": {
        "label": "Transformer\n(no cross-attn)",
        "color": "#e05252",
        "indist_eval_dirs": [
            "results/eval/tfm_5M_s42_multi",
            "results/eval/tfm_5M_s1_multi",   # populated after s1 training completes
        ],
        "ood_eval_dirs": [
            "results/main_runs/tfm_no_xattn_5M_s42/eval_ood",
            "results/main_runs/tfm_no_xattn_5M_s1/eval_ood",  # after s1 completes
        ],
    },
}

INDIST_TRAJS  = ["moving_target", "circle", "figure8"]
OOD_TRAJS     = ["square", "rectangle"]
ALL_TRAJS     = INDIST_TRAJS + OOD_TRAJS
TRAJ_LABELS   = ["Moving\nTarget", "Circle", "Figure‑8", "Square\n(OOD)", "Rectangle\n(OOD)"]

# IK baseline from multi-seed eval (consistent seed=0 for OOD)
IK_MM = {
    "moving_target": 48.6,   # 10-seed mean
    "circle":        11.5,
    "figure8":        7.7,
    "square":        10.4,   # seed=0
    "rectangle":     10.9,   # seed=0
}
IK_STD = {"moving_target": 8.0}   # only MT has meaningful stochastic variance


def load_eval(eval_dir: str) -> dict | None:
    json_p = pathlib.Path(eval_dir) / "ablation.json"
    if not json_p.exists():
        return None
    return json.loads(json_p.read_text())


def get_rmse(data: dict, traj: str) -> tuple[float | None, float | None]:
    """Return (mean, std_or_None) from an ablation.json entry."""
    t = data.get(traj, {})
    mean = t.get("residual_settled_rmse_mm")
    std  = t.get("residual_settled_rmse_mm_std")
    return mean, std


def collect_results(arch_cfg: dict) -> dict:
    """Collect per-trajectory (mean, std) lists across all eval dirs for an arch."""
    results = {t: [] for t in ALL_TRAJS}

    for eval_dir in arch_cfg.get("indist_eval_dirs", []):
        data = load_eval(eval_dir)
        if data is None:
            continue
        for traj in INDIST_TRAJS:
            mean, std = get_rmse(data, traj)
            if mean is not None:
                results[traj].append((mean, std))

    for eval_dir in arch_cfg.get("ood_eval_dirs", []):
        data = load_eval(eval_dir)
        if data is None:
            continue
        for traj in OOD_TRAJS:
            mean, std = get_rmse(data, traj)
            if mean is not None:
                results[traj].append((mean, std))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arch_results = {arch: collect_results(cfg) for arch, cfg in MODELS.items()}

    # ── layout ────────────────────────────────────────────────────────────────
    n_traj  = len(ALL_TRAJS)
    n_arch  = len(MODELS)
    bar_w   = 0.20
    gap     = 0.08
    group_w = gap + (n_arch + 1) * bar_w  # +1 for IK bar
    x_base  = np.arange(n_traj, dtype=float) * (group_w + 0.30)

    # Widen gap between in-dist and OOD groups
    ood_start = len(INDIST_TRAJS)
    x_base[ood_start:] += 0.35

    offsets = np.linspace(-(n_arch) * bar_w / 2,
                           (n_arch) * bar_w / 2,
                           n_arch + 1)   # includes IK slot at offsets[0]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # OOD shaded background
    ood_left  = x_base[ood_start] - group_w * 0.55
    ood_right = x_base[-1] + group_w * 0.55
    ax.axvspan(ood_left, ood_right, color="#f5f0ff", zorder=0, alpha=0.9)
    ax.text((ood_left + ood_right) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50,
            "out-of-distribution", ha="center", va="top",
            fontsize=8.5, color="#9966cc", style="italic")

    legend_patches = []
    ik_color = "#aaaaaa"

    # IK bars
    for i, traj in enumerate(ALL_TRAJS):
        ik_val = IK_MM[traj]
        ik_std = IK_STD.get(traj)
        xi = x_base[i] + offsets[0]
        yerr = [[ik_std], [ik_std]] if ik_std else None
        ax.bar(xi, ik_val, width=bar_w, color=ik_color, edgecolor="white",
               lw=0.8, zorder=3, yerr=yerr,
               error_kw=dict(ecolor="#888888", lw=1.0, capsize=2.5, capthick=1.0))
        top = ik_val + (ik_std if ik_std else 0)
        ax.text(xi, top + 0.4, f"{ik_val:.0f}",
                ha="center", va="bottom", fontsize=6.5, color="#666666")
    legend_patches.append(mpatches.Patch(color=ik_color, label="IK (100ms delay)"))

    # Model bars
    for j, (arch, cfg) in enumerate(MODELS.items()):
        color = cfg["color"]
        label = cfg["label"]
        x_pos = x_base + offsets[j + 1]

        res       = arch_results[arch]
        n_indist  = min(len(res[t]) for t in INDIST_TRAJS) if INDIST_TRAJS else 0
        n_ood     = min(len(res[t]) for t in OOD_TRAJS)    if OOD_TRAJS    else 0
        has2_indist = n_indist >= 2
        has2_ood    = n_ood >= 2

        for i, traj in enumerate(ALL_TRAJS):
            entries = res[traj]
            if not entries:
                continue

            means = [e[0] for e in entries]
            stds  = [e[1] for e in entries if e[1] is not None]
            grand = float(np.mean(means))

            # Error bar strategy:
            # - stochastic (MT): mean ± mean_std across seeds
            # - deterministic multi-seed: inter-seed range
            yerr = None
            if stds:
                s = float(np.mean(stds))
                yerr = [[s], [s]]
            elif len(means) >= 2:
                yerr = [[grand - min(means)], [max(means) - grand]]

            top = grand + (yerr[1][0] if yerr else 0)
            ax.bar(x_pos[i], grand, width=bar_w, color=color, edgecolor="white",
                   lw=0.8, zorder=3, yerr=yerr,
                   error_kw=dict(ecolor="#333333", lw=1.1, capsize=2.5, capthick=1.1))

            is_ood  = traj in OOD_TRAJS
            has2    = has2_ood if is_ood else has2_indist
            suffix  = "" if has2 else "*"
            ax.text(x_pos[i], top + 0.35, f"{grand:.1f}{suffix}",
                    ha="center", va="bottom", fontsize=7,
                    color="#111111", fontweight="bold")

        # Seeds annotation for legend
        indist_note = f"2 seeds" if has2_indist else "1 seed*"
        ood_note    = f"2 seeds" if has2_ood    else "1 seed*"
        seeds_note  = indist_note if indist_note == ood_note else f"{indist_note} / OOD {ood_note}"
        legend_patches.append(mpatches.Patch(color=color,
                                              label=f"{label} 5M ({seeds_note})"))

    # Divider line between in-dist and OOD
    mid = (x_base[ood_start - 1] + x_base[ood_start]) / 2 - 0.05
    ax.axvline(mid, color="#ccbbee", lw=1.2, ls="--", zorder=1)

    ax.set_xticks(x_base)
    ax.set_xticklabels(TRAJ_LABELS, fontsize=10)
    ax.set_ylabel("Settled RMSE (mm) — lower is better", fontsize=11)
    ax.set_title("MLP vs Transformer — 5M steps  |  in-distribution & OOD generalization\n"
                 "Error bars: ±std for moving target (10 seeds); seed range for deterministic. *=single seed.",
                 fontsize=10.5, pad=10)
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.92, loc="upper right")
    ax.grid(axis="y", color="#e0e0e0", lw=0.8, zorder=0)
    ax.set_xlim(x_base[0] - group_w * 0.65, x_base[-1] + group_w * 0.65)
    ax.set_ylim(0, 60)

    # Re-draw OOD label after ylim is set
    ax.text((ood_left + ood_right) / 2, 57.5,
            "out-of-distribution", ha="center", va="top",
            fontsize=8.5, color="#7744aa", style="italic", fontweight="bold")

    plt.tight_layout()
    out = out_dir / "comparison_5M_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
