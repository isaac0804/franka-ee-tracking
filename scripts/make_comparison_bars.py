"""Per-trajectory RMSE bar chart: IK vs MLP@5M vs Transformer@5M (2-seed means ± range).

Usage:
    python scripts/make_comparison_bars.py
    python scripts/make_comparison_bars.py --out results/figures

Requires both seed=42 and seed=1 models to be evaluated first (ablation.json).
Runs eval automatically if json is missing.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── model registry ────────────────────────────────────────────────────────────
MODELS = {
    "mlp": {
        "label": "MLP",
        "color": "#4e79a7",
        "runs": [
            "results/sweep/rs012_5M",
            "results/sweep/rs012_seed1_5M",
        ],
        # Pre-computed multi-seed eval dirs (10-seed MT averaging)
        "multi_eval_dirs": [
            "results/eval/mlp_5M_s42_multi",
            "results/eval/mlp_5M_s1_multi",
        ],
    },
    "transformer": {
        "label": "Transformer\n(no cross-attn)",
        "color": "#e05252",
        "runs": [
            "results/main_runs/tfm_no_xattn_5M_s42",
            "results/main_runs/tfm_no_xattn_5M_s1",
        ],
        "multi_eval_dirs": [
            "results/eval/tfm_5M_s42_multi",
            "results/eval/tfm_5M_s1_multi",   # populated after s1 training completes
        ],
    },
}

TRAJECTORIES = ["moving_target", "circle", "figure8"]
TRAJ_LABELS  = ["Moving Target", "Circle", "Figure‑8"]

IK_MM = {"moving_target": 38.1, "circle": 12.1, "figure8": 7.7}


# ── helpers ───────────────────────────────────────────────────────────────────
def load_multi_eval(eval_dir: str) -> dict | None:
    """Load pre-computed multi-seed ablation.json from an eval directory."""
    json_p = pathlib.Path(eval_dir) / "ablation.json"
    if not json_p.exists():
        return None
    return json.loads(json_p.read_text())


def get_rmse(data: dict, traj: str) -> tuple[float | None, float | None]:
    """Return (mean, std) RMSE. std is None for deterministic trajectories."""
    t = data.get(traj, {})
    mean = t.get("residual_settled_rmse_mm")
    std  = t.get("residual_settled_rmse_mm_std")   # present only for stochastic trajs
    return mean, std


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Fail instead of running evaluate.py automatically")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── gather per-seed results from multi-seed eval dirs ────────────────────
    # arch → traj → list of (mean, std_or_None) per model seed
    arch_results = {}
    for arch, cfg in MODELS.items():
        arch_results[arch] = {t: [] for t in TRAJECTORIES}
        for eval_dir in cfg.get("multi_eval_dirs", []):
            data = load_multi_eval(eval_dir)
            if data is None:
                print(f"  [skip] {eval_dir} — no ablation.json")
                continue
            for traj in TRAJECTORIES:
                mean, std = get_rmse(data, traj)
                if mean is not None:
                    arch_results[arch][traj].append((mean, std))

    # ── plot ──────────────────────────────────────────────────────────────────
    n_traj   = len(TRAJECTORIES)
    n_arch   = len(MODELS)
    bar_w    = 0.22
    group_w  = 0.10 + n_arch * bar_w
    x_base   = np.arange(n_traj) * (group_w + 0.25)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    offsets = np.linspace(-(n_arch - 1) * bar_w / 2,
                           (n_arch - 1) * bar_w / 2,
                           n_arch)

    legend_patches = []

    # IK baseline bars (leftmost position in each group, lighter)
    ik_color = "#aaaaaa"
    ik_x = x_base + offsets[0] - bar_w
    for i, traj in enumerate(TRAJECTORIES):
        ax.bar(ik_x[i], IK_MM[traj], width=bar_w, color=ik_color,
               edgecolor="white", lw=0.8, zorder=3)
        ax.text(ik_x[i], IK_MM[traj] + 0.3, f"{IK_MM[traj]:.1f}",
                ha="center", va="bottom", fontsize=7, color="#666666")
    legend_patches.append(mpatches.Patch(color=ik_color, label="IK baseline (100ms delay)"))

    for j, (arch, cfg) in enumerate(MODELS.items()):
        color  = cfg["color"]
        label  = cfg["label"]
        x_pos  = x_base + offsets[j]

        seed_data = arch_results[arch]
        n_seeds_available = min(len(seed_data[t]) for t in TRAJECTORIES)
        has_2seeds = n_seeds_available >= 2

        for i, traj in enumerate(TRAJECTORIES):
            entries_for_traj = seed_data[traj]   # list of (mean, std_or_None)
            if not entries_for_traj:
                continue

            means = [e[0] for e in entries_for_traj]
            stds  = [e[1] for e in entries_for_traj if e[1] is not None]
            grand_mean = float(np.mean(means))

            # Error bar: for stochastic trajectories use mean±std from the
            # multi-seed eval (more honest than inter-seed range).
            # For deterministic trajectories show inter-seed range.
            yerr = None
            if stds:
                # Use the mean std across model seeds (MT)
                avg_std = float(np.mean(stds))
                yerr = [[avg_std], [avg_std]]
            elif len(means) >= 2:
                yerr = [[grand_mean - min(means)], [max(means) - grand_mean]]

            top = grand_mean + (yerr[1][0] if yerr else 0)
            ax.bar(x_pos[i], grand_mean, width=bar_w, color=color,
                   edgecolor="white", lw=0.8, zorder=3,
                   yerr=yerr, error_kw=dict(ecolor="#333333", lw=1.2,
                                            capsize=3, capthick=1.2))

            suffix = "" if has_2seeds else "*"
            ax.text(x_pos[i], top + 0.3, f"{grand_mean:.1f}{suffix}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#222222", fontweight="bold")

        seeds_note = "(2 seeds)" if has_2seeds else "(1 seed*)"
        legend_patches.append(mpatches.Patch(color=color,
                                              label=f"{label} 5M {seeds_note}"))

    ax.set_xticks(x_base)
    ax.set_xticklabels(TRAJ_LABELS, fontsize=11)
    ax.set_ylabel("Settled RMSE (mm) — lower is better", fontsize=11)
    ax.set_title("MLP vs Transformer — 5M steps, 2 seeds\n"
                 "Error bars = seed range. * = single seed.",
                 fontsize=11, pad=10)
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9,
              loc="upper right")
    ax.grid(axis="y", color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlim(x_base[0] - group_w * 0.7, x_base[-1] + group_w * 0.7)
    ax.set_ylim(0, max(IK_MM.values()) * 1.15)

    plt.tight_layout()
    out = out_dir / "comparison_5M_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
