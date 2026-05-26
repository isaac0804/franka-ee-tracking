"""Evaluate both architectures at 1M/2M/3M/5M checkpoints and plot scaling curves.

Builds a compute-matched comparison: FLOPs/step are equal, so steps == FLOPs proxy.
Outputs:
  results/scaling/results.json          — raw per-checkpoint results
  results/figures/scaling_curves.png    — RMSE vs training steps

Usage:
    python scripts/scaling_eval.py               # run evals + plot
    python scripts/scaling_eval.py --plot-only   # skip evals, replot from saved json
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── config ────────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    "mlp": {
        "label": "MLP",
        "color": "#4e79a7",
        "model_dir": "results/sweep/rs012_5M",
        "steps": [1_000_000, 2_000_000, 3_000_000, 5_000_000],
    },
    "transformer": {
        "label": "Transformer (no cross-attn)",
        "color": "#e05252",
        "model_dir": "results/main_runs/tfm_no_xattn_5M_s42",
        "steps": [1_000_000, 2_000_000, 3_000_000, 5_000_000],
    },
}

# Hardcoded 300k points from earlier ablation runs (single seed, same hyperparams)
PRIOR_300K = {
    "mlp":         {"circle": 10.7, "figure8": 8.7, "moving_target": 25.9},
    "transformer": {"circle":  4.9, "figure8": 4.8, "moving_target": 23.6},
}

# IK reference (100ms delay), from multi-seed eval
IK_MM = {"circle": 11.5, "figure8": 7.7, "moving_target": 48.6}

TRAJECTORIES = "circle,figure8,moving_target"
N_SEEDS_MT   = 5      # moving_target is stochastic; 5 seeds for speed
OUT_BASE     = pathlib.Path("results/scaling")
FIGURES_DIR  = pathlib.Path("results/figures")


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_checkpoint(arch: str, model_dir: str, step: int) -> dict | None:
    """Run evaluate.py on one checkpoint; return ablation.json dict."""
    ckpt_dir   = pathlib.Path(model_dir) / "checkpoints"
    model_zip  = ckpt_dir / f"ppo_{step}_steps.zip"
    vecnorm    = ckpt_dir / f"ppo_vecnormalize_{step}_steps.pkl"

    if not model_zip.exists():
        print(f"  [skip] checkpoint not found: {model_zip}")
        return None
    if not vecnorm.exists():
        print(f"  [skip] vecnorm not found: {vecnorm}")
        return None

    out_dir = OUT_BASE / arch / f"{step // 1_000_000}M"
    out_dir.mkdir(parents=True, exist_ok=True)

    # evaluate.py looks for vecnormalize.pkl in the *same dir as the model*.
    # Create a thin staging dir with both files symlinked under expected names.
    staging = out_dir / "staging"
    staging.mkdir(exist_ok=True)

    config_yaml = pathlib.Path(model_dir) / "config.yaml"
    files_to_link = [("final_model.zip", model_zip), ("vecnormalize.pkl", vecnorm)]
    if config_yaml.exists():
        files_to_link.append(("config.yaml", config_yaml))

    for fname, src in files_to_link:
        dst = staging / fname
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    eval_out = out_dir / "eval"
    cmd = [
        sys.executable, "evaluate.py", "ablation",
        "--model",        str(staging / "final_model.zip"),
        "--trajectories", TRAJECTORIES,
        "--n-seeds",      str(N_SEEDS_MT),
        "--out",          str(eval_out),
    ]
    print(f"  running: {arch} @ {step//1_000_000}M steps …", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-800:]}")
        return None

    json_path = eval_out / "ablation.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return None


def load_or_run(arch: str, model_dir: str, step: int) -> dict | None:
    """Return cached result if present, otherwise run eval."""
    json_path = OUT_BASE / arch / f"{step // 1_000_000}M" / "eval" / "ablation.json"
    if json_path.exists():
        print(f"  [cached] {arch} @ {step//1_000_000}M")
        return json.loads(json_path.read_text())
    return eval_checkpoint(arch, model_dir, step)


# ── plotting ──────────────────────────────────────────────────────────────────

TRAJ_STYLES = {
    "circle":       {"ls": "-",  "marker": "o", "label": "Circle"},
    "figure8":      {"ls": "--", "marker": "s", "label": "Figure-8"},
    "moving_target":{"ls": ":",  "marker": "^", "label": "Moving Target"},
}


def plot_scaling(all_results: dict, out_path: pathlib.Path):
    """
    all_results: {arch: {step_M_float: {traj: rmse_mm}}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor("white")

    trajs = ["circle", "figure8", "moving_target"]
    traj_labels = ["Circle", "Figure-8", "Moving Target"]

    for ax, traj, tlabel in zip(axes, trajs, traj_labels):
        ax.set_facecolor("#fafafa")
        ax.set_title(tlabel, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training steps (millions)", fontsize=10)
        ax.set_ylabel("Settled RMSE (mm)", fontsize=10)
        ax.grid(axis="y", color="#e0e0e0", lw=0.8)
        ax.grid(axis="x", color="#eeeeee", lw=0.6)

        # IK reference
        ax.axhline(IK_MM[traj], color="#aaaaaa", lw=1.1, ls="-.",
                   label=f"IK 100ms ({IK_MM[traj]:.0f} mm)", zorder=2)

        for arch, cfg in CHECKPOINTS.items():
            color = cfg["color"]
            label = cfg["label"]

            steps_M = []
            rmse    = []

            # Prepend 300k data point
            if arch in PRIOR_300K and traj in PRIOR_300K[arch]:
                steps_M.append(0.3)
                rmse.append(PRIOR_300K[arch][traj])

            # Add checkpoint evals
            arch_data = all_results.get(arch, {})
            for step_M in sorted(arch_data):
                v = arch_data[step_M].get(traj)
                if v is not None:
                    steps_M.append(step_M)
                    rmse.append(v)

            if not steps_M:
                continue

            ax.plot(steps_M, rmse, color=color, lw=2.2, marker="o",
                    markersize=6, label=label, zorder=3)

            # Annotate final value
            ax.annotate(f"{rmse[-1]:.1f}",
                        xy=(steps_M[-1], rmse[-1]),
                        xytext=(6, 3), textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold")

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.1f}M" if x != int(x) else f"{int(x)}M"))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Scaling curves: MLP vs Transformer — RMSE vs training compute\n"
        "(seed=42; 300k points from ablation runs; 1–5M from 5M-run checkpoints; MT averaged over 5 seeds)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved → {out_path}")


# ── advantage table ───────────────────────────────────────────────────────────

def print_advantage_table(all_results: dict):
    print("\n=== Transformer advantage over MLP (%) at each compute level ===")
    print(f"{'Steps':>7}  {'CI adv':>8}  {'F8 adv':>8}  {'MT adv':>8}")
    print("-" * 40)

    mlp_data = all_results.get("mlp", {})
    tfm_data = all_results.get("transformer", {})

    # Include 300k
    rows = [(0.3, PRIOR_300K.get("mlp", {}), PRIOR_300K.get("transformer", {}))]
    for step_M in sorted(set(mlp_data) | set(tfm_data)):
        rows.append((step_M, mlp_data.get(step_M, {}), tfm_data.get(step_M, {})))

    for step_M, mlp, tfm in rows:
        advs = []
        for traj in ["circle", "figure8", "moving_target"]:
            m, t = mlp.get(traj), tfm.get(traj)
            if m and t:
                advs.append(f"{(m-t)/m*100:+.0f}%")
            else:
                advs.append("  n/a")
        print(f"{step_M:>6.1f}M  {advs[0]:>8}  {advs[1]:>8}  {advs[2]:>8}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip evals, replot from saved results.json")
    args = ap.parse_args()

    results_json = OUT_BASE / "results.json"

    if args.plot_only:
        if not results_json.exists():
            print(f"No saved results at {results_json}; run without --plot-only first.")
            return
        all_results = json.loads(results_json.read_text())
    else:
        all_results = {}
        for arch, cfg in CHECKPOINTS.items():
            print(f"\n── {cfg['label']} ──────────────────────────────")
            arch_results = {}
            for step in cfg["steps"]:
                data = load_or_run(arch, cfg["model_dir"], step)
                if data is None:
                    continue
                step_M = step / 1_000_000
                arch_results[step_M] = {
                    traj: data[traj]["residual_settled_rmse_mm"]
                    for traj in ["circle", "figure8", "moving_target"]
                    if traj in data
                }
            all_results[arch] = arch_results

        OUT_BASE.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(all_results, indent=2))
        print(f"\nsaved → {results_json}")

    print_advantage_table(all_results)
    plot_scaling(all_results, FIGURES_DIR / "scaling_curves.png")


if __name__ == "__main__":
    main()
