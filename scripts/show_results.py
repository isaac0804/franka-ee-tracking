#!/usr/bin/env python3
"""
show_results.py — Experiment viewer for Franka EE Tracking residual PPO
========================================================================

Usage:
    python show_results.py                # terminal table, all runs
    python show_results.py --sort mt      # sort by moving_target RMSE (default)
    python show_results.py --sort ev      # sort by explained variance
    python show_results.py --filter probe # only show probe/* runs
    python show_results.py --filter sweep # only show sweep/* runs
    python show_results.py --chart        # also regenerate results/probe_progress.png
    python show_results.py --tb           # print tensorboard launch command

Discovers experiments by walking results/ for TB event files.
For each run, reads:
  - config.yaml       → key hyperparams (rs, delay, pool, steps)
  - ablation.json     → eval RMSE (moving_target / circle / figure8)
  - TB events         → final pos_err_mm, final EV, FPS
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
RESULTS_ROOT = Path("results")
IK_MT_MM     = 38.1
IK_CI_MM     = 12.1
IK_F8_MM     =  7.7

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── helpers ───────────────────────────────────────────────────────────────────

def color_mm(val, ik_ref, *, lower_is_better=True):
    """Return coloured string: green if beats IK, red if worse."""
    if val is None:
        return GRAY + "  —  " + RESET
    s = f"{val:6.1f}"
    if lower_is_better:
        c = GREEN if val < ik_ref else (RED if val > ik_ref * 1.05 else YELLOW)
    else:
        c = GREEN if val > ik_ref else RED
    return c + s + RESET


def read_tb_finals(tb_dir: Path):
    """Return {tag: last_value} for a TB event directory (reads first PPO_N subdir found)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    # Find any PPO_* subdir
    subdirs = sorted(tb_dir.glob("PPO_*"))
    if not subdirs:
        subdirs = [tb_dir]

    result = {}
    for subdir in subdirs:
        try:
            ea = EventAccumulator(str(subdir), size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                if events:
                    result[tag] = events[-1].value
                    result[f"{tag}__step"] = events[-1].step
        except Exception:
            pass
    return result


def read_config(config_path: Path):
    """Return parsed config dict or {}."""
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def find_ablation(model_dir: Path, run_name: str, phase: str):
    """
    Try several candidate locations for ablation.json given a model dir.
    Returns (path, data) or (None, {}).
    """
    candidates = [
        model_dir / "eval" / "ablation.json",
        model_dir / "ablation.json",
        RESULTS_ROOT / "eval" / phase / run_name / "ablation.json",
        RESULTS_ROOT / "eval" / run_name / "ablation.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return p, json.loads(p.read_text())
            except Exception:
                pass
    return None, {}


def pool_str(cfg: dict):
    """Short string for trajectory pool."""
    pool = cfg.get("env", {}).get("trajectory_pool", [])
    if isinstance(pool, list):
        if len(pool) == 3:
            return "mixed"
        elif len(pool) == 1:
            return pool[0][:4]
        else:
            return "+".join(p[:3] for p in pool)
    return str(pool)[:8]


# ── experiment discovery ──────────────────────────────────────────────────────

def discover_experiments():
    """
    Walk results/ for directories containing 'config.yaml' or TB events.
    Returns list of experiment dicts.
    """
    exps = []
    seen_dirs = set()

    # Strategy: find all config.yaml files inside results/ — these mark model dirs
    for config_path in sorted(RESULTS_ROOT.rglob("config.yaml")):
        model_dir = config_path.parent
        if model_dir in seen_dirs:
            continue
        seen_dirs.add(model_dir)

        # Determine phase/name
        rel = model_dir.relative_to(RESULTS_ROOT)
        parts = rel.parts
        if len(parts) == 1:
            phase, run_name = "root", parts[0]
        else:
            phase, run_name = parts[0], "/".join(parts[1:])

        # Read config
        cfg = read_config(config_path)
        env  = cfg.get("env", {})
        trn  = cfg.get("train", {})

        rs        = env.get("residual_scale", "?")
        cmd_delay = env.get("disturbance", {}).get("cmd_delay", "?")
        steps     = trn.get("total_timesteps", None)
        lr        = trn.get("learning_rate", None)
        lr_final  = trn.get("lr_final", None)
        n_envs    = trn.get("n_envs", None)
        seed      = trn.get("seed", 42)

        # TB data
        tb_dir = model_dir / "tb"
        tb = read_tb_finals(tb_dir) if tb_dir.exists() else {}

        final_pos_err = tb.get("tracking/pos_err_mm")
        final_ev      = tb.get("train/explained_variance")
        final_steps   = tb.get("tracking/pos_err_mm__step")  # last logged step
        fps           = tb.get("time/fps")

        # Eval data
        _, abl = find_ablation(model_dir, run_name, phase)
        mt = abl.get("moving_target", {}).get("residual_settled_rmse_mm")
        ci = abl.get("circle", {}).get("residual_settled_rmse_mm")
        f8 = abl.get("figure8", {}).get("residual_settled_rmse_mm")

        exps.append({
            "phase":      phase,
            "run_name":   run_name,
            "model_dir":  str(model_dir),
            "rs":         rs,
            "cmd_delay":  cmd_delay,
            "pool":       pool_str(cfg),
            "steps":      steps,
            "steps_done": final_steps,
            "lr":         lr,
            "lr_final":   lr_final,
            "n_envs":     n_envs,
            "seed":       seed,
            "pos_err_mm": final_pos_err,
            "ev":         final_ev,
            "fps":        fps,
            "mt":         mt,
            "ci":         ci,
            "f8":         f8,
        })

    return exps


# ── display ───────────────────────────────────────────────────────────────────

def fmt_steps(s):
    if s is None:
        return "  —  "
    if s >= 1_000_000:
        return f"{s/1_000_000:.1f}M"
    if s >= 1_000:
        return f"{s/1_000:.0f}K"
    return str(s)


def print_table(exps, sort_by="mt", filter_phase=None):
    if filter_phase:
        exps = [e for e in exps if filter_phase in e["phase"] or filter_phase in e["run_name"]]

    if sort_by == "mt":
        exps = sorted(exps, key=lambda e: (e["mt"] or 999))
    elif sort_by == "ev":
        exps = sorted(exps, key=lambda e: -(e["ev"] or -999))
    elif sort_by == "name":
        exps = sorted(exps, key=lambda e: (e["phase"], e["run_name"]))

    # Header
    hdr = (
        f"{'Run':<42}  {'rs':>5}  {'dly':>4}  {'pool':>6}  "
        f"{'steps':>6}  {'pos_err':>8}  {'EV':>6}  "
        f"{'MT(mm)':>8}  {'CI(mm)':>8}  {'F8(mm)':>8}"
    )
    print()
    print(BOLD + "=" * len(hdr) + RESET)
    print(BOLD + hdr + RESET)
    print(BOLD + "=" * len(hdr) + RESET)

    prev_phase = None
    for e in exps:
        if e["phase"] != prev_phase:
            prev_phase = e["phase"]
            print(CYAN + f"\n  [{e['phase']}]" + RESET)

        run_label = e["run_name"]
        if len(run_label) > 41:
            run_label = "…" + run_label[-40:]

        pos_err_s = f"{e['pos_err_mm']:7.1f}" if e["pos_err_mm"] else "      —"
        ev_s      = f"{e['ev']:.3f}"           if e["ev"]         else "     —"
        steps_s   = fmt_steps(e["steps_done"] or e["steps"])

        mt_s = color_mm(e["mt"], IK_MT_MM)
        ci_s = color_mm(e["ci"], IK_CI_MM)
        f8_s = color_mm(e["f8"], IK_F8_MM)

        print(
            f"  {run_label:<42}  {str(e['rs']):>5}  {str(e['cmd_delay']):>4}  "
            f"{e['pool']:>6}  {steps_s:>6}  "
            f"{pos_err_s:>8}  {ev_s:>6}  "
            f"{mt_s:>8}  {ci_s:>8}  {f8_s:>8}"
        )

    print()
    print(BOLD + f"  IK + 100ms delay baseline:  MT={IK_MT_MM}mm  CI={IK_CI_MM}mm  F8={IK_F8_MM}mm" + RESET)
    print(BOLD + "=" * len(hdr) + RESET)
    print(f"  {GREEN}green{RESET} = beats IK  {YELLOW}yellow{RESET} = within 5%  {RED}red{RESET} = worse than IK\n")


# ── chart ─────────────────────────────────────────────────────────────────────

def regenerate_chart(exps):
    """Regenerate results/probe_progress.png with all eval'd experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping chart")
        return

    # Only include experiments with at least MT eval
    eval_exps = [e for e in exps if e["mt"] is not None]
    # Sort by something meaningful for x-axis (try to preserve chronological order)
    eval_exps = sorted(eval_exps, key=lambda e: (e["phase"], e["run_name"]))

    xs   = list(range(len(eval_exps)))
    mts  = [e["mt"] for e in eval_exps]
    cis  = [e["ci"] for e in eval_exps]
    f8s  = [e["f8"] for e in eval_exps]

    # Running best on MT
    running_best = []
    best = float("inf")
    for v in mts:
        best = min(best, v)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    # IK reference lines
    ax.axhline(IK_MT_MM, color="#cc0000", lw=1.5, ls="--", alpha=0.7, label=f"IK+delay MT ({IK_MT_MM}mm)")
    ax.axhline(IK_CI_MM, color="#cc6600", lw=1.0, ls=":",  alpha=0.5, label=f"IK+delay CI ({IK_CI_MM}mm)")
    ax.axhline(IK_F8_MM, color="#aa8800", lw=1.0, ls=":",  alpha=0.5, label=f"IK+delay F8 ({IK_F8_MM}mm)")

    # Running best step line
    ax.step(xs, running_best, where="post", color="#1a7a1a", lw=2, alpha=0.5, label="Running best (MT)")

    # Secondary trajectories (where available)
    ci_xs  = [x for x, v in zip(xs, cis)  if v is not None]
    ci_vs  = [v for v in cis  if v is not None]
    f8_xs  = [x for x, v in zip(xs, f8s)  if v is not None]
    f8_vs  = [v for v in f8s  if v is not None]

    ax.scatter(ci_xs, ci_vs, marker="s", s=40, color="#1565C0", alpha=0.6, label="circle eval", zorder=4)
    ax.scatter(f8_xs, f8_vs, marker="^", s=40, color="#E65100", alpha=0.6, label="figure8 eval", zorder=4)

    # MT main dots — green if new best, gray otherwise
    new_best_flags = [mts[i] < (min(mts[:i]) if i > 0 else float("inf")) for i in range(len(mts))]
    for xi, yi, nb in zip(xs, mts, new_best_flags):
        color = "#2ca02c" if nb else "#888888"
        ax.scatter(xi, yi, s=80, color=color, zorder=5)

    # Add experiment name labels (rotated, small)
    labels = [f"{i}\n{e['run_name'].split('/')[-1][:12]}" for i, e in enumerate(eval_exps)]
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [e["run_name"].split("/")[-1][:14] for e in eval_exps],
        rotation=45, ha="right", fontsize=7
    )

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Eval RMSE (mm)", fontsize=11)
    ax.set_title("Franka EE Tracking — All Eval Results", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3, color="#cccccc")
    ax.set_ylim(0, max(mts) * 1.1)

    plt.tight_layout()
    out = RESULTS_ROOT / "probe_progress.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved → {out}")


# ── tensorboard hint ──────────────────────────────────────────────────────────

def print_tb_hint():
    print()
    print(BOLD + "TensorBoard" + RESET)
    print("  All training runs log to results/*/tb/ — point TB at the root:")
    print()
    print(f"    {CYAN}.venv/bin/tensorboard --logdir results/ --port 6006{RESET}")
    print()
    print("  Useful tags available:")
    tags = [
        "tracking/pos_err_mm   — live EE tracking error",
        "train/explained_variance — critic quality (1.0 = perfect)",
        "train/std              — policy entropy proxy",
        "train/clip_fraction    — PPO clipping rate",
        "time/fps               — samples per second",
    ]
    for t in tags:
        print(f"    • {t}")
    print()
    print("  Compare specific runs:")
    print(f"    {CYAN}.venv/bin/tensorboard --logdir_spec=probe:results/probe,sweep:results/sweep{RESET}")
    print()


# ── summary stats ─────────────────────────────────────────────────────────────

def print_summary(exps):
    eval_exps = [e for e in exps if e["mt"] is not None]
    beats_ik  = [e for e in eval_exps if e["mt"] < IK_MT_MM]

    print(f"\n  Total experiments tracked : {len(exps)}")
    print(f"  With eval (ablation.json) : {len(eval_exps)}")
    print(f"  Beat IK on moving_target  : {len(beats_ik)}")

    if beats_ik:
        best = min(beats_ik, key=lambda e: e["mt"])
        print(f"  Best MT RMSE              : {GREEN}{best['mt']:.1f}mm{RESET}  ({best['phase']}/{best['run_name']})")

    ci_beats = [e for e in eval_exps if e["ci"] is not None and e["ci"] < IK_CI_MM]
    f8_beats = [e for e in eval_exps if e["f8"] is not None and e["f8"] < IK_F8_MM]
    print(f"  Beat IK on circle         : {len(ci_beats)}")
    print(f"  Beat IK on figure8        : {len(f8_beats)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Show experiment results table")
    parser.add_argument("--sort",   default="mt",  choices=["mt", "ev", "name"],
                        help="Sort by: mt=moving_target RMSE, ev=explained variance, name=run name")
    parser.add_argument("--filter", default=None,
                        help="Filter by phase or string in run name (e.g. 'probe', 'sweep', 'rs012')")
    parser.add_argument("--chart",  action="store_true",
                        help="Regenerate results/probe_progress.png")
    parser.add_argument("--tb",     action="store_true",
                        help="Print TensorBoard launch command and tag reference")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    print(f"\n{BOLD}Scanning experiments in results/ ...{RESET}", end="", flush=True)
    exps = discover_experiments()
    print(f" {len(exps)} found.")

    print_summary(exps)
    print_table(exps, sort_by=args.sort, filter_phase=args.filter)

    if args.chart:
        print("Regenerating chart ...")
        regenerate_chart(exps)

    if args.tb:
        print_tb_hint()


if __name__ == "__main__":
    main()
