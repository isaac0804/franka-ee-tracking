#!/usr/bin/env python3
"""Overnight hyperparameter sweep — runs configs sequentially, logs results.

Usage:
    python sweep.py                        # run all 5 configs
    python sweep.py --runs base_5M lrdecay_5M   # run a subset by name

Results land in results/sweep/<run_name>/.
A summary table is printed at the end and saved to results/sweep/summary.csv.

Each config is 5M steps (~35 min each); total wall time ~3 hours.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------

SWEEP_DIR  = Path("ee_tracking/configs/sweep")
OUT_BASE   = Path("results/sweep")

RUNS: list[tuple[str, str, str]] = [
    # (name,               config_file,                   description)
    ("base_5M",
     "base_5M.yaml",
     "control — 5M steps, all else baseline"),
    ("lrdecay_5M",
     "lrdecay_5M.yaml",
     "LR 1e-3 → 1e-4 linear decay"),
    ("gamma99_5M",
     "gamma99_5M.yaml",
     "gamma 0.97 → 0.99"),
    ("nosmooth_5M",
     "nosmooth_5M.yaml",
     "w_smooth=0, w_jerk=0 (unconstrained corrections)"),
    ("lrdecay_gamma99_5M",
     "lrdecay_gamma99_5M.yaml",
     "LR decay + gamma=0.99 (combo best-guess)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tb_metrics(run_dir: Path) -> dict:
    """Pull final and best pos_err + final EV/KL from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    tb_dir = run_dir / "tb" / "PPO_1"
    if not tb_dir.exists():
        return {}

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    def last(tag: str) -> float:
        try:
            evs = ea.Scalars(tag)
            return evs[-1].value if evs else float("nan")
        except KeyError:
            return float("nan")

    def best_min(tag: str) -> float:
        try:
            return min(e.value for e in ea.Scalars(tag))
        except KeyError:
            return float("nan")

    return {
        "pos_err_final": last("tracking/pos_err_mm"),
        "pos_err_best":  best_min("tracking/pos_err_mm"),
        "EV_final":      last("train/explained_variance"),
        "KL_final":      last("train/approx_kl"),
        "residual_norm": last("tracking/residual_norm"),
    }


def fmt(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}" if v == v else "nan"


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    header = (f"{'run':<26} {'final mm':>9} {'best mm':>9}"
              f" {'EV':>7} {'KL':>7} {'time':>8}")
    sep = "=" * len(header)
    print(f"\n{sep}\nSWEEP SUMMARY\n{sep}")
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("status") == "ok":
            line = (
                f"{r['name']:<26}"
                f" {fmt(r.get('pos_err_final', float('nan'))):>9}"
                f" {fmt(r.get('pos_err_best',  float('nan'))):>9}"
                f" {fmt(r.get('EV_final',  float('nan')), 3):>7}"
                f" {fmt(r.get('KL_final',  float('nan')), 3):>7}"
                f" {r.get('elapsed_min', 0):>7.1f}m"
            )
        else:
            line = f"{r['name']:<26}  FAILED  ({r.get('status','?')})"
        print(line)
    print(sep)
    print(f"\n  Reference:  IK+delay=37.2 mm  |"
          f"  policy@500K=27.4 mm  |  IK_floor=14.7 mm")


def save_summary(results: list[dict], out: Path) -> None:
    keys = ["name", "status", "pos_err_final", "pos_err_best",
            "EV_final", "KL_final", "residual_norm", "elapsed_min",
            "config", "description"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Subset of run names (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = RUNS
    if args.runs:
        names = set(args.runs)
        selected = [r for r in RUNS if r[0] in names]
        if not selected:
            print(f"No matching runs for: {args.runs}")
            sys.exit(1)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SWEEP: {len(selected)} runs × 5M steps  (~35 min/run)")
    print(f"  Output: {OUT_BASE.resolve()}")
    print(f"{'='*60}")
    for name, _, desc in selected:
        print(f"  • {name:<26}  {desc}")
    print()

    all_results: list[dict] = []
    sweep_t0 = time.time()

    for idx, (name, cfg_file, desc) in enumerate(selected, 1):
        cfg_path = SWEEP_DIR / cfg_file
        out_dir  = OUT_BASE / name
        log_path = out_dir / "train.log"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "train.py",
               "--config", str(cfg_path),
               "--out",    str(out_dir)]

        print(f"\n{'─'*60}")
        print(f"[{idx}/{len(selected)}]  {name}")
        print(f"   desc   : {desc}")
        print(f"   config : {cfg_path}")
        print(f"   log    : {log_path}")

        if args.dry_run:
            print("   [DRY RUN — skipped]")
            continue

        result: dict = {"name": name, "config": cfg_file, "description": desc}
        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_path.write_text(proc.stdout)

            elapsed = time.time() - t0
            result["elapsed_min"] = elapsed / 60.0

            # Print key lines so the terminal isn't silent for 35 min
            for line in proc.stdout.splitlines():
                s = line.strip()
                if any(s.startswith(p) for p in (
                    "| total_timesteps", "| pos_err_mm",
                    "| explained_variance", "Done in"
                )):
                    print(f"   {s}")

            if proc.returncode == 0:
                result["status"] = "ok"
                metrics = read_tb_metrics(out_dir)
                result.update(metrics)
                print(f"\n   ✓  {elapsed/60:.1f} min  |"
                      f"  pos_err {fmt(metrics.get('pos_err_final', float('nan')))} mm"
                      f"  (best {fmt(metrics.get('pos_err_best', float('nan')))} mm)"
                      f"  EV={fmt(metrics.get('EV_final', float('nan')), 3)}")
            else:
                result["status"] = f"exit_{proc.returncode}"
                print(f"\n   ✗  FAILED exit={proc.returncode}")
                print('\n'.join(proc.stdout.splitlines()[-20:]))

        except Exception as exc:
            result["status"] = f"exc:{exc}"
            result["elapsed_min"] = (time.time() - t0) / 60.0
            print(f"\n   ✗  EXCEPTION: {exc}")

        all_results.append(result)
        # Incremental save so partial results survive a crash
        save_summary(all_results, OUT_BASE / "summary.csv")

    print_summary(all_results)
    total_min = (time.time() - sweep_t0) / 60.0
    print(f"\nTotal sweep time: {total_min:.1f} min")

    with open(OUT_BASE / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"summary.csv  → {OUT_BASE / 'summary.csv'}")
    print(f"summary.json → {OUT_BASE / 'summary.json'}")


if __name__ == "__main__":
    main()
