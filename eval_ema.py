#!/usr/bin/env python3
"""Post-hoc EMA action-smoothing sweep across trained sweep models.

NOTE — LEGACY SCRIPT (pre-delay architecture):
    This script was written when the main disturbance was residual-only delay
    (act_delay=1, 20 ms) and action noise was the dominant problem.  Post-hoc
    EMA smoothing helped because the policy produced noisy corrections that
    needed filtering at inference time.

    The current architecture uses a whole-pipeline cmd_delay=5 (100 ms) applied
    equally to IK and residual.  The policy's job is now delay compensation via
    lookahead, not noise filtering.  Post-hoc EMA on top of that is likely
    counterproductive (adds phase lag on top of the delay the policy already
    compensates for).

    Use this script only to reproduce old results or as a reference.
    For current models use evaluate.py directly.

No retraining.  For each model found under --sweep-dir, runs evaluation at
several EMA alpha values and reports which (model, alpha) combinations improve
over the raw (alpha=1.0) baseline and over pure IK.

EMA: smoothed_t = (1 - alpha) * smoothed_{t-1} + alpha * raw_t
  alpha=1.0  → no smoothing (raw policy output)
  alpha=0.5  → half-life ~1.4 steps  (28 ms at 50 Hz)
  alpha=0.3  → half-life ~1.9 steps  (38 ms)
  alpha=0.2  → half-life ~3.1 steps  (62 ms)
  alpha=0.1  → half-life ~6.6 steps  (132 ms)

Usage:
    python eval_ema.py                                   # all sweep models
    python eval_ema.py --models rs012_wr05 rs008_wr01   # specific models only
    python eval_ema.py --alphas 1.0 0.5 0.3 0.2         # custom alpha grid
    python eval_ema.py --seeds 42 43 44                  # average over seeds
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from evaluate import load_model, run_ik, run_residual, _env_kwargs_from_cfg


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
DEFAULT_SEEDS  = [42]
TRAJECTORY     = "moving_target"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def half_life_ms(alpha: float, hz: float = 50.0) -> str:
    if alpha >= 1.0:
        return "off"
    hl_steps = math.log(0.5) / math.log(1.0 - alpha)
    return f"{hl_steps / hz * 1000:.0f} ms"


def eval_model(model_path: Path, alphas: list[float], seeds: list[int]) -> list[dict]:
    """Return one row per alpha, averaged over seeds."""
    model, vn_ref, saved_cfg = load_model(str(model_path))
    env_kwargs = _env_kwargs_from_cfg(saved_cfg)

    # IK baseline — same disturbance, average over seeds
    ik_errs = []
    for seed in seeds:
        r = run_ik(TRAJECTORY, seed=seed, disturbance=env_kwargs["disturbance"])
        ik_errs.append(r["settled_rmse_mm"])
    ik_mm = float(np.mean(ik_errs))

    rows = []
    for alpha in alphas:
        res_errs = []
        for seed in seeds:
            r = run_residual(
                model, vn_ref, TRAJECTORY,
                seed=seed, action_ema=alpha,
                **env_kwargs,
            )
            res_errs.append(r["settled_rmse_mm"])
        res_mm = float(np.mean(res_errs))
        improv = (ik_mm - res_mm) / ik_mm * 100.0
        rows.append({
            "name":    model_path.parent.name,
            "alpha":   alpha,
            "hl_ms":   half_life_ms(alpha),
            "ik_mm":   ik_mm,
            "res_mm":  res_mm,
            "delta":   improv,
        })
    return rows


def print_table(all_rows: list[dict]):
    # Sort by delta descending
    all_rows = sorted(all_rows, key=lambda r: -r["delta"])

    col_name  = max(len(r["name"]) for r in all_rows)
    print()
    print("=" * 74)
    print(f"  {'model':<{col_name}}  {'alpha':>6}  {'hl':>7}  {'IK mm':>7}  {'res mm':>8}  {'Δ %':>8}  notes")
    print("=" * 74)
    for r in all_rows:
        beat_ik  = " ✓" if r["delta"] > 0    else ""
        raw_row  = r["alpha"] >= 1.0
        marker   = beat_ik + (" (raw)" if raw_row else "")
        print(
            f"  {r['name']:<{col_name}}  {r['alpha']:>6.1f}  {r['hl_ms']:>7}  "
            f"{r['ik_mm']:>7.1f}  {r['res_mm']:>8.1f}  {r['delta']:>+7.1f}%{marker}"
        )
    print("=" * 74)
    print()


def save_csv(all_rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMA smoothing sweep on trained models")
    parser.add_argument(
        "--sweep-dir", default="results/sweep",
        help="Root sweep output directory (default: results/sweep)",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model names to evaluate (default: all with final_model.zip)",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
        help=f"EMA alpha values to try (default: {DEFAULT_ALPHAS})",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Random seeds to average over (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--out", default="results/ema_sweep/summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_dir)
    if args.models:
        model_paths = [sweep_root / name / "final_model.zip" for name in args.models]
    else:
        model_paths = sorted(sweep_root.glob("*/final_model.zip"))

    missing = [p for p in model_paths if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} model(s) not found:")
        for p in missing:
            print(f"  {p}")
        model_paths = [p for p in model_paths if p.exists()]

    if not model_paths:
        print("No models found — exiting.")
        return

    hl_info = "  ".join(f"{a}→{half_life_ms(a)}" for a in args.alphas)
    print(f"\nEMA sweep: {len(model_paths)} models × {len(args.alphas)} alphas × {len(args.seeds)} seed(s)")
    print(f"Alphas: {hl_info}")
    print(f"Trajectory: {TRAJECTORY}\n")

    all_rows: list[dict] = []
    for i, model_path in enumerate(model_paths, 1):
        name = model_path.parent.name
        print(f"[{i}/{len(model_paths)}] {name} ...", flush=True)
        try:
            rows = eval_model(model_path, args.alphas, args.seeds)
            for r in rows:
                raw = next(x for x in rows if x["alpha"] >= 1.0)
                r["vs_raw_delta"] = r["delta"] - raw["delta"]
            all_rows.extend(rows)
            # Print per-model summary
            best = max(rows, key=lambda r: r["delta"])
            raw  = next(r for r in rows if r["alpha"] >= 1.0)
            print(f"  raw(1.0): {raw['res_mm']:.1f}mm  Δ={raw['delta']:+.1f}%  |  "
                  f"best(α={best['alpha']}): {best['res_mm']:.1f}mm  Δ={best['delta']:+.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

    if all_rows:
        print_table(all_rows)
        save_csv(all_rows, Path(args.out))


if __name__ == "__main__":
    main()
