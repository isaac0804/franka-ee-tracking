#!/usr/bin/env python3
"""Short diagnostic probes — run several configs at 300k steps and compare.

Each probe writes its final SB3 training metrics + eval RMSE to a summary
table, so we can speculate about what a 1.5M run would do without waiting.

Usage:
    python probe.py                    # run all probes (≈15 min)
    python probe.py --only baseline    # single probe
    python probe.py --timesteps 150000 # quick smoke-test (≈90s/probe)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Probe grid
# ---------------------------------------------------------------------------
# Each probe is a delta over default.yaml — only the keys listed here are
# overridden; everything else inherits from the YAML.
PROBES: list[dict] = [
    {
        "name": "baseline",
        "desc": "default — cmd_delay=5, Butterworth 2Hz, rs=0.05, wr=0.50",
        "env": {},
    },
    {
        "name": "wr020",
        "desc": "lower regularization: w_residual 0.50 → 0.20",
        "env": {"w_residual": 0.20},
    },
    {
        "name": "wr010",
        "desc": "very low regularization: w_residual 0.50 → 0.10",
        "env": {"w_residual": 0.10},
    },
    {
        "name": "rs010",
        "desc": "more authority: residual_scale 0.05 → 0.10",
        "env": {"residual_scale": 0.10},
    },
    {
        "name": "rs010_wr020",
        "desc": "combined: residual_scale=0.10, w_residual=0.20",
        "env": {"residual_scale": 0.10, "w_residual": 0.20},
    },
    {
        "name": "nofilter",
        "desc": "no baked filter (action_filter_hz=0) — diagnose filter overhead",
        "env": {"action_filter_hz": 0.0},
    },
    {
        "name": "hz5",
        "desc": "looser Butterworth: 2Hz → 5Hz",
        "env": {"action_filter_hz": 5.0},
    },
    {
        "name": "delay_3",
        "desc": "shorter delay: cmd_delay=5 → 3 (60 ms)",
        "env": {"disturbance": {"obs_pos_noise": 0.005, "obs_jnt_noise": 0.002, "cmd_delay": 3}},
    },
    {
        "name": "delay_8",
        "desc": "longer delay: cmd_delay=5 → 8 (160 ms)",
        "env": {"disturbance": {"obs_pos_noise": 0.005, "obs_jnt_noise": 0.002, "cmd_delay": 8}},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(d: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(d, f)


def parse_last_block(log_text: str) -> dict:
    """Pull scalar metrics from the last SB3 progress table in the log."""
    metrics: dict[str, float] = {}
    in_block = False
    for line in log_text.splitlines():
        if "---" in line and "---" in line:
            in_block = not in_block
            continue
        if in_block and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) == 2:
                key, val = parts
                key = key.strip().rstrip("/").replace("/", ".").replace(" ", "_")
                try:
                    metrics[key] = float(val)
                except ValueError:
                    pass
    return metrics


def run_probe(probe: dict, base_cfg: dict, out_root: Path,
              timesteps: int, n_envs: int) -> dict:
    name = probe["name"]
    out_dir = out_root / name
    log_path = out_root / f"{name}.log"

    # Build config
    cfg = {
        "env": {**base_cfg["env"], **probe.get("env", {})},
        "train": {**base_cfg["train"], "total_timesteps": timesteps,
                  "n_envs": n_envs},
    }
    cfg_path = out_root / f"{name}_config.yaml"
    save_yaml(cfg, cfg_path)

    print(f"\n{'='*60}")
    print(f"  PROBE: {name}")
    print(f"  {probe['desc']}")
    print(f"  timesteps={timesteps:,}  n_envs={n_envs}")
    print(f"  out → {out_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_path, "w") as logf:
        ret = subprocess.run(
            [sys.executable, "train.py", "--config", str(cfg_path),
             "--out", str(out_dir), "--timesteps", str(timesteps)],
            stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0

    result = {
        "name": name,
        "desc": probe["desc"],
        "elapsed_s": round(elapsed),
        "returncode": ret.returncode,
    }

    # Parse training metrics from log
    log_text = log_path.read_text()
    metrics = parse_last_block(log_text)
    result["train_metrics"] = metrics

    # Quick eval (ablation, moving_target only)
    model_zip = out_dir / "final_model.zip"
    if model_zip.exists():
        eval_out = subprocess.run(
            [sys.executable, "evaluate.py", "ablation",
             "--model", str(model_zip),
             "--trajectories", "moving_target"],
            capture_output=True, text=True,
        )
        # parse the table line for moving_target
        for line in eval_out.stdout.splitlines():
            if "moving_target" in line:
                parts = line.split()
                try:
                    result["ik_mm"] = float(parts[1])
                    result["res_mm"] = float(parts[2])
                    result["delta_pct"] = float(parts[3].rstrip("%"))
                except (IndexError, ValueError):
                    result["eval_raw"] = line.strip()
                break
        if "ik_mm" not in result:
            result["eval_stdout"] = eval_out.stdout[-500:]

    return result


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"  {'PROBE':<16}  {'IK mm':>7}  {'Res mm':>7}  {'Δ%':>7}  "
          f"{'ev':>6}  {'std':>6}  {'clip%':>6}  {'res_norm':>9}  DESC")
    print("=" * 90)
    for r in results:
        m = r.get("train_metrics", {})
        ev = m.get("explained_variance", float("nan"))
        std = m.get("std", float("nan"))
        clip = m.get("clip_fraction", float("nan"))
        rn = m.get("tracking.residual_norm", float("nan"))
        ik = r.get("ik_mm", float("nan"))
        res = r.get("res_mm", float("nan"))
        dp = r.get("delta_pct", float("nan"))
        print(f"  {r['name']:<16}  {ik:>7.1f}  {res:>7.1f}  {dp:>+7.1f}  "
              f"{ev:>6.3f}  {std:>6.3f}  {clip:>6.3f}  {rn:>9.4f}  {r['desc'][:40]}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ee_tracking/configs/default.yaml")
    parser.add_argument("--out", default="results/probes")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--only", nargs="+",
                        help="Run only named probes (space-separated)")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    probes = PROBES
    if args.only:
        probes = [p for p in PROBES if p["name"] in args.only]

    results = []
    for probe in probes:
        r = run_probe(probe, base_cfg, out_root, args.timesteps, args.n_envs)
        results.append(r)
        # save incrementally
        with open(out_root / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print_summary(results)

    print(f"\nFull summary → {out_root}/summary.json")


if __name__ == "__main__":
    main()
