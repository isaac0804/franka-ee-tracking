#!/usr/bin/env python3
"""Overnight hyperparameter sweep for residual PPO on moving_target.

Runs all configs sequentially, evaluates each one, and keeps a live
ranked summary in results/sweep/summary.csv.

Usage:
    python sweep.py                        # full sweep, 2M steps/run (~7–8 h)
    python sweep.py --timesteps 500000     # quick smoke-test (~2 h)
    python sweep.py --dry-run              # print configs and estimated time, exit

Resumable: any run whose final_model.zip already exists is skipped.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Base config (mirrors default.yaml — the sweep overrides values on top)
# ---------------------------------------------------------------------------

BASE = {
    "env": {
        "control_hz": 50.0,
        "episode_seconds": 6.0,
        "randomize_trajectory": True,
        "trajectory_pool": ["moving_target"],
        "residual_scale": 0.05,
        "use_residual": True,
        "w_pos": 5.0,
        "w_vel": 0.1,
        "w_residual": 0.5,
        "w_jerk": 0.001,
        "w_smooth": 0.05,
        "w_delta_pos": 0.0,
        "w_bonus": 0.0,
        "bonus_sharpness": 0.0,
        "fail_pos_err": 0.30,
        "lookahead_horizon": 5,
        "lookahead_dt": 0.10,
        "action_ema": 0.2,
        "disturbance": {
            "obs_pos_noise": 0.005,
            "obs_jnt_noise": 0.002,
            "act_delay": 1,
        },
    },
    "train": {
        "total_timesteps": 2_000_000,
        "n_envs": 10,
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": [256, 256]},
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.97,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 0,
    },
}

# ---------------------------------------------------------------------------
# Sweep configs
# Each entry: (name, env_overrides, train_overrides)
#
# Groups:
#   G1  residual_scale × w_residual grid  — the core tradeoff
#   G2  learning rate
#   G3  network architecture
#   G4  lookahead horizon/dt
#   G5  action delay
#   G6  reward weights (w_pos, w_smooth)
# ---------------------------------------------------------------------------

CONFIGS: list[tuple[str, dict, dict]] = [

    # ── G1: residual_scale × w_residual ──────────────────────────────────
    # baseline lives at rs005 / wr05
    ("baseline",        {"residual_scale": 0.05, "w_residual": 0.5},  {}),

    ("rs002_wr01",      {"residual_scale": 0.02, "w_residual": 0.1},  {}),
    ("rs002_wr03",      {"residual_scale": 0.02, "w_residual": 0.3},  {}),
    ("rs002_wr05",      {"residual_scale": 0.02, "w_residual": 0.5},  {}),
    ("rs002_wr10",      {"residual_scale": 0.02, "w_residual": 1.0},  {}),

    ("rs005_wr01",      {"residual_scale": 0.05, "w_residual": 0.1},  {}),
    ("rs005_wr03",      {"residual_scale": 0.05, "w_residual": 0.3},  {}),
    ("rs005_wr10",      {"residual_scale": 0.05, "w_residual": 1.0},  {}),

    ("rs008_wr01",      {"residual_scale": 0.08, "w_residual": 0.1},  {}),
    ("rs008_wr03",      {"residual_scale": 0.08, "w_residual": 0.3},  {}),
    ("rs008_wr05",      {"residual_scale": 0.08, "w_residual": 0.5},  {}),
    ("rs008_wr10",      {"residual_scale": 0.08, "w_residual": 1.0},  {}),

    ("rs012_wr01",      {"residual_scale": 0.12, "w_residual": 0.1},  {}),
    ("rs012_wr03",      {"residual_scale": 0.12, "w_residual": 0.3},  {}),
    ("rs012_wr05",      {"residual_scale": 0.12, "w_residual": 0.5},  {}),
    ("rs012_wr10",      {"residual_scale": 0.12, "w_residual": 1.0},  {}),

    # ── G2: learning rate ────────────────────────────────────────────────
    ("lr_1e4",          {},  {"learning_rate": 1e-4}),
    ("lr_1e3",          {},  {"learning_rate": 1e-3}),

    # ── G3: network architecture ─────────────────────────────────────────
    ("net_128x128",     {},  {"policy_kwargs": {"net_arch": [128, 128]}}),
    ("net_256x256x256", {},  {"policy_kwargs": {"net_arch": [256, 256, 256]}}),

    # ── G4: lookahead ────────────────────────────────────────────────────
    # ablation: does lookahead actually help?
    ("look_none",       {"lookahead_horizon": 1, "lookahead_dt": 0.50}, {}),  # only current step
    ("look_near",       {"lookahead_horizon": 3, "lookahead_dt": 0.05}, {}),  # 0.15 s ahead
    ("look_far",        {"lookahead_horizon": 8, "lookahead_dt": 0.10}, {}),  # 0.8 s ahead
    ("look_wide",       {"lookahead_horizon": 5, "lookahead_dt": 0.20}, {}),  # 1.0 s ahead

    # ── G5: action delay ─────────────────────────────────────────────────
    # delay=0: IK is perfect reactive — lookahead has less value
    # delay=2: harder for IK, more room for residual to compensate
    ("delay_0",         {"disturbance": {"obs_pos_noise": 0.005, "obs_jnt_noise": 0.002, "act_delay": 0}}, {}),
    ("delay_2",         {"disturbance": {"obs_pos_noise": 0.005, "obs_jnt_noise": 0.002, "act_delay": 2}}, {}),

    # ── G6: reward weights ───────────────────────────────────────────────
    ("wpos_3",          {"w_pos": 3.0},   {}),
    ("wpos_8",          {"w_pos": 8.0},   {}),
    ("wsmooth_0",       {"w_smooth": 0.0}, {}),
    ("wsmooth_20",      {"w_smooth": 0.2}, {}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def build_cfg(env_override: dict, train_override: dict, timesteps: int) -> dict:
    cfg = deep_merge(BASE, {"env": env_override, "train": train_override})
    cfg["train"]["total_timesteps"] = timesteps
    return cfg


def run_training(cfg: dict, out_dir: Path) -> tuple[bool, float, str]:
    """Write config yaml, call train.py, return (success, elapsed_s, stdout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "sweep_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "train.py", "--config", str(cfg_path), "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    stdout = result.stdout + result.stderr
    return result.returncode == 0, elapsed, stdout


def run_eval(model_path: Path, eval_dir: Path, trajectory_pool: list[str]) -> dict | None:
    """Call evaluate.py ablation and return the JSON results."""
    result = subprocess.run(
        [sys.executable, "evaluate.py", "ablation",
         "--model", str(model_path),
         "--trajectories", "moving_target",
         "--out", str(eval_dir)],
        capture_output=True, text=True,
    )
    json_path = eval_dir / "ablation.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def parse_final_pos_err(stdout: str) -> float | None:
    """Extract last pos_err_mm value from training stdout."""
    val = None
    for line in stdout.splitlines():
        if "pos_err_mm" in line:
            try:
                val = float(line.split("|")[-2].strip())
            except Exception:
                pass
    return val


def write_summary(rows: list[dict], path: Path):
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_leaderboard(rows: list[dict]):
    finished = [r for r in rows if r["improvement_pct"] != "FAILED"]
    if not finished:
        return
    finished.sort(key=lambda r: -float(r["improvement_pct"]))
    print("\n" + "=" * 72)
    print(f"  {'RANK':<5} {'name':<20} {'IK mm':>7} {'res mm':>8} {'Δ %':>8}  notes")
    print("=" * 72)
    for i, r in enumerate(finished, 1):
        marker = " ✓" if float(r["improvement_pct"]) > 0 else ""
        print(f"  {i:<5} {r['name']:<20} {float(r['ik_mm']):>7.1f} "
              f"{float(r['res_mm']):>8.1f} {float(r['improvement_pct']):>+7.1f}%{marker}")
    print("=" * 72)
    failed = [r["name"] for r in rows if r["improvement_pct"] == "FAILED"]
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overnight hyperparameter sweep")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Steps per run (default 2M ≈ 13 min each)")
    parser.add_argument("--out", default="results/sweep",
                        help="Root output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs and time estimate only")
    args = parser.parse_args()

    out_root = Path(args.out)
    summary_path = out_root / "summary.csv"
    n = len(CONFIGS)
    secs_per_run = args.timesteps / 2500  # empirical ~2500 steps/sec on this machine
    total_h = n * secs_per_run / 3600

    print(f"\n{'='*60}")
    print(f"  Sweep: {n} configs × {args.timesteps/1e6:.1f}M steps")
    print(f"  Estimated time: {total_h:.1f} h  ({secs_per_run/60:.0f} min/run)")
    print(f"  Output: {out_root}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for i, (name, env_ov, train_ov) in enumerate(CONFIGS, 1):
            overrides = {**env_ov, **train_ov}
            desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
            print(f"  {i:2d}. {name:<22}  {desc}")
        return

    rows: list[dict] = []

    # Load existing summary for resume
    if summary_path.exists():
        import csv
        with open(summary_path) as f:
            rows = list(csv.DictReader(f))
        done_names = {r["name"] for r in rows}
        print(f"Resuming: {len(done_names)} runs already done\n")
    else:
        done_names = set()

    t_sweep_start = time.time()

    for idx, (name, env_ov, train_ov) in enumerate(CONFIGS, 1):
        run_dir = out_root / name
        model_path = run_dir / "final_model.zip"

        if name in done_names and model_path.exists():
            print(f"[{idx}/{n}] {name}  — skipped (already done)")
            continue

        cfg = build_cfg(env_ov, train_ov, args.timesteps)
        overrides_desc = ", ".join(
            f"{k}={v}" for d in (env_ov, train_ov) for k, v in d.items()
        ) or "baseline"

        elapsed_so_far = time.time() - t_sweep_start
        remaining_runs = n - idx
        eta_h = remaining_runs * secs_per_run / 3600

        print(f"\n[{idx}/{n}] {name}  —  {overrides_desc}")
        print(f"  Elapsed: {elapsed_so_far/3600:.1f}h  |  ETA: {eta_h:.1f}h remaining")

        success, elapsed, stdout = run_training(cfg, run_dir)
        final_pos_err = parse_final_pos_err(stdout)

        row: dict = {
            "name": name,
            "group": name.split("_")[0] if "_" in name else name,
            "overrides": overrides_desc,
            "train_ok": success,
            "train_time_s": f"{elapsed:.0f}",
            "final_pos_err_mm": f"{final_pos_err:.1f}" if final_pos_err else "?",
            "ik_mm": "FAILED",
            "res_mm": "FAILED",
            "improvement_pct": "FAILED",
        }

        if not success:
            print(f"  ✗ Training failed after {elapsed:.0f}s")
            print(stdout[-800:])
        else:
            print(f"  ✓ Training done in {elapsed:.0f}s  (final pos_err={final_pos_err:.1f}mm)"
                  if final_pos_err else f"  ✓ Training done in {elapsed:.0f}s")

            eval_dir = run_dir / "eval"
            pool = cfg["env"].get("trajectory_pool", ["moving_target"])
            eval_results = run_eval(model_path, eval_dir, pool)

            if eval_results and "moving_target" in eval_results:
                mt = eval_results["moving_target"]
                row["ik_mm"] = f"{mt['ik_settled_rmse_mm']:.2f}"
                row["res_mm"] = f"{mt['residual_settled_rmse_mm']:.2f}"
                row["improvement_pct"] = f"{mt['improvement_pct']:.2f}"
                marker = " ✓" if mt["improvement_pct"] > 0 else ""
                print(f"  moving_target:  IK={mt['ik_settled_rmse_mm']:.1f}mm  "
                      f"residual={mt['residual_settled_rmse_mm']:.1f}mm  "
                      f"Δ={mt['improvement_pct']:+.1f}%{marker}")
            else:
                print("  ✗ Eval failed or missing moving_target results")

        rows.append(row)
        write_summary(rows, summary_path)

    # Final leaderboard
    total_elapsed = time.time() - t_sweep_start
    print(f"\nAll done in {total_elapsed/3600:.1f}h")
    print_leaderboard(rows)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
