#!/usr/bin/env python3
"""Evaluate post-hoc inference techniques on trained models.

NOTE — LEGACY SCRIPT (pre-delay architecture):
    This script was written when the dominant error was action noise and
    residual-only delay (act_delay=1).  Butterworth, deadzone, error-gain,
    and obs-smoothing filters were evaluated as post-hoc patches at inference.

    The current architecture uses whole-pipeline cmd_delay=5 (100 ms), making
    delay compensation via lookahead the main task.  Post-hoc filters applied
    to the policy output add latency on top of the delay and are likely harmful.

    Use this script only to reproduce old results or as a reference.
    For current models use evaluate.py directly.

No retraining required.  Each technique is a stateful filter applied at
inference time; techniques can be stacked.

Techniques implemented:
  ema          -- 1st-order IIR on policy output (already in eval_ema.py, baseline here)
  butter2      -- 2nd-order Butterworth at same cutoff; better roll-off, same phase lag
  deadzone     -- zero action components below threshold after filtering
  errorgain    -- scale residual magnitude by current tracking error / IK floor
  obs_ema      -- smooth normalised observations before policy inference

Usage:
    python eval_posthoc.py                           # best 3 models, all configs
    python eval_posthoc.py --models rs012_wr05       # single model
    python eval_posthoc.py --seeds 42 43 44          # average over seeds
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from evaluate import load_model, run_ik, _env_kwargs_from_cfg, _eval_config
from ee_tracking.env.franka_tracking_env import FrankaTrackingEnv
from ee_tracking.env.disturbances import DisturbanceConfig
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from evaluate import wrap_eval_env, _metrics

# ---------------------------------------------------------------------------
# Filter classes — all share the same interface
# ---------------------------------------------------------------------------

class EMAFilter:
    """1st-order exponential moving average."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._s: np.ndarray | None = None

    def reset(self):
        self._s = None

    def __call__(self, action: np.ndarray, prev_err_m: float) -> np.ndarray:
        if self._s is None:
            self._s = action.copy()
        else:
            self._s = (1.0 - self.alpha) * self._s + self.alpha * action
        return self._s.copy()


class ButterworthFilter:
    """2nd-order Butterworth low-pass — sharper roll-off than EMA at same cutoff."""
    def __init__(self, cutoff_hz: float, fs: float = 50.0, order: int = 2):
        from scipy.signal import butter, sosfilt_zi
        self.sos = butter(order, cutoff_hz / (fs / 2.0), btype="low", output="sos")
        # zi template: (n_sections, 2) → broadcast to (n_sections, 2, n_channels)
        self._zi_1d = sosfilt_zi(self.sos)       # (n_sections, 2)
        self._n_ch: int | None = None
        self.zi: np.ndarray | None = None

    def reset(self):
        self.zi = None

    def __call__(self, action: np.ndarray, prev_err_m: float) -> np.ndarray:
        from scipy.signal import sosfilt
        n = action.shape[-1]
        if self.zi is None or self._n_ch != n:
            self._n_ch = n
            # (n_sections, 2, n_channels) — all zeros (start from rest)
            self.zi = self._zi_1d[:, :, np.newaxis] * np.zeros((1, 1, n))
        x = action.reshape(1, n)                  # (1, n) — one time step
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y.reshape(n)


class DeadzoneFilter:
    """Zero out action components below threshold, then optionally run base filter."""
    def __init__(self, threshold: float, base: EMAFilter | ButterworthFilter | None = None):
        self.threshold = threshold
        self.base = base

    def reset(self):
        if self.base is not None:
            self.base.reset()

    def __call__(self, action: np.ndarray, prev_err_m: float) -> np.ndarray:
        a = np.where(np.abs(action) < self.threshold, 0.0, action)
        if self.base is not None:
            a = self.base(a, prev_err_m)
        return a


class ErrorGainFilter:
    """Scale residual magnitude by how far error is above the IK floor.

    gain = clip(prev_err / ik_floor, 0, 1)
    When tracking error ≤ ik_floor the policy is silent; when error is 2× the
    floor the policy is at full authority.  Prevents corrections when IK is
    already at its natural performance ceiling.
    """
    def __init__(self, ik_floor_m: float, base: EMAFilter | ButterworthFilter | None = None):
        self.ik_floor_m = ik_floor_m
        self.base = base
        self._prev_err: float = ik_floor_m * 2.0  # start with full authority

    def reset(self):
        self._prev_err = self.ik_floor_m * 2.0
        if self.base is not None:
            self.base.reset()

    def __call__(self, action: np.ndarray, prev_err_m: float) -> np.ndarray:
        a = action
        if self.base is not None:
            a = self.base(a, prev_err_m)
        gain = float(np.clip(prev_err_m / self.ik_floor_m, 0.0, 1.0))
        self._prev_err = prev_err_m
        return a * gain


# ---------------------------------------------------------------------------
# Evaluation runner with optional obs smoothing + action filter
# ---------------------------------------------------------------------------

def run_with_filter(
    model,
    vn_ref,
    trajectory: str,
    env_kwargs: dict,
    action_filter,          # callable: (action_7d, prev_err_m) → action_7d
    obs_ema: float = 1.0,   # smooth normalised obs before predict; 1.0 = off
    seed: int = 42,
) -> dict:
    cfg = _eval_config(
        trajectory, use_residual=True, seed=seed, **env_kwargs
    )
    env = FrankaTrackingEnv(cfg)
    venv = wrap_eval_env(env, vn_ref)

    action_filter.reset()
    obs = venv.reset()
    smoothed_obs = obs.copy()
    prev_err_m = 0.020          # 20 mm — reasonable starting assumption
    ee_pos, tgt_pos, err_mm, res_norms = [], [], [], []

    while True:
        # ── optional obs smoothing ─────────────────────────────────────────
        if obs_ema < 1.0:
            smoothed_obs = (1.0 - obs_ema) * smoothed_obs + obs_ema * obs
            policy_obs = smoothed_obs
        else:
            policy_obs = obs

        raw_action, _ = model.predict(policy_obs, deterministic=True)
        action = action_filter(raw_action.reshape(-1), prev_err_m).reshape(raw_action.shape)

        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        prev_err_m = float(info["pos_err"])
        ee_pos.append(info["ee_pos"].copy())
        tgt_pos.append(info["target_pos"].copy())
        err_mm.append(prev_err_m * 1000.0)
        res_norms.append(float(info.get("residual_norm", 0.0)))
        if dones[0]:
            break

    venv.close()
    return _metrics(
        np.array(ee_pos), np.array(tgt_pos),
        np.array(err_mm), np.array(res_norms),
    )


# ---------------------------------------------------------------------------
# Filter configurations to sweep
# ---------------------------------------------------------------------------

# cutoff matching EMA α=0.2 at 50 Hz:
#   EMA -3dB ≈ -fs * ln(1-α) / (2π) ≈ 1.78 Hz → round to 2.0 Hz
BUTTER_HZ = 2.0

CONFIGS: list[tuple[str, dict]] = [
    # ── baselines ──────────────────────────────────────────────────────────
    ("raw",
     dict(action_filter=lambda: EMAFilter(1.0), obs_ema=1.0)),

    ("ema_02",
     dict(action_filter=lambda: EMAFilter(0.2), obs_ema=1.0)),

    # ── 2nd-order Butterworth ───────────────────────────────────────────────
    ("butter2",
     dict(action_filter=lambda: ButterworthFilter(BUTTER_HZ), obs_ema=1.0)),

    # ── deadzone (applied before EMA so jitter is zero'd first) ────────────
    ("dz005+ema",
     dict(action_filter=lambda: DeadzoneFilter(0.05, EMAFilter(0.2)), obs_ema=1.0)),

    ("dz010+ema",
     dict(action_filter=lambda: DeadzoneFilter(0.10, EMAFilter(0.2)), obs_ema=1.0)),

    ("dz005+butter2",
     dict(action_filter=lambda: DeadzoneFilter(0.05, ButterworthFilter(BUTTER_HZ)), obs_ema=1.0)),

    # ── error-conditioned gain (floor = IK natural performance ~15.6 mm) ───
    ("errorgain+ema",
     dict(action_filter=lambda: ErrorGainFilter(0.0156, EMAFilter(0.2)), obs_ema=1.0)),

    ("errorgain+butter2",
     dict(action_filter=lambda: ErrorGainFilter(0.0156, ButterworthFilter(BUTTER_HZ)), obs_ema=1.0)),

    ("errorgain_only",
     dict(action_filter=lambda: ErrorGainFilter(0.0156, EMAFilter(1.0)), obs_ema=1.0)),

    # ── obs smoothing ───────────────────────────────────────────────────────
    ("obs07",
     dict(action_filter=lambda: EMAFilter(1.0), obs_ema=0.7)),

    ("obs07+ema",
     dict(action_filter=lambda: EMAFilter(0.2), obs_ema=0.7)),

    ("obs07+butter2",
     dict(action_filter=lambda: ButterworthFilter(BUTTER_HZ), obs_ema=0.7)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], ik_mm: float):
    rows = sorted(rows, key=lambda r: r["res_mm"])
    col = max(len(r["name"]) for r in rows)
    print()
    print("=" * 68)
    print(f"  IK baseline: {ik_mm:.2f} mm")
    print(f"  {'config':<{col}}  {'res mm':>8}  {'Δ vs IK':>9}  {'Δ vs raw':>9}")
    print("=" * 68)
    raw_mm = next(r["res_mm"] for r in rows if r["name"] == "raw")
    for r in rows:
        vs_ik  = (ik_mm - r["res_mm"]) / ik_mm * 100
        vs_raw = (raw_mm - r["res_mm"]) / raw_mm * 100
        beat = " ✓" if vs_ik > 0 else ""
        print(f"  {r['name']:<{col}}  {r['res_mm']:>8.2f}  {vs_ik:>+8.1f}%  {vs_raw:>+8.1f}%{beat}")
    print("=" * 68)
    print()


def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_MODELS = ["rs012_wr05", "rs008_wr01", "rs012_wr03"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="results/sweep")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--out", default="results/posthoc/summary.csv")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_dir)
    model_names = args.models or DEFAULT_MODELS
    model_paths = [sweep_root / n / "final_model.zip" for n in model_names]

    missing = [p for p in model_paths if not p.exists()]
    if missing:
        print(f"Missing: {[str(p) for p in missing]}")
        model_paths = [p for p in model_paths if p.exists()]

    all_rows: list[dict] = []

    for model_path in model_paths:
        name = model_path.parent.name
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        model, vn_ref, saved_cfg = load_model(str(model_path))
        env_kwargs = _env_kwargs_from_cfg(saved_cfg)

        # IK baseline (averaged over seeds)
        ik_errs = [
            run_ik("moving_target", seed=s, disturbance=env_kwargs["disturbance"])["settled_rmse_mm"]
            for s in args.seeds
        ]
        ik_mm = float(np.mean(ik_errs))
        print(f"  IK baseline: {ik_mm:.2f} mm  (avg over {len(args.seeds)} seed(s))\n")

        model_rows: list[dict] = []
        for cfg_name, cfg in CONFIGS:
            errs = []
            for seed in args.seeds:
                filt = cfg["action_filter"]()   # fresh instance each seed
                r = run_with_filter(
                    model, vn_ref, "moving_target",
                    env_kwargs=env_kwargs,
                    action_filter=filt,
                    obs_ema=cfg["obs_ema"],
                    seed=seed,
                )
                errs.append(r["settled_rmse_mm"])
            res_mm = float(np.mean(errs))
            vs_ik = (ik_mm - res_mm) / ik_mm * 100
            model_rows.append({"model": name, "name": cfg_name,
                                "ik_mm": ik_mm, "res_mm": res_mm, "delta_ik": vs_ik})
            print(f"  {cfg_name:<20}  {res_mm:.2f} mm  ({vs_ik:+.1f}% vs IK)")

        print_table(model_rows, ik_mm)
        all_rows.extend(model_rows)

    if all_rows:
        save_csv(all_rows, Path(args.out))


if __name__ == "__main__":
    main()
