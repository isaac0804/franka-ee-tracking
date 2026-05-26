#!/usr/bin/env python3
"""Evaluate a trained model (or IK-only baseline) on the 5 orientation tasks.

Reports:
  - Settled position RMSE (mm)      — after 0.5 s warmup
  - Settled orientation RMSE (deg)  — axis-angle magnitude
  - Action roughness                 — mean |a_t − a_{t-1}| per joint
  - Saturation rate                  — fraction of |action| > 0.9

Usage examples
--------------
  # IK baseline only (no model)
  python evaluate_orientation.py --ik-only

  # Trained model vs IK on all tasks
  python evaluate_orientation.py --model results/orient_300k/final_model.zip

  # Single task
  python evaluate_orientation.py --model results/orient_300k/final_model.zip \
      --tasks tilted_circle rotating_grasp

  # More seeds for stochastic tasks
  python evaluate_orientation.py --model results/orient_300k/final_model.zip \
      --n-seeds 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ee_tracking.env.franka_tracking_6dof_env import FrankaTracking6DoFEnv, Env6DoFConfig
from ee_tracking.env.disturbances import DisturbanceConfig
from ee_tracking.env.trajectories import quat_error, _IDENTITY_QUAT
from ee_tracking.env import trajectories as traj_module


# ── task definitions ──────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "tilted_circle": dict(
        stochastic=False,
        traj_class="TiltedCircle",
        traj_kwargs=dict(center=(0.5, 0.0, 0.5), radius=0.12, period=6.0),
        label="Tilted Circle",
    ),
    "look_at": dict(
        stochastic=True,
        traj_class="LookAt",
        traj_kwargs=None,          # built per-seed in make_env
        label="Look-At",
    ),
    "rotating_grasp": dict(
        stochastic=False,
        traj_class="RotatingGrasp",
        traj_kwargs=dict(position=(0.5, 0.05, 0.52), omega=0.5, duration=12.0),
        label="Rotating Grasp",
    ),
    "random_walk_6dof": dict(
        stochastic=True,
        traj_class="RandomWalk6DoF",
        traj_kwargs=None,          # seed-specific
        label="6-DoF Random Walk",
    ),
    "upright_constraint": dict(
        stochastic=False,
        traj_class="UprightConstraint",
        traj_kwargs=None,          # wraps a circle
        label="Upright Constraint",
    ),
}

SETTLE_STEPS = 25   # 0.5 s at 50 Hz


# ── helpers ───────────────────────────────────────────────────────────────────

# Training pool — must match the pool used during training so the onehot size is
# consistent with the model's expected input.
# Use the full 5-task pool for models trained with tfm_6dof_5task_300k.yaml (143-D obs).
# Pass pool explicitly when evaluating older 3-task models (141-D obs).
_TRAIN_POOL_5 = ("tilted_circle", "look_at", "random_walk_6dof",
                 "rotating_grasp", "upright_constraint")
_TRAIN_POOL_3 = ("tilted_circle", "look_at", "random_walk_6dof")


def make_fixed_env(task: str, pool: tuple = _TRAIN_POOL_5) -> FrankaTracking6DoFEnv:
    """Build an env locked to a single task trajectory (no randomisation).

    The trajectory_pool is kept at its training size (3) so the one-hot obs
    dimension matches the model's expected input even for OOD eval tasks.
    If `task` is not already in `pool`, it is prepended and the last pool
    entry is dropped to maintain a fixed pool size.
    """
    if task not in pool:
        pool = (task,) + pool[: len(pool) - 1]
    cfg = Env6DoFConfig()
    cfg.randomize_trajectory = False
    cfg.trajectory = task
    cfg.trajectory_pool = pool
    cfg.disturbance = DisturbanceConfig(cmd_delay=5, obs_pos_noise=0.005, obs_jnt_noise=0.002)
    cfg.episode_seconds = 10.0
    return FrankaTracking6DoFEnv(cfg)


def run_episode(env, policy_fn, seed: int = 0) -> dict:
    """Run one episode; return per-step metrics."""
    obs, info = env.reset(seed=seed)
    traj_name = info.get("trajectory", "unknown")

    pos_errs, ori_errs, actions = [], [], []
    prev_action = np.zeros(7)
    done = False
    step = 0

    while not done:
        action = policy_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if step >= SETTLE_STEPS:
            pos_errs.append(float(info["pos_err"]) * 1000.0)     # mm
            ori_errs.append(float(info.get("ori_err_deg", 0.0)))  # deg
            actions.append(action.copy())

        prev_action = action.copy()
        step += 1

    actions = np.array(actions) if actions else np.zeros((1, 7))
    roughness = float(np.mean(np.abs(np.diff(actions, axis=0)))) if len(actions) > 1 else 0.0
    sat_rate  = float(np.mean(np.abs(actions) > 0.9))

    return dict(
        traj=traj_name,
        pos_rmse=float(np.sqrt(np.mean(np.array(pos_errs) ** 2))) if pos_errs else 0.0,
        ori_rmse=float(np.sqrt(np.mean(np.array(ori_errs) ** 2))) if ori_errs else 0.0,
        roughness=roughness,
        sat_rate=sat_rate,
        n_steps=step,
    )


def ik_policy(obs):
    """Zero residual — pure IK baseline."""
    return np.zeros(7, dtype=np.float32)


def load_model_policy(model_path: str):
    """Load an SB3 model and return a deterministic policy function."""
    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from ee_tracking.policies.transformer_policy import TRANSFORMER_POLICY_REGISTRY
    from ee_tracking.policies.gelu_policy import POLICY_REGISTRY
    registry = {**POLICY_REGISTRY, **TRANSFORMER_POLICY_REGISTRY}

    model_path = pathlib.Path(model_path)
    model_dir  = model_path.parent

    # Try to find a matching vecnorm and config
    vecnorm_path = model_dir / "vecnormalize.pkl"
    config_path  = model_dir / "config.yaml"

    # Build a representative env matching the training obs shape.
    # Read the saved config.yaml so the trajectory_pool (and therefore
    # the onehot size) matches exactly what the model was trained on.
    cfg = Env6DoFConfig()
    if config_path.exists():
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        env_d = saved.get("env", {})
        pool = env_d.get("trajectory_pool", None)
        if pool:
            cfg.trajectory_pool = tuple(pool)
    dummy_vec = DummyVecEnv([lambda: FrankaTracking6DoFEnv(cfg)])

    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), dummy_vec)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = dummy_vec

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    # Expose the training pool so the eval env can match obs shape exactly.
    model_pool = tuple(cfg.trajectory_pool)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_t = vec_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_t, deterministic=True)
        return action.flatten()

    policy_fn.pool = model_pool  # type: ignore[attr-defined]
    return policy_fn


# ── main evaluation loop ──────────────────────────────────────────────────────

def evaluate_task(task: str, policy_fn, n_seeds: int = 3,
                  stochastic: bool = False) -> dict:
    """Run n_seeds episodes on `task`; return aggregated metrics."""
    pool = getattr(policy_fn, "pool", _TRAIN_POOL_5)
    env = make_fixed_env(task, pool=pool)
    seeds = range(n_seeds) if stochastic else [42]

    results = [run_episode(env, policy_fn, seed=s) for s in seeds]
    env.close()

    pos  = [r["pos_rmse"]   for r in results]
    ori  = [r["ori_rmse"]   for r in results]
    rough = np.mean([r["roughness"] for r in results])
    sat   = np.mean([r["sat_rate"]  for r in results])

    return dict(
        pos_mean=float(np.mean(pos)),
        pos_std=float(np.std(pos)),
        ori_mean=float(np.mean(ori)),
        ori_std=float(np.std(ori)),
        roughness=float(rough),
        sat_rate=float(sat),
        n_seeds=len(seeds),
    )


def print_table(results: dict):
    """Pretty-print evaluation results."""
    tasks = list(results.keys())
    methods = list(results[tasks[0]].keys())

    # ── header ──
    col = 22
    print(f"\n{'Task':<22} {'Method':<12} {'Pos RMSE':>12} {'Ori RMSE':>12} {'Roughness':>10} {'Sat%':>7}")
    print("─" * 80)

    for task in tasks:
        label = TASK_CONFIGS[task]["label"]
        for mi, method in enumerate(methods):
            r = results[task][method]
            pos_str = f"{r['pos_mean']:6.1f}±{r['pos_std']:4.1f}" if r["pos_std"] > 0.05 else f"{r['pos_mean']:6.1f}    "
            ori_str = f"{r['ori_mean']:6.1f}±{r['ori_std']:4.1f}" if r["ori_std"] > 0.05 else f"{r['ori_mean']:6.1f}    "
            print(f"  {label if mi==0 else '':<20} {method:<12} {pos_str:>12} {ori_str:>12} "
                  f"{r['roughness']:>10.3f} {r['sat_rate']*100:>6.1f}%")
        if task != tasks[-1]:
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to trained model .zip")
    parser.add_argument("--ik-only",  action="store_true",
                        help="Evaluate IK baseline only (no model needed)")
    parser.add_argument("--tasks",    nargs="*",
                        default=list(TASK_CONFIGS.keys()),
                        help="Which tasks to evaluate")
    parser.add_argument("--n-seeds",  type=int, default=3,
                        help="Seeds for stochastic tasks (look_at, random_walk_6dof)")
    args = parser.parse_args()

    if args.model is None and not args.ik_only:
        parser.error("Provide --model or --ik-only")

    policies = {"IK": ik_policy}
    if args.model and not args.ik_only:
        print(f"Loading model: {args.model}")
        policies["Policy"] = load_model_policy(args.model)

    print(f"\nEvaluating tasks: {args.tasks}")
    print(f"Seeds for stochastic tasks: {args.n_seeds}\n")

    all_results = {}
    for task in args.tasks:
        if task not in TASK_CONFIGS:
            print(f"Unknown task: {task}  (choose from {list(TASK_CONFIGS)})")
            continue
        stochastic = TASK_CONFIGS[task]["stochastic"]
        print(f"  {TASK_CONFIGS[task]['label']}...", end="", flush=True)
        all_results[task] = {}
        for name, policy_fn in policies.items():
            r = evaluate_task(task, policy_fn,
                              n_seeds=args.n_seeds, stochastic=stochastic)
            all_results[task][name] = r
            print(f"  {name}: {r['pos_mean']:.1f}mm/{r['ori_mean']:.1f}°", end="", flush=True)
        print()

    print_table(all_results)


if __name__ == "__main__":
    main()
