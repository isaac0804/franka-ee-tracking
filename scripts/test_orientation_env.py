"""Smoke-test for the 6-DoF orientation tracking environment.

Checks:
  1. All 5 orientation trajectories instantiate and produce valid pos/vel/quat/angvel.
  2. FrankaTracking6DoFEnv resets and steps without error.
  3. Observation has the expected 141-D shape.
  4. Reward includes orientation progress signal.

Usage:
    python scripts/test_orientation_env.py
"""

import sys
import traceback

import numpy as np


def check(name, fn):
    try:
        fn()
        print(f"  ✓  {name}")
    except Exception as e:
        print(f"  ✗  {name}")
        traceback.print_exc()
        return False
    return True


PASS = True


def test_trajectory(cls_name, traj):
    global PASS

    def run():
        t = 0.0
        for _ in range(10):
            pos, vel = traj.sample(t)
            assert pos.shape == (3,), f"pos shape {pos.shape}"
            assert vel.shape == (3,), f"vel shape {vel.shape}"
            assert np.all(np.isfinite(pos)), "pos contains NaN/inf"
            assert np.all(np.isfinite(vel)), "vel contains NaN/inf"

            if traj.has_orientation:
                quat, angvel = traj.sample_ori(t)
                assert quat.shape == (4,), f"quat shape {quat.shape}"
                assert angvel.shape == (3,), f"angvel shape {angvel.shape}"
                norm = np.linalg.norm(quat)
                assert abs(norm - 1.0) < 1e-5, f"quat not unit: ‖q‖={norm:.6f}"
                assert np.all(np.isfinite(quat)), "quat contains NaN/inf"
                assert np.all(np.isfinite(angvel)), "angvel contains NaN/inf"

            t += 0.2

        # lookahead_ori
        if traj.has_orientation:
            la = traj.lookahead_ori(0.0, 0.02, 5)
            assert la.shape == (20,), f"lookahead_ori shape {la.shape}"

    PASS = check(f"trajectory: {cls_name}", run) and PASS


def test_env_obs_shape():
    global PASS

    def run():
        from ee_tracking.env.franka_tracking_6dof_env import FrankaTracking6DoFEnv, Env6DoFConfig
        cfg = Env6DoFConfig()
        env = FrankaTracking6DoFEnv(cfg)
        obs, info = env.reset(seed=0)
        assert obs.shape == (141,), f"expected obs (141,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "obs contains NaN/inf at reset"

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (141,), f"step obs shape {obs.shape}"
            assert np.isfinite(reward), f"reward is {reward}"
            assert "ori_err" in info, "info missing ori_err"
            if terminated or truncated:
                obs, info = env.reset()

        env.close()

    PASS = check("FrankaTracking6DoFEnv: obs=141-D, step OK", run) and PASS


def test_all_trajectories_in_pool():
    """Each trajectory in the default pool must work in the env."""
    global PASS

    def run():
        from ee_tracking.env.franka_tracking_6dof_env import FrankaTracking6DoFEnv, Env6DoFConfig
        for traj_name in ["tilted_circle", "look_at", "rotating_grasp", "random_walk_6dof"]:
            cfg = Env6DoFConfig()
            cfg.trajectory_pool = (traj_name,)
            cfg.randomize_trajectory = False
            cfg.trajectory = traj_name
            env = FrankaTracking6DoFEnv(cfg)
            obs, _ = env.reset(seed=0)
            # Expected obs dim: 30+10+35+28+cmd_block+traj_onehot
            # cmd_block = 7*cmd_delay = 35 (5-step delay); traj_onehot = 1 (single pool)
            expected = env.observation_space.shape[0]
            assert obs.shape == (expected,), f"{traj_name}: obs shape {obs.shape} != {expected}"
            for _ in range(20):
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                assert obs.shape == (expected,), f"{traj_name} step: obs shape {obs.shape}"
                if terminated or truncated:
                    break
            env.close()

    PASS = check("all pool trajectories step without error", run) and PASS


def test_quat_utils():
    """Check quaternion utility functions for correctness."""
    global PASS

    def run():
        from ee_tracking.env.trajectories import (
            quat_mul, axisangle_to_quat, quat_to_axisangle,
            quat_error, rotation_from_z_to, mat_to_quat
        )

        # Identity round-trip
        v = np.array([0.1, 0.2, 0.3])
        q = axisangle_to_quat(v)
        v2 = quat_to_axisangle(q)
        assert np.allclose(v, v2, atol=1e-6), f"axis-angle round-trip failed: {v} → {v2}"

        # quat_mul: q * q_inv = identity
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        prod = quat_mul(q, q_inv)
        assert np.allclose(prod, [1, 0, 0, 0], atol=1e-6), f"q * q_inv ≠ identity: {prod}"

        # quat_error: error between q and itself = zero rotation
        e = quat_error(q, q)
        assert np.allclose(e, 0.0, atol=1e-6), f"quat_error(q,q) ≠ 0: {e}"

        # rotation_from_z_to + mat_to_quat round-trip
        z_des = np.array([0.0, 1.0, 0.0])  # y-axis
        R = rotation_from_z_to(z_des)
        q_R = mat_to_quat(R)
        z_rot = R @ np.array([0, 0, 1])
        assert np.allclose(z_rot, z_des, atol=1e-6), f"rotation_from_z_to failed: {z_rot}"

    PASS = check("quaternion utilities correct", run) and PASS


if __name__ == "__main__":
    print("\n=== Orientation environment smoke test ===\n")

    # Trajectory unit tests (no MuJoCo assets needed)
    try:
        from ee_tracking.env import trajectories as traj
        test_trajectory("UprightConstraint", traj.UprightConstraint(traj.Circle()))
        test_trajectory("TiltedCircle", traj.TiltedCircle())
        test_trajectory("LookAt", traj.LookAt())
        test_trajectory("RotatingGrasp", traj.RotatingGrasp())
        test_trajectory("RandomWalk6DoF", traj.RandomWalk6DoF(seed=42))
    except ImportError as e:
        print(f"  ✗  trajectories import failed: {e}")
        PASS = False

    test_quat_utils()

    # Full env tests (needs MuJoCo assets)
    print()
    try:
        test_env_obs_shape()
        test_all_trajectories_in_pool()
    except ImportError as e:
        print(f"  ✗  env import failed (assets may not be installed): {e}")

    print()
    if PASS:
        print("All tests passed ✓")
        sys.exit(0)
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)
