# Franka EE Tracking — Clean Slate

7-DoF Franka Panda end-effector tracking in MuJoCo.
The environment ships with a damped-least-squares IK baseline; everything else is to be built.

## What's here

```
ee_tracking/
  env/
    franka_tracking_env.py   # Gym env — obs / action / reward
    ik_controller.py         # DLS inverse-kinematics baseline
    trajectories.py          # circle, figure-8, moving_target, unreachable
    disturbances.py          # obs noise, joint noise, action delay
  configs/
    default.yaml             # env + placeholder train hyperparameters
assets/
  mujoco_menagerie/franka_emika_panda/   # Franka XML + meshes
```

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install mujoco gymnasium "stable-baselines3>=2.3" torch numpy \
               matplotlib pyyaml imageio imageio-ffmpeg tqdm tensorboard
```

Franka assets (if not already present):
```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
```

## Baseline: IK-only performance

| Trajectory    | RMSE (settled) |
|---------------|---------------|
| circle        | ~6 mm         |
| figure-8      | ~6 mm         |
| moving_target | ~47 mm        |
| unreachable   | ~9 mm         |

`moving_target` is the primary benchmark — IK alone struggles because
the target velocity is stochastic and the IK has no lookahead.

## Goal

Beat IK on every trajectory, with `moving_target` as the primary target.
