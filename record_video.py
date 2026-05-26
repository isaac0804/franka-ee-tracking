#!/usr/bin/env python3
"""Record side-by-side IK vs Residual PPO video for a given trajectory.

3D EE and target trails are drawn directly inside the MuJoCo scene using
mjv_initGeom — no matplotlib overlays needed.

Usage:
    python record_video.py --model results/sweep/rs012_10M/final_model.zip
    python record_video.py --model results/sweep/rs012_10M/final_model.zip \
                           --trajectory circle --out results/videos
    python record_video.py --model results/sweep/rs012_10M/final_model.zip \
                           --all --out results/videos

Outputs per trajectory:
    <out>/ik_vs_residual_<traj>.mp4    (H.264, ~2 MB per 30 s)
    <out>/ik_vs_residual_<traj>.gif    (palette-optimised, README-ready)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from evaluate import load_model, _eval_config, wrap_eval_env
from ee_tracking.env.franka_tracking_env import FrankaTrackingEnv


# ── constants ─────────────────────────────────────────────────────────────────

TARGET_R   = 0.020       # current target sphere radius
COL_TARGET = np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32)  # bright red


# ── scene geom injection ──────────────────────────────────────────────────────


def _add_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> bool:
    """Append one sphere geom to the scene. Returns False if scene is full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),   # size passed directly — more reliable
        np.array(pos, dtype=np.float64),
        np.eye(3).ravel(),
        rgba.astype(np.float32),
    )
    scene.ngeom += 1
    return True


def inject_target(scene, tgt_pos: np.ndarray):
    """Add a single red sphere at the current target position."""
    _add_sphere(scene, tgt_pos, TARGET_R, COL_TARGET)


def _render_frame(env: FrankaTrackingEnv, tgt_pos: np.ndarray) -> np.ndarray:
    """Render the scene with just a target marker sphere — no trails."""
    r = env._renderer
    if r is None:
        env.render()
        r = env._renderer
    r.update_scene(env.data, camera=env._cam)
    inject_target(r.scene, tgt_pos)
    return r.render()


# ── PIL overlay ───────────────────────────────────────────────────────────────

_FONT = None
def _get_font(size: int = 15):
    global _FONT
    if _FONT is None:
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]:
            try:
                _FONT = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT


def _add_overlay(img: np.ndarray, label: str, rmse: float | None, color: tuple) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _get_font(15)
    text = label if rmse is None else f"{label}   {rmse:.1f} mm"
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 10
    draw.rectangle([6, 6, 6 + w, 6 + h], fill=(0, 0, 0, 200))
    draw.text((14, 10), text, fill=color, font=font)
    return np.array(pil)


def _hstack(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1)


# ── episode runners ───────────────────────────────────────────────────────────

def record_ik(trajectory: str, seed: int = 42, disturbance=None):
    """Run IK-only episode, yield (env, info) per step for frame capture."""
    cfg = _eval_config(trajectory, use_residual=False, seed=seed, disturbance=disturbance)
    env = FrankaTrackingEnv(cfg, render_mode="rgb_array")
    env.reset(seed=seed)
    env.render()   # warm renderer + cam

    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        yield env, info
        if terminated or truncated:
            break

    env.close()


def record_residual(model, vn_ref, env_kwargs: dict, trajectory: str, seed: int = 42):
    """Run residual policy episode, yield (env, info) per step."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = _eval_config(trajectory, use_residual=True, seed=seed, **env_kwargs)
    raw_env = FrankaTrackingEnv(cfg, render_mode="rgb_array")
    raw_env.reset(seed=seed)
    raw_env.render()   # warm renderer + cam

    venv = DummyVecEnv([lambda: raw_env])  # noqa: B023
    if vn_ref is not None:
        venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
        venv.obs_rms = vn_ref.obs_rms
        venv.ret_rms = vn_ref.ret_rms
        venv.clip_obs = vn_ref.clip_obs

    obs = venv.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        yield raw_env, infos[0]
        if dones[0]:
            break

    venv.close()


# ── main recording function ───────────────────────────────────────────────────

def record(
    model_path: str,
    trajectory: str,
    out_dir: Path,
    fps: int = 50,
    save_gif: bool = True,
    gif_scale: float = 0.5,
    seed: int = 42,
):
    import yaml
    from ee_tracking.env.disturbances import DisturbanceConfig

    out_dir.mkdir(parents=True, exist_ok=True)

    model, vn_ref, saved_cfg = load_model(model_path)

    cfg_path = Path(model_path).parent / "config.yaml"
    env_kwargs: dict = {}
    disturbance = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        env_cfg = raw.get("env", {})
        dist    = env_cfg.get("disturbance", {})
        env_kwargs = dict(
            trajectory_pool=tuple(env_cfg.get("trajectory_pool", ["moving_target"])),
            lookahead_horizon=int(env_cfg.get("lookahead_horizon", 5)),
            lookahead_dt=float(env_cfg.get("lookahead_dt", 0.02)),
            lookahead_coarse_horizon=int(env_cfg.get("lookahead_coarse_horizon", 4)),
            lookahead_coarse_dt=float(env_cfg.get("lookahead_coarse_dt", 0.10)),
            residual_scale=float(env_cfg.get("residual_scale", 0.12)),
            action_filter_hz=float(env_cfg.get("action_filter_hz", 0.0)),
        )
        disturbance = DisturbanceConfig(
            obs_pos_noise=float(dist.get("obs_pos_noise", 0.005)),
            obs_jnt_noise=float(dist.get("obs_jnt_noise", 0.002)),
            cmd_delay=int(dist.get("cmd_delay", dist.get("act_delay", 5))),
        )

    print(f"[record_video]  trajectory={trajectory}  fps={fps}")

    ik_gen  = record_ik(trajectory, seed=seed, disturbance=disturbance)
    res_gen = record_residual(model, vn_ref, env_kwargs, trajectory, seed=seed)

    combined_frames = []
    ik_errs, res_errs = [], []

    WINDOW = 25
    for (ik_env, ik_info), (res_env, res_info) in zip(ik_gen, res_gen):
        # Render: robot + red target sphere, no trails
        ik_frame  = _render_frame(ik_env,  ik_info["target_pos"].copy())
        res_frame = _render_frame(res_env, res_info["target_pos"].copy())

        # Rolling RMSE overlay
        ik_errs.append(float(ik_info["pos_err"]) * 1000.0)
        res_errs.append(float(res_info["pos_err"]) * 1000.0)
        live_ik  = float(np.sqrt(np.mean(np.array(ik_errs[-WINDOW:])  ** 2)))
        live_res = float(np.sqrt(np.mean(np.array(res_errs[-WINDOW:]) ** 2)))

        left  = _add_overlay(ik_frame,  "IK + 100ms delay", live_ik,  (255, 120, 120))
        right = _add_overlay(res_frame, "Transformer PPO",  live_res, (100, 210, 255))
        combined_frames.append(_hstack(left, right))

    n = len(combined_frames)
    final_ik  = float(np.sqrt(np.mean(np.array(ik_errs)  ** 2)))
    final_res = float(np.sqrt(np.mean(np.array(res_errs) ** 2)))
    print(f"  IK RMSE: {final_ik:.1f} mm   PPO RMSE: {final_res:.1f} mm   ({n} frames)")

    # ── MP4 ──
    mp4_path = out_dir / f"ik_vs_residual_{trajectory}.mp4"
    with imageio.get_writer(str(mp4_path), fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p",
                            macro_block_size=1) as w:
        for f in combined_frames:
            w.append_data(f)
    print(f"  MP4 → {mp4_path}  ({mp4_path.stat().st_size // 1024} KB)")

    # ── GIF ──
    if save_gif:
        gif_path = out_dir / f"ik_vs_residual_{trajectory}.gif"
        h, w_px = combined_frames[0].shape[:2]
        nh, nw = int(h * gif_scale), int(w_px * gif_scale)
        # Keep every 3rd frame (50fps → ~17fps) for smaller file size
        gif_frames = [
            np.array(Image.fromarray(f).resize((nw, nh), Image.LANCZOS))
            for f in combined_frames[::3]
        ]
        # Quantise to 128 colours for smaller GIF
        gif_pil = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
                   for f in gif_frames]
        gif_pil[0].save(
            str(gif_path), save_all=True, append_images=gif_pil[1:],
            loop=0, duration=int(1000 / (fps // 3)), optimize=True,
        )
        print(f"  GIF → {gif_path}  ({gif_path.stat().st_size // 1024} KB)")

    return mp4_path


# ── CLI ───────────────────────────────────────────────────────────────────────

TRAJECTORIES = ["moving_target", "circle", "figure8"]


def main():
    p = argparse.ArgumentParser(description="Record IK vs Residual PPO side-by-side video")
    p.add_argument("--model",      required=True)
    p.add_argument("--trajectory", choices=TRAJECTORIES, default="moving_target")
    p.add_argument("--all",        action="store_true", dest="all_trajs")
    p.add_argument("--out",        default="results/videos")
    p.add_argument("--fps",        type=int,   default=50)
    p.add_argument("--no-gif",     action="store_true")
    p.add_argument("--gif-scale",  type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    trajs   = TRAJECTORIES if args.all_trajs else [args.trajectory]
    out_dir = Path(args.out)

    for traj in trajs:
        print(f"\n{'='*60}\n  Recording: {traj}\n{'='*60}")
        record(
            model_path=args.model,
            trajectory=traj,
            out_dir=out_dir,
            fps=args.fps,
            save_gif=not args.no_gif,
            gif_scale=args.gif_scale,
            seed=args.seed,
        )

    print(f"\nDone — videos in {out_dir}/")


if __name__ == "__main__":
    main()
