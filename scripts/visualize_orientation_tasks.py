"""Matplotlib 3-D animated GIFs for each orientation tracking task.

Generates one GIF per task showing:
  - EE position trajectory (coloured path)
  - Orientation frames: an RGB triad showing the EE orientation target
    (R=x, G=y, B=z axes of the desired EE frame)
  - For look_at: the beacon point and the look direction
  - For upright_constraint: the upright arrow constraint visualised

Output: results/figures/orient_task_*.gif
Usage:  python scripts/visualize_orientation_tasks.py
"""

import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ee_tracking.env import trajectories as traj_module
from ee_tracking.env.trajectories import (
    axisangle_to_quat, quat_to_axisangle,
    rotation_from_z_to, mat_to_quat,
    _IDENTITY_QUAT,
)

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def draw_frame(ax, origin, quat, length=0.03, alpha=0.9, lw=2.0):
    """Draw an RGB orientation triad at `origin` for quaternion `quat`."""
    R = quat_to_R(quat)
    colours = ['#e74c3c', '#2ecc71', '#3498db']  # R=x, G=y, B=z
    for i, c in enumerate(colours):
        ax.quiver(origin[0], origin[1], origin[2],
                  R[0, i], R[1, i], R[2, i],
                  length=length, color=c, alpha=alpha, linewidth=lw,
                  arrow_length_ratio=0.25)


def sample_trajectory(traj, dt=0.02, n_frames=100):
    """Sample position + orientation at `n_frames` evenly spaced times."""
    ts   = np.linspace(0, min(traj.duration, 8.0), n_frames)
    pos  = np.array([traj.sample(t)[0] for t in ts])
    quats = np.array([traj.sample_ori(t)[0] for t in ts])
    return ts, pos, quats


def make_gif(name, traj, extra_fn=None, n_frames=80, fps=15, elev=20, azim=-60,
             frame_stride=4, title=""):
    """Create a 3-D animated GIF for `traj`.

    Parameters
    ----------
    name        : output filename stem
    traj        : Trajectory instance
    extra_fn    : optional callable(ax, origin, quat, t) for per-frame extras
    n_frames    : total frames to generate
    fps         : GIF frame rate
    frame_stride: draw orientation frame every N position samples
    """
    ts, pos, quats = sample_trajectory(traj, n_frames=n_frames)

    # Determine axis limits with 15% padding
    lo = pos.min(axis=0) - 0.04
    hi = pos.max(axis=0) + 0.04
    mid = (lo + hi) / 2
    half = np.max(hi - lo) / 2 + 0.02

    fig = plt.figure(figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)
    ax.set_xlabel('X (m)', fontsize=8, labelpad=2)
    ax.set_ylabel('Y (m)', fontsize=8, labelpad=2)
    ax.set_zlabel('Z (m)', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6, pad=1)
    ax.set_title(title or name.replace("_", " ").title(), fontsize=10, pad=6)

    # Static ghost: full path in light grey
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#cccccc', lw=0.8, zorder=1)

    # Pre-compute: draw orientation frames at strided points
    frame_indices = list(range(0, n_frames, frame_stride))

    # Artists that update each animation frame
    trail_line, = ax.plot([], [], [], lw=2.0, color='#3498db', zorder=3)
    dot, = ax.plot([], [], [], 'o', color='#e74c3c', ms=6, zorder=5)

    # Orientation frame quivers — recreated each frame
    _quiver_handles = []

    def init():
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        dot.set_data([], [])
        dot.set_3d_properties([])
        return trail_line, dot

    def animate(fi):
        nonlocal _quiver_handles
        # Remove old orientation frames
        for h in _quiver_handles:
            h.remove()
        _quiver_handles = []

        i = fi  # current frame index
        trail_line.set_data(pos[:i+1, 0], pos[:i+1, 1])
        trail_line.set_3d_properties(pos[:i+1, 2])
        dot.set_data([pos[i, 0]], [pos[i, 1]])
        dot.set_3d_properties([pos[i, 2]])

        # Draw orientation frames at recent strided points (fade: old=faint, new=bright)
        visible_frames = [j for j in frame_indices if j <= i][-8:]
        n_vis = max(len(visible_frames) - 1, 1)
        for idx, j in enumerate(visible_frames):
            alpha = float(np.clip(0.2 + 0.7 * (idx / n_vis), 0.0, 1.0))
            R = quat_to_R(quats[j])
            length = 0.028
            for dim, c in enumerate(['#e74c3c', '#2ecc71', '#3498db']):
                q = ax.quiver(
                    pos[j, 0], pos[j, 1], pos[j, 2],
                    R[0, dim] * length, R[1, dim] * length, R[2, dim] * length,
                    color=c, alpha=alpha, linewidth=1.5,
                    arrow_length_ratio=0.3,
                )
                _quiver_handles.append(q)

        # Task-specific extras
        if extra_fn is not None:
            extra_fn(ax, pos[i], quats[i], ts[i], _quiver_handles)

        return trail_line, dot

    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=False
    )

    out = OUT_DIR / f"orient_task_{name}.gif"
    ani.save(str(out), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ── Task 1 — Upright Constraint ──────────────────────────────────────────────

def task1_upright():
    # Position: figure-8. Orientation: always upright (identity)
    inner = traj_module.FigureEight(center=(0.5, 0.0, 0.5), size=0.12, period=8.0)
    traj = traj_module.UprightConstraint(inner)

    def extra(ax, pos, quat, t, handles):
        # Draw the "desired upright" z-axis as a thick blue arrow
        h = ax.quiver(pos[0], pos[1], pos[2],
                      0, 0, 0.05,
                      color='#3498db', alpha=0.5, linewidth=2, arrow_length_ratio=0.3)
        handles.append(h)
        # Draw a cup symbol (circle at top) — just a label
        # (too complex to animate, skip)

    make_gif("upright_constraint", traj, extra_fn=extra, n_frames=90, fps=15,
             title="Task 1: Upright Constraint\n(keep EE z-axis pointing up while tracking)")


# ── Task 2 — Tilted Circle ───────────────────────────────────────────────────

def task2_tilted_circle():
    traj = traj_module.TiltedCircle(
        center=(0.5, 0.0, 0.5), radius=0.12, period=6.0
    )

    def extra(ax, pos, quat, t, handles):
        # Draw a line from EE to circle centre showing the "look inward" direction
        centre = np.array([0.5, 0.0, 0.5])
        d = centre - pos
        d_norm = d / (np.linalg.norm(d) + 1e-9)
        h = ax.quiver(pos[0], pos[1], pos[2],
                      d_norm[0]*0.04, d_norm[1]*0.04, d_norm[2]*0.04,
                      color='#f39c12', alpha=0.7, linewidth=2,
                      arrow_length_ratio=0.3)
        handles.append(h)

    make_gif("tilted_circle", traj, extra_fn=extra, n_frames=90, fps=15,
             title="Task 2: Tilted Circle\n(EE z-axis points toward circle centre)")


# ── Task 3 — Look-At ─────────────────────────────────────────────────────────

def task3_look_at():
    inner = traj_module.MovingTarget(
        center=(0.5, 0.0, 0.5), extent=0.10, duration=12.0,
        cutoff_hz=0.08, seed=7
    )
    beacon = np.array([0.5, 0.22, 0.68])
    traj = traj_module.LookAt(position_traj=inner, beacon=beacon)

    beacon_plotted = [False]

    def extra(ax, pos, quat, t, handles):
        # Beacon marker (only on first call)
        if not beacon_plotted[0]:
            ax.scatter(*beacon, color='#f39c12', s=80, zorder=6, marker='*')
            beacon_plotted[0] = True
        # Look direction arrow
        d = beacon - pos
        d_n = d / (np.linalg.norm(d) + 1e-9)
        h = ax.quiver(pos[0], pos[1], pos[2],
                      d_n[0]*0.05, d_n[1]*0.05, d_n[2]*0.05,
                      color='#f39c12', alpha=0.8, linewidth=1.5,
                      arrow_length_ratio=0.3)
        handles.append(h)

    make_gif("look_at", traj, extra_fn=extra, n_frames=90, fps=15, azim=-50,
             title="Task 3: Look-At Tracking\n(EE always points toward fixed beacon ★)")


# ── Task 4 — Rotating Grasp ──────────────────────────────────────────────────

def task4_rotating_grasp():
    traj = traj_module.RotatingGrasp(
        position=(0.5, 0.05, 0.52), omega=0.6, duration=12.0
    )

    # Draw the rotation axis (z-axis of EE frame)
    def extra(ax, pos, quat, t, handles):
        R = quat_to_R(quat)
        h = ax.quiver(pos[0], pos[1], pos[2],
                      R[0, 2]*0.06, R[1, 2]*0.06, R[2, 2]*0.06,
                      color='#9b59b6', alpha=0.8, linewidth=2.5,
                      arrow_length_ratio=0.25)
        handles.append(h)
        # Rotation sweep arc — draw a small curved arrow hint
        theta = t * traj._omega
        arc_pts = 12
        r = 0.025
        thetas = np.linspace(theta, theta + np.pi / 3, arc_pts)
        arc_x = pos[0] + r * np.cos(thetas)
        arc_y = pos[1] + r * np.sin(thetas)
        arc_z = np.full(arc_pts, pos[2])
        h2, = ax.plot(arc_x, arc_y, arc_z, color='#9b59b6', alpha=0.4, lw=1.2)
        handles.append(h2)

    make_gif("rotating_grasp", traj, extra_fn=extra, n_frames=100, fps=15,
             frame_stride=5, azim=-70, elev=30,
             title="Task 4: Rotating Grasp\n(fixed position, wrist rotates around z-axis)")


# ── Task 5 — 6-DoF Random Walk ───────────────────────────────────────────────

def task5_random_walk():
    traj = traj_module.RandomWalk6DoF(
        center=(0.5, 0.0, 0.5), extent=0.10, max_angle=np.pi/4,
        duration=12.0, cutoff_hz=0.08, seed=3
    )

    make_gif("random_walk_6dof", traj, n_frames=90, fps=15, azim=-55,
             title="Task 5: 6-DoF Random Walk\n(independent random walks in R³ × SO(3))")


# ── Overview panel GIF ───────────────────────────────────────────────────────

def make_overview():
    """Single figure with all 5 tasks side-by-side, static (no animation)."""
    tasks = [
        ("Upright\nConstraint",
         traj_module.UprightConstraint(traj_module.FigureEight(center=(0.5, 0., 0.5), size=0.12, period=8.))),
        ("Tilted\nCircle",
         traj_module.TiltedCircle(center=(0.5, 0., 0.5), radius=0.12, period=6.)),
        ("Look-At\nTracking",
         traj_module.LookAt(position_traj=traj_module.MovingTarget(center=(0.5, 0., 0.5), extent=0.10, duration=12., cutoff_hz=0.08, seed=7),
                            beacon=(0.5, 0.22, 0.68))),
        ("Rotating\nGrasp",
         traj_module.RotatingGrasp(position=(0.5, 0.05, 0.52), omega=0.6, duration=10.)),
        ("6-DoF\nRandom Walk",
         traj_module.RandomWalk6DoF(center=(0.5, 0., 0.5), extent=0.10, max_angle=np.pi/4, duration=12., seed=3)),
    ]

    fig = plt.figure(figsize=(18, 4.5), dpi=100)
    fig.suptitle("Orientation Tracking Tasks — 6-DoF Residual PPO Branch",
                 fontsize=13, fontweight='bold', y=1.01)

    for k, (label, traj) in enumerate(tasks):
        ax = fig.add_subplot(1, 5, k+1, projection='3d')
        n = 120
        ts = np.linspace(0, min(traj.duration, 8.0), n)
        pos  = np.array([traj.sample(t)[0] for t in ts])
        quats = np.array([traj.sample_ori(t)[0] for t in ts])

        # Colour gradient: blue→red along time
        colours = plt.cm.plasma(np.linspace(0.1, 0.9, n - 1))
        for i in range(n - 1):
            ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                    color=colours[i], lw=1.5, alpha=0.85)

        # Orientation frames at every 8th point
        for i in range(0, n, 8):
            R = quat_to_R(quats[i])
            for dim, c in enumerate(['#c0392b', '#27ae60', '#2980b9']):
                ax.quiver(
                    pos[i, 0], pos[i, 1], pos[i, 2],
                    R[0, dim]*0.025, R[1, dim]*0.025, R[2, dim]*0.025,
                    color=c, alpha=0.65, linewidth=1.2,
                    arrow_length_ratio=0.35,
                )

        # Task-specific extras
        if k == 2:  # Look-At beacon
            ax.scatter(0.5, 0.22, 0.68, color='#f39c12', s=60, zorder=6, marker='*')

        lo = pos.min(0) - 0.035; hi = pos.max(0) + 0.035
        mid = (lo + hi) / 2; half = max(hi - lo) / 2 + 0.02
        ax.set_xlim(mid[0]-half, mid[0]+half)
        ax.set_ylim(mid[1]-half, mid[1]+half)
        ax.set_zlim(mid[2]-half, mid[2]+half)
        ax.set_title(f"Task {k+1}\n{label}", fontsize=9, pad=3)
        ax.set_xlabel('X', fontsize=7, labelpad=0)
        ax.set_ylabel('Y', fontsize=7, labelpad=0)
        ax.set_zlabel('Z', fontsize=7, labelpad=0)
        ax.tick_params(labelsize=5, pad=0)
        ax.view_init(elev=20, azim=-60 + k * 8)

    plt.tight_layout()
    out = OUT_DIR / "orient_tasks_overview.png"
    plt.savefig(str(out), bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    print("\nGenerating orientation task visualizations...\n")
    print("Overview panel:")
    make_overview()

    print("\nAnimated GIFs (each ~5 s):")
    task1_upright()
    task2_tilted_circle()
    task3_look_at()
    task4_rotating_grasp()
    task5_random_walk()

    print("\nDone. Files in results/figures/")
