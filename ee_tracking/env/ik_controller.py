"""Damped least-squares (DLS) inverse-kinematics controller.

Produces joint-velocity commands that drive the end-effector toward a
target position with a feed-forward term from the desired EE velocity.

The damping factor keeps the controller well-behaved near singularities
and outside the reachable workspace (it just slows down, instead of
diverging). This is the "baseline" the residual RL policy improves upon.

6-DoF extension
---------------
`DLS6DoFController` subclasses `DLSController` to also command orientation.
It uses the full 6×7 Jacobian (translational + rotational) and a separate
orientation gain `kp_ori`.  The 6-D task-space velocity is:

    v_des = [kp_pos * e_pos + v_pos_ff,
             kp_ori * e_ori + v_ori_ff]    # e_ori is the 3-D axis-angle error

The DLS solve is identical in structure, just larger:
    q_dot = J^T (J J^T + λ²I₆)^{-1} v_des   (J is 6×7)
"""

from __future__ import annotations

import mujoco
import numpy as np


class DLSController:
    """Damped least-squares IK over the 7 arm joints of the Franka.

    Args:
        model:       mjModel
        ee_body:     name of the end-effector body (its xpos is the target).
        arm_dof:     joint indices on the arm we are allowed to command.
        damping:     DLS damping (m). Larger => slower but safer.
        kp:          proportional gain on position error (1/s).
        max_jntvel:  per-joint velocity clip (rad/s).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_body: str = "hand",
        arm_dof: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        damping: float = 0.08,
        kp: float = 6.0,
        max_jntvel: float = 1.5,
    ):
        self.model = model
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_id < 0:
            raise ValueError(f"body {ee_body!r} not in model")
        self.arm_dof = np.asarray(arm_dof, dtype=np.int32)
        self.damping = float(damping)
        self.kp = float(kp)
        self.max_jntvel = float(max_jntvel)

        # Scratch buffers reused every step.
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def ee_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.ee_id].copy()

    def jacobian(self, data: mujoco.MjData) -> np.ndarray:
        """3x(arm_dof) translational Jacobian at the EE."""
        mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, self.ee_id)
        return self._jacp[:, self.arm_dof].copy()

    def compute(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return joint velocities (rad/s) for the 7 arm joints."""
        if target_vel is None:
            target_vel = np.zeros(3)

        p = self.ee_pos(data)
        e = target_pos - p
        v_des = target_vel + self.kp * e  # 3-vec EE velocity command

        J = self.jacobian(data)            # (3, 7)
        # DLS: q_dot = J^T (J J^T + lambda^2 I)^-1 v_des
        lam2 = self.damping ** 2
        A = J @ J.T + lam2 * np.eye(3)
        q_dot = J.T @ np.linalg.solve(A, v_des)
        return np.clip(q_dot, -self.max_jntvel, self.max_jntvel)


# ---------------------------------------------------------------------------
# 6-DoF DLS controller (position + orientation)
# ---------------------------------------------------------------------------

class DLS6DoFController(DLSController):
    """Damped least-squares IK with full 6-DoF (position + orientation) control.

    Extends `DLSController` to use the combined 6×7 Jacobian (translational
    rows stacked above rotational rows) and a separate orientation gain.

    Args:
        kp_ori:   proportional gain on orientation error (rad/s per rad).
                  Smaller than kp_pos is usually better: orientation errors
                  converge more slowly due to larger inertia in distal joints.
        *args, **kwargs: forwarded to `DLSController`.
    """

    def __init__(self, *args, kp_ori: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kp_ori = float(kp_ori)
        # Additional scratch buffer for rotational Jacobian (reused in __init__)
        # _jacp and _jacr already allocated by DLSController.__init__

    def jacobian6d(self, data: mujoco.MjData) -> np.ndarray:
        """6×7 Jacobian: translational rows stacked above rotational rows."""
        mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, self.ee_id)
        J_pos = self._jacp[:, self.arm_dof]   # (3, 7)
        J_rot = self._jacr[:, self.arm_dof]   # (3, 7)
        return np.vstack([J_pos, J_rot])       # (6, 7)

    def ee_quat(self, data: mujoco.MjData) -> np.ndarray:
        """Current EE orientation as (w, x, y, z) unit quaternion."""
        # MuJoCo stores xquat as (w, x, y, z) — same convention we use throughout
        return data.xquat[self.ee_id].copy()

    def compute6d(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_vel: np.ndarray | None = None,
        target_quat: np.ndarray | None = None,
        target_angvel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return 7 joint velocities (rad/s) for full 6-DoF tracking.

        Args:
            target_pos:   desired EE position (3D).
            target_vel:   desired EE translational velocity (3D, optional).
            target_quat:  desired EE orientation as (w,x,y,z) quaternion.
                          If None, falls back to position-only IK.
            target_angvel: desired EE angular velocity (3D, optional).
        """
        from .trajectories import quat_error   # lazy import to avoid circularity

        if target_vel is None:
            target_vel = np.zeros(3)

        # --- Position task ---
        p_cur = self.ee_pos(data)
        e_pos = target_pos - p_cur
        v_pos = target_vel + self.kp * e_pos

        if target_quat is None:
            # Fallback: position-only (3×7 Jacobian, same as base class)
            return self.compute(data, target_pos, target_vel)

        # --- Orientation task ---
        if target_angvel is None:
            target_angvel = np.zeros(3)

        q_cur = self.ee_quat(data)
        e_ori = quat_error(target_quat, q_cur)           # 3-D axis-angle error
        v_ori = target_angvel + self.kp_ori * e_ori

        # --- Combined 6-D DLS solve ---
        v_des = np.concatenate([v_pos, v_ori])            # (6,)
        J     = self.jacobian6d(data)                     # (6, 7)
        lam2  = self.damping ** 2
        A     = J @ J.T + lam2 * np.eye(6)               # (6, 6)
        q_dot = J.T @ np.linalg.solve(A, v_des)          # (7,)
        return np.clip(q_dot, -self.max_jntvel, self.max_jntvel)
