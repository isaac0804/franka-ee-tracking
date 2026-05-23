"""Damped least-squares (DLS) inverse-kinematics controller.

Produces joint-velocity commands that drive the end-effector toward a
target position with a feed-forward term from the desired EE velocity.

The damping factor keeps the controller well-behaved near singularities
and outside the reachable workspace (it just slows down, instead of
diverging). This is the "baseline" the residual RL policy improves upon.
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
