"""Inverse kinematics — Jacobian-based IK solver."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def solve_ik(
    backend: NewtonBackend,
    robot_name: str,
    target_position: Tuple[float, float, float],
    target_rotation: Optional[Tuple[float, float, float, float]] = None,
    end_effector_body: Optional[int] = None,
    max_iterations: int = 100,
    tolerance: float = 0.001,
) -> Dict[str, Any]:
    """Solve inverse kinematics using Jacobian-based method.

    Phase 4: Following Newton's cloth_franka IK pattern.
    """
    if backend._model is None:
        from ._scene import finalize_model

        finalize_model(backend)

    if robot_name not in backend._robots:
        return {"success": False, "message": f"Robot '{robot_name}' not found."}

    wp = backend._wp
    newton = backend._newton
    target = np.array(target_position)

    try:
        for i in range(max_iterations):
            # Forward kinematics
            newton.eval_fk(
                backend._model,
                backend._state_0.joint_q,
                backend._state_0.joint_qd,
                backend._state_0,
            )
            body_q = backend._state_0.body_q.numpy()

            # Get end effector position
            ee_idx = end_effector_body or (backend._bodies_per_world - 1)
            ee_pos = body_q[ee_idx][:3]
            error = target - ee_pos
            error_norm = float(np.linalg.norm(error))

            if error_norm < tolerance:
                return {
                    "success": True,
                    "iterations": i + 1,
                    "error": error_norm,
                    "joint_positions": backend._state_0.joint_q.numpy().copy(),
                }

            # Gradient-based IK step
            joint_q = backend._state_0.joint_q.numpy().copy()
            step_size = min(0.1, error_norm)
            delta_q = np.zeros(len(joint_q), dtype=np.float32)

            # Numerical Jacobian
            for j in range(min(len(joint_q), backend._joints_per_world)):
                q_plus = joint_q.copy()
                q_plus[j] += 0.001
                temp_state = backend._model.state()
                temp_state.joint_q.assign(
                    wp.array(q_plus, dtype=wp.float32, device=backend._config.device)
                )
                temp_state.joint_qd.assign(backend._state_0.joint_qd)
                newton.eval_fk(
                    backend._model, temp_state.joint_q, temp_state.joint_qd, temp_state
                )
                body_out = temp_state.body_q.numpy()
                new_pos = body_out[ee_idx][:3]
                jac_col = (new_pos - ee_pos) / 0.001
                delta_q[j] = np.dot(jac_col, error) * step_size

            current_q = joint_q + delta_q
            backend._state_0.joint_q.assign(
                wp.array(current_q, dtype=wp.float32, device=backend._config.device)
            )

        return {
            "success": True,
            "iterations": max_iterations,
            "error": float(np.linalg.norm(error)),
            "joint_positions": backend._state_0.joint_q.numpy().copy(),
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}
