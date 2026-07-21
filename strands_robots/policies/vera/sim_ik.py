"""Inverse-kinematics bridge: VERA EE-delta action chunk -> MuJoCo joint targets.

VERA's ``mimicgen`` (``eef_delta``) and ``droid`` (``cartesian_delta``)
embodiments emit, per step, a **6-DoF end-effector delta** (translation +
rotation) plus an optional gripper column. MuJoCo arm actuators are commanded in
**joint space**, so closing the sim loop needs an IK step that maps each
Cartesian *delta* onto an absolute target pose and solves it to joint angles.

This module is **copied** from (and intentionally independent of) the cosmos3
IK bridge, :mod:`~strands_robots.policies.cosmos3.sim_ik`: that module decodes
an *absolute* EE pose trajectory
(translation + rot6d) for its in-process diffusers backend, whereas VERA emits
*relative* deltas. Copying (rather than sharing) keeps the two providers'
kinematics decoupled - a change to one model's action semantics can never
silently break the other. The shared piece is only the generic ``mink`` solver
wrapper (:class:`MinkIKBridge`); the VERA-specific decode lives in
:func:`decode_vera_delta_chunk_to_targets`.

``mink`` + ``mujoco`` are imported lazily so importing the VERA provider in the
light base env (no torch / no sim) stays cheap; a missing stack raises an
actionable install error rather than a silent default.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco

logger = logging.getLogger(__name__)


def _install_hint() -> str:
    """Actionable message when the IK stack (mink + mujoco) is not importable."""
    return (
        "The VERA eef-delta IK-to-MuJoCo bridge needs 'mink' + 'mujoco', which "
        "were not importable. Install the sim extra:\n"
        "  uv pip install 'strands-robots[sim-mujoco]' mink\n"
        "This turns VERA's end-effector delta chunk (mimicgen/droid) into joint "
        "targets the MuJoCo arm can track. For joint_position embodiments "
        "(allegro) no IK is needed - the action maps directly to joints."
    )


_PREFERRED_QP_SOLVERS = ("daqp", "quadprog", "osqp", "proxqp", "cvxopt", "scs")


def _resolve_qp_solver(requested: str | None) -> str:
    """Pick an installed ``qpsolvers`` backend for ``mink.solve_ik``.

    ``mink`` pins ``daqp``, but many envs ship only ``quadprog``. Auto-select
    from ``qpsolvers.available_solvers`` (prefer daqp, then quadprog) so the IK
    works everywhere; honour an explicit ``requested`` name when installed, else
    fail with an actionable error (no silent fallback to an unrequested solver).
    """
    try:
        from qpsolvers import available_solvers
    except ImportError as e:
        raise ImportError(_install_hint()) from e
    available = list(available_solvers)
    if not available:
        raise RuntimeError(
            "No qpsolvers backend is installed; the VERA IK bridge needs one "
            "(e.g. 'daqp' or 'quadprog'). Install: "
            "uv pip install 'strands-robots[sim-mujoco]' 'qpsolvers[quadprog]'."
        )
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Requested qpsolvers backend {requested!r} is not installed. Available: {available}.")
        return requested
    for name in _PREFERRED_QP_SOLVERS:
        if name in available:
            return name
    return available[0]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a 6D rotation representation into a ``(3, 3)`` matrix.

    The 6D rep (Zhou et al. 2019) is the first two columns of the rotation
    matrix; the third is their cross product. Robust to non-orthonormal input.
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = r[:3], r[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert an axis-angle 3-vector (rotation vector) to a ``(3, 3)`` matrix."""
    v = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def delta_to_matrix(rot_delta: np.ndarray, rotation_dim: int) -> np.ndarray:
    """Map a rotation delta (``rotation_dim`` ∈ {3 axis-angle, 6 rot6d}) -> (3,3)."""
    if rotation_dim == 6:
        return rot6d_to_matrix(rot_delta)
    if rotation_dim == 3:
        return axis_angle_to_matrix(rot_delta)
    raise ValueError(f"unsupported rotation_dim {rotation_dim!r}; use 3 (axis-angle) or 6 (rot6d)")


class MinkIKBridge:
    """Differential-IK bridge from EE poses to MuJoCo joint configurations.

    A copy of the cosmos3 bridge's generic solver wrapper (the model-agnostic
    part). See module docstring for why VERA keeps its own copy.

    Args:
        model: The ``mujoco.MjModel`` for the arm being controlled.
        ee_frame_name: End-effector frame the Cartesian task tracks.
        ee_frame_type: ``"body"`` (default), ``"site"`` or ``"geom"``.
        position_cost / orientation_cost / posture_cost: task weights.
        solver: qpsolvers backend (``None`` auto-selects).
        damping / max_iters / dt / pos_threshold / ori_threshold: solver knobs.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_frame_name: str,
        ee_frame_type: str = "body",
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        solver: str | None = None,
        damping: float = 1e-3,
        max_iters: int = 20,
        dt: float = 1e-2,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-3,
    ) -> None:
        try:
            import mink
        except ImportError as e:
            raise ImportError(_install_hint()) from e

        self._mink = mink
        self.model = model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_type = ee_frame_type
        self.solver = _resolve_qp_solver(solver)
        self.damping = damping
        self.max_iters = max_iters
        self.dt = dt
        self.pos_threshold = pos_threshold
        self.ori_threshold = ori_threshold

        self._configuration = mink.Configuration(model)
        self._frame_task = mink.FrameTask(
            frame_name=ee_frame_name,
            frame_type=ee_frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self._tasks = [self._frame_task, self._posture_task]
        logger.info(
            "VERA MinkIKBridge ready [ee=%s/%s solver=%s nq=%d]",
            ee_frame_type,
            ee_frame_name,
            self.solver,
            model.nq,
        )

    def ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Forward kinematics: the EE frame's absolute ``(4, 4)`` pose at ``qpos``."""
        self._configuration.update(np.asarray(qpos, dtype=np.float64))
        transform = self._configuration.get_transform_frame_to_world(self.ee_frame_name, self.ee_frame_type)
        return np.asarray(transform.as_matrix(), dtype=np.float64)

    def solve(self, target_pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Solve IK for one Cartesian target from a seed configuration."""
        mink = self._mink
        q = np.asarray(q_init, dtype=np.float64).copy()
        self._configuration.update(q)
        self._posture_task.set_target(q)
        target = mink.SE3.from_matrix(np.asarray(target_pose, dtype=np.float64))
        self._frame_task.set_target(target)
        for _ in range(self.max_iters):
            velocity = mink.solve_ik(self._configuration, self._tasks, self.dt, self.solver, self.damping)
            self._configuration.integrate_inplace(velocity, self.dt)
            err = self._frame_task.compute_error(self._configuration)
            if np.linalg.norm(err[:3]) <= self.pos_threshold and np.linalg.norm(err[3:]) <= self.ori_threshold:
                break
        return np.asarray(self._configuration.q, dtype=np.float64).copy()


def decode_vera_delta_chunk_to_targets(
    action_chunk: np.ndarray,
    ik_bridge: MinkIKBridge,
    q_init: np.ndarray,
    *,
    rotation_dim: int = 3,
    has_gripper: bool = True,
    gripper_dim_index: int = -1,
    translation_scale: float = 1.0,
) -> dict[str, Any]:
    """Turn a VERA EE-**delta** action chunk into MuJoCo joint targets via IK.

    VERA emits, per step, ``[translation(3), rotation(rotation_dim), gripper?]``
    as a delta on the *current* end-effector pose. We re-anchor each delta on the
    arm's **achieved** EE pose (closed loop - the FK of the previous IK solve),
    mirroring how robot deploy servers anchor on the observed pose so per-step
    tracking error stays bounded instead of compounding down the chunk.

    Args:
        action_chunk: ``[T, D]`` VERA action chunk (per-step EE delta + gripper).
        ik_bridge: A :class:`MinkIKBridge` over the target arm's MuJoCo model.
        q_init: Seed joint config (length ``model.nq``) - the robot's current pose.
        rotation_dim: 3 (axis-angle) or 6 (rot6d) rotation delta encoding.
        has_gripper: Whether the chunk carries a trailing gripper column.
        gripper_dim_index: Index of the gripper column (``-1`` => last when
            ``has_gripper``); the value is passed through (binarized by caller).
        translation_scale: Optional scale on the translation delta (units match).

    Returns:
        ``{"qpos": [T, nq], "gripper": [T] | None, "tracking_error": {...}}``.
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk must be [T, D]; got {action_chunk.shape}")
    T, D = action_chunk.shape

    # Split gripper column off.
    gripper = None
    pose_block = action_chunk
    if has_gripper:
        gidx = gripper_dim_index if gripper_dim_index >= 0 else D - 1
        gripper = action_chunk[:, gidx].copy()
        pose_block = np.delete(action_chunk, gidx, axis=1)

    expected = 3 + rotation_dim
    if pose_block.shape[1] < expected:
        raise ValueError(
            f"VERA eef-delta needs >= {expected} pose dims (3 trans + {rotation_dim} rot); "
            f"got {pose_block.shape[1]} after removing gripper. Check rotation_dim/action_space."
        )

    q = np.asarray(q_init, dtype=np.float64).copy()
    achieved = ik_bridge.ee_pose(q)
    qpos_list: list[np.ndarray] = []
    err_list: list[float] = []
    for step in pose_block:
        # Robosuite OSC_POSE maps the policy's [-1,1] action to metric deltas via
        # output_max: translation *= 0.05 m, rotation *= 0.5 rad (control_delta=true,
        # input_max=1). VERA emits these normalized OSC actions, so we apply the
        # same scaling before IK -- without it the raw [-1,1] values are treated as
        # ~0.4 m steps, producing unreachable IK targets (track err > 1 m) and the
        # arm never descends to the object. translation_scale composes on top of
        # the OSC position scale for callers that need a further tweak.
        _OSC_POS_SCALE = 0.05
        _OSC_ROT_SCALE = 0.5
        trans = step[:3] * (_OSC_POS_SCALE * float(translation_scale))
        rot = step[3 : 3 + rotation_dim] * _OSC_ROT_SCALE
        # VERA/MimicGen eef_delta follows robosuite OSC_POSE: translation deltas
        # are in the WORLD/base frame (added to the EE position), not the tool
        # frame. Rotation deltas premultiply (world-frame) the current EE
        # orientation. Composing translation in the tool frame (achieved @ delta)
        # rotates a "move down" command by the gripper's orientation, so the arm
        # barely descends -- the cube never gets reached. Apply world-frame.
        rot_delta = delta_to_matrix(rot, rotation_dim)
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = rot_delta @ achieved[:3, :3]  # world-frame rotation delta
        target[:3, 3] = achieved[:3, 3] + trans  # world-frame translation delta
        q = ik_bridge.solve(target, q)
        achieved_new = ik_bridge.ee_pose(q)
        err_list.append(float(np.linalg.norm(achieved_new[:3, 3] - target[:3, 3])))
        achieved = achieved_new
        qpos_list.append(q.copy())

    nq = ik_bridge.model.nq
    qpos = np.stack(qpos_list) if qpos_list else np.empty((0, nq), dtype=np.float64)
    err_arr = np.asarray(err_list, dtype=np.float64)
    tracking = {
        "mean_mm": float(err_arr.mean() * 1000.0) if err_arr.size else 0.0,
        "max_mm": float(err_arr.max() * 1000.0) if err_arr.size else 0.0,
    }
    return {"qpos": qpos, "gripper": gripper, "tracking_error": tracking}
