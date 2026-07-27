"""Inverse-kinematics bridge: VERA EE-delta action chunk -> MuJoCo joint targets.

VERA's ``mimicgen`` (``eef_delta``) and ``droid`` (``cartesian_delta``)
embodiments emit, per step, a **6-DoF end-effector delta** (translation +
rotation) plus an optional gripper column. MuJoCo arm actuators are commanded in
**joint space**, so closing the sim loop needs an IK step that maps each
Cartesian *delta* onto an absolute target pose and solves it to joint angles.

The generic damped-least-squares solver wrapper is the shared
:class:`strands_robots.simulation.ik.MinkIKBridge` (one home for the mink
``FrameTask`` + ``PostureTask`` solve loop; the cosmos3 provider - which
decodes *absolute* EE pose trajectories in
:mod:`~strands_robots.policies.cosmos3.sim_ik` - and the simulation motion
primitives use the same class). This module subclasses it only to brand the
install errors with the ``sim-mujoco`` extra, and keeps the VERA-specific
decode glue (:func:`decode_vera_delta_chunk_to_targets`) local so a change to
one model's action semantics can never silently break the other.

``mink`` + ``mujoco`` are imported lazily so importing the VERA provider in the
light base env (no torch / no sim) stays cheap; a missing stack raises an
actionable install error rather than a silent default.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from strands_robots.simulation.ik import MinkIKBridge as _SharedMinkIKBridge
from strands_robots.simulation.ik import resolve_qp_solver

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


_NO_BACKEND_MSG = (
    "No qpsolvers backend is installed; the VERA IK bridge needs one "
    "(e.g. 'daqp' or 'quadprog'). Install: "
    "uv pip install 'strands-robots[sim-mujoco]' 'qpsolvers[quadprog]'."
)


def _resolve_qp_solver(requested: str | None) -> str:
    """Pick an installed ``qpsolvers`` backend for ``mink.solve_ik``.

    Delegates to the shared :func:`strands_robots.simulation.ik.resolve_qp_solver`
    with VERA-branded errors: the install hint and no-backend message name the
    ``sim-mujoco`` extra so a clean-install user is pointed at the right
    dependency set (no silent fallback to an unrequested solver).
    """
    return resolve_qp_solver(requested, install_hint=_install_hint(), no_backend_msg=_NO_BACKEND_MSG)


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


class MinkIKBridge(_SharedMinkIKBridge):
    """Differential-IK bridge from EE poses to MuJoCo joint configurations.

    The VERA branding of the shared
    :class:`strands_robots.simulation.ik.MinkIKBridge` (same solver, tasks, and
    convergence behavior): a missing ``mink``/``qpsolvers`` stack raises the
    ``sim-mujoco`` install hint. See the shared class for the full
    constructor/solve contract.
    """

    _INSTALL_HINT: ClassVar[str] = _install_hint()
    _NO_BACKEND_MSG: ClassVar[str] = _NO_BACKEND_MSG
    _LOG_LABEL: ClassVar[str] = "VERA MinkIKBridge"


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
