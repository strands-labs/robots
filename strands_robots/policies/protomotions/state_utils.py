"""Pure-numpy quaternion + state derivation utilities for ProtoMotions tracking.

Clean-room from ProtoMotions (Apache 2.0) deployment/state_utils.py.
No torch, no PyTorch tensors - a caller a strands-robots policy can invoke
without a GPU or a heavy tensor stack.

Quaternion convention throughout: ``xyzw`` (ProtoMotions common format).
MuJoCo's own quaternions are ``wxyz`` - :func:`mujoco_wxyz_to_xyzw` bridges
the two at the read boundary so the rest of this module a caller mixes with
MuJoCo state can stay in one convention.

Unit-quaternion contract: every rotation formula here is derived for a unit
quaternion, so :func:`quat_rotate_inverse` and :func:`extract_yaw_quat`
normalise their input rather than trusting it, and refuse a quaternion that
cannot define a rotation. Scale does not cancel in either formula - both mix a
quadratic term with a constant - so a caller's raw quaternion would otherwise
be read as a *different* rotation instead of the same one. That matters
because the quaternions reaching this module are not all guaranteed unit: a
hand-assembled ``body_rot_xyzw`` batch, a real-robot IMU reading and an
orientation obtained by linearly interpolating two samples are all slightly
off unit (linear interpolation is why the sibling
:mod:`strands_robots.policies.protomotions.motion_utils` normalises inside its
slerp). This matches the world-to-body helpers elsewhere in the package, which
all normalise internally.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "mujoco_wxyz_to_xyzw",
    "quat_rotate_inverse",
    "compute_anchor_rot",
    "compute_root_local_ang_vel",
    "extract_yaw_quat",
    "quat_mul",
    "quat_conjugate",
    "compute_yaw_offset",
    "apply_heading_offset",
]


# ---------------------------------------------------------------------------
# Convention conversion
# ---------------------------------------------------------------------------


def mujoco_wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo ``wxyz`` quaternion (or array of them) to ``xyzw``.

    Args:
        wxyz: Shape ``[..., 4]`` in MuJoCo's ``wxyz`` order.

    Returns:
        Same shape, reordered to ``xyzw``.
    """
    return wxyz[..., [1, 2, 3, 0]]


# ---------------------------------------------------------------------------
# Quaternion math
# ---------------------------------------------------------------------------


#: Smallest quaternion norm treated as a rotation. Below this the direction is
#: numerical noise, so scaling to unit length would amplify noise into an
#: arbitrary orientation rather than recover the caller's intent.
_MIN_QUAT_NORM = 1e-8


def _unit_quat(q_xyzw: np.ndarray, *, context: str) -> np.ndarray:
    """Scale ``q_xyzw`` to unit length, refusing one that is not a rotation.

    Args:
        q_xyzw: Shape ``[4]`` quaternion, not required to be unit.
        context: Name of the calling helper, quoted in the refusal so the
            caller learns which input was rejected.

    Returns:
        The same quaternion scaled to unit length, in the input's dtype.

    Raises:
        ValueError: If the norm is not finite or is below
            :data:`_MIN_QUAT_NORM`. An all-zero quaternion is the common
            spelling of a dropped or never-written orientation, and it cannot
            be repaired by scaling - every direction is equally consistent with
            it - so it is refused instead of silently standing in for one.
    """
    q = np.asarray(q_xyzw)
    norm = float(np.linalg.norm(np.asarray(q, dtype=np.float64)))
    if not math.isfinite(norm) or norm < _MIN_QUAT_NORM:
        raise ValueError(
            f"{context}: quaternion {np.asarray(q, dtype=np.float64).tolist()} has norm "
            f"{norm!r} and cannot define a rotation. An all-zero or non-finite quaternion "
            "usually means the orientation was never written - supply the body's real "
            "orientation (MuJoCo reports it as wxyz; convert with mujoco_wxyz_to_xyzw)."
        )
    # Dividing by a Python float keeps the caller's dtype (float32 stays
    # float32), which the ONNX session downstream expects.
    return q / norm


def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the INVERSE of the rotation ``q`` encodes.

    Computes ``R(q)^T @ v`` - the standard "rotate by the inverse" that
    expresses a world-frame vector in the body frame.

    Args:
        q_xyzw: Shape ``[4]`` (xyzw). Need not be exactly unit; it is
            normalised internally, so a quaternion scaled by any positive
            factor gives the same answer.
        v: Shape ``[3]`` world-frame vector.

    Returns:
        Shape ``[3]`` local-frame vector.

    Raises:
        ValueError: If ``q_xyzw`` cannot define a rotation (non-finite or
            ~zero norm).
    """
    q = _unit_quat(q_xyzw, context="quat_rotate_inverse")
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_mul(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``xyzw`` quaternions.

    Args:
        a_xyzw: Shape ``[..., 4]``.
        b_xyzw: Shape ``[..., 4]`` (broadcastable with ``a``).

    Returns:
        Product ``a * b``, shape ``[..., 4]``.
    """
    ax, ay, az, aw = a_xyzw[..., 0], a_xyzw[..., 1], a_xyzw[..., 2], a_xyzw[..., 3]
    bx, by, bz, bw = b_xyzw[..., 0], b_xyzw[..., 1], b_xyzw[..., 2], b_xyzw[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_conjugate(q_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse, for unit quats) of an ``xyzw`` quaternion."""
    result = q_xyzw.copy()
    result[..., :3] *= -1.0
    return result


# ---------------------------------------------------------------------------
# State derivation on a full body-rotation array
# ---------------------------------------------------------------------------


def compute_anchor_rot(rigid_body_rot: np.ndarray, anchor_body_index: int) -> np.ndarray:
    """Slice a single anchor-body orientation out of a full body-rotation array.

    Args:
        rigid_body_rot: Shape ``[num_bodies, 4]`` (xyzw).
        anchor_body_index: Row index (e.g. ``16`` for ``torso_link`` on the G1).

    Returns:
        Shape ``[4]`` xyzw anchor orientation.
    """
    return rigid_body_rot[anchor_body_index]


def compute_root_local_ang_vel(
    rigid_body_rot: np.ndarray,
    rigid_body_ang_vel: np.ndarray,
    root_body_index: int = 0,
) -> np.ndarray:
    """Rotate a root-body world-frame angular velocity into the local frame.

    Use ONLY when the source is a WORLD-frame angular velocity (e.g.
    ``data.cvel[root_body, :3]`` from MuJoCo, which is a world-frame twist).
    Do NOT use for a source that is already local (``data.qvel[3:6]`` on a
    freejoint or a real-robot IMU gyro reading is a body-frame quantity - pass
    it straight through).

    Args:
        rigid_body_rot: Shape ``[num_bodies, 4]`` (xyzw).
        rigid_body_ang_vel: Shape ``[num_bodies, 3]`` (world frame).
        root_body_index: Row index of the root (default ``0``).

    Returns:
        Shape ``[3]`` local-frame angular velocity of the root body.

    Raises:
        ValueError: If the root row of ``rigid_body_rot`` cannot define a
            rotation (non-finite or ~zero norm).
    """
    root_rot = rigid_body_rot[root_body_index]
    root_ang_vel = rigid_body_ang_vel[root_body_index]
    return quat_rotate_inverse(root_rot, root_ang_vel)


# ---------------------------------------------------------------------------
# Heading alignment
# ---------------------------------------------------------------------------


def extract_yaw_quat(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract a yaw-only unit quaternion from a full orientation ``q_xyzw``.

    Discards pitch and roll - the residual is a rotation about the world Z axis
    only, which is the natural way to align a motion clip's ground-plane
    heading with the robot's own heading without touching torso lean.

    Args:
        q_xyzw: Shape ``[4]``. Need not be exactly unit; it is normalised
            internally, so a quaternion scaled by any positive factor yields
            the same yaw.

    Returns:
        Shape ``[4]`` yaw-only quaternion ``(0, 0, sin(yaw/2), cos(yaw/2))``.

    Raises:
        ValueError: If ``q_xyzw`` cannot define a rotation (non-finite or
            ~zero norm).
    """
    q = _unit_quat(q_xyzw, context="extract_yaw_quat")
    x, y, z, w = q[0], q[1], q[2], q[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def compute_yaw_offset(robot_quat_xyzw: np.ndarray, motion_quat_xyzw: np.ndarray) -> np.ndarray:
    """Compute the yaw-only offset that maps ``motion`` heading onto ``robot``.

    Both inputs are extracted to yaw-only quaternions first (so torso lean does
    not contaminate the alignment), then the offset is ``robot_yaw *
    motion_yaw^-1`` in Hamilton convention.

    Args:
        robot_quat_xyzw: Shape ``[4]`` - robot's current anchor orientation.
        motion_quat_xyzw: Shape ``[4]`` - motion clip's first-frame anchor
            orientation (a caller uses the same anchor body on both sides).

    Returns:
        Shape ``[4]`` yaw-only quaternion offset - apply with
        :func:`apply_heading_offset` to align a motion body-rot batch.

    Raises:
        ValueError: If either input cannot define a rotation (non-finite or
            ~zero norm).
    """
    robot_yaw = extract_yaw_quat(robot_quat_xyzw)
    motion_yaw = extract_yaw_quat(motion_quat_xyzw)
    return quat_mul(robot_yaw, quat_conjugate(motion_yaw))


def apply_heading_offset(offset_quat_xyzw: np.ndarray, body_rots_xyzw: np.ndarray) -> np.ndarray:
    """Apply a single yaw offset to a batch of body rotations.

    Args:
        offset_quat_xyzw: Shape ``[4]``.
        body_rots_xyzw: Shape ``[..., 4]``.

    Returns:
        Same shape as ``body_rots_xyzw``, each row multiplied on the left by
        ``offset_quat_xyzw``.
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = quat_mul(offset_broadcast, flat)
    return aligned.reshape(original_shape)
