"""Pure-NumPy control + quaternion helpers for the WBC policy.

These functions reproduce the math in NVIDIA's GR00T-WholeBodyControl reference
controller (``decoupled_wbc/sim2mujoco``) with no torch / onnxruntime
dependency, so they are unit-testable against hand-computed values on any
machine (issue #466 acceptance criterion: "PD-control + quat helpers match
hand-computed values").

Conventions:
* Quaternions are ``[w, x, y, z]`` (scalar-first), matching MuJoCo's
  ``data.qpos`` free-joint layout and the upstream controller.
* The gravity vector points down: ``g = [0, 0, -1]`` in the world frame.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# World-frame gravity direction (unit, pointing down). The controller feeds the
# *projected* gravity (this vector rotated into the robot base frame) to the
# network as a 3-vector orientation cue.
_GRAVITY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def pd_control(
    target_q: NDArray[np.float64],
    q: NDArray[np.float64],
    kp: NDArray[np.float64],
    target_dq: NDArray[np.float64],
    dq: NDArray[np.float64],
    kd: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-joint PD torque law: ``tau = (target_q - q) * kp + (target_dq - dq) * kd``.

    This is the exact law the upstream reference loop writes to
    ``data.ctrl[:15]`` (torque actuators). All arguments are 1-D arrays of the
    same length (``num_actions``); the return is the same shape.

    Args:
        target_q: Desired joint positions.
        q: Measured joint positions.
        kp: Proportional gains (per joint).
        target_dq: Desired joint velocities (usually zeros for a position hold).
        dq: Measured joint velocities.
        kd: Derivative gains (per joint).

    Returns:
        Joint torques, same shape as the inputs.

    Raises:
        ValueError: If the input arrays do not all share one length.
    """
    arrays = {"target_q": target_q, "q": q, "kp": kp, "target_dq": target_dq, "dq": dq, "kd": kd}
    lengths = {name: np.asarray(a).shape for name, a in arrays.items()}
    n = np.asarray(target_q).shape
    if any(shape != n for shape in lengths.values()):
        raise ValueError(f"pd_control: all inputs must share one shape; got {lengths}")
    return (np.asarray(target_q) - np.asarray(q)) * np.asarray(kp) + (
        np.asarray(target_dq) - np.asarray(dq)
    ) * np.asarray(kd)


def compute_targets(
    default_angles: NDArray[np.float64],
    raw_action: NDArray[np.float64],
    action_scale: float,
) -> NDArray[np.float64]:
    """Convert a raw network action (joint-position offset) to absolute targets.

    ``target_q = default_angles + action_scale * raw_action`` - the exact form
    the upstream reference controller uses before the PD law turns targets into
    torques. Pulled into ``control`` (rather than the policy) so it is covered
    by the same hand-computed unit tests as :func:`pd_control`.

    Args:
        default_angles: Per-joint nominal stance angles.
        raw_action: Raw ONNX output (per-joint offset).
        action_scale: Scalar applied to the offset (upstream ``action_scale``).

    Returns:
        Absolute joint-position targets, same shape as the inputs.

    Raises:
        ValueError: If ``default_angles`` and ``raw_action`` differ in shape.
    """
    d = np.asarray(default_angles, dtype=np.float64)
    r = np.asarray(raw_action, dtype=np.float64)
    if d.shape != r.shape:
        raise ValueError(f"compute_targets: default_angles {d.shape} and raw_action {r.shape} must match")
    return d + float(action_scale) * r


def quat_rotate_inverse(quat_wxyz: NDArray[np.float64], vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate ``vec`` from the world frame into the body frame given ``quat``.

    Computes ``R(q)^T @ vec`` where ``R(q)`` is the rotation matrix of the
    scalar-first quaternion ``q = [w, x, y, z]``. This is the standard
    "rotate by the inverse" used to express a world-frame vector in the body
    frame, matching the upstream controller's ``quat_rotate_inverse``.

    Args:
        quat_wxyz: Body orientation quaternion ``[w, x, y, z]`` (need not be
            exactly unit; it is normalised internally).
        vec: World-frame 3-vector.

    Returns:
        The 3-vector expressed in the body frame.

    Raises:
        ValueError: If ``quat_wxyz`` is not length 4 or ``vec`` is not length 3,
            or the quaternion has ~zero norm.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quat_rotate_inverse: quat must be length-4 [w,x,y,z], got shape {q.shape}")
    if v.shape != (3,):
        raise ValueError(f"quat_rotate_inverse: vec must be length-3, got shape {v.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quat_rotate_inverse: quaternion has ~zero norm; cannot normalise")
    q = q / norm
    w, x, y, z = q

    # Standard derivation (Rodrigues form used by IsaacGym/Lab and the upstream
    # WBC controller). For inverse rotation the cross-product term flips sign
    # relative to the forward rotation.
    q_vec = np.array([x, y, z], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def projected_gravity(quat_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the world gravity direction expressed in the body frame.

    This is the 3-vector orientation cue the WBC network consumes at
    observation indices ``[10:13]`` (upstream). For an upright base it is
    approximately ``[0, 0, -1]``; tilting the base rotates the vector,
    giving the policy a gravity-aligned attitude signal without a full IMU.

    Args:
        quat_wxyz: Body orientation quaternion ``[w, x, y, z]``.

    Returns:
        Gravity direction in the body frame (unit 3-vector).
    """
    return quat_rotate_inverse(quat_wxyz, _GRAVITY_DIR)


__all__ = ["pd_control", "compute_targets", "quat_rotate_inverse", "projected_gravity"]
