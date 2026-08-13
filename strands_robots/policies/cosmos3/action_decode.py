"""Decode a Cosmos 3 raw unified-action chunk into an absolute EE-pose trajectory.

The in-process ``diffusers`` backend (:mod:`policy_diffusers`) returns the
model's **raw unified action** of width ``embodiment.raw_action_dim`` -
``[tx, ty, tz, r0..r5, grasp]`` for the DROID/Franka domain - which is

* **quantile-normalized to ``[-1, 1]``** (the model's output space), and
* a **relative end-effector pose delta** per step (3D translation + 6D rotation
  of Zhou et al. 2019), encoded ``backward_framewise``,

*not* joint radians. Feeding it straight into MuJoCo joint actuators is
physically meaningless (normalized values land arbitrarily inside/outside real
joint limits). This module turns that chunk into a usable absolute Cartesian
trajectory in two pure-NumPy steps that mirror ``cosmos_framework`` exactly:

1. :func:`denormalize_quantile` - invert the quantile normalization with the
   embodiment's bundled ``q01``/``q99`` action stats
   (``0.5 * (a + 1) * (q99 - q01) + q01``; see
   ``cosmos_framework.data.vfm.action.action_normalization.denormalize_action``).
2. :func:`decode_pose_trajectory` - integrate the per-step relative EE-pose
   deltas into an absolute ``(T+1, 4, 4)`` SE3 trajectory anchored at the
   robot's current pose (the ``midtrain`` decode the RoboLab server applies in
   ``action_policy_server_robolab.RobolabPolicyService.infer``).

The resulting absolute poses are then handed to :class:`~strands_robots.policies.cosmos3.sim_ik.MinkIKBridge`
for inverse kinematics to MuJoCo joint targets. No ``cosmos_framework`` import
is needed (it runs server-side, in its own env); the decode is reimplemented
here against the same conventions so the ``diffusers`` backend can close the sim
loop in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Bundled per-embodiment action normalization stats (q01/q99), copied verbatim
# from cosmos_framework/data/vfm/action/datasets/stats/. Tiny JSON files; keyed
# by the embodiment ``domain_name``.
_STATS_DIR = Path(__file__).parent / "stats"


def load_action_stats(domain_name: str) -> dict[str, np.ndarray]:
    """Load bundled quantile action stats (``q01``/``q99``) for a domain.

    Args:
        domain_name: Cosmos 3 conditioning domain (e.g. ``"droid_lerobot"``).

    Returns:
        ``{"q01": np.ndarray[D], "q99": np.ndarray[D]}`` for the domain.

    Raises:
        FileNotFoundError: If no bundled stats file exists for the domain. The
            message lists the domains that *are* bundled so the caller can pick a
            supported embodiment or supply stats explicitly (no silent default).
    """
    path = _STATS_DIR / f"{domain_name}_stats.json"
    if not path.exists():
        available = sorted(p.name.replace("_stats.json", "") for p in _STATS_DIR.glob("*_stats.json"))
        raise FileNotFoundError(
            f"No bundled Cosmos 3 action stats for domain {domain_name!r}. "
            f"Bundled domains: {available}. The de-normalization quantiles "
            "(q01/q99) are required to turn the model's [-1, 1] action into "
            "physical EE-pose deltas. Either use a bundled embodiment, or pass "
            "this domain's own quantiles to decode_cosmos_chunk_to_targets as "
            f"stats=... together with stats_domain={domain_name!r} - another "
            "domain's stats are not a substitute even when the action width "
            "matches."
        )
    with path.open("r") as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items() if k in ("q01", "q99")}


def denormalize_quantile(action: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Invert Cosmos 3 quantile normalization: ``[-1, 1]`` -> physical units.

    Mirrors ``cosmos_framework`` ``denormalize_action(method="quantile")``::

        denorm = 0.5 * (action + 1.0) * (q99 - q01) + q01

    Args:
        action: Normalized action of shape ``[..., D]`` (values nominally in
            ``[-1, 1]``).
        q01: Per-column 1st-percentile stat of shape ``[D]``.
        q99: Per-column 99th-percentile stat of shape ``[D]``.

    Returns:
        De-normalized action with the same shape as ``action`` (``float32``).

    Raises:
        ValueError: If the action's last dim does not match the stats width.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != q01.shape[-1] or action.shape[-1] != q99.shape[-1]:
        raise ValueError(
            f"action width {action.shape[-1]} does not match stats width "
            f"q01={q01.shape[-1]} q99={q99.shape[-1]}; the de-normalization "
            "stats must describe every action column."
        )
    return (0.5 * (action + 1.0) * (q99 - q01) + q01).astype(np.float32)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode a 6D rotation (Zhou et al. 2019) to a ``(3, 3)`` rotation matrix.

    The 6D vector stores the first two columns ``a, b`` of the rotation matrix;
    the third is ``a x b``. Matches ``cosmos_framework`` ``convert_rotation``'s
    ``rot6d`` -> ``matrix`` branch (column-major ``stack((c0, c1, c2), axis=-1)``).
    """
    col0 = rot6d[:3]
    col1 = rot6d[3:6]
    col2 = np.cross(col0, col1)
    return np.stack((col0, col1, col2), axis=-1).astype(np.float32)


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Project an approximate ``(3, 3)`` matrix onto ``SO(3)`` via SVD.

    Mirrors ``cosmos_framework`` ``pose_utils._normalize_rotation_matrices``: the
    decoded ``rot6d`` columns are not perfectly orthonormal, so the RoboLab
    server normalizes with ``normalize_rotation=True`` before composing the
    trajectory. We replicate that (``U @ Vt`` with a determinant-+1 guard) so the
    in-process decode matches the server's absolute poses.
    """
    u, _, vt = np.linalg.svd(matrix)
    proj = u @ vt
    if np.linalg.det(proj) < 0:
        u = u.copy()
        u[:, -1] *= -1
        proj = u @ vt
    return proj.astype(np.float32)


def decode_pose_delta(pose_step: np.ndarray, *, rotation_dim: int = 6) -> np.ndarray:
    """Decode one de-normalized relative-pose action step into a ``(4, 4)`` delta.

    A single row of the de-normalized pose block (``[translation(3),
    rotation(rotation_dim)]``) becomes the homogeneous relative transform
    ``delta_T`` that the ``backward_framewise`` convention composes onto the
    running pose (``T_{i+1} = T_i @ delta_T``; see :func:`decode_pose_trajectory`
    and ``cosmos_framework`` ``pose_utils._pose_vector_to_delta_transform``).

    Exposed as a primitive so a closed-loop sim bridge can re-anchor each step on
    the arm's *achieved* end-effector pose (the
    :func:`~strands_robots.policies.cosmos3.sim_ik.decode_cosmos_chunk_to_targets`
    fix for compounding Cartesian tracking error), rather than only as part of
    the open-loop full-trajectory decode.

    Args:
        pose_step: A single de-normalized relative-pose action of shape
            ``[3 + rotation_dim]`` (translation block + rotation block).
        rotation_dim: Width of the rotation block (6 for ``rot6d``).

    Returns:
        The ``(4, 4)`` homogeneous relative transform (``float32``).

    Raises:
        ValueError: If ``pose_step`` is not 1-D of length ``3 + rotation_dim``.
    """
    pose_step = np.asarray(pose_step, dtype=np.float32)
    if pose_step.ndim != 1 or pose_step.shape[0] != 3 + rotation_dim:
        raise ValueError(
            f"pose_step must be [{3 + rotation_dim}] (3 translation + {rotation_dim} rotation); got {pose_step.shape}"
        )
    delta = np.eye(4, dtype=np.float32)
    delta[:3, 3] = pose_step[:3]
    delta[:3, :3] = _project_to_so3(_rot6d_to_matrix(pose_step[3 : 3 + rotation_dim]))
    return delta


def decode_pose_trajectory(
    pose_chunk: np.ndarray,
    initial_pose: np.ndarray,
    *,
    rotation_dim: int = 6,
) -> np.ndarray:
    """Integrate per-step relative EE-pose deltas into an absolute SE3 trajectory.

    The de-normalized DROID action's pose block is ``[translation(3),
    rot6d(6)]`` encoded ``backward_framewise``: each step is a relative transform
    ``delta_T`` composed onto the running pose as ``T_{i+1} = T_i @ delta_T``
    (mirrors ``cosmos_framework`` ``pose_utils.pose_rel_to_abs`` with
    ``rotation_format="rot6d"``, ``pose_convention="backward_framewise"``,
    ``normalize_rotation=True``).

    Args:
        pose_chunk: De-normalized relative-pose action of shape ``[T, 3 +
            rotation_dim]`` (translation block + rotation block; the trailing
            gripper column, if any, must be sliced off by the caller).
        initial_pose: Absolute ``(4, 4)`` EE pose for the first frame (the
            robot's current end-effector pose). The trajectory is anchored here
            so the IK targets are reachable from the current configuration.
        rotation_dim: Width of the rotation block (6 for ``rot6d``).

    Returns:
        Absolute poses of shape ``[T + 1, 4, 4]`` (``float32``); index 0 is
        ``initial_pose`` and indices ``1..T`` are the predicted trajectory.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    pose_chunk = np.asarray(pose_chunk, dtype=np.float32)
    if pose_chunk.ndim != 2 or pose_chunk.shape[1] != 3 + rotation_dim:
        raise ValueError(
            f"pose_chunk must be [T, {3 + rotation_dim}] (3 translation + "
            f"{rotation_dim} rotation); got {pose_chunk.shape}"
        )
    initial_pose = np.asarray(initial_pose, dtype=np.float32)
    if initial_pose.shape != (4, 4):
        raise ValueError(f"initial_pose must be (4, 4); got {initial_pose.shape}")

    poses = [initial_pose]
    current = initial_pose
    for step in pose_chunk:
        delta = decode_pose_delta(step, rotation_dim=rotation_dim)
        current = (current @ delta).astype(np.float32)
        poses.append(current)
    return np.stack(poses).astype(np.float32)
