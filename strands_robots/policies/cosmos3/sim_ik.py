"""Inverse-kinematics bridge: Cosmos 3 EE-pose trajectory -> MuJoCo joint targets.

A Cosmos 3 action chunk decodes (see :mod:`action_decode`) to an absolute
end-effector pose trajectory in Cartesian space. MuJoCo arm actuators are
commanded in **joint space**, so closing the sim loop needs an IK step that
maps each Cartesian target to joint angles.

The generic damped-least-squares solver wrapper is the shared
:class:`strands_robots.simulation.ik.MinkIKBridge` (one home for the mink
``FrameTask`` + ``PostureTask`` solve loop; the VERA provider and the
simulation motion primitives use the same class). This module subclasses it
only to brand the install errors with the ``cosmos3-sim`` extra, and keeps the
Cosmos-specific decode glue (:func:`decode_cosmos_chunk_to_targets`) local so a
change to another model's action semantics can never silently break Cosmos.

``mink`` + ``mujoco`` are imported lazily so the ``cosmos3-diffusers`` extra
alone (no sim) stays importable; a missing stack raises an actionable install
error (AGENTS.md key convention #6, no silent default).

This is a geometric post-step applied *after* Cosmos, not part of the model.
The Cosmos "modes" (``policy`` / ``forward_dynamics`` / ``inverse_dynamics``)
are world-model conditioning modes, not robot kinematics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from strands_robots.simulation.ik import MinkIKBridge as _SharedMinkIKBridge
from strands_robots.simulation.ik import resolve_qp_solver

if TYPE_CHECKING:
    from .embodiments import Cosmos3Embodiment

logger = logging.getLogger(__name__)


def _install_hint() -> str:
    """Actionable message when the IK stack (mink + mujoco) is not importable."""
    return (
        "Cosmos 3 IK-to-MuJoCo bridge needs the 'cosmos3-sim' extra (mink + "
        "mujoco), which was not importable. Install it with:\n"
        "  uv pip install strands-robots[cosmos3-sim]\n"
        "This pulls mink (differential IK on the MuJoCo model) and mujoco. It "
        "turns the Cosmos end-effector pose trajectory into joint targets the "
        "MuJoCo arm can track."
    )


_NO_BACKEND_MSG = (
    "No qpsolvers backend is installed; the Cosmos 3 IK bridge needs one "
    "(e.g. 'daqp' or 'quadprog'). Install the cosmos3-sim extra: "
    "uv pip install 'strands-robots[cosmos3-sim]'."
)


def _resolve_qp_solver(requested: str | None) -> str:
    """Pick an installed ``qpsolvers`` backend for ``mink.solve_ik``.

    Delegates to the shared :func:`strands_robots.simulation.ik.resolve_qp_solver`
    with Cosmos 3-branded errors: the install hint and no-backend message name
    the ``cosmos3-sim`` extra so a clean-install user is pointed at the right
    dependency set (AGENTS.md #6 - no silent fallback to an unrequested solver,
    but also no opaque KeyError deep in qpsolvers).
    """
    return resolve_qp_solver(requested, install_hint=_install_hint(), no_backend_msg=_NO_BACKEND_MSG)


class MinkIKBridge(_SharedMinkIKBridge):
    """Differential-IK bridge from EE poses to MuJoCo joint configurations.

    The Cosmos 3 branding of the shared
    :class:`strands_robots.simulation.ik.MinkIKBridge` (same solver, tasks, and
    convergence behavior): a missing ``mink``/``qpsolvers`` stack raises the
    ``cosmos3-sim`` install hint instead of the generic ``sim-mujoco`` one.
    See the shared class for the full constructor/solve contract.
    """

    _INSTALL_HINT: ClassVar[str] = _install_hint()
    _NO_BACKEND_MSG: ClassVar[str] = _NO_BACKEND_MSG
    _LOG_LABEL: ClassVar[str] = "Cosmos3 MinkIKBridge"

    def ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Forward kinematics: ``(4, 4)`` EE pose at a joint configuration.

        Cosmos 3 contract: returns ``float32`` (the dtype the Cosmos decode
        pipeline was built around; the shared bridge returns ``float64``).
        """
        return super().ee_pose(qpos).astype(np.float32)


def decode_cosmos_chunk_to_targets(
    action_chunk: np.ndarray,
    embodiment: Cosmos3Embodiment,
    ik_bridge: MinkIKBridge,
    q_init: np.ndarray,
    *,
    stats: dict[str, np.ndarray] | None = None,
    stats_domain: str | None = None,
    reanchor: bool = True,
) -> dict[str, Any]:
    """Turn a Cosmos 3 raw action chunk into MuJoCo joint targets via IK.

    The full sim-loop bridge for the in-process ``diffusers`` backend, composing
    the three honest steps (no fabricated joint targets): de-normalize ->
    decode relative EE poses to an absolute trajectory -> inverse kinematics.

    1. **De-normalize** the model's ``[-1, 1]`` quantile-normalized action back
       to physical units with the embodiment's bundled ``q01``/``q99`` stats
       (:func:`~strands_robots.policies.cosmos3.action_decode.denormalize_quantile`).
    2. **Decode + IK, step by step, re-anchored** (``reanchor=True``, default).
       Each relative ``[translation(3), rot6d(6)]`` pose delta is composed onto
       the arm's **achieved** end-effector pose (the forward kinematics of the
       previous IK solution), not onto the previous *ideal* target, then solved
       to joints. This mirrors how ``cosmos_framework``'s RoboLab server anchors
       every decode on the observed ``eef_pos``/``eef_quat``
       (``action_policy_server_robolab`` -> ``pose_rel_to_abs(initial_pose=
       current_observed_pose)``). When IK under-tracks a step (workspace edge,
       joint limit), the next target is built from where the arm *actually is*,
       so Cartesian tracking error stays bounded per step instead of compounding
       down the chunk. With ``reanchor=False`` the legacy open-loop decode is
       used (full target trajectory integrated up front via
       :func:`~strands_robots.policies.cosmos3.action_decode.decode_pose_trajectory`,
       then IK each frame) - retained only for comparison/diagnostics.

    Args:
        action_chunk: Raw unified action ``[T, raw_action_dim]`` from the
            diffusers backend (normalized ``[-1, 1]``; last column is grasp for
            gripper embodiments).
        embodiment: Active :class:`Cosmos3Embodiment` (provides ``domain_name``
            for the stats lookup, ``raw_action_layout`` for the gripper column,
            and ``normalization``).
        ik_bridge: A :class:`MinkIKBridge` over the target arm's MuJoCo model.
        q_init: Seed joint configuration (length ``model.nq``) - the robot's
            current pose; the EE trajectory is anchored at its forward
            kinematics and each IK solve warm-starts from the previous step.
        stats: Optional explicit ``{"q01", "q99"}`` stats override. When ``None``
            the bundled per-domain stats are loaded for ``embodiment.domain_name``.
            When supplied, ``stats_domain`` must name the domain they were
            measured on: several Cosmos 3 domains share an action width
            (``umi``, ``droid_lerobot``, ``bridge_orig_lerobot`` and
            ``openarm_lerobot`` are all 10 columns), so the width check in
            :func:`~strands_robots.policies.cosmos3.action_decode.denormalize_quantile`
            cannot tell one domain's quantiles from another's.
        stats_domain: Domain the explicit ``stats`` describe. Required whenever
            ``stats`` is supplied and must equal ``embodiment.domain_name``.
        reanchor: When ``True`` (default) re-anchor each decoded pose delta on
            the arm's achieved EE pose (closed loop), bounding tracking error.
            When ``False`` use the legacy open-loop decode (targets integrated
            up front, then IK) - kept for diagnostics only.

    Returns:
        ``{"qpos": np.ndarray[T, nq], "gripper": np.ndarray[T] | None,
        "poses": np.ndarray[T, 4, 4], "tracking_error": {"mean_mm", "max_mm"}}``.
        ``gripper`` is ``None`` for grasp-less embodiments. ``poses`` is the
        sequence of Cartesian targets actually commanded: when ``reanchor=True``
        each is anchored on the realized EE pose of the prior step.

    Raises:
        ValueError: If ``embodiment.normalization`` is not ``"quantile"`` (the
            only method the current Cosmos 3 domains and bundled stats use), if
            ``stats`` is supplied without ``stats_domain``, or if
            ``stats_domain`` names a domain other than
            ``embodiment.domain_name`` - de-normalizing with another domain's
            quantiles silently rescales every commanded pose delta.
    """
    from .action_decode import (
        decode_pose_delta,
        decode_pose_trajectory,
        denormalize_quantile,
        load_action_stats,
    )

    if embodiment.normalization != "quantile":
        raise ValueError(
            f"decode_cosmos_chunk_to_targets supports normalization='quantile' "
            f"(the bundled Cosmos 3 stats), not {embodiment.normalization!r}."
        )
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk must be [T, D]; got {action_chunk.shape}")

    if stats is None:
        stats = load_action_stats(embodiment.domain_name)
    elif stats_domain is None:
        raise ValueError(
            "decode_cosmos_chunk_to_targets: explicit stats= must declare the domain "
            "they were measured on via stats_domain=. Cosmos 3 domains share action "
            "widths (umi, droid_lerobot, bridge_orig_lerobot and openarm_lerobot are "
            "all 10 columns), so a width check cannot tell one domain's quantiles "
            "from another's, and "
            f"de-normalizing a {embodiment.domain_name!r} action with the wrong "
            "domain's quantiles silently rescales every commanded pose delta. Pass "
            f"stats_domain={embodiment.domain_name!r}."
        )
    elif stats_domain != embodiment.domain_name:
        raise ValueError(
            f"decode_cosmos_chunk_to_targets: stats_domain={stats_domain!r} does not "
            f"describe embodiment {embodiment.name!r} (domain "
            f"{embodiment.domain_name!r}). Quantile stats are per-domain physical "
            "ranges, so using another domain's would rescale every commanded pose "
            "delta while the action width still matches."
        )
    denorm = denormalize_quantile(action_chunk, stats["q01"], stats["q99"])

    # Split off a trailing grasp/gripper column when the layout has one.
    layout = embodiment.raw_action_layout
    has_grasp = bool(layout) and layout[-1] == "grasp"
    pose_block = denorm[:, :-1] if has_grasp else denorm
    gripper = denorm[:, -1] if has_grasp else None

    q0 = np.asarray(q_init, dtype=np.float64)

    if not reanchor:
        # Legacy open-loop: integrate the whole target trajectory up front, then
        # solve IK per frame. Tracking error compounds when IK under-tracks.
        initial_pose = ik_bridge.ee_pose(q0).astype(np.float64)
        abs_poses = decode_pose_trajectory(pose_block, initial_pose, rotation_dim=6)
        target_poses = abs_poses[1:]
        qpos = ik_bridge.solve_trajectory(target_poses, q0)
        return {
            "qpos": qpos,
            "gripper": gripper,
            "poses": target_poses,
            "tracking_error": ik_bridge.tracking_error(target_poses, qpos),
        }

    # Closed-loop re-anchoring: compose each pose delta onto the realized EE pose
    # of the previous IK solve (not the prior ideal target), warm-starting the
    # next solve from that joint config. Mirrors the RoboLab server's
    # observed-pose anchoring so per-step error does not accumulate.
    q = q0.copy()
    achieved = ik_bridge.ee_pose(q).astype(np.float64)
    qpos_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    for step in pose_block:
        delta = decode_pose_delta(step, rotation_dim=6).astype(np.float64)
        target = achieved @ delta
        q = ik_bridge.solve(target, q)
        achieved = ik_bridge.ee_pose(q).astype(np.float64)
        qpos_list.append(q.copy())
        target_list.append(target)
    nq = ik_bridge.model.nq
    qpos = np.stack(qpos_list) if qpos_list else np.empty((0, nq), dtype=np.float64)
    target_poses = np.stack(target_list).astype(np.float32) if target_list else np.empty((0, 4, 4), dtype=np.float32)
    return {
        "qpos": qpos,
        "gripper": gripper,
        "poses": target_poses,
        "tracking_error": ik_bridge.tracking_error(target_poses, qpos),
    }
