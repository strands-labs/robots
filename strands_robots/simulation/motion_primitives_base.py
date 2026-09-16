"""Backend-agnostic core of the motion primitives - ``move_to`` / ``set_gripper`` / ``rotate_wrist``.

The primitives were introduced for the MuJoCo backend (GH #1645, see
:mod:`strands_robots.simulation.mujoco.motion_primitives` for the full
agent-facing contract). This module is the half that never touches an engine
(extracted per GH #2153, step 1 of the Isaac parity work GH #2123): the
parameter domains, the registry gripper-metadata contract, the shared name
heuristics, and the success / timeout envelope builders. Everything here
operates on plain values so backend adapters cannot drift apart on wording or
payload shape - an agent reading a ``move_to`` refusal sees the same sentence
whichever backend produced it.

What stays in a backend adapter
(:class:`strands_robots.simulation.mujoco.motion_primitives.MotionPrimitivesMixin`
today): robot/world resolution against the backend's own state, actuation
(``data.ctrl`` writes and the ``mj_step`` tick loop), the kinematics source
(mink IK on the compiled ``MjModel``), the actuator-name half of gripper
resolution, and the mid-run abort checks, which read backend-owned state.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from strands_robots.registry.robots import get_robot
from strands_robots.utils import (
    coerce_orientation_quaternion,
    coerce_pose_vector,
    refusal_repr,
)

# Name hints (lowercased substring match on the gripper DOF's name) used to
# resolve the gripper when the robot registry carries no gripper metadata for
# the robot's ``data_config``. Matches the existing runtime precedent (the
# cosmos3 policy uses gripper|finger; SO-100's gripper joint is the
# "Jaw"). Registry metadata (``robots.json`` -> ``<robot>.gripper``, GH #1658)
# is authoritative when present; the heuristic is the zero-config fallback for
# user URDFs / injected MJCF, and an unresolvable gripper returns a structured
# error listing the candidates so the agent can fall back to send_action.
_GRIPPER_HINTS = ("gripper", "finger", "jaw")

# Valid values for the registry gripper metadata ``closed`` / ``open`` fields:
# which END of the gripper's set-point range the state maps to. The registry
# integrity tests shape-check the shipped robots.json against the same two
# values.
_CTRLRANGE_ENDS = ("low", "high")

# Wrist-yaw joint hints, most-specific first. Fallback: the last non-gripper
# hinge joint in the robot's chain (the distal roll joint on most serial
# arms). "Non-gripper" is decided by the shared registry-metadata-first
# classification, not by _GRIPPER_HINTS alone.
_WRIST_HINTS = ("wrist_roll", "wrist_yaw", "wrist_rotate", "wrist")

# Hard ceiling on max_steps / steps to prevent unbounded primitive runtime.
_MAX_PRIMITIVE_STEPS = 10_000

# Workspace sanity radius: a move_to target further than this from the robot's
# base is rejected up front (meters). Generous on purpose - it guards against
# unit mistakes (mm vs m), not reachability; true reachability is checked by
# the kinematics residual.
_WORKSPACE_SANITY_RADIUS_M = 5.0

# Deterministic IK restart seeds tried when the direct solve from the live
# configuration stalls in a local minimum (see move_to). Bounded so the worst
# case is still a sub-second solve budget.
_IK_RESTART_SEEDS = 8

# Default orientation convergence tolerance for ``move_to`` (radians), applied
# when the caller supplies an ``orientation`` target and no explicit
# ``orientation_tol``. A pose target is two independent quantities, so the
# rotational half needs its own domain: ``tol`` is documented in METERS and
# cannot bound an angle, and gating a pose request on the translation alone
# accepts a solve that reached the point with the wrong wrist orientation.
#
# The value clears the compliant-servo floor with margin. The primitives drive
# position servos under gravity, which settle to a small steady-state
# rotational offset that no further stepping removes; held to steady state on
# the shipped models that floor is 0.0123 rad (0.70 deg) for panda (7 DOF) and
# 0.0343 rad (1.97 deg) for so101 (5 DOF). 0.1 rad (5.7 deg) is roughly 3x the
# worse of the two, so a realizable orientation is never refused for servo
# compliance, while an orientation the arm cannot realize still fails the gate.
_DEFAULT_ORIENTATION_TOL_RAD = 0.1


def _is_finite_real(value: Any) -> bool:
    """True when ``value`` is a real, finite scalar (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _quat_angle_error(target_wxyz: Any, actual_wxyz: Any) -> float:
    """Rotation angle between two ``[w, x, y, z]`` quaternions, in radians.

    Both inputs are normalized first, and the double-cover sign ambiguity is
    resolved by taking ``abs`` of the dot product: ``q`` and ``-q`` denote the
    SAME orientation, so an ``arccos`` of the signed dot would report ``pi``
    for two identical rotations and make an exactly-met orientation look
    maximally wrong. The result is the absolute geodesic angle in ``[0, pi]``.

    Returns ``0.0`` when either input has ~zero norm; ``move_to`` refuses such
    a target up front (:meth:`MotionPrimitivesCore._validate_move_to_args`), so
    this only keeps the helper total for callers that measure a readback.
    """
    target = np.asarray(target_wxyz, dtype=np.float64)
    actual = np.asarray(actual_wxyz, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    actual_norm = float(np.linalg.norm(actual))
    if target_norm < 1e-8 or actual_norm < 1e-8:
        return 0.0
    dot = abs(float(np.dot(target / target_norm, actual / actual_norm)))
    return float(2.0 * math.acos(min(1.0, dot)))


OBSTRUCTION_MAX_CONTACTS = 3
"""Contacts named in a not-reached ``move_to`` refusal; the total is still reported."""

JOINT_LIMIT_MARGIN_FRACTION = 0.01
"""A commanded joint within this fraction of its range (floor 1e-3) of a bound counts as at the limit."""


def _err(text: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structured error tool-result, optionally with a json details block."""
    content: list[dict[str, Any]] = [{"text": text}]
    if payload is not None:
        content.append({"json": payload})
    return {"status": "error", "content": content}


class MotionPrimitivesCore:
    """Backend-agnostic half of the motion primitives.

    Pure helpers only: nothing here reads engine state, takes locks, or holds
    attributes, so the class mixes into any simulation backend. The
    parameter-domain wording and the result payloads live here precisely so
    every backend answers identically (AGENTS.md: "Match docstrings to
    semantics").
    """

    # -- parameter validation -------------------------------------------------

    @staticmethod
    def _validate_step_budget(action: str, param: str, value: Any) -> dict[str, Any] | None:
        """Validate an integer control-tick budget (1..``_MAX_PRIMITIVE_STEPS``)."""
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            return _err(f"{action}: '{param}' must be an integer, got {type(value).__name__}.")
        if not (1 <= int(value) <= _MAX_PRIMITIVE_STEPS):
            return _err(f"{action}: '{param}' must be between 1 and {_MAX_PRIMITIVE_STEPS}, got {int(value)}.")
        return None

    def _validate_move_to_args(
        self,
        position: Any,
        orientation: Any,
        tol: Any,
        max_steps: Any,
        orientation_tol: Any = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int, float | None, dict[str, Any] | None]:
        """Shared ``move_to`` parameter domain (before any engine state is read).

        Returns ``(target, orientation, max_steps, orientation_tol, None)`` on
        success - ``target`` a float64 ``(3,)`` array, ``orientation`` a float64
        ``(4,)`` array or ``None`` when omitted, ``orientation_tol`` the
        resolved rotational tolerance in radians or ``None`` when no
        orientation was requested - or ``(None, None, 0, None, error)`` when a
        parameter is off-domain. The pose vectors are validated by the same
        rule the scene-construction calls use
        (:func:`strands_robots.utils.coerce_pose_vector`): three/four finite
        real components, a NumPy array accepted, a ``bool`` refused.

        ``orientation_tol`` without an ``orientation`` target is REFUSED rather
        than ignored: the value would have no quantity to bound, and silently
        dropping a tolerance the caller set is how an unmet request goes
        unnoticed. When an orientation IS given and no tolerance is passed the
        default is :data:`_DEFAULT_ORIENTATION_TOL_RAD`.
        """
        if position is None:
            return None, None, 0, None, _err("move_to requires 'position' ([x, y, z] target in meters).")
        # Same guard the scene-construction entry points use, so a pose
        # `add_object`/`move_object` refuses is refused here too. `len()` on a
        # value with no length (a scalar, an iterator) raises a bare TypeError,
        # which would escape the primitives' documented never-raises contract.
        position, pos_err = coerce_pose_vector("move_to", "position", position, 3)
        if pos_err is not None:
            return None, None, 0, None, _err(pos_err)
        assert position is not None  # non-None input yields a non-None result
        orientation, quat_err = coerce_orientation_quaternion("move_to", "orientation", orientation)
        if quat_err is not None:
            return None, None, 0, None, _err(quat_err)
        if not _is_finite_real(tol) or float(tol) <= 0.0:
            return (
                None,
                None,
                0,
                None,
                _err(f"move_to: 'tol' must be a positive number of meters, got {refusal_repr(tol)}."),
            )
        if orientation_tol is not None and orientation is None:
            return (
                None,
                None,
                0,
                None,
                _err(
                    "move_to: 'orientation_tol' only bounds an 'orientation' target, and none was "
                    "given. Pass orientation=[w, x, y, z] to command a full pose, or drop "
                    "orientation_tol - a position-only solve has no orientation to converge."
                ),
            )
        if orientation_tol is not None and (not _is_finite_real(orientation_tol) or float(orientation_tol) <= 0.0):
            return (
                None,
                None,
                0,
                None,
                _err(
                    f"move_to: 'orientation_tol' must be a positive number of radians, got {refusal_repr(orientation_tol)}."
                ),
            )
        err = self._validate_step_budget("move_to", "max_steps", max_steps)
        if err is not None:
            return None, None, 0, None, err
        target = np.asarray(position, dtype=np.float64)
        quat = None if orientation is None else np.asarray(orientation, dtype=np.float64)
        resolved_orientation_tol = (
            None
            if quat is None
            else (_DEFAULT_ORIENTATION_TOL_RAD if orientation_tol is None else float(orientation_tol))
        )
        return target, quat, int(max_steps), resolved_orientation_tol, None

    def _validate_set_gripper_args(self, state: Any, steps: Any) -> tuple[int, dict[str, Any] | None]:
        """Shared ``set_gripper`` parameter domain: ``(steps, None)`` or ``(0, error)``."""
        if state not in ("open", "close"):
            return 0, _err(f'set_gripper: \'state\' must be "open" or "close", got {refusal_repr(state)}.')
        err = self._validate_step_budget("set_gripper", "steps", steps)
        if err is not None:
            return 0, err
        return int(steps), None

    def _validate_rotate_wrist_args(
        self, target_yaw: Any, tol: Any, max_steps: Any
    ) -> tuple[float, int, dict[str, Any] | None]:
        """Shared ``rotate_wrist`` parameter domain: ``(target_yaw, max_steps, None)`` or ``(0, 0, error)``."""
        if target_yaw is None:
            return 0.0, 0, _err("rotate_wrist requires 'target_yaw' (wrist joint set-point in radians).")
        if not _is_finite_real(target_yaw):
            return (
                0.0,
                0,
                _err(f"rotate_wrist: 'target_yaw' must be a finite number of radians, got {refusal_repr(target_yaw)}."),
            )
        if not _is_finite_real(tol) or float(tol) <= 0.0:
            return 0.0, 0, _err(f"rotate_wrist: 'tol' must be a positive number of radians, got {refusal_repr(tol)}.")
        err = self._validate_step_budget("rotate_wrist", "max_steps", max_steps)
        if err is not None:
            return 0.0, 0, err
        return float(target_yaw), int(max_steps), None

    @staticmethod
    def _workspace_sanity_error(robot_name: str, target: np.ndarray, base: np.ndarray) -> dict[str, Any] | None:
        """Reject a ``move_to`` target outside the workspace sanity box.

        A unit-mistake guard (mm vs m), not a reachability check - see
        :data:`_WORKSPACE_SANITY_RADIUS_M`. ``base`` is the robot's base /
        spawn position in world coordinates, from whichever source the
        backend owns it in.
        """
        base_dist = float(np.linalg.norm(target - np.asarray(base, dtype=np.float64)))
        if base_dist > _WORKSPACE_SANITY_RADIUS_M:
            return _err(
                f"move_to: target {target.tolist()} is {base_dist:.2f} m from robot "
                f"'{robot_name}' base - outside the {_WORKSPACE_SANITY_RADIUS_M:.0f} m workspace "
                "sanity box. Check units (meters, world frame)."
            )
        return None

    # -- registry gripper metadata (GH #1658) ---------------------------------

    @staticmethod
    def _get_registry_robot(data_config: str) -> dict[str, Any] | None:
        """Registry lookup seam for :meth:`_registry_gripper_metadata`.

        Resolves through this module's global ``get_robot``. A backend adapter
        may override it to keep its historical patch point alive - the MuJoCo
        mixin does, so patching
        ``strands_robots.simulation.mujoco.motion_primitives.get_robot`` keeps
        working for tests and user code.
        """
        return get_robot(data_config)

    def _registry_gripper_metadata(self, robot: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Registry ``gripper`` block for *robot*'s ``data_config``, shape-checked.

        Returns ``(metadata, None)`` when the robot's ``data_config`` resolves
        to a registry entry with a well-formed ``gripper`` block,
        ``(None, None)`` when there is no metadata (heuristic fallback
        applies), and ``(None, reason)`` when a block exists but is malformed
        (possible via the user-local registry overlay; the shipped
        ``robots.json`` is shape-checked by tests). A malformed block is a
        loud error, never a silent heuristic fallback - half-applying it
        could reintroduce the silent-DOF bug this metadata exists to fix.
        """
        data_config = getattr(robot, "data_config", None)
        if not data_config:
            return None, None
        info = self._get_registry_robot(data_config)
        if not info or "gripper" not in info:
            return None, None
        meta = info["gripper"]
        actuators = meta.get("actuators") if isinstance(meta, dict) else None
        closed_end = meta.get("closed", "low") if isinstance(meta, dict) else None
        open_end = meta.get("open", "high") if isinstance(meta, dict) else None
        if (
            isinstance(actuators, list)
            and actuators
            and all(isinstance(a, str) and a for a in actuators)
            and closed_end in _CTRLRANGE_ENDS
            and open_end in _CTRLRANGE_ENDS
            and closed_end != open_end
        ):
            return meta, None
        return None, (
            f"registry gripper metadata for data_config '{data_config}' is malformed: {meta!r}. "
            "Expected {'actuators': [non-empty names], 'closed': 'low'|'high', "
            "'open': 'low'|'high'} with 'closed' != 'open'. Fix the registry entry "
            "(user overlay: user_robots.json), or drive the gripper directly with "
            "action='send_action'."
        )

    @staticmethod
    def _gripper_state_end(state: str, meta: dict[str, Any] | None) -> str:
        """Which END of the set-point range ``state`` maps to (``"low"``/``"high"``).

        The registry metadata's ``closed``/``open`` fields when present, else
        the open=HIGH / close=LOW convention (correct for SO-100/SO-101 and
        Franka, but a convention, not a law - the metadata field exists to
        remove that sign trap for robots with an inverted gripper).
        """
        if meta is not None:
            return str(meta.get("open", "high") if state == "open" else meta.get("closed", "low"))
        return "high" if state == "open" else "low"

    # -- shared pose-error measurement ----------------------------------------

    @staticmethod
    def _pose_violation(
        position_error: float,
        tol: float,
        orientation_error: float | None = None,
        orientation_tol: float | None = None,
    ) -> float:
        """Worst of the pose error's two components, normalized by its tolerance.

        ``move_to`` has to rank candidate IK solutions and decide convergence
        against a target that is up to TWO independent quantities. Comparing
        the raw numbers is meaningless (metres against radians), so each
        component is divided by its own tolerance and the worse ratio wins:
        the result is ``<= 1.0`` exactly when EVERY requested component is
        within tolerance, which makes one scalar usable both as the accept
        gate and as the "is this restart better" ordering.

        With no orientation requested this degenerates to
        ``position_error / tol``, i.e. ranking by the position residual - the
        position-only behaviour, unchanged.
        """
        violation = position_error / tol
        if orientation_error is not None and orientation_tol is not None:
            violation = max(violation, orientation_error / orientation_tol)
        return float(violation)

    # -- shared result envelopes ----------------------------------------------
    @staticmethod
    def _move_to_unreachable_error(
        robot_name: str,
        target: np.ndarray,
        tol: float,
        *,
        ik_residual: float,
        frame_name: str,
        frame_type: str,
        orientation_tol: float | None = None,
        ik_orientation_residual: float | None = None,
        position_only_residual: float | None = None,
        unrestricted_residual: float | None = None,
        uncommanded_joints: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Refusal for a ``move_to`` target the kinematics cannot reach.

        Two independent things put a target out of reach, and the refusal names
        whichever one applies instead of always advising a closer point.

        *Which half of a pose is short.* A damped least-squares solve given a
        full pose trades position against orientation, so the residual alone
        cannot say WHICH half is out of reach - on an arm with too few DOF to
        realize the rotation it satisfies the ROTATION and leaves the position
        short, so a position-keyed message blames the point when the point is
        fine. ``position_only_residual`` (the same target solved with the
        orientation task switched off) settles that, and it selects the remedy:
        omitting ``orientation`` is the fix when the position alone is
        reachable, and no tolerance change is.

        *Whose reach is short - the robot's, or only this primitive's.*
        ``ik_residual`` is measured on a solve restricted to the joints the
        primitive commands, so it is the error the servo descent would actually
        be left with. ``unrestricted_residual`` is the same target solved with
        the whole model free: when that one fits ``tol`` the target is inside
        the reach of the ROBOT but outside the reach of this PRIMITIVE, and the
        remedy is to move the borrowed degrees of freedom rather than to pick a
        closer point or loosen the tolerance.

        The two are reported independently, so a caller gets whichever
        diagnosis its backend could measure. With no orientation requested the
        wording is the position-only one.

        Args:
            robot_name: Robot the target was requested for.
            target: The requested Cartesian target, in the frame the message
                reports it in (world for both MuJoCo and Isaac).
            tol: Position tolerance the request was judged against (m).
            ik_residual: Best residual over the commanded joints (m).
            frame_name: End-effector frame the solve tracked.
            frame_type: ``"site"`` / ``"body"`` / ``"geom"``.
            orientation_tol: Orientation tolerance (rad), or ``None`` for a
                position-only request - which selects the position-only
                wording.
            ik_orientation_residual: Best orientation residual (rad), when an
                orientation was requested.
            position_only_residual: The same target solved with the orientation
                task off (m), when the backend could measure it.
            unrestricted_residual: The same target solved with every model DOF
                free (m), when the backend could measure it.
            uncommanded_joints: Uncommanded joints that unrestricted solve
                moved, as named by :meth:`_uncommanded_joints_moved`.

        Returns:
            The structured error envelope, json block included.
        """
        payload: dict[str, Any] = {
            "reached": False,
            "steps": 0,
            "ik_residual_m": ik_residual,
            "frame": frame_name,
            "frame_type": frame_type,
        }
        if orientation_tol is not None:
            payload["orientation_tol_rad"] = orientation_tol
            payload["ik_orientation_residual_rad"] = ik_orientation_residual
        if position_only_residual is not None:
            payload["position_only_ik_residual_m"] = position_only_residual
        if unrestricted_residual is not None:
            payload["unrestricted_ik_residual_m"] = unrestricted_residual
            payload["uncommanded_joints_moved"] = list(uncommanded_joints)

        # A borrowed-DOF solve only explains the miss when it BOTH clears the
        # tolerance and needed a joint this primitive does not command; an
        # unmeasurable one (bridge unavailable) reports math.inf and so cannot.
        borrowed_dofs_would_reach = (
            bool(uncommanded_joints) and unrestricted_residual is not None and unrestricted_residual <= float(tol)
        )
        borrowed_dof_text = ""
        if borrowed_dofs_would_reach:
            borrowed_dof_text = (
                f" The same target solves to {unrestricted_residual:.4f} m once the "
                f"{len(uncommanded_joints)} degree(s) of freedom move_to does not command are "
                f"free too ({', '.join(uncommanded_joints)}), so the point is not outside the "
                "robot's workspace: reaching it needs motion this primitive cannot produce. "
                f"move_to drives the {frame_type} '{frame_name}' with position servos only, so "
                "move those degrees of freedom first (a mobile base has to drive there), then "
                "call move_to."
            )

        if orientation_tol is None:
            text = (
                f"move_to: target {target.tolist()} is unreachable for '{robot_name}' EE "
                f"({frame_type} '{frame_name}') within tol={float(tol)} m - the best solve over "
                f"the joints move_to commands leaves a residual of {ik_residual:.4f} m."
            )
            text += borrowed_dof_text if borrowed_dofs_would_reach else " Choose a closer target or loosen tol."
            return _err(text, payload)

        missed = []
        if ik_residual > float(tol):
            missed.append(f"position by {ik_residual:.4f} m (tol={float(tol)} m)")
        if ik_orientation_residual is not None and ik_orientation_residual > float(orientation_tol):
            missed.append(
                f"orientation by {ik_orientation_residual:.4f} rad (orientation_tol={float(orientation_tol)} rad)"
            )
        missed_text = " and ".join(missed) if missed else "the requested pose"
        text = (
            f"move_to: the requested POSE is not achievable for '{robot_name}' EE "
            f"({frame_type} '{frame_name}') - the best IK solution misses {missed_text}."
        )
        if position_only_residual is not None and position_only_residual <= float(tol):
            text += (
                f" The position {target.tolist()} on its own IS reachable "
                f"(residual {position_only_residual:.4f} m): it is the orientation that does not "
                "fit. Omit 'orientation' for a position-only solve, or command this pose on an "
                "arm with enough DOF to realize it. Loosening 'tol' would only accept a solve "
                "that still points the wrong way."
            )
        elif borrowed_dofs_would_reach:
            text += borrowed_dof_text
        else:
            text += (
                f" The position {target.tolist()} is out of reach as well"
                + (
                    f" (position-only residual {position_only_residual:.4f} m)"
                    if position_only_residual is not None
                    else ""
                )
                + ". Choose a closer target."
            )
        return _err(text, payload)

    @staticmethod
    def _move_to_result(
        robot_name: str,
        target: np.ndarray,
        tol: float,
        max_steps: int,
        *,
        reached: bool,
        steps_used: int,
        position_error: float,
        ik_residual: float,
        ee_pos: Any,
        ee_quat: Any,
        frame_name: str,
        frame_type: str,
        orientation_error: float | None = None,
        orientation_tol: float | None = None,
        ik_orientation_residual: float | None = None,
        obstruction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Success / not-reached envelope for ``move_to``, shared across backends.

        ``orientation_error`` / ``orientation_tol`` /
        ``ik_orientation_residual`` are the rotational half of the pose error
        and are ``None`` for a position-only call, which has no orientation to
        report. When they are present ``reached`` must already account for
        them - this builder reports the measurement, it does not decide it.

        ``obstruction`` is what the engine saw at the final servo tick, in the
        shape :meth:`_obstruction_text` reads (``contacts`` on the robot and
        ``joints_at_limit`` among the commanded joints). It is only read on
        the not-reached path, where it turns "the pose fights joint
        limits/contacts" into the name of the contact or the joint that
        stopped the servo. ``None`` means the engine did not look; an empty
        report means it looked and found nothing, which is itself an answer
        (the servo just needs more steps).
        """
        payload: dict[str, Any] = {
            "reached": reached,
            "steps": steps_used,
            "position_error_m": position_error,
            "ik_residual_m": ik_residual,
            "ee_position": [float(v) for v in ee_pos],
            "ee_orientation_wxyz": [float(v) for v in ee_quat],
            "frame": frame_name,
            "frame_type": frame_type,
        }
        # Both halves of the rotational measurement travel together: an
        # orientation error without the tolerance it was judged against is not
        # a reportable number, so a caller cannot see one without the other.
        pose_detail = ""
        if orientation_error is not None and orientation_tol is not None:
            payload["orientation_error_rad"] = orientation_error
            payload["orientation_tol_rad"] = orientation_tol
            payload["ik_orientation_residual_rad"] = ik_orientation_residual
            pose_detail = f" and orientation within {float(orientation_tol)} rad (error {orientation_error:.4f} rad)"
        if reached:
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"move_to: '{robot_name}' EE ({frame_type} '{frame_name}') reached "
                            f"{target.tolist()} within {float(tol)} m in {steps_used} steps "
                            f"(error {position_error:.4f} m){pose_detail}."
                        )
                    },
                    {"json": payload},
                ],
            }
        residuals = f"residual {position_error:.4f} m"
        if orientation_error is not None and orientation_tol is not None:
            residuals += f", orientation residual {orientation_error:.4f} rad (tol {float(orientation_tol)} rad)"
        if obstruction is not None:
            payload["obstruction"] = obstruction
        return _err(
            f"move_to: '{robot_name}' EE ({frame_type} '{frame_name}') did not reach "
            f"{target.tolist()} within tol={float(tol)} m "
            f"after max_steps={max_steps} ({residuals}; IK residual was "
            f"{ik_residual:.4f} m). " + MotionPrimitivesCore._obstruction_text(obstruction),
            payload,
        )

    @staticmethod
    def _joints_at_limit(mj: Any, model: Any, qpos: Any, joint_ids: Iterable[int]) -> list[dict[str, Any]]:
        """The commanded joints sitting at a bound of their range, by name.

        Engine-agnostic: reads a full ``qpos`` vector against the MuJoCo model
        every backend already holds for IK, so the MuJoCo servo (live
        ``data.qpos``) and the Isaac servo (its FK readback ``q_fk``) answer
        the same question the same way. A joint counts when it has limits and
        its position is within :data:`JOINT_LIMIT_MARGIN_FRACTION` of its
        range (floor ``1e-3``) of either bound.

        Args:
            mj: The ``mujoco`` module.
            model: The ``mujoco.MjModel`` the joints live in.
            qpos: Full generalized position vector indexed by ``jnt_qposadr``.
            joint_ids: The joints the primitive commanded.

        Returns:
            ``[{"joint", "pos", "limit", "side"}]`` ascending by joint id.
        """
        out: list[dict[str, Any]] = []
        for jnt_id in sorted(int(j) for j in joint_ids):
            if not bool(model.jnt_limited[jnt_id]):
                continue
            lo, hi = float(model.jnt_range[jnt_id][0]), float(model.jnt_range[jnt_id][1])
            margin = max((hi - lo) * JOINT_LIMIT_MARGIN_FRACTION, 1e-3)
            pos = float(qpos[int(model.jnt_qposadr[jnt_id])])
            side = "lower" if pos <= lo + margin else "upper" if pos >= hi - margin else None
            if side is None:
                continue
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id) or f"joint_{jnt_id}"
            out.append({"joint": name, "pos": pos, "limit": lo if side == "lower" else hi, "side": side})
        return out

    @staticmethod
    def _obstruction_text(obstruction: dict[str, Any] | None) -> str:
        """Say what stopped a ``move_to`` servo, by name, or that nothing visible did.

        Measured with an agent on the bundled so100 (v0.5.2 devx replay, s09):
        three ``move_to`` calls in a row ended in "the pose fights joint
        limits/contacts" while ``get_contacts`` showed the actual cause, a
        self-collision ``Fixed_Jaw <-> Base`` - the target sat too close to the
        base. The agent needed an extra tool call per failure to learn that,
        and the generic line named neither of the two things it could have.

        Args:
            obstruction: ``{"contacts": [{"a", "b", "dist_m"}], "joints_at_limit":
                [{"joint", "pos", "limit", "side"}]}`` as the engine read it at
                the final tick, at most :data:`OBSTRUCTION_MAX_CONTACTS`
                contacts. ``None`` when the engine did not look.

        Returns:
            One clause naming the obstruction and the remedy it implies; the
            legacy generic clause when the engine did not look.
        """
        if obstruction is None:
            return "The servo may need more steps, or the pose fights joint limits/contacts."
        contacts = list(obstruction.get("contacts") or [])
        limits = list(obstruction.get("joints_at_limit") or [])
        parts: list[str] = []
        if contacts:
            named = "; ".join(f"'{c['a']}' <-> '{c['b']}' (d={float(c['dist_m']):.4f} m)" for c in contacts)
            more = obstruction.get("contacts_total")
            suffix = f" and {int(more) - len(contacts)} more" if more and int(more) > len(contacts) else ""
            parts.append(f"the robot is in contact: {named}{suffix}")
        if limits:
            named = "; ".join(
                f"'{j['joint']}' at its {j['side']} limit ({float(j['pos']):.4f} vs {float(j['limit']):.4f})"
                for j in limits
            )
            parts.append(f"commanded joint(s) at a limit: {named}")
        if not parts:
            if obstruction.get("contacts_total") is None:
                return (
                    "At the final step no commanded joint was at a limit (contacts were not read on "
                    "this backend): the servo may need more steps (raise max_steps), or the pose "
                    "fights a contact."
                )
            return (
                "At the final step no contact involved the robot and no commanded joint was at a "
                "limit, so nothing visible is blocking it: the servo most likely needs more steps "
                "(raise max_steps) or a slightly looser tol."
            )
        remedy = (
            " Clear what it touches or command a value that avoids it, or loosen tol."
            if contacts
            else " Command a value that joint can reach inside its range, or loosen tol."
        )
        return "The servo was stopped: " + "; ".join(parts) + "." + remedy

    @staticmethod
    def _commanded_dof_indices(model: Any, commanded_joint_ids: Iterable[int]) -> list[int]:
        """Velocity-space indices of the joints a primitive's servos command.

        The position servos ``move_to`` drives command one scalar per joint, so
        the commandable subspace is the first DOF of each of those joints. This
        is the mask handed to
        :class:`strands_robots.simulation.ik.MinkIKBridge` as ``commanded_dofs``
        so the solve cannot answer with motion the servo loop never sends.

        Args:
            model: The ``mujoco.MjModel`` the IK bridge solves on.
            commanded_joint_ids: MuJoCo joint ids the primitive commands.

        Returns:
            The corresponding ``nv``-space indices, ascending.
        """
        return sorted(int(model.jnt_dofadr[joint_id]) for joint_id in commanded_joint_ids)

    @staticmethod
    def _uncommanded_joints_moved(
        mj: Any,
        model: Any,
        commanded_joint_ids: Iterable[int],
        q_before: np.ndarray,
        q_after: np.ndarray,
        namespace: str = "",
    ) -> list[str]:
        """Names of the uncommanded joints a solve moved, for the refusal text.

        Used only on the unreachable path, to say WHY a target the arm cannot
        reach is nonetheless solvable: an unrestricted solve reports which
        degrees of freedom it had to borrow (a mobile base, a held gripper),
        which is the difference between "outside the workspace" and "needs
        motion this primitive does not command".

        Args:
            mj: The ``mujoco`` module.
            model: The model both configurations belong to.
            commanded_joint_ids: Joint ids the primitive commands (excluded).
            q_before: Seed configuration (length ``model.nq``).
            q_after: Solved configuration (length ``model.nq``).
            namespace: Robot namespace prefix to strip from reported names.

        Returns:
            Joint names (namespace-stripped, ascending by joint id) whose
            configuration the solve changed and which are not commanded. An
            unnamed joint (a bare ``<freejoint/>``) is reported by index.
        """
        commanded = {int(joint_id) for joint_id in commanded_joint_ids}
        moved: list[str] = []
        for joint_id in range(int(model.njnt)):
            if joint_id in commanded:
                continue
            start = int(model.jnt_qposadr[joint_id])
            end = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < int(model.njnt) else int(model.nq)
            if np.allclose(q_before[start:end], q_after[start:end], atol=1e-9, rtol=0.0):
                continue
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if namespace and name.startswith(namespace):
                name = name[len(namespace) :]
            moved.append(name or f"unnamed joint {joint_id}")
        return moved

    @staticmethod
    def _set_gripper_result(
        robot_name: str,
        state: str,
        steps: int,
        actuators: list[str],
        targets: dict[str, float],
        setpoint_sources: dict[str, str],
        gripper_joint_positions: dict[str, float],
        *,
        held: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Success envelope for ``set_gripper``, shared across backends.

        All mappings are keyed by the (namespace-stripped) actuator / joint
        name, so the payload is meaningful to the agent whichever backend
        resolved the ids.

        ``held`` is what the fingers touch after a close, body name to contact
        count, for backends that can read contacts; ``None`` means the backend
        did not look (or the state was ``"open"``), and the reply stays as it
        was. An empty dict is a close that touched nothing, and the text says
        so - a grasp that missed reads exactly like one that landed otherwise.
        """
        payload: dict[str, Any] = {
            "state": state,
            "actuators": actuators,
            "targets": targets,
            "setpoint_sources": setpoint_sources,
            "gripper_joint_positions": gripper_joint_positions,
        }
        text = f"set_gripper: '{robot_name}' gripper commanded {state} ({steps} ticks, actuators {actuators})."
        if held is not None:
            payload["holding"] = sorted(held)
            payload["finger_contacts"] = dict(sorted(held.items()))
            if held:
                what = ", ".join(f"'{name}' ({n} contact{'s' if n != 1 else ''})" for name, n in sorted(held.items()))
                text += f" Closed on {what}."
            else:
                text += (
                    " Closed on nothing: no object is touching the fingers, so a lift now carries nothing - "
                    "move_to the object first (get_body_state gives its position)."
                )
        return {
            "status": "success",
            "content": [{"text": text}, {"json": payload}],
        }

    @staticmethod
    def _rotate_wrist_result(
        robot_name: str,
        tol: float,
        max_steps: int,
        *,
        reached: bool,
        steps_used: int,
        wrist_name: str,
        target_yaw: float,
        final_yaw: float,
        yaw_error: float,
        uncommanded_drives: list[str] | None = None,
        obstruction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Success / not-reached envelope for ``rotate_wrist``, shared across backends.

        *uncommanded_drives* names actuators on this robot that the primitive
        deliberately left alone because their ``ctrl`` is not a joint pose (a
        wheel ``<velocity>`` drive on a mobile manipulator, say), so "holds every
        other joint" is reported for the joints it can hold rather than claimed
        for all of them. Omitted from the payload when there are none, which
        keeps the envelope byte-identical for a fully position-servo robot.

        ``obstruction`` is read on the not-reached path only, in the shape
        :meth:`_obstruction_text` reads, and is the same measurement
        ``move_to`` reports: a wrist that cannot turn is usually pressing the
        gripper into something. ``None`` means the engine did not look, which
        keeps the generic clause.
        """
        payload: dict[str, Any] = {
            "reached": reached,
            "steps": steps_used,
            "wrist_joint": wrist_name,
            "target_yaw": target_yaw,
            "final_yaw": final_yaw,
            "yaw_error_rad": yaw_error,
        }
        if uncommanded_drives:
            payload["uncommanded_drives"] = list(uncommanded_drives)
        if reached:
            text = (
                f"rotate_wrist: '{robot_name}' joint '{wrist_name}' reached "
                f"{target_yaw:.3f} rad within {float(tol)} rad in {steps_used} steps."
            )
            if uncommanded_drives:
                text += (
                    f" {len(uncommanded_drives)} actuator(s) were left uncommanded because their "
                    f"ctrl is not a joint pose ({list(uncommanded_drives)}); command those in "
                    "their own units with action='send_action'."
                )
            return {
                "status": "success",
                "content": [{"text": text}, {"json": payload}],
            }
        if obstruction is not None:
            payload["obstruction"] = obstruction
        return _err(
            f"rotate_wrist: '{robot_name}' joint '{wrist_name}' did not reach {target_yaw:.3f} rad "
            f"within tol={float(tol)} rad after max_steps={max_steps} (residual {yaw_error:.4f} rad). "
            + MotionPrimitivesCore._obstruction_text(obstruction),
            payload,
        )
