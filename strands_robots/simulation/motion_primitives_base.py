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
from typing import Any

import numpy as np

from strands_robots.registry.robots import get_robot
from strands_robots.utils import coerce_pose_vector

# Name hints (lowercased substring match on the gripper DOF's name) used to
# resolve the gripper when the robot registry carries no gripper metadata for
# the robot's ``data_config``. Matches the existing runtime precedent (vera
# provider / cosmos3 policy use gripper|finger; SO-100's gripper joint is the
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


def _is_finite_real(value: Any) -> bool:
    """True when ``value`` is a real, finite scalar (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


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
    ) -> tuple[np.ndarray | None, np.ndarray | None, int, dict[str, Any] | None]:
        """Shared ``move_to`` parameter domain (before any engine state is read).

        Returns ``(target, orientation, max_steps, None)`` on success -
        ``target`` a float64 ``(3,)`` array, ``orientation`` a float64 ``(4,)``
        array or ``None`` when omitted - or ``(None, None, 0, error)`` when a
        parameter is off-domain. The pose vectors are validated by the same
        rule the scene-construction calls use
        (:func:`strands_robots.utils.coerce_pose_vector`): three/four finite
        real components, a NumPy array accepted, a ``bool`` refused.
        """
        if position is None:
            return None, None, 0, _err("move_to requires 'position' ([x, y, z] target in meters).")
        # Same guard the scene-construction entry points use, so a pose
        # `add_object`/`move_object` refuses is refused here too. `len()` on a
        # value with no length (a scalar, an iterator) raises a bare TypeError,
        # which would escape the primitives' documented never-raises contract.
        position, pos_err = coerce_pose_vector("move_to", "position", position, 3)
        if pos_err is not None:
            return None, None, 0, _err(pos_err)
        assert position is not None  # non-None input yields a non-None result
        orientation, quat_err = coerce_pose_vector("move_to", "orientation", orientation, 4)
        if quat_err is not None:
            return None, None, 0, _err(quat_err)
        if orientation is not None:
            quat_norm = float(np.linalg.norm(np.asarray(orientation, dtype=np.float64)))
            if quat_norm < 1e-8:
                return (
                    None,
                    None,
                    0,
                    _err("move_to: 'orientation' quaternion has ~zero norm; pass a valid [w, x, y, z]."),
                )
        if not _is_finite_real(tol) or float(tol) <= 0.0:
            return None, None, 0, _err(f"move_to: 'tol' must be a positive number of meters, got {tol!r}.")
        err = self._validate_step_budget("move_to", "max_steps", max_steps)
        if err is not None:
            return None, None, 0, err
        target = np.asarray(position, dtype=np.float64)
        quat = None if orientation is None else np.asarray(orientation, dtype=np.float64)
        return target, quat, int(max_steps), None

    def _validate_set_gripper_args(self, state: Any, steps: Any) -> tuple[int, dict[str, Any] | None]:
        """Shared ``set_gripper`` parameter domain: ``(steps, None)`` or ``(0, error)``."""
        if state not in ("open", "close"):
            return 0, _err(f'set_gripper: \'state\' must be "open" or "close", got {state!r}.')
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
            return 0.0, 0, _err(f"rotate_wrist: 'target_yaw' must be a finite number of radians, got {target_yaw!r}.")
        if not _is_finite_real(tol) or float(tol) <= 0.0:
            return 0.0, 0, _err(f"rotate_wrist: 'tol' must be a positive number of radians, got {tol!r}.")
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

    # -- shared result envelopes ----------------------------------------------

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
    ) -> dict[str, Any]:
        """Success / not-reached envelope for ``move_to``, shared across backends."""
        payload = {
            "reached": reached,
            "steps": steps_used,
            "position_error_m": position_error,
            "ik_residual_m": ik_residual,
            "ee_position": [float(v) for v in ee_pos],
            "ee_orientation_wxyz": [float(v) for v in ee_quat],
            "frame": frame_name,
            "frame_type": frame_type,
        }
        if reached:
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"move_to: '{robot_name}' EE ({frame_type} '{frame_name}') reached "
                            f"{target.tolist()} within {float(tol)} m in {steps_used} steps "
                            f"(error {position_error:.4f} m)."
                        )
                    },
                    {"json": payload},
                ],
            }
        return _err(
            f"move_to: '{robot_name}' did not reach {target.tolist()} within tol={float(tol)} m "
            f"after max_steps={max_steps} (residual {position_error:.4f} m; IK residual was "
            f"{ik_residual:.4f} m). The servo may need more steps, or the pose fights joint "
            "limits/contacts.",
            payload,
        )

    @staticmethod
    def _set_gripper_result(
        robot_name: str,
        state: str,
        steps: int,
        actuators: list[str],
        targets: dict[str, float],
        setpoint_sources: dict[str, str],
        gripper_joint_positions: dict[str, float],
    ) -> dict[str, Any]:
        """Success envelope for ``set_gripper``, shared across backends.

        All mappings are keyed by the (namespace-stripped) actuator / joint
        name, so the payload is meaningful to the agent whichever backend
        resolved the ids.
        """
        payload = {
            "state": state,
            "actuators": actuators,
            "targets": targets,
            "setpoint_sources": setpoint_sources,
            "gripper_joint_positions": gripper_joint_positions,
        }
        return {
            "status": "success",
            "content": [
                {
                    "text": f"set_gripper: '{robot_name}' gripper commanded {state} ({steps} ticks, actuators {actuators})."
                },
                {"json": payload},
            ],
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
    ) -> dict[str, Any]:
        """Success / not-reached envelope for ``rotate_wrist``, shared across backends."""
        payload = {
            "reached": reached,
            "steps": steps_used,
            "wrist_joint": wrist_name,
            "target_yaw": target_yaw,
            "final_yaw": final_yaw,
            "yaw_error_rad": yaw_error,
        }
        if reached:
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"rotate_wrist: '{robot_name}' joint '{wrist_name}' reached "
                            f"{target_yaw:.3f} rad within {float(tol)} rad in {steps_used} steps."
                        )
                    },
                    {"json": payload},
                ],
            }
        return _err(
            f"rotate_wrist: '{robot_name}' joint '{wrist_name}' did not reach {target_yaw:.3f} rad "
            f"within tol={float(tol)} rad after max_steps={max_steps} (residual {yaw_error:.4f} rad).",
            payload,
        )
