"""Analytic motion primitives - ``move_to`` / ``set_gripper`` / ``rotate_wrist``.

Agent-facing staging/transport/release vocabulary (GH #1645, motivated by
Harness VLA, arXiv:2607.08448): a fixed library of analytic primitives lets an
LLM agent position the robot *around* a learned policy's competence region
instead of asking the policy to absorb every distribution shift. On LIBERO the
paper reports ``move_to`` alone is 61.8% of all primitive calls; the frozen VLA
is invoked as a contact-rich primitive inside a larger analytic scaffold.

Primitives (mirroring the paper's Table 1, minimal set first):

* :meth:`MotionPrimitivesMixin.move_to` - Cartesian end-effector transport:
  solve IK to a world-frame target (shared mink damped-least-squares bridge,
  :class:`strands_robots.simulation.ik.MinkIKBridge`; position-only when no
  orientation is given - important for 5-DOF arms like SO-101), then drive the
  position-servo actuators until the EE is within ``tol`` or ``max_steps``
  control ticks elapse.
* :meth:`MotionPrimitivesMixin.set_gripper` - atomic open/close set-point on
  the gripper actuator(s).
* :meth:`MotionPrimitivesMixin.rotate_wrist` - wrist-yaw joint set-point while
  the other arm joints hold their current positions (so the Cartesian EE
  position is preserved up to servo compliance).

Contract notes (shared by all three):

* **Not collision-aware.** ``move_to`` hides the solver backend (the paper's
  contract: "operational-space servo, Jacobian controller, or IK planner ...
  exposed semantics identical"); a future collision-aware backend (curobo) can
  replace the solver without changing this surface.
* **Errors are structured dicts** (``{"status": "error", ...}``), never raised
  through the tool surface. A target the IK cannot reach returns the residual.
* **Refuse while a policy runs** on the same robot
  (``_require_no_running_policy``, per-robot scope) - a primitive and a
  ``PolicyRunner`` would otherwise race on ``data.ctrl``.
* **Self-locking**: these actions are in ``_SELF_LOCKING_ACTIONS`` and acquire
  ``self._lock`` per control tick (the ``step()`` pattern), so ``stop_policy``,
  renders, and MJPEG threads can interleave during a long primitive. Runtime is
  bounded by ``max_steps`` (hard cap ``_MAX_PRIMITIVE_STEPS``); if the world is
  destroyed/recompiled or a policy starts mid-run, the loop aborts with a
  structured error instead of stepping a stale model.
* **Dataset-recording interplay** (pinned by test): primitive motion does NOT
  feed frames into an active ``start_recording`` dataset session - only
  ``run_policy``'s per-frame hook records episodes. Camera MP4 recording
  (``start_cameras_recording``) still captures primitive motion, since it
  samples the live scene on its own thread.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.registry.robots import get_robot
from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG

logger = logging.getLogger(__name__)

# Name hints (lowercased substring match on the namespace-stripped actuator /
# joint name) used to resolve the gripper DOF when the robot registry carries
# no gripper metadata for the robot's ``data_config``. Matches the existing
# runtime precedent (vera provider / cosmos3 policy use gripper|finger;
# SO-100's gripper joint is the "Jaw"). Registry metadata
# (``robots.json`` -> ``<robot>.gripper``, GH #1658) is authoritative when
# present; the heuristic is the zero-config fallback for user URDFs / injected
# MJCF, and an unresolvable gripper returns a structured error listing the
# actuators so the agent can fall back to send_action.
_GRIPPER_HINTS = ("gripper", "finger", "jaw")

# Valid values for the registry gripper metadata ``closed`` / ``open`` fields:
# which ctrlrange END the state maps to. The registry integrity tests
# shape-check the shipped robots.json against the same two values.
_CTRLRANGE_ENDS = ("low", "high")

# Wrist-yaw joint hints, most-specific first. Fallback: the last non-gripper
# hinge joint in the robot's chain (the distal roll joint on most serial
# arms). "Non-gripper" is decided by the shared registry-metadata-first
# classification (_resolve_gripper_actuators), not by _GRIPPER_HINTS alone.
_WRIST_HINTS = ("wrist_roll", "wrist_yaw", "wrist_rotate", "wrist")

# Physics substeps per control tick. Each move_to/rotate_wrist "step" (and each
# set_gripper "step") re-asserts ctrl then advances this many mj_step calls, so
# the default budgets stay bounded: move_to(max_steps=200) is at most
# 200 * 5 = 1000 physics steps (2 s of sim time at the 0.002 s default).
_SUBSTEPS_PER_TICK = 5

# Hard ceiling on max_steps / steps to prevent unbounded primitive runtime.
_MAX_PRIMITIVE_STEPS = 10_000

# Workspace sanity radius: a move_to target further than this from the robot's
# spawn position is rejected up front (meters). Generous on purpose - it guards
# against unit mistakes (mm vs m), not reachability; true reachability is
# checked by the IK residual.
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


class MotionPrimitivesMixin:
    """Analytic motion primitives mixed into ``Simulation``.

    **Coupling** (see the :mod:`simulation` top-level docstring): reaches into
    ``self._world``, ``self._lock``, ``self._mj``, and the guard helpers.
    ``TYPE_CHECKING`` stubs below exist so mypy accepts those lookups; they are
    a documentary contract, not an enforceable protocol.
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: SimWorld | None
        _lock: Any  # threading.RLock from Simulation
        _mj: Any  # cached ``mujoco`` module reference from Simulation

        def _require_no_running_policy(self, action_name: str, robot_name: str | None = None) -> dict[str, Any] | None:
            """Provided by ``Simulation``; declared here for type-checkers."""

        def _unknown_robot_msg(self, requested: str) -> str:
            """Provided by ``Simulation``; declared here for type-checkers."""

        def _resolve_single_robot(self, robot_name: str | None) -> str:
            """Provided by ``SimEngine``; declared here for type-checkers."""

        def _apply_kinematic_attachments(self) -> None:
            """Provided by ``Simulation``; declared here for type-checkers."""

    # -- shared guards / resolution -----------------------------------------

    def _primitive_resolve_robot(self, action: str, robot_name: str | None) -> tuple[str | None, dict[str, Any] | None]:
        """Common primitive preamble: world + robot + no-running-policy guards.

        Returns ``(robot_name, None)`` on success or ``(None, error_dict)``.
        Callers should hold ``self._lock`` (the checks read shared state).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return None, _err(_NO_WORLD_MSG)
        if robot_name is None:
            try:
                robot_name = self._resolve_single_robot(None)
            except ValueError as e:
                return None, _err(str(e))
        if robot_name not in self._world.robots:
            return None, _err(self._unknown_robot_msg(robot_name))
        guard = self._require_no_running_policy(action, robot_name=robot_name)
        if guard is not None:
            return None, guard
        return robot_name, None

    def _primitive_abort_reason(self, action: str, robot_name: str, model: Any) -> dict[str, Any] | None:
        """Mid-loop cancellation check (call under ``self._lock``).

        The primitive loops release the lock between control ticks, so the
        world can legitimately be destroyed, the model recompiled, or a policy
        started while a primitive runs. Each of those aborts the primitive
        with a structured error rather than stepping a stale/contended model.
        """
        world = self._world
        if world is None or world._model is not model or world._data is None:
            return _err(f"{action}: world was destroyed or the model was recompiled mid-run; aborting.")
        robot = world.robots.get(robot_name)
        if robot is None:
            return _err(f"{action}: robot '{robot_name}' was removed mid-run; aborting.")
        if robot.policy_running:
            return _err(
                f"{action}: a policy started on '{robot_name}' mid-run; aborting. "
                f"Stop it first: action='stop_policy', name='{robot_name}'."
            )
        return None

    def _primitive_tick(self, model: Any, data: Any, ctrl_targets: dict[int, float]) -> None:
        """One control tick: assert ctrl targets, advance physics, refresh FK.

        Must be called under ``self._lock``. Mirrors ``step()``'s bookkeeping
        (kinematic attachments, ``sim_time`` / ``step_count``) and finishes
        with ``mj_kinematics`` so the caller reads current frame poses.
        """
        mj = self._mj
        assert self._world is not None  # callers must check
        for act_id, value in ctrl_targets.items():
            data.ctrl[act_id] = value
        has_kinematic_attachments = bool(self._world._backend_state.get("kinematic_attachments"))
        for _ in range(_SUBSTEPS_PER_TICK):
            mj.mj_step(model, data)
            if has_kinematic_attachments:
                self._apply_kinematic_attachments()
        self._world.sim_time = data.time
        self._world.step_count += _SUBSTEPS_PER_TICK
        mj.mj_kinematics(model, data)

    def _joint_actuator_map(self, model: Any, robot: Any) -> dict[int, int]:
        """Map ``jnt_id -> act_id`` for the robot's joint-transmission actuators.

        Only hinge/slide joints with a JOINT / JOINTINPARENT transmission
        actuator are included (the DOFs a position set-point can drive).
        Tendon-driven actuators (e.g. a Franka split gripper) have no matching
        joint and are excluded - ``set_gripper`` resolves those by name instead.
        """
        mj = self._mj
        joint_trn = {int(mj.mjtTrn.mjTRN_JOINT), int(mj.mjtTrn.mjTRN_JOINTINPARENT)}
        settable = {int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)}
        joint_ids = {int(j) for j in (robot.joint_ids or [])}
        out: dict[int, int] = {}
        for act_id in range(model.nu):
            if int(model.actuator_trntype[act_id]) not in joint_trn:
                continue
            jnt_id = int(model.actuator_trnid[act_id, 0])
            if jnt_id in joint_ids and int(model.jnt_type[jnt_id]) in settable:
                out[jnt_id] = act_id
        return out

    def _short_name(self, name: str | None, namespace: str) -> str:
        """Strip the robot namespace prefix for hint matching."""
        if name and namespace and name.startswith(namespace):
            return name[len(namespace) :]
        return name or ""

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
        info = get_robot(data_config)
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

    def _resolve_gripper_actuators(
        self, model: Any, robot: Any
    ) -> tuple[set[int], dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve *robot*'s gripper actuator ids (registry first, heuristic fallback).

        The shared classification behind ``set_gripper`` (which commands these
        actuators) and ``move_to`` (which must NOT command them - the gripper
        DOF is kinematically irrelevant to the EE task, so an IK solve would
        pass whatever seed value it was given straight through, and a random
        restart seed would then servo the jaw to a random point of its range,
        dropping a held object mid-transport). Resolution order (GH #1658):

        1. Registry ``gripper`` metadata for the robot's ``data_config``,
           when present (authoritative): an actuator qualifies when its
           namespace-stripped name matches one of ``gripper.actuators``
           exactly (case-insensitive). Metadata that matches NO actuator in
           the model is a loud structured error, never a silent heuristic
           fallback - a stale registry entry degrading to the heuristic
           would reintroduce exactly the misclassification the metadata
           exists to prevent.
        2. Name heuristic (zero-config fallback for user URDFs / injected
           MJCF): the actuator's short name, or its driven joint's short
           name, contains one of :data:`_GRIPPER_HINTS`.

        Returns ``(actuator_ids, metadata, error)``: ``metadata`` is the
        registry block when path 1 resolved (``None`` on the heuristic path);
        ``error`` is a structured error dict when metadata exists but is
        unusable (malformed, or matching nothing). Callers must hold
        ``self._lock``.
        """
        mj = self._mj
        namespace = robot.namespace or ""
        jnt_by_act = {act_id: jnt_id for jnt_id, act_id in self._joint_actuator_map(model, robot).items()}
        act_ids = [int(a) for a in (robot.actuator_ids or [])]

        meta, malformed_reason = self._registry_gripper_metadata(robot)
        if malformed_reason is not None:
            return set(), None, _err(f"Cannot resolve the gripper for '{robot.name}': {malformed_reason}")
        if meta is not None:
            wanted = {str(a).lower() for a in meta["actuators"]}
            matched: set[int] = set()
            short_names: list[str] = []
            for act_id in act_ids:
                short = self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id), namespace)
                short_names.append(short)
                if short.lower() in wanted:
                    matched.add(act_id)
            if not matched:
                return (
                    set(),
                    None,
                    _err(
                        f"Registry gripper metadata for '{robot.name}' (data_config "
                        f"'{robot.data_config}') names actuators {meta['actuators']} but none "
                        f"exist in the model. Model actuators: {short_names}. The registry "
                        "entry is stale for this model; fix it, or drive the gripper "
                        "directly with action='send_action'."
                    ),
                )
            return matched, meta, None

        out: set[int] = set()
        for act_id in act_ids:
            act_name = self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id), namespace).lower()
            jnt_id = jnt_by_act.get(act_id)
            jnt_name = ""
            if jnt_id is not None:
                jnt_name = self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id), namespace).lower()
            if any(h in act_name or (jnt_name and h in jnt_name) for h in _GRIPPER_HINTS):
                out.add(act_id)
        return out, None, None

    def _frame_world_pose(
        self, model: Any, data: Any, frame_name: str, frame_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """World position + wxyz quaternion of a site/body frame from live data.

        Callers must have run a kinematics pass so ``xpos``/``xmat`` are
        current, and must hold ``self._lock``.
        """
        mj = self._mj
        if frame_type == "site":
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, frame_name)
            pos = np.array(data.site_xpos[sid], dtype=np.float64)
            quat = np.zeros(4, dtype=np.float64)
            mj.mju_mat2Quat(quat, np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(9))
            return pos, quat
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, frame_name)
        return np.array(data.xpos[bid], dtype=np.float64), np.array(data.xquat[bid], dtype=np.float64)

    # -- primitives ----------------------------------------------------------

    def move_to(
        self,
        robot_name: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        tol: float = 0.01,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Move the end-effector to a world-frame Cartesian target via IK.

        Composite analytic primitive (the staging/transport verb): solves
        inverse kinematics to the target with the shared mink
        damped-least-squares bridge, then drives the arm's position-servo
        actuators toward the solved configuration until the end-effector is
        within ``tol`` meters of the target or ``max_steps`` control ticks
        elapse (each tick advances a few physics substeps).

        The end-effector frame is auto-discovered per robot namespace
        (:func:`strands_robots.simulation.ik.discover_ee_frame`: TCP-like site,
        else hand/tool body, else the chain's leaf body) - the same heuristic
        eef-delta policies use, so multi-robot scenes resolve the right arm.

        GRASP PRESERVATION (contract): gripper actuators (resolved by the same
        registry-metadata-first classification ``set_gripper`` uses, see
        :meth:`_resolve_gripper_actuators`) are excluded from the IK solve
        and its restart seeding, and are HELD at their live position for the
        whole servo descent - a closed gripper stays closed through staging
        and transport, so ``set_gripper("close") -> move_to(...)`` carries the
        held object rather than releasing it.

        NOT collision-aware: the straight servo descent can sweep through
        obstacles. Collision-aware transport is the curobo provider's job; this
        primitive deliberately hides the solver backend so that upgrade cannot
        change the surface. Motion is NOT recorded into an active dataset
        recording session (see module docstring).

        Args:
            robot_name: Robot to move; defaults to the single robot in the
                world (errors if ambiguous).
            position: World-frame target ``[x, y, z]`` in meters (required).
            orientation: Optional target orientation quaternion ``[w, x, y, z]``.
                When omitted the solve is position-only - the right choice for
                arms with fewer than 6 DOF (e.g. SO-100/SO-101), which cannot
                realize an arbitrary full pose.
            tol: Position convergence tolerance in meters (> 0).
            max_steps: Max control ticks before returning a not-reached error
                (1..10000).

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{reached, steps, position_error_m, ik_residual_m, ee_position,
            ee_orientation_wxyz, frame, frame_type}`` on arrival;
            ``{"status": "error", ...}`` with the same json block (including
            the residual) when the target is unreachable (IK residual > tol)
            or servo convergence times out. Never raises.
        """
        # ---- parameter validation (before touching the world) ----
        if position is None:
            return _err("move_to requires 'position' ([x, y, z] target in meters).")
        if len(position) != 3 or not all(_is_finite_real(c) for c in position):
            return _err("move_to: 'position' must be 3 finite numbers [x, y, z] in meters.")
        if orientation is not None:
            if len(orientation) != 4 or not all(_is_finite_real(c) for c in orientation):
                return _err("move_to: 'orientation' must be 4 finite numbers [w, x, y, z] (unit quaternion).")
            quat_norm = float(np.linalg.norm(np.asarray(orientation, dtype=np.float64)))
            if quat_norm < 1e-8:
                return _err("move_to: 'orientation' quaternion has ~zero norm; pass a valid [w, x, y, z].")
        if not _is_finite_real(tol) or float(tol) <= 0.0:
            return _err(f"move_to: 'tol' must be a positive number of meters, got {tol!r}.")
        err = self._validate_step_budget("move_to", "max_steps", max_steps)
        if err is not None:
            return err
        max_steps = int(max_steps)
        target = np.asarray([float(c) for c in position], dtype=np.float64)

        # ---- setup under the lock: guards, EE frame, IK solve ----
        with self._lock:
            robot_name_resolved, error = self._primitive_resolve_robot("move_to", robot_name)
            if error is not None:
                return error
            assert robot_name_resolved is not None
            robot_name = robot_name_resolved
            assert self._world is not None
            model, data = self._world._model, self._world._data
            robot = self._world.robots[robot_name]
            namespace = robot.namespace or ""

            base = np.asarray(robot.position or (0.0, 0.0, 0.0), dtype=np.float64)
            base_dist = float(np.linalg.norm(target - base))
            if base_dist > _WORKSPACE_SANITY_RADIUS_M:
                return _err(
                    f"move_to: target {target.tolist()} is {base_dist:.2f} m from robot "
                    f"'{robot_name}' base - outside the {_WORKSPACE_SANITY_RADIUS_M:.0f} m workspace "
                    "sanity box. Check units (meters, world frame)."
                )

            from strands_robots.simulation.ik import discover_ee_frame

            frame = discover_ee_frame(model, namespace or None)
            if frame is None:
                return _err(
                    f"move_to: could not auto-discover an end-effector frame for robot "
                    f"'{robot_name}' (namespace {namespace!r}). The model has no TCP-like "
                    "site or hand/tool body to track."
                )
            frame_name, frame_type = frame

            jact = self._joint_actuator_map(model, robot)
            if not jact:
                return _err(
                    f"move_to: robot '{robot_name}' has no joint-transmission actuators to "
                    "drive. If it was loaded from a bare URDF, add position servos first: "
                    "action='actuate_robot'."
                )
            # Split arm vs gripper DOFs: the gripper is kinematically
            # irrelevant to the EE task (mink passes its seed value straight
            # through), so it must be excluded from IK seeding and from the
            # solved command set - otherwise a restart seed would randomize
            # the jaw and move_to would servo it there, dropping a held
            # object mid-transport. The gripper channel is instead HELD at
            # its live position (not omitted: _primitive_tick writes
            # data.ctrl every tick, and an unwritten channel would let a
            # stale ctrl from another path drive it).
            grip_acts, _, grip_err = self._resolve_gripper_actuators(model, robot)
            if grip_err is not None:
                return grip_err
            arm_jact = {j: a for j, a in jact.items() if a not in grip_acts}
            if not arm_jact:
                return _err(
                    f"move_to: robot '{robot_name}' has no non-gripper joint-transmission "
                    "actuators to drive - every actuated joint is classified as a gripper "
                    f"drive (registry metadata or the name heuristic {list(_GRIPPER_HINTS)}). "
                    "Nothing can move the end-effector."
                )

            try:
                from strands_robots.simulation.ik import MinkIKBridge

                # max_iters is raised well above the bridge default (20): the
                # policy decode paths solve small re-anchored per-step deltas,
                # whereas move_to jumps from the current pose to an arbitrary
                # workspace point in one solve and needs the extra integration
                # budget to converge from far seeds.
                bridge = MinkIKBridge(
                    model,
                    frame_name,
                    frame_type,
                    orientation_cost=1.0 if orientation is not None else 0.0,
                    max_iters=200,
                )
            except (ImportError, RuntimeError, ValueError) as e:
                return _err(f"move_to: IK bridge unavailable: {e}")

            q0 = np.array(data.qpos, dtype=np.float64, copy=True)
            target_pose = np.eye(4, dtype=np.float64)
            target_pose[:3, 3] = target
            if orientation is not None:
                quat = np.asarray([float(c) for c in orientation], dtype=np.float64)
                quat = quat / np.linalg.norm(quat)
                rot = np.zeros(9, dtype=np.float64)
                self._mj.mju_quat2Mat(rot, quat)
                target_pose[:3, :3] = rot.reshape(3, 3)
            else:
                # Position-only: keep the current EE orientation in the target
                # pose (the zero orientation cost makes it a soft no-op).
                target_pose[:3, :3] = bridge.ee_pose(q0)[:3, :3]

            q_star = bridge.solve(target_pose, q0)
            ik_residual = float(np.linalg.norm(bridge.ee_pose(q_star)[:3, 3] - target))

            # Damped-least-squares IK is a local method: from a distant seed it
            # can stall in a joint-limit / elbow-branch local minimum even for a
            # reachable target. When the direct solve misses, retry from a few
            # DETERMINISTIC restart seeds and keep the best solution: the first
            # restart uses the model's home keyframe when one exists (for a
            # from-home arm that is usually the branch that converges, and it
            # is free), the rest set the robot's own ARM hinge/slide joints
            # uniformly within their ranges. Gripper DOFs and everything else
            # (other robots, object free joints) stay at the live state.
            # Bounded and reproducible: same target, same answer.
            if ik_residual > float(tol):
                # The RNG is deliberately reconstructed with a fixed seed PER
                # CALL (not module-level): identical calls draw identical seed
                # sequences, which is what makes move_to reproducible. Two
                # calls to different targets sharing the sequence is fine -
                # do not "fix" this into a shared RNG, that would make a
                # call's outcome depend on call history.
                rng = np.random.default_rng(0)
                settable_qadr = [int(model.jnt_qposadr[jnt_id]) for jnt_id in arm_jact]
                ranges = [
                    (float(model.jnt_range[jnt_id][0]), float(model.jnt_range[jnt_id][1]))
                    if bool(model.jnt_limited[jnt_id])
                    else (-np.pi, np.pi)
                    for jnt_id in arm_jact
                ]
                for restart in range(_IK_RESTART_SEEDS):
                    q_seed = q0.copy()
                    if restart == 0 and int(model.nkey) > 0:
                        for qadr in settable_qadr:
                            q_seed[qadr] = float(model.key_qpos[0][qadr])
                    else:
                        for qadr, (lo, hi) in zip(settable_qadr, ranges, strict=True):
                            q_seed[qadr] = rng.uniform(lo, hi)
                    q_try = bridge.solve(target_pose, q_seed)
                    residual_try = float(np.linalg.norm(bridge.ee_pose(q_try)[:3, 3] - target))
                    if residual_try < ik_residual:
                        q_star, ik_residual = q_try, residual_try
                    if ik_residual <= float(tol):
                        break

            if ik_residual > float(tol):
                return _err(
                    f"move_to: target {target.tolist()} is unreachable for '{robot_name}' "
                    f"within tol={float(tol)} m - best IK solution leaves a residual of "
                    f"{ik_residual:.4f} m. Choose a closer target or loosen tol.",
                    {
                        "reached": False,
                        "steps": 0,
                        "ik_residual_m": ik_residual,
                        "frame": frame_name,
                        "frame_type": frame_type,
                    },
                )

            # Command ARM joints to the solve; HOLD gripper joints at their
            # live position (see the arm/gripper split above - this is the
            # grasp-preservation contract, and mirrors rotate_wrist's
            # hold-everything-else behaviour).
            ctrl_targets = {
                act_id: float(q_star[int(model.jnt_qposadr[jnt_id])]) for jnt_id, act_id in arm_jact.items()
            }
            ctrl_targets.update(
                {
                    act_id: float(data.qpos[int(model.jnt_qposadr[jnt_id])])
                    for jnt_id, act_id in jact.items()
                    if act_id in grip_acts
                }
            )

        # ---- servo loop: self-locking per control tick ----
        steps_used = 0
        reached = False
        position_error = math.inf
        ee_pos = target
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(max_steps):
            with self._lock:
                abort = self._primitive_abort_reason("move_to", robot_name, model)
                if abort is not None:
                    return abort
                self._primitive_tick(model, data, ctrl_targets)
                ee_pos, ee_quat = self._frame_world_pose(model, data, frame_name, frame_type)
            steps_used += 1
            position_error = float(np.linalg.norm(ee_pos - target))
            if position_error <= float(tol):
                reached = True
                break

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

    def set_gripper(
        self,
        robot_name: str | None = None,
        state: str | None = None,
        steps: int = 12,
    ) -> dict[str, Any]:
        """Drive the gripper to an open/close set-point (atomic primitive).

        Resolves the gripper actuator(s) via :meth:`_resolve_gripper_actuators`
        (registry ``gripper`` metadata for the robot's ``data_config`` when
        present, else the ``gripper`` / ``finger`` / ``jaw`` name heuristic)
        and commands the actuator ctrlrange end-point. With no metadata,
        ``"open"`` drives to the HIGH end and ``"close"`` to the LOW end -
        the convention for SO-100/SO-101 (the jaw closes toward the
        low/negative end of its range - the sign trap documented in the
        so101_curobo example) and for the Franka split gripper (0=closed,
        255=open). Registry metadata's ``closed``/``open`` fields override
        that convention per robot. Motion is NOT recorded into an active
        dataset recording session (see module docstring).

        Args:
            robot_name: Robot whose gripper to drive; defaults to the single
                robot in the world (errors if ambiguous).
            state: ``"open"`` or ``"close"`` (required).
            steps: Control ticks to hold the set-point (1..10000); each tick
                advances a few physics substeps so the fingers actually travel.

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{state, actuators, targets, gripper_joint_positions}``;
            structured error when the gripper cannot be resolved or the
            ctrlrange gives no usable set-points. Never raises.
        """
        if state not in ("open", "close"):
            return _err(f'set_gripper: \'state\' must be "open" or "close", got {state!r}.')
        err = self._validate_step_budget("set_gripper", "steps", steps)
        if err is not None:
            return err
        steps = int(steps)

        with self._lock:
            robot_name_resolved, error = self._primitive_resolve_robot("set_gripper", robot_name)
            if error is not None:
                return error
            assert robot_name_resolved is not None
            robot_name = robot_name_resolved
            assert self._world is not None
            model, data = self._world._model, self._world._data
            robot = self._world.robots[robot_name]
            namespace = robot.namespace or ""
            mj = self._mj

            jact = self._joint_actuator_map(model, robot)
            jnt_by_act = {act_id: jnt_id for jnt_id, act_id in jact.items()}
            act_ids = [int(a) for a in (robot.actuator_ids or [])]
            # Shared arm/gripper classification (also the exclusion set that
            # keeps move_to's IK from commanding the jaw): registry metadata
            # first, name heuristic as the zero-config fallback.
            grip_set, grip_meta, grip_err = self._resolve_gripper_actuators(model, robot)
            if grip_err is not None:
                return grip_err
            gripper_acts = sorted(grip_set)
            if not gripper_acts:
                names = [
                    self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a), namespace) for a in act_ids
                ]
                return _err(
                    f"set_gripper: could not resolve a gripper actuator for '{robot_name}' - the "
                    f"registry carries no gripper metadata for this robot and no actuator/joint "
                    f"name contains {list(_GRIPPER_HINTS)}. Actuators: {names}. "
                    "Drive it directly with action='send_action' instead."
                )

            # Which ctrlrange END each state maps to: the registry metadata's
            # `closed`/`open` fields when present, else the open=HIGH /
            # close=LOW convention (correct for SO-100/SO-101 and Franka, but
            # a convention, not a law - the metadata field exists to remove
            # that sign trap for robots with an inverted gripper).
            if grip_meta is not None:
                end = grip_meta.get("open", "high") if state == "open" else grip_meta.get("closed", "low")
            else:
                end = "high" if state == "open" else "low"

            targets: dict[int, float] = {}
            for act_id in gripper_acts:
                lo = float(model.actuator_ctrlrange[act_id][0])
                hi = float(model.actuator_ctrlrange[act_id][1])
                if not (hi > lo):
                    act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id)
                    return _err(
                        f"set_gripper: actuator '{act_name}' has no usable ctrlrange "
                        f"({lo}, {hi}); cannot infer open/close set-points. Drive it directly "
                        "with action='send_action'."
                    )
                targets[act_id] = hi if end == "high" else lo

        for _ in range(steps):
            with self._lock:
                abort = self._primitive_abort_reason("set_gripper", robot_name, model)
                if abort is not None:
                    return abort
                self._primitive_tick(model, data, targets)

        with self._lock:
            joint_positions: dict[str, float] = {}
            for act_id in gripper_acts:
                jnt_id = jnt_by_act.get(act_id)
                if jnt_id is not None:
                    jname = self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id), namespace)
                    joint_positions[jname] = float(data.qpos[int(model.jnt_qposadr[jnt_id])])
            act_names = [
                self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a), namespace) for a in gripper_acts
            ]
        payload = {
            "state": state,
            "actuators": act_names,
            "targets": {n: targets[a] for n, a in zip(act_names, gripper_acts, strict=True)},
            "gripper_joint_positions": joint_positions,
        }
        return {
            "status": "success",
            "content": [
                {
                    "text": f"set_gripper: '{robot_name}' gripper commanded {state} ({steps} ticks, actuators {act_names})."
                },
                {"json": payload},
            ],
        }

    def rotate_wrist(
        self,
        robot_name: str | None = None,
        target_yaw: float | None = None,
        tol: float = 0.02,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Rotate the wrist-yaw joint to a set-point, holding the arm posture.

        Atomic primitive: resolves the wrist-roll/yaw joint by name heuristic
        (``wrist_roll`` / ``wrist_yaw`` / ``wrist_rotate`` / ``wrist``, else
        the last non-gripper hinge joint - the distal roll joint on most serial
        arms; gripper DOFs are excluded via the shared registry-metadata-first
        classification, :meth:`_resolve_gripper_actuators`), commands it to
        ``target_yaw`` while every other arm actuator
        holds its current joint position, and servoes until the joint is within
        ``tol`` radians or ``max_steps`` control ticks elapse. Holding the
        other joints preserves the Cartesian EE position up to servo
        compliance. Motion is NOT recorded into an active dataset recording
        session (see module docstring).

        Args:
            robot_name: Robot whose wrist to rotate; defaults to the single
                robot in the world (errors if ambiguous).
            target_yaw: Wrist joint set-point in radians (required; must lie
                within the joint's range when the joint is limited).
            tol: Joint-angle convergence tolerance in radians (> 0).
            max_steps: Max control ticks before returning a not-reached error
                (1..10000).

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{reached, steps, wrist_joint, target_yaw, final_yaw,
            yaw_error_rad}``; structured error (with the residual) when the
            joint cannot be resolved, the registry gripper metadata is
            stale/malformed (same contract as ``set_gripper``/``move_to``),
            the target is out of range, or servo convergence times out.
            Never raises.
        """
        if target_yaw is None:
            return _err("rotate_wrist requires 'target_yaw' (wrist joint set-point in radians).")
        if not _is_finite_real(target_yaw):
            return _err(f"rotate_wrist: 'target_yaw' must be a finite number of radians, got {target_yaw!r}.")
        if not _is_finite_real(tol) or float(tol) <= 0.0:
            return _err(f"rotate_wrist: 'tol' must be a positive number of radians, got {tol!r}.")
        err = self._validate_step_budget("rotate_wrist", "max_steps", max_steps)
        if err is not None:
            return err
        max_steps = int(max_steps)
        target_yaw = float(target_yaw)

        with self._lock:
            robot_name_resolved, error = self._primitive_resolve_robot("rotate_wrist", robot_name)
            if error is not None:
                return error
            assert robot_name_resolved is not None
            robot_name = robot_name_resolved
            assert self._world is not None
            model, data = self._world._model, self._world._data
            robot = self._world.robots[robot_name]
            namespace = robot.namespace or ""
            mj = self._mj

            jact = self._joint_actuator_map(model, robot)
            if not jact:
                return _err(
                    f"rotate_wrist: robot '{robot_name}' has no joint-transmission actuators to "
                    "drive. If it was loaded from a bare URDF, add position servos first: "
                    "action='actuate_robot'."
                )

            def jnt_short(jnt_id: int) -> str:
                return self._short_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id), namespace)

            # Exclude gripper DOFs via the shared registry-first classification
            # (GH #1661, follow-up to #1658), not a raw name-hint match. On
            # so101's shipped sim MJCF the joints are named 1..6 - no gripper
            # hint matches, so the raw heuristic would let the last-hinge
            # fallback below pick joint 6, which IS the jaw: rotate_wrist
            # would open/close the gripper instead of rotating the wrist.
            # Stale/malformed registry metadata is the same loud structured
            # error set_gripper / move_to return, never a silent fallback.
            grip_acts, _, grip_err = self._resolve_gripper_actuators(model, robot)
            if grip_err is not None:
                return grip_err

            hinge = int(mj.mjtJoint.mjJNT_HINGE)
            candidates = [j for j in jact if int(model.jnt_type[j]) == hinge]
            non_gripper = [j for j in candidates if jact[j] not in grip_acts]
            wrist_jnt: int | None = None
            for hint in _WRIST_HINTS:
                matches = [j for j in non_gripper if hint in jnt_short(j).lower()]
                if matches:
                    wrist_jnt = matches[-1]  # most distal on a name tie
                    break
            if wrist_jnt is None and non_gripper:
                # Fallback: the last (most distal) non-gripper hinge joint.
                wrist_jnt = max(non_gripper)
            if wrist_jnt is None:
                names = [jnt_short(j) for j in jact]
                return _err(
                    f"rotate_wrist: could not resolve a wrist joint for '{robot_name}'. "
                    f"Actuated hinge joints: {names}. Drive one directly with "
                    "action='send_action' instead."
                )
            wrist_name = jnt_short(wrist_jnt)

            if bool(model.jnt_limited[wrist_jnt]):
                lo = float(model.jnt_range[wrist_jnt][0])
                hi = float(model.jnt_range[wrist_jnt][1])
                if not (lo <= target_yaw <= hi):
                    return _err(
                        f"rotate_wrist: target_yaw={target_yaw} rad is outside joint "
                        f"'{wrist_name}' range [{lo:.3f}, {hi:.3f}] rad."
                    )

            # Hold every other actuated joint at its CURRENT position; command
            # only the wrist to the set-point.
            ctrl_targets: dict[int, float] = {}
            for jnt_id, act_id in jact.items():
                qadr = int(model.jnt_qposadr[jnt_id])
                ctrl_targets[act_id] = target_yaw if jnt_id == wrist_jnt else float(data.qpos[qadr])
            wrist_qadr = int(model.jnt_qposadr[wrist_jnt])

        steps_used = 0
        reached = False
        yaw_error = math.inf
        final_yaw = 0.0
        for _ in range(max_steps):
            with self._lock:
                abort = self._primitive_abort_reason("rotate_wrist", robot_name, model)
                if abort is not None:
                    return abort
                self._primitive_tick(model, data, ctrl_targets)
                final_yaw = float(data.qpos[wrist_qadr])
            steps_used += 1
            yaw_error = abs(final_yaw - target_yaw)
            if yaw_error <= float(tol):
                reached = True
                break

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

    # -- validation helpers ---------------------------------------------------

    @staticmethod
    def _validate_step_budget(action: str, param: str, value: Any) -> dict[str, Any] | None:
        """Validate an integer control-tick budget (1..``_MAX_PRIMITIVE_STEPS``)."""
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            return _err(f"{action}: '{param}' must be an integer, got {type(value).__name__}.")
        if not (1 <= int(value) <= _MAX_PRIMITIVE_STEPS):
            return _err(f"{action}: '{param}' must be between 1 and {_MAX_PRIMITIVE_STEPS}, got {int(value)}.")
        return None
