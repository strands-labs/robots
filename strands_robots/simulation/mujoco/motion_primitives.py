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
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.registry.robots import get_robot
from strands_robots.simulation.models import registered, registry_entry

# The backend-agnostic half (parameter domains, registry gripper metadata,
# shared envelopes and name-hint constants) lives in
# :mod:`strands_robots.simulation.motion_primitives_base` (GH #2123) so the
# Isaac adapter answers identically. The redundant aliases re-export the
# moved names at their historical import path.
from strands_robots.simulation.motion_primitives_base import (
    _GRIPPER_HINTS as _GRIPPER_HINTS,
)
from strands_robots.simulation.motion_primitives_base import (
    _IK_RESTART_SEEDS as _IK_RESTART_SEEDS,
)
from strands_robots.simulation.motion_primitives_base import (
    _WRIST_HINTS as _WRIST_HINTS,
)
from strands_robots.simulation.motion_primitives_base import (
    MotionPrimitivesCore,
    _err,
    _quat_angle_error,
)
from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, mj_name_to_id

logger = logging.getLogger(__name__)

# Physics substeps per control tick. Each move_to/rotate_wrist "step" (and each
# set_gripper "step") re-asserts ctrl then advances this many mj_step calls, so
# the default budgets stay bounded: move_to(max_steps=200) is at most
# 200 * 5 = 1000 physics steps (2 s of sim time at the 0.002 s default).
# Deliberately NOT in motion_primitives_base: it is a MuJoCo physics-substep
# detail, not part of the backend-agnostic primitive contract.
_SUBSTEPS_PER_TICK = 5


class MotionPrimitivesMixin(MotionPrimitivesCore):
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

        def _unknown_robot_msg(self, requested: object) -> str:
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
        if not registered(self._world.robots, robot_name):
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
        robot = registry_entry(world.robots, robot_name)
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

        That exclusion is load-bearing twice over: it is also what scopes
        :meth:`_gripper_setpoint_range`'s driven-joint substitution to the
        transmissions where ``ctrl`` is the joint target, so a tendon gripper
        with an unusable ctrlrange keeps refusing rather than being commanded
        in the wrong units.
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

    def _gripper_setpoint_range(
        self, model: Any, act_id: int, jnt_id: int | None
    ) -> tuple[tuple[float, float] | None, str]:
        """Open/close set-point bounds for a gripper actuator, and their source.

        Returns ``((lo, hi), source)`` - *source* naming where the bounds came
        from, for the success payload - or ``(None, reason)`` when every source
        is exhausted, *reason* then naming each one that was tried so the
        refusal says what it looked at.

        The actuator ``ctrlrange`` is authoritative whenever it is usable. When
        it is not, MuJoCo's encoding is the thing to read carefully: a position
        servo whose MJCF declares neither ``ctrlrange`` nor ``inheritrange="1"``
        compiles to ``ctrlrange == (0, 0)`` with ``actuator_ctrllimited == 0``,
        and that is the UNLIMITED actuator - a different claim from "this
        actuator accepts nothing". For a JOINT / JOINTINPARENT transmission
        ``ctrl`` IS the joint target, so the driven joint's own limits are the
        open/close set-points, and they are precisely what ``inheritrange="1"``
        would have compiled the ctrlrange to. Both sibling primitives already
        make that substitution (``rotate_wrist`` and ``move_to`` read
        ``jnt_range`` under ``jnt_limited``); ``set_gripper`` read only the
        ctrlrange and so refused on so101, whose shipped MJCF authors neither
        attribute while so100's sets ``inheritrange="1"`` on every actuator -
        the only reason so100 was unaffected (GH #1942).

        Three shapes keep refusing, and none of them is an omission to repair:

        * ``actuator_ctrllimited == 1`` alongside a degenerate range is a claim
          about the actuator, so it is respected rather than second-guessed.
          The MJCF compiler cannot produce that combination - it rejects an
          explicit ``ctrllimited="true"`` whose range is not strictly increasing
          with *invalid control range for actuator*, and it compiles a bare
          degenerate range (``"0 0"``, ``"0.5 0.5"``) to ``ctrllimited == 0`` -
          so this guard bites only on a model mutated after compilation, which
          this package does do: :mod:`strands_robots.policies.wbc.sim_control`
          rewrites ``ctrlrange`` to hand control to a whole-body controller and
          restores it afterwards.
        * A driven joint that is itself unlimited has no limits to lend.
        * A tendon actuator's ctrlrange is a normalised command space, not joint
          units - the shipped Franka gripper is ``(0, 255)`` - so a joint range
          would command the wrong quantity. *jnt_id* is ``None`` for one by
          construction: only JOINT / JOINTINPARENT transmissions appear in
          :meth:`_joint_actuator_map`.

        A degenerate range *stored* under ``ctrllimited == 0`` is inert rather
        than restrictive - MuJoCo clamps ``ctrl`` only when
        ``ctrllimited == 1`` - so such an actuator genuinely accepts any
        command, and substituting the joint range restricts nothing that was
        previously free and widens nothing that was previously enforced.
        """
        lo = float(model.actuator_ctrlrange[act_id][0])
        hi = float(model.actuator_ctrlrange[act_id][1])
        if hi > lo:
            return (lo, hi), "actuator ctrlrange"
        if bool(model.actuator_ctrllimited[act_id]):
            return None, (
                f"its ctrlrange ({lo}, {hi}) is degenerate and ctrllimited=1 declares that "
                "as a real limit rather than an unset one"
            )
        if jnt_id is None:
            return None, (
                f"its ctrlrange ({lo}, {hi}) is unset (ctrllimited=0) and it drives no joint "
                "whose limits could substitute - a tendon actuator's ctrlrange is a normalised "
                "command space, not joint units"
            )
        if not bool(model.jnt_limited[jnt_id]):
            return None, (
                f"its ctrlrange ({lo}, {hi}) is unset (ctrllimited=0) and the joint it drives is itself unlimited"
            )
        jnt_lo = float(model.jnt_range[jnt_id][0])
        jnt_hi = float(model.jnt_range[jnt_id][1])
        if jnt_hi > jnt_lo:
            return (jnt_lo, jnt_hi), "driven joint range"
        return None, (
            f"its ctrlrange ({lo}, {hi}) is unset (ctrllimited=0) and the joint it drives has "
            f"a degenerate range ({jnt_lo}, {jnt_hi})"
        )

    def _short_name(self, name: str | None, namespace: str) -> str:
        """Strip the robot namespace prefix for hint matching."""
        if name and namespace and name.startswith(namespace):
            return name[len(namespace) :]
        return name or ""

    @staticmethod
    def _get_registry_robot(data_config: str) -> dict[str, Any] | None:
        """Registry lookup seam (see ``MotionPrimitivesCore._get_registry_robot``).

        Resolves through this module's ``get_robot`` global so the historical
        patch point
        (``strands_robots.simulation.mujoco.motion_primitives.get_robot``)
        keeps working for tests and user code.
        """
        return get_robot(data_config)

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

        For ``frame_type="body"`` this reads the body's FRAME ORIGIN
        (``data.xpos`` / ``data.xquat``), not its inertial/CoM frame
        (``data.xipos``) - the two are distinct whenever the body's mass is not
        centred on its origin. mink optimizes the frame origin, and
        :meth:`move_to` decides ``reached`` by comparing this readback against
        the same target the solver was given, so reading the inertial frame here
        would leave the solver and the convergence check measuring points that
        are metres-scale apart on some models and never converge.

        Callers must have run a kinematics pass so ``xpos``/``xmat`` are
        current, and must hold ``self._lock``.
        """
        mj = self._mj
        if frame_type == "site":
            sid = mj_name_to_id(model, mj.mjtObj.mjOBJ_SITE, frame_name)
            pos = np.array(data.site_xpos[sid], dtype=np.float64)
            quat = np.zeros(4, dtype=np.float64)
            mj.mju_mat2Quat(quat, np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(9))
            return pos, quat
        bid = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, frame_name)
        return np.array(data.xpos[bid], dtype=np.float64), np.array(data.xquat[bid], dtype=np.float64)

    def _diagnose_unreachable(
        self,
        model: Any,
        frame_name: str,
        frame_type: str,
        target_quat: np.ndarray | None,
        target_pose: np.ndarray,
        target: np.ndarray,
        q0: np.ndarray,
        arm_jact: dict[int, int],
        namespace: str,
    ) -> tuple[float, list[str]]:
        """Best residual with every model DOF free, and the DOFs that took it.

        Runs on the ``move_to`` refusal path only (see
        :meth:`MotionPrimitivesCore._move_to_unreachable_error`): the primitive
        solves over its commanded joints, and this reports what an unrestricted
        solve could have done, so the refusal distinguishes a point outside the
        robot's workspace from one that needs uncommanded motion.

        Args:
            model: The world ``mujoco.MjModel``.
            frame_name: End-effector frame the solve tracks.
            frame_type: ``"site"`` / ``"body"`` / ``"geom"``.
            target_quat: Requested orientation, or ``None`` for position-only.
            target_pose: The ``(4, 4)`` target the restricted solve was given.
            target: World-frame target position.
            q0: The live configuration the solve seeded from.
            arm_jact: Commanded joint id -> actuator id map.
            namespace: Robot namespace, stripped from reported joint names.

        Returns:
            ``(residual_m, uncommanded_joint_names)``. On a fully actuated
            fixed-base arm the name list is empty and the residual matches the
            restricted one, which is what keeps the refusal text unchanged
            there.
        """
        from strands_robots.simulation.ik import MinkIKBridge

        try:
            reference = MinkIKBridge(
                model,
                frame_name,
                frame_type,
                orientation_cost=1.0 if target_quat is not None else 0.0,
                max_iters=200,
            )
        except (ImportError, RuntimeError, ValueError):  # pragma: no cover - the restricted build succeeded
            return math.inf, []
        q_free = reference.solve(target_pose, q0)
        residual = float(np.linalg.norm(reference.ee_pose(q_free)[:3, 3] - target))
        moved = self._uncommanded_joints_moved(self._mj, model, arm_jact, q0, q_free, namespace)
        return residual, moved

    # -- primitives ----------------------------------------------------------

    def move_to(
        self,
        robot_name: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        tol: float = 0.01,
        max_steps: int = 200,
        orientation_tol: float | None = None,
    ) -> dict[str, Any]:
        """Move the end-effector to a world-frame Cartesian target via IK.

        Composite analytic primitive (the staging/transport verb): solves
        inverse kinematics to the target with the shared mink
        damped-least-squares bridge, then drives the arm's position-servo
        actuators toward the solved configuration until the end-effector is
        within ``tol`` meters of the target or ``max_steps`` control ticks
        elapse (each tick advances a few physics substeps).

        The end-effector frame is auto-discovered per robot namespace
        (:func:`strands_robots.simulation.ik.discover_ee_frame`: a site naming
        the tool point or the end effector, else a body naming the end effector,
        else the chain's leaf body) - the same heuristic eef-delta policies use,
        so multi-robot scenes resolve the right arm. A site outranks a body of
        the same name: it is placed at the tool point, while the body origin
        sits at the link's mount.

        GRASP PRESERVATION (contract): gripper actuators (resolved by the same
        registry-metadata-first classification ``set_gripper`` uses, see
        :meth:`_resolve_gripper_actuators`) are excluded from the IK solve
        and its restart seeding, and are HELD at their live position for the
        whole servo descent - a closed gripper stays closed through staging
        and transport, so ``set_gripper("close") -> move_to(...)`` carries the
        held object rather than releasing it.

        COMMANDED-DOF SOLVE (contract): the IK solve is restricted to the
        joints this primitive drives, so ``ik_residual_m`` is the error the
        servo descent is actually left with. mink optimizes over every degree
        of freedom in the model, and an unrestricted solve borrows whatever is
        cheapest - a floating/mobile base, a second robot in the same world
        model, the gripper this primitive holds - none of which ``move_to``
        commands. A borrowed solve reports a near-zero residual for a pose the
        arm cannot hold, which reads as a solved target whose servo merely
        "needs more steps". When the restricted solve cannot reach the target,
        the refusal re-solves unrestricted and names the degrees of freedom
        that would have to move first, so an out-of-workspace point is
        distinguishable from one that needs base motion.

        NOT collision-aware: the straight servo descent can sweep through
        obstacles. Collision-aware transport is the curobo provider's job; this
        primitive deliberately hides the solver backend so that upgrade cannot
        change the surface. Motion is NOT recorded into an active dataset
        recording session (see module docstring).

        Args:
            robot_name: Robot to move; defaults to the single robot in the
                world (errors if ambiguous).
            position: World-frame target ``[x, y, z]`` in meters (required).
                Validated by the same rule the scene-construction calls use
                (:func:`strands_robots.utils.coerce_pose_vector`): three finite
                real components, a NumPy array accepted, a ``bool`` refused.
            orientation: Optional target orientation quaternion ``[w, x, y, z]``,
                validated the same way and normalized before it enters the solve
                (a non-unit quaternion is fine; a ~zero-norm one is refused).
                When omitted the solve is position-only - the right choice for
                arms with fewer than 6 DOF (e.g. SO-100/SO-101), which cannot
                realize an arbitrary full pose.
            tol: Position convergence tolerance in meters (> 0). Bounds the
                TRANSLATION only; ``orientation_tol`` bounds the rotation.
            max_steps: Max control ticks before returning a not-reached error
                (1..10000).
            orientation_tol: Orientation convergence tolerance in radians
                (> 0), defaulting to
                :data:`~strands_robots.simulation.motion_primitives_base._DEFAULT_ORIENTATION_TOL_RAD`.
                Only meaningful alongside an ``orientation`` target, and
                REFUSED without one rather than silently ignored.

        POSE CONVERGENCE (contract): a requested ``orientation`` is measured,
        not merely fed to the solver. Both the IK accept gate and the servo
        descent require the position within ``tol`` AND the orientation within
        ``orientation_tol``, so ``reached`` is never ``True`` with the wrist
        pointing somewhere the caller did not ask for. This matters because the
        servo stops at the tick both components converge: gating on the
        position alone cut the descent short while the orientation was still
        settling, which made the achieved orientation a function of ``tol`` - a
        tolerance documented in meters - with the miss reported nowhere.

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{reached, steps, position_error_m, ik_residual_m, ee_position,
            ee_orientation_wxyz, frame, frame_type}`` on arrival, plus
            ``{orientation_error_rad, orientation_tol_rad,
            ik_orientation_residual_rad}`` when an ``orientation`` was
            requested (absent for a position-only call, which has no
            orientation to report);
            ``{"status": "error", ...}`` with the same json block (including
            the residuals) when the pose is unreachable or servo convergence
            times out. An unreachable refusal names what is out of reach on two
            independent axes. WHICH HALF of a pose: a damped least-squares solve
            trades position against orientation, so a full-pose request on an
            arm with too few DOF typically satisfies the ROTATION and leaves the
            POSITION short - the refusal reports the same point solved
            position-only (``position_only_ik_residual_m``) and recommends
            omitting ``orientation`` when that alone is reachable. WHOSE REACH:
            ``unrestricted_ik_residual_m`` and ``uncommanded_joints_moved``
            report what a solve over the whole model could have reached and
            which uncommanded joints it needed, so the caller can tell an
            out-of-workspace target from one needing base motion. Never raises.
        """
        # ---- parameter validation (before touching the world) ----
        # Shared with the Isaac adapter (motion_primitives_base): same
        # pose-vector rule the scene-construction calls use, same tol /
        # max_steps domains, same wording.
        target, target_quat, max_steps, orientation_tol, arg_err = self._validate_move_to_args(
            position, orientation, tol, max_steps, orientation_tol
        )
        if arg_err is not None:
            return arg_err
        assert target is not None  # no error implies a coerced target

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
            sanity_err = self._workspace_sanity_error(robot_name, target, base)
            if sanity_err is not None:
                return sanity_err

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
                # commanded_dofs restricts the solve to the joints the servo
                # loop below actually drives. mink optimizes over the whole
                # world model, so an unrestricted solve satisfies the Cartesian
                # task with whatever DOF is cheapest - a floating base, a
                # second robot, the held gripper - and then reports a residual
                # for a configuration move_to never commands. On a mobile
                # manipulator that reads as a solved target followed by a servo
                # that "just needs more steps"; it never arrives.
                bridge = MinkIKBridge(
                    model,
                    frame_name,
                    frame_type,
                    orientation_cost=1.0 if target_quat is not None else 0.0,
                    max_iters=200,
                    commanded_dofs=self._commanded_dof_indices(model, arm_jact),
                )
            except (ImportError, RuntimeError, ValueError) as e:
                return _err(f"move_to: IK bridge unavailable: {e}")

            q0 = np.array(data.qpos, dtype=np.float64, copy=True)
            target_pose = np.eye(4, dtype=np.float64)
            target_pose[:3, 3] = target
            if target_quat is not None:
                quat = target_quat / np.linalg.norm(target_quat)
                rot = np.zeros(9, dtype=np.float64)
                self._mj.mju_quat2Mat(rot, quat)
                target_pose[:3, :3] = rot.reshape(3, 3)
            else:
                # Position-only: keep the current EE orientation in the target
                # pose (the zero orientation cost makes it a soft no-op).
                target_pose[:3, :3] = bridge.ee_pose(q0)[:3, :3]

            def pose_residuals(q: np.ndarray) -> tuple[float, float | None]:
                """(position residual in m, orientation residual in rad) of a solve.

                The orientation half is measured only when one was requested;
                a position-only solve has no rotational target to miss.
                """
                ee = bridge.ee_pose(q)
                pos_res = float(np.linalg.norm(ee[:3, 3] - target))
                if target_quat is None:
                    return pos_res, None
                ee_quat_solved = np.zeros(4, dtype=np.float64)
                self._mj.mju_mat2Quat(ee_quat_solved, np.ascontiguousarray(ee[:3, :3], dtype=np.float64).reshape(9))
                return pos_res, _quat_angle_error(target_quat, ee_quat_solved)

            q_star = bridge.solve(target_pose, q0)
            ik_residual, ik_orientation_residual = pose_residuals(q_star)
            # ONE scalar ranks a candidate solve against a target that is up to
            # two independent quantities, and is <= 1 exactly when every
            # requested component is within its own tolerance. Position-only
            # calls reduce to ik_residual / tol, i.e. the historical ordering.
            violation = self._pose_violation(ik_residual, float(tol), ik_orientation_residual, orientation_tol)

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
            if violation > 1.0:
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
                    residual_try, orientation_residual_try = pose_residuals(q_try)
                    violation_try = self._pose_violation(
                        residual_try, float(tol), orientation_residual_try, orientation_tol
                    )
                    if violation_try < violation:
                        q_star, ik_residual, ik_orientation_residual, violation = (
                            q_try,
                            residual_try,
                            orientation_residual_try,
                            violation_try,
                        )
                    if violation <= 1.0:
                        break

            # `violation` is the pose-aware miss metric: max(position/tol,
            # orientation/orientation_tol). With no orientation requested it
            # degenerates to position/tol, so this is exactly `ik_residual >
            # tol` there - and with one it also catches a solve that hit the
            # point while pointing the wrong way.
            if violation > 1.0:
                # Two independent refusal-path diagnoses. Neither may turn a
                # structured refusal into a raise.
                #
                # (a) WHICH HALF: a pose solve trades position against
                # orientation, so the residual alone cannot say which half is
                # out of reach. Solve the same point with the orientation task
                # switched off - that residual is the evidence for the remedy
                # the refusal recommends.
                position_only_residual: float | None = None
                if target_quat is not None:
                    try:
                        reference = MinkIKBridge(model, frame_name, frame_type, orientation_cost=0.0, max_iters=200)
                        reference_pose = np.eye(4, dtype=np.float64)
                        reference_pose[:3, 3] = target
                        reference_pose[:3, :3] = reference.ee_pose(q0)[:3, :3]
                        position_only_residual = float(
                            np.linalg.norm(reference.ee_pose(reference.solve(reference_pose, q0))[:3, 3] - target)
                        )
                    except (ImportError, RuntimeError, ValueError) as e:
                        logger.debug("move_to: position-only reference solve unavailable: %s", e)
                # (b) WHOSE REACH: solve the same target with every model DOF
                # free. If THAT fits tol, the point is inside the robot's reach
                # and outside this primitive's, so the refusal can name the
                # degrees of freedom that have to move first instead of
                # advising a closer target.
                unrestricted_residual, uncommanded = self._diagnose_unreachable(
                    model, frame_name, frame_type, target_quat, target_pose, target, q0, arm_jact, namespace
                )
                return self._move_to_unreachable_error(
                    robot_name,
                    target,
                    float(tol),
                    ik_residual=ik_residual,
                    frame_name=frame_name,
                    frame_type=frame_type,
                    orientation_tol=orientation_tol,
                    ik_orientation_residual=ik_orientation_residual,
                    position_only_residual=position_only_residual,
                    unrestricted_residual=unrestricted_residual,
                    uncommanded_joints=uncommanded,
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
        orientation_error: float | None = None if target_quat is None else math.inf
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
            # Convergence is measured on EVERY component the caller asked for.
            # Breaking on the position alone stops the descent while the
            # orientation is still settling, which made the achieved
            # orientation a function of `tol` - a tolerance documented in
            # meters - and left the miss unreported.
            if target_quat is not None:
                orientation_error = _quat_angle_error(target_quat, ee_quat)
            if self._pose_violation(position_error, float(tol), orientation_error, orientation_tol) <= 1.0:
                reached = True
                break

        return self._move_to_result(
            robot_name,
            target,
            float(tol),
            max_steps,
            reached=reached,
            steps_used=steps_used,
            position_error=position_error,
            ik_residual=ik_residual,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            frame_name=frame_name,
            frame_type=frame_type,
            orientation_error=orientation_error,
            orientation_tol=orientation_tol,
            ik_orientation_residual=ik_orientation_residual,
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
        and commands an end-point of its set-point range - the actuator
        ctrlrange, or the driven joint's limits when the MJCF left the
        ctrlrange unset (see :meth:`_gripper_setpoint_range`). With no metadata,
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
            ``{state, actuators, targets, setpoint_sources,
            gripper_joint_positions}`` - ``setpoint_sources`` naming, per
            actuator, where its bounds came from, so a substituted joint range
            is visible rather than silent; structured error when the gripper
            cannot be resolved or no source gives usable set-points. Never
            raises.
        """
        steps, arg_err = self._validate_set_gripper_args(state, steps)
        if arg_err is not None:
            return arg_err
        assert state is not None  # narrowed by the shared validator

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

            # Which END of the set-point range each state maps to: shared
            # registry-metadata-first mapping (open=HIGH / close=LOW
            # convention when no metadata; see
            # MotionPrimitivesCore._gripper_state_end).
            end = self._gripper_state_end(state, grip_meta)

            targets: dict[int, float] = {}
            setpoint_sources: dict[int, str] = {}
            for act_id in gripper_acts:
                span, detail = self._gripper_setpoint_range(model, act_id, jnt_by_act.get(act_id))
                if span is None:
                    act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id)
                    return _err(
                        f"set_gripper: actuator '{act_name}' has no usable open/close set-points - "
                        f"{detail}. Drive it directly with action='send_action'."
                    )
                lo, hi = span
                targets[act_id] = hi if end == "high" else lo
                setpoint_sources[act_id] = detail

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
        return self._set_gripper_result(
            robot_name,
            state,
            steps,
            act_names,
            {n: targets[a] for n, a in zip(act_names, gripper_acts, strict=True)},
            {n: setpoint_sources[a] for n, a in zip(act_names, gripper_acts, strict=True)},
            joint_positions,
        )

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
        target_yaw, max_steps, arg_err = self._validate_rotate_wrist_args(target_yaw, tol, max_steps)
        if arg_err is not None:
            return arg_err

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

        return self._rotate_wrist_result(
            robot_name,
            float(tol),
            max_steps,
            reached=reached,
            steps_used=steps_used,
            wrist_name=wrist_name,
            target_yaw=target_yaw,
            final_yaw=final_yaw,
            yaw_error=yaw_error,
        )
