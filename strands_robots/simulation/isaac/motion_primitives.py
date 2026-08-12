"""Joint-space motion primitives for the Isaac backend - ``set_gripper`` / ``rotate_wrist``.

The Isaac half of the analytic motion primitives (GH #1645; Isaac parity work
GH #2123, this module is #2154). Before this module the only motion path on
Isaac was the raw kinematic ``set_joint_positions`` write - no blocking move,
no timeout/abort-reason contract, no registry-driven gripper semantics. The
two joint-space primitives now share their backend-neutral half with the
MuJoCo reference implementation
(:mod:`strands_robots.simulation.mujoco.motion_primitives`) through
:class:`~strands_robots.simulation.motion_primitives_base.MotionPrimitivesCore`:
parameter domains, registry gripper metadata (``closed``/``open`` ->
set-point-range end), the ``_GRIPPER_HINTS`` / ``_WRIST_HINTS`` name
fallbacks, step budgets, and the structured success / timeout envelopes. An
agent reading a refusal or a result payload sees the same sentence and the
same json keys whichever backend produced it (AGENTS.md: "Match docstrings to
semantics").

``move_to`` (IK-backed, Cartesian) is deliberately not here - it lands in a
follow-up child of #2123.

What this adapter owns (the backend-specific half):

* robot / world resolution against ``IsaacSimulation``'s own state
  (``_robots`` / ``_world_created``), with this backend's established guard
  wording;
* the gripper / wrist DOF resolution against the articulation's ``joint_names``
  - the demangled URDF vocabulary (#1900), further stripped of any
  robot-namespace path prefix before the shared hints apply;
* actuation: PD position targets via
  ``articulation.apply_action(ArticulationAction(...))`` (the same Isaac Sim
  6.0 surface ``send_action`` drives - the pre-6.0
  ``set_joint_position_targets`` does not exist there), ticked by
  ``world.step``;
* the threading marshal: Isaac writes must happen on the thread that owns the
  Kit pump, so the drive loop runs inline on the owning thread or is submitted
  through :meth:`IsaacSimulation.run_on_main` when the pump is engaged.
  Unlike :meth:`IsaacSimulation._marshal_main_thread_affine` (whose callers do
  not all inspect the envelope), a primitive called off-thread with no pump
  returns a structured error - the primitives' documented never-raises
  contract.

Contract notes shared with the MuJoCo mixin: errors are structured dicts
(``{"status": "error", ...}``), never raised through the tool surface, and a
failure is never reported as a zero-valued/silent-default success. The loops
take ``self._lock`` per control tick, and if the world is destroyed or the
robot removed mid-run the loop aborts with a structured error rather than
stepping a torn-down stage.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.registry.robots import get_robot
from strands_robots.simulation.models import registered, registry_entry
from strands_robots.simulation.motion_primitives_base import (
    _GRIPPER_HINTS,
    _WRIST_HINTS,
    MotionPrimitivesCore,
    _err,
)


class IsaacMotionPrimitivesMixin(MotionPrimitivesCore):
    """Joint-space motion primitives mixed into ``IsaacSimulation``.

    **Coupling** (documentary contract, mirrored from the MuJoCo mixin):
    reaches into ``self._robots``, ``self._world``, ``self._lock``, the
    config, and the main-thread marshal helpers. The ``TYPE_CHECKING`` stubs
    below exist so mypy accepts those lookups.
    """

    if TYPE_CHECKING:
        _lock: Any  # threading.RLock from IsaacSimulation
        _world: Any
        _world_created: bool
        _robots: dict[str, Any]  # name -> _RobotState
        _config: Any  # IsaacConfig
        _sim_time: float
        _step_count: int
        _pump_running: bool

        def _on_main_thread(self) -> bool:
            """Provided by ``IsaacSimulation``; declared here for type-checkers."""
            raise NotImplementedError

        def run_on_main(self, fn: Any, timeout: float | None = None) -> Any:
            """Provided by ``IsaacSimulation``; declared here for type-checkers."""
            raise NotImplementedError

    # -- registry seam ---------------------------------------------------------

    @staticmethod
    def _get_registry_robot(data_config: str) -> dict[str, Any] | None:
        """Registry lookup seam (see ``MotionPrimitivesCore._get_registry_robot``).

        Resolves through this module's ``get_robot`` global so tests and user
        code have a patch point local to the Isaac adapter
        (``strands_robots.simulation.isaac.motion_primitives.get_robot``),
        mirroring the MuJoCo mixin's historical seam.
        """
        return get_robot(data_config)

    # -- shared guards / resolution --------------------------------------------

    @staticmethod
    def _short_joint_name(name: str | None) -> str:
        """Strip any robot-namespace path prefix for hint / metadata matching.

        Isaac articulations report the demangled URDF vocabulary
        (:mod:`strands_robots.simulation.isaac.joint_names`, #1900), but a
        robot loaded under a prim namespace can still report path-qualified
        names (``so101/wrist_roll``). The shared hints and the registry
        gripper metadata both speak the bare joint name, so matching happens
        on the segment after the final ``/`` - the counterpart of the MuJoCo
        mixin's ``robot.namespace`` strip.
        """
        return (name or "").rsplit("/", 1)[-1]

    def _primitive_resolve_robot(self, robot_name: str | None) -> tuple[str | None, Any | None, dict[str, Any] | None]:
        """Common primitive preamble: world + robot + articulation guards.

        Returns ``(robot_name, robot_state, None)`` on success or
        ``(None, None, error_dict)``. Callers must hold ``self._lock`` (the
        checks read shared state). The guard wording is this backend's own
        (``send_action`` / ``set_joint_positions`` established it) - robot and
        world resolution is the half of the primitives that stays
        backend-specific, per the ``MotionPrimitivesCore`` split.
        """
        if not self._world_created or self._world is None:
            return None, None, _err("No world created.")
        if robot_name is None:
            if len(self._robots) == 1:
                robot_name = next(iter(self._robots))
            elif not self._robots:
                return None, None, _err("No robots in the world.")
            else:
                return (
                    None,
                    None,
                    _err(f"Multiple robots present; specify robot_name. Available: {sorted(self._robots)}"),
                )
        if not registered(self._robots, robot_name):
            return None, None, _err(f"Robot '{robot_name}' not found.")
        robot = self._robots[robot_name]
        if robot.articulation is None:
            return None, None, _err(f"Robot {robot_name!r} not initialized.")
        return robot_name, robot, None

    def _primitive_abort_reason(self, action: str, robot_name: str) -> dict[str, Any] | None:
        """Mid-loop cancellation check (call under ``self._lock``).

        The primitive loops release the lock between control ticks, so the
        world can legitimately be destroyed or the robot removed while a
        primitive runs. Either aborts the primitive with a structured error -
        the same abort contract as the MuJoCo mixin - rather than stepping a
        torn-down stage. (Isaac has no in-place model recompile, and policy
        rollouts are driven externally by ``PolicyRunner``, so those MuJoCo
        abort branches have no Isaac counterpart.)
        """
        if not self._world_created or self._world is None:
            return _err(f"{action}: world was destroyed mid-run; aborting.")
        robot = registry_entry(self._robots, robot_name)
        if robot is None or robot.articulation is None:
            return _err(f"{action}: robot '{robot_name}' was removed mid-run; aborting.")
        return None

    def _run_primitive_on_kit(self, action: str, fn: Any) -> dict[str, Any]:
        """Run a drive loop on the thread that owns the Kit pump, never raising.

        Isaac Sim only pumps kit updates on the thread that created
        ``SimulationApp``, so ``world.step`` called from a worker thread blocks
        forever (see :meth:`IsaacSimulation._marshal_main_thread_affine`). On
        the owning thread ``fn`` runs inline; off it with the pump engaged it
        is submitted through :meth:`IsaacSimulation.run_on_main`; off it with
        no pump the primitive returns a structured error naming the recipe -
        a raise here would escape the primitives' documented never-raises
        contract, which is why this does not reuse the raising marshal.
        """
        if self._on_main_thread():
            return fn()
        if self._pump_running:
            return self.run_on_main(fn)
        return _err(
            f"{action}: called from a worker thread with no main-thread pump running. "
            "Isaac Sim only pumps kit updates on the thread that created SimulationApp, "
            "so this call would block forever. Either call it from the owning thread, or "
            "have the owning thread run `run_pump_forever(stop_event=...)` and submit the "
            "call from the worker via `run_on_main(lambda: ...)`."
        )

    # -- articulation readers / writers -----------------------------------------

    @staticmethod
    def _read_joint_positions(articulation: Any) -> np.ndarray | None:
        """Current joint positions as a flat float64 array, or ``None``.

        Tolerates the surfaces the rest of this backend already handles: a
        torch tensor (``.cpu().numpy()``, the pump's joint-cache read does the
        same) and the exception set a torn-down articulation raises. ``None``
        means "could not be read" - callers must answer that loudly, never by
        substituting zeros.
        """
        try:
            q = articulation.get_joint_positions()
        except (RuntimeError, ValueError, AttributeError, TypeError):
            return None
        if q is None:
            return None
        arr = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    @staticmethod
    def _articulation_dof_limits(articulation: Any, n_dofs: int) -> list[tuple[float, float] | None]:
        """Per-DOF ``(lower, upper)`` limits, ``None`` where none are usable.

        The Isaac counterpart of the MuJoCo mixin's ``jnt_range`` reads: the
        articulation's ``dof_properties`` structured array (``lower`` /
        ``upper``, honoring ``hasLimits`` when the field is present) is
        authoritative, with the view-shaped ``get_dof_limits()`` as the
        fallback surface. A DOF whose bounds are absent, non-finite, or
        degenerate (``upper <= lower``) reports ``None`` - the caller refuses
        loudly rather than mapping ``open``/``close`` onto a range that does
        not exist.
        """
        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        has_limits: np.ndarray | None = None
        props = getattr(articulation, "dof_properties", None)
        if props is not None:
            try:
                lower = np.asarray(props["lower"], dtype=np.float64).reshape(-1)
                upper = np.asarray(props["upper"], dtype=np.float64).reshape(-1)
            except (KeyError, ValueError, IndexError, TypeError):
                lower = upper = None
            else:
                try:
                    has_limits = np.asarray(props["hasLimits"], dtype=bool).reshape(-1)
                except (KeyError, ValueError, IndexError, TypeError):
                    has_limits = None
        if lower is None or upper is None:
            get_limits = getattr(articulation, "get_dof_limits", None)
            if get_limits is not None:
                try:
                    raw = get_limits()
                    arr = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
                    arr = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
                    lower, upper = arr[:, 0], arr[:, 1]
                except (RuntimeError, ValueError, AttributeError, TypeError, IndexError):
                    lower = upper = None
        out: list[tuple[float, float] | None] = []
        for dof in range(n_dofs):
            if lower is None or upper is None or dof >= lower.size or dof >= upper.size:
                out.append(None)
                continue
            if has_limits is not None and dof < has_limits.size and not bool(has_limits[dof]):
                out.append(None)
                continue
            lo, hi = float(lower[dof]), float(upper[dof])
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                out.append(None)
                continue
            out.append((lo, hi))
        return out

    def _apply_position_targets(
        self, action: str, robot_name: str, articulation: Any, targets: dict[int, float]
    ) -> dict[str, Any] | None:
        """Assert PD position targets on a DOF subset, as a structured result.

        The same Isaac Sim 6.0 surface ``send_action`` drives:
        ``apply_action(ArticulationAction(joint_positions=..., joint_indices=...))``
        commands ONLY the indexed DOFs and leaves the rest at their current PD
        targets. Same narrow exception set as ``send_action``'s write:
        ``RuntimeError`` (torn-down articulation), ``ValueError`` (shape
        mismatch), ``AttributeError`` (omni surface drift), ``ImportError``
        (isaacsim runtime not importable). Returns ``None`` on success.
        """
        try:
            from isaacsim.core.utils.types import (  # type: ignore[import-not-found]
                ArticulationAction,
            )

            indices = sorted(targets)
            articulation.apply_action(
                ArticulationAction(
                    joint_positions=np.array([targets[i] for i in indices], dtype=np.float32),
                    joint_indices=np.array(indices, dtype=np.int32),
                )
            )
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            return _err(f"{action}: failed to set joint position targets on '{robot_name}': {e}")
        return None

    def _primitive_tick(self) -> None:
        """One control tick: advance physics, keep the clock bookkeeping.

        Must be called under ``self._lock`` on the Kit-owning thread (the
        drive loops run inside :meth:`_run_primitive_on_kit`). One
        ``world.step`` per control tick - an Isaac step is a full kit update,
        so there is no MuJoCo-style substep multiplier here; the budget a
        caller passes IS the physics-step budget. Renders when the config is
        not headless, matching ``send_action``.
        """
        self._world.step(render=self._config.render_mode != "headless")
        self._sim_time += self._config.physics_dt
        self._step_count += 1

    def _resolve_gripper_dofs(self, robot: Any) -> tuple[list[int], dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve *robot*'s gripper DOF indices (registry first, heuristic fallback).

        The shared classification behind ``set_gripper`` (which commands these
        DOFs) and ``rotate_wrist`` (which must NOT pick one as the wrist).
        Resolution order, mirroring the MuJoCo mixin (GH #1658):

        1. Registry ``gripper`` metadata for the robot's ``data_config``, when
           present (authoritative): a DOF qualifies when its
           namespace-stripped joint name matches one of ``gripper.actuators``
           exactly (case-insensitive) - on Isaac the articulation drives
           joints directly, so the metadata's actuator names resolve against
           the joint vocabulary. Metadata that matches NO joint is a loud
           structured error, never a silent heuristic fallback.
        2. Name heuristic (zero-config fallback): the stripped joint name
           contains one of
           :data:`~strands_robots.simulation.motion_primitives_base._GRIPPER_HINTS`.

        Returns ``(dof_indices, metadata, error)`` on the same contract as the
        MuJoCo mixin's ``_resolve_gripper_actuators``. Callers must hold
        ``self._lock``.
        """
        short_names = [self._short_joint_name(n) for n in robot.joint_names]
        meta, malformed_reason = self._registry_gripper_metadata(robot)
        if malformed_reason is not None:
            return [], None, _err(f"Cannot resolve the gripper for '{robot.name}': {malformed_reason}")
        if meta is not None:
            wanted = {str(a).lower() for a in meta["actuators"]}
            matched = [i for i, short in enumerate(short_names) if short.lower() in wanted]
            if not matched:
                return (
                    [],
                    None,
                    _err(
                        f"Registry gripper metadata for '{robot.name}' (data_config "
                        f"'{robot.data_config}') names actuators {meta['actuators']} but none "
                        f"match a joint on the articulation. Articulation joints: {short_names}. "
                        "The registry entry is stale for this robot; fix it, or drive the "
                        "gripper directly with action='send_action'."
                    ),
                )
            return matched, meta, None
        matched = [i for i, short in enumerate(short_names) if any(h in short.lower() for h in _GRIPPER_HINTS)]
        return matched, None, None

    # -- primitives --------------------------------------------------------------

    def set_gripper(
        self,
        robot_name: str | None = None,
        state: str | None = None,
        steps: int = 12,
    ) -> dict[str, Any]:
        """Drive the gripper to an open/close set-point (atomic primitive).

        Resolves the gripper DOF(s) via :meth:`_resolve_gripper_dofs`
        (registry ``gripper`` metadata for the robot's ``data_config`` when
        present, else the ``gripper`` / ``finger`` / ``jaw`` name heuristic on
        the articulation's joint names) and commands an end-point of the
        joint's limit range as a PD position target on the Kit loop. With no
        metadata, ``"open"`` drives to the HIGH end and ``"close"`` to the LOW
        end - the SO-100/SO-101/Franka convention; registry metadata's
        ``closed``/``open`` fields override it per robot. Same public API and
        result semantics as the MuJoCo backend's
        :meth:`~strands_robots.simulation.mujoco.motion_primitives.MotionPrimitivesMixin.set_gripper`.

        Args:
            robot_name: Robot whose gripper to drive; defaults to the single
                robot in the world (errors if ambiguous).
            state: ``"open"`` or ``"close"`` (required).
            steps: Control ticks to hold the set-point (1..10000); each tick
                advances one physics step so the fingers actually travel.

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{state, actuators, targets, setpoint_sources,
            gripper_joint_positions}`` (keyed by the namespace-stripped joint
            names); structured error when the gripper cannot be resolved
            (listing the joint names), the registry metadata is
            stale/malformed, or the resolved joint has no usable limits to map
            ``open``/``close`` onto. Never raises.
        """
        steps, arg_err = self._validate_set_gripper_args(state, steps)
        if arg_err is not None:
            return arg_err
        assert state is not None  # narrowed by the shared validator
        # Rebound under a plain-``str`` name for the closure below (mypy does
        # not carry an Optional narrowing across a closure boundary).
        gripper_state: str = state

        with self._lock:
            robot_name_resolved, robot, error = self._primitive_resolve_robot(robot_name)
            if error is not None:
                return error
            assert robot_name_resolved is not None and robot is not None
            # Rebound under a plain-``str`` name: the drive closure below
            # captures it, and mypy does not carry an Optional narrowing
            # across a closure boundary.
            name: str = robot_name_resolved
            short_names = [self._short_joint_name(n) for n in robot.joint_names]

            grip_dofs, grip_meta, grip_err = self._resolve_gripper_dofs(robot)
            if grip_err is not None:
                return grip_err
            if not grip_dofs:
                return _err(
                    f"set_gripper: could not resolve a gripper joint for '{name}' - the "
                    f"registry carries no gripper metadata for this robot and no joint name "
                    f"contains {list(_GRIPPER_HINTS)}. Joints: {short_names}. "
                    "Drive it directly with action='send_action' instead."
                )

            # Which END of the set-point range each state maps to: shared
            # registry-metadata-first mapping (open=HIGH / close=LOW convention
            # when no metadata; see MotionPrimitivesCore._gripper_state_end).
            end = self._gripper_state_end(gripper_state, grip_meta)

            limits = self._articulation_dof_limits(robot.articulation, len(short_names))
            targets: dict[int, float] = {}
            setpoint_sources: dict[int, str] = {}
            for dof in grip_dofs:
                span = limits[dof]
                if span is None:
                    return _err(
                        f"set_gripper: joint '{short_names[dof]}' has no usable open/close "
                        "set-points - the articulation reports no finite joint limits for it. "
                        "Drive it directly with action='send_action'."
                    )
                lo, hi = span
                targets[dof] = hi if end == "high" else lo
                setpoint_sources[dof] = "articulation dof limits"
            articulation = robot.articulation

        def _drive() -> dict[str, Any]:
            for _ in range(steps):
                with self._lock:
                    abort = self._primitive_abort_reason("set_gripper", name)
                    if abort is not None:
                        return abort
                    apply_err = self._apply_position_targets("set_gripper", name, articulation, targets)
                    if apply_err is not None:
                        return apply_err
                    self._primitive_tick()
            with self._lock:
                q = self._read_joint_positions(articulation)
            if q is None or q.size < len(short_names):
                # The drive ran, but the readback the success payload promises
                # is gone - reporting success with fabricated positions would
                # be a silent default.
                return _err(
                    f"set_gripper: commanded '{name}' {gripper_state} but could not read the "
                    "gripper joint positions back from the articulation; the final state is "
                    "unverified."
                )
            return self._set_gripper_result(
                name,
                gripper_state,
                steps,
                [short_names[d] for d in grip_dofs],
                {short_names[d]: float(targets[d]) for d in grip_dofs},
                {short_names[d]: setpoint_sources[d] for d in grip_dofs},
                {short_names[d]: float(q[d]) for d in grip_dofs},
            )

        return self._run_primitive_on_kit("set_gripper", _drive)

    def rotate_wrist(
        self,
        robot_name: str | None = None,
        target_yaw: float | None = None,
        tol: float = 0.02,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Rotate the wrist joint to a set-point, holding the arm posture.

        Atomic primitive: resolves the wrist joint by name heuristic
        (``wrist_roll`` / ``wrist_yaw`` / ``wrist_rotate`` / ``wrist``, else
        the last non-gripper joint - the distal roll joint on most serial
        arms; gripper DOFs are excluded via the shared registry-metadata-first
        classification, :meth:`_resolve_gripper_dofs`), commands it to
        ``target_yaw`` while every other articulated DOF holds its current
        position as a PD target, and servoes on the Kit loop until the joint
        is within ``tol`` radians or ``max_steps`` control ticks elapse.
        Holding the other joints preserves the Cartesian EE position up to
        servo compliance. Same public API and result semantics as the MuJoCo
        backend's
        :meth:`~strands_robots.simulation.mujoco.motion_primitives.MotionPrimitivesMixin.rotate_wrist`.

        Args:
            robot_name: Robot whose wrist to rotate; defaults to the single
                robot in the world (errors if ambiguous).
            target_yaw: Wrist joint set-point in radians (required; must lie
                within the joint's limits when the articulation reports them).
            tol: Joint-angle convergence tolerance in radians (> 0).
            max_steps: Max control ticks before returning a not-reached error
                (1..10000).

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{reached, steps, wrist_joint, target_yaw, final_yaw,
            yaw_error_rad}``; structured error (with the residual) when the
            joint cannot be resolved, the registry gripper metadata is
            stale/malformed (same contract as ``set_gripper``), the target is
            out of range, or servo convergence times out. Never raises.
        """
        target_yaw, max_steps, arg_err = self._validate_rotate_wrist_args(target_yaw, tol, max_steps)
        if arg_err is not None:
            return arg_err

        with self._lock:
            robot_name_resolved, robot, error = self._primitive_resolve_robot(robot_name)
            if error is not None:
                return error
            assert robot_name_resolved is not None and robot is not None
            # Rebound under a plain-``str`` name: the servo closure below
            # captures it, and mypy does not carry an Optional narrowing
            # across a closure boundary.
            name: str = robot_name_resolved
            short_names = [self._short_joint_name(n) for n in robot.joint_names]

            # Exclude gripper DOFs via the shared registry-first classification
            # (GH #1661), not a raw name-hint match: on so101 the joints are
            # named 1..6 with no gripper hint, so the last-joint fallback below
            # would otherwise pick joint 6 - which IS the jaw. Stale/malformed
            # registry metadata is the same loud structured error set_gripper
            # returns, never a silent fallback.
            grip_dofs, _, grip_err = self._resolve_gripper_dofs(robot)
            if grip_err is not None:
                return grip_err
            grip_set = set(grip_dofs)

            # Candidates are every non-gripper DOF. The MuJoCo mixin narrows to
            # hinge joints; the Isaac articulation's per-DOF joint-type
            # reporting is not stable across API generations, so this adapter
            # relies on the gripper classification alone - the case that bit
            # (so101's revolute jaw) is excluded by it either way.
            non_gripper = [i for i in range(len(short_names)) if i not in grip_set]
            wrist_dof: int | None = None
            for hint in _WRIST_HINTS:
                matches = [i for i in non_gripper if hint in short_names[i].lower()]
                if matches:
                    wrist_dof = matches[-1]  # most distal on a name tie
                    break
            if wrist_dof is None and non_gripper:
                # Fallback: the last (most distal) non-gripper DOF.
                wrist_dof = non_gripper[-1]
            if wrist_dof is None:
                return _err(
                    f"rotate_wrist: could not resolve a wrist joint for '{name}'. "
                    f"Articulation joints: {short_names}. Drive one directly with "
                    "action='send_action' instead."
                )
            # Rebound as a plain ``int`` for the closure below, same reason
            # as ``name``.
            wrist_index: int = wrist_dof
            wrist_name = short_names[wrist_index]

            span = self._articulation_dof_limits(robot.articulation, len(short_names))[wrist_index]
            if span is not None:
                lo, hi = span
                if not (lo <= target_yaw <= hi):
                    return _err(
                        f"rotate_wrist: target_yaw={target_yaw} rad is outside joint "
                        f"'{wrist_name}' range [{lo:.3f}, {hi:.3f}] rad."
                    )

            q0 = self._read_joint_positions(robot.articulation)
            if q0 is None or q0.size < len(short_names):
                return _err(
                    f"rotate_wrist: could not read joint positions from '{name}' - the "
                    "articulation did not report a usable joint-position vector."
                )
            start_yaw = float(q0[wrist_index])
            # Hold every other DOF at its CURRENT position; command only the
            # wrist to the set-point (grasp preservation: a closed gripper's
            # DOF is held closed, mirroring the MuJoCo mixin).
            targets = {i: (target_yaw if i == wrist_index else float(q0[i])) for i in range(len(short_names))}
            articulation = robot.articulation

        def _servo() -> dict[str, Any]:
            steps_used = 0
            reached = False
            yaw_error = math.inf
            final_yaw = start_yaw
            for _ in range(max_steps):
                with self._lock:
                    abort = self._primitive_abort_reason("rotate_wrist", name)
                    if abort is not None:
                        return abort
                    apply_err = self._apply_position_targets("rotate_wrist", name, articulation, targets)
                    if apply_err is not None:
                        return apply_err
                    self._primitive_tick()
                    q = self._read_joint_positions(articulation)
                if q is None or wrist_index >= q.size:
                    return _err(f"rotate_wrist: could not read joint positions from '{name}' mid-run; aborting.")
                steps_used += 1
                final_yaw = float(q[wrist_index])
                yaw_error = abs(final_yaw - target_yaw)
                if yaw_error <= float(tol):
                    reached = True
                    break
            return self._rotate_wrist_result(
                name,
                float(tol),
                max_steps,
                reached=reached,
                steps_used=steps_used,
                wrist_name=wrist_name,
                target_yaw=target_yaw,
                final_yaw=final_yaw,
                yaw_error=yaw_error,
            )

        return self._run_primitive_on_kit("rotate_wrist", _servo)
