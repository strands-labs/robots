"""Motion primitives for the Isaac backend - ``move_to`` / ``set_gripper`` / ``rotate_wrist``.

The Isaac half of the analytic motion primitives (GH #1645; Isaac parity work
GH #2123: the joint-space pair is #2154, the IK-backed ``move_to`` is #2155).
Before this module the only motion path on Isaac was the raw kinematic
``set_joint_positions`` write - no blocking move, no timeout/abort-reason
contract, no registry-driven gripper semantics. The primitives share their
backend-neutral half with the MuJoCo reference implementation
(:mod:`strands_robots.simulation.mujoco.motion_primitives`) through
:class:`~strands_robots.simulation.motion_primitives_base.MotionPrimitivesCore`:
parameter domains, registry gripper metadata (``closed``/``open`` ->
set-point-range end), the ``_GRIPPER_HINTS`` / ``_WRIST_HINTS`` name
fallbacks, step budgets, and the structured success / timeout envelopes. An
agent reading a refusal or a result payload sees the same sentence and the
same json keys whichever backend produced it (AGENTS.md: "Match docstrings to
semantics").

``move_to`` reuses the shared damped-least-squares IK bridge
(:class:`strands_robots.simulation.ik.MinkIKBridge`), which operates on the
MuJoCo model of the robot: Isaac registry robots carry MJCF sources, so the
kinematic model the solve runs on is resolved from the robot's
``data_config``. The Isaac articulation's joint ordering/namespacing is
reconciled with the MJCF-side solution through an explicit NAME-KEYED map
(MJCF joint name -> articulation DOF index); a solved joint that cannot be
mapped is a structured refusal, never a positional/flat-index write
(AGENTS.md: "Per-name state copy, not flat index").

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
from strands_robots.simulation.model_registry import resolve_model
from strands_robots.simulation.models import registered, registry_entry
from strands_robots.simulation.motion_primitives_base import (
    _GRIPPER_HINTS,
    _IK_RESTART_SEEDS,
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

    def _primitive_resolve_robot(
        self, action: str, robot_name: str | None
    ) -> tuple[str | None, Any | None, dict[str, Any] | None]:
        """Common primitive preamble: world + robot + articulation + policy guards.

        Returns ``(robot_name, robot_state, None)`` on success or
        ``(None, None, error_dict)``. Callers must hold ``self._lock`` (the
        checks read shared state). The guard wording is this backend's own
        (``send_action`` / ``set_joint_positions`` established it) - robot and
        world resolution is the half of the primitives that stays
        backend-specific, per the ``MotionPrimitivesCore`` split.

        The no-running-policy refusal lives here so *every* primitive carries
        it, at the same point the MuJoCo mixin's preamble calls
        ``_require_no_running_policy``: a primitive and the policy loop would
        race on the articulation's PD targets, and ``policy_running`` is the
        per-robot flag every Isaac policy-driving loop sets
        (:meth:`IsaacSimulation.run_multi_policy`, the recording rollout hook).
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
        if robot.policy_running:
            return (
                None,
                None,
                _err(
                    f"Cannot '{action}' on '{robot_name}' while its policy is running - a primitive "
                    "and the policy loop would race on the articulation's PD targets. Wait "
                    "for the rollout to finish (Isaac policy loops clear the flag on exit)."
                ),
            )
        return robot_name, robot, None

    def _primitive_abort_reason(self, action: str, robot_name: str) -> dict[str, Any] | None:
        """Mid-loop cancellation check (call under ``self._lock``).

        The primitive loops release the lock between control ticks, so the
        world can legitimately be destroyed, the robot removed, or a policy
        started while a primitive runs. Each of those aborts the primitive with
        a structured error - the same abort contract as the MuJoCo mixin -
        rather than stepping a torn-down stage or writing PD targets the policy
        loop is also writing. (Isaac has no in-place model recompile, so that
        MuJoCo abort branch alone has no Isaac counterpart.)
        """
        if not self._world_created or self._world is None:
            return _err(f"{action}: world was destroyed mid-run; aborting.")
        robot = registry_entry(self._robots, robot_name)
        if robot is None or robot.articulation is None:
            return _err(f"{action}: robot '{robot_name}' was removed mid-run; aborting.")
        if robot.policy_running:
            return _err(f"{action}: a policy started on '{robot_name}' mid-run; aborting.")
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

    # -- move_to kinematics plumbing ---------------------------------------------

    @staticmethod
    def _articulation_base_pose(articulation: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """World pose of the articulation base: ``(position (3,), quat wxyz (4,))`` or ``None``.

        Reads ``articulation.get_world_pose()`` - the read counterpart of the
        write :meth:`IsaacSimulation.set_robot_pose` drives - tolerating torch
        tensors and the exception set a torn-down articulation raises.
        ``None`` means "could not be read"; callers must answer that loudly,
        never by substituting an origin base (a wrong base makes every
        world-frame target silently wrong). The returned quaternion is
        normalized.
        """
        try:
            pose = articulation.get_world_pose()
        except (RuntimeError, ValueError, AttributeError, TypeError):
            return None
        if pose is None:
            return None
        try:
            pos_raw, quat_raw = pose
        except (TypeError, ValueError):
            return None
        if pos_raw is None or quat_raw is None:
            return None
        pos_arr = pos_raw.cpu().numpy() if hasattr(pos_raw, "cpu") else pos_raw
        quat_arr = quat_raw.cpu().numpy() if hasattr(quat_raw, "cpu") else quat_raw
        pos = np.asarray(pos_arr, dtype=np.float64).reshape(-1)
        quat = np.asarray(quat_arr, dtype=np.float64).reshape(-1)
        if pos.size != 3 or quat.size != 4:
            return None
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return None
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            return None
        return pos, quat / norm

    def _load_ik_mjcf(self, robot: Any) -> tuple[Any, Any, dict[str, Any] | None]:
        """Compile the MuJoCo model the IK solve runs on: ``(mujoco, model, None)`` or an error.

        The shared IK bridge (:class:`strands_robots.simulation.ik.MinkIKBridge`)
        operates on a compiled ``mujoco.MjModel``; Isaac registry robots carry
        MJCF sources, so the kinematic model is resolved from the robot's
        ``data_config`` through the same
        :func:`~strands_robots.simulation.model_registry.resolve_model` lookup
        the MuJoCo backend's ``add_robot`` uses (this module's ``resolve_model``
        global is the patch point for tests). Every failure - no
        ``data_config``, nothing resolves, ``mujoco`` not importable, the file
        does not compile - is a structured error naming the remedy, never a
        raise or a silent identity model.
        """
        data_config = getattr(robot, "data_config", None)
        if not data_config:
            return (
                None,
                None,
                _err(
                    f"move_to: robot '{robot.name}' has no data_config, so the registry MJCF the "
                    "IK solve runs on cannot be resolved. Re-add the robot with data_config=..., "
                    "or drive the joints directly with action='send_action'."
                ),
            )
        path = resolve_model(data_config)
        if path is None:
            return (
                None,
                None,
                _err(
                    f"move_to: no MJCF/URDF model resolves for data_config '{data_config}', so "
                    "there is no kinematic model for the IK solve. Register one "
                    "(strands_robots.simulation.model_registry.register_urdf), or drive the "
                    "joints directly with action='send_action'."
                ),
            )
        try:
            import mujoco as mj
        except ImportError:
            return (
                None,
                None,
                _err(
                    "move_to: the IK solve runs on the MuJoCo model of the robot and needs the "
                    "'mujoco' + 'mink' stack, which is not importable. Install the sim extra: "
                    "uv pip install 'strands-robots[sim-mujoco]'."
                ),
            )
        try:
            model = mj.MjModel.from_xml_path(path)
        except (ValueError, OSError, RuntimeError, mj.FatalError) as e:
            return (
                None,
                None,
                _err(f"move_to: could not compile the IK model for data_config '{data_config}' from {path}: {e}"),
            )
        return mj, model, None

    def _mjcf_articulation_joint_map(
        self, mj: Any, model: Any, robot: Any
    ) -> tuple[dict[int, int], dict[int, int], dict[str, Any] | None]:
        """Name-keyed map from MJCF joints onto articulation DOF indices.

        The key risk of running the IK on the MJCF while writing targets to
        the Isaac articulation is joint ORDER (#2123): the URDF importer and
        the MJCF compiler need not agree on DOF ordering, so the solved
        configuration is reconciled per NAME - MJCF joint name (namespace
        stripped on both sides, see :meth:`_short_joint_name`) -> articulation
        DOF index - and a solved joint that cannot be mapped is a structured
        refusal, never a positional/flat-index write (AGENTS.md: "Per-name
        state copy, not flat index").

        Returns ``(arm_map, grip_map, error)``: ``arm_map`` / ``grip_map`` map
        MJCF joint id -> articulation DOF index for the hinge/slide joints,
        split by the shared registry-metadata-first gripper classification
        (same vocabulary on both sides, so the split cannot disagree with
        :meth:`_resolve_gripper_dofs`). Every ARM joint must map - the solve
        commands them all; a gripper joint with no counterpart merely stays at
        the model default in the FK reads (it is held on the articulation
        side, not commanded from the solve). Callers must hold ``self._lock``.
        """
        short_names = [self._short_joint_name(n) for n in robot.joint_names]
        dof_by_name: dict[str, int] = {}
        for i, short in enumerate(short_names):
            dof_by_name.setdefault(short, i)

        meta, malformed_reason = self._registry_gripper_metadata(robot)
        if malformed_reason is not None:
            return {}, {}, _err(f"Cannot resolve the gripper for '{robot.name}': {malformed_reason}")
        wanted = {str(a).lower() for a in meta["actuators"]} if meta is not None else None

        def _is_gripper(short: str) -> bool:
            if wanted is not None:
                return short.lower() in wanted
            return any(h in short.lower() for h in _GRIPPER_HINTS)

        settable = {int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)}
        arm_map: dict[int, int] = {}
        grip_map: dict[int, int] = {}
        unmapped: list[str] = []
        for jnt_id in range(int(model.njnt)):
            if int(model.jnt_type[jnt_id]) not in settable:
                continue
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            short = self._short_joint_name(jname)
            dof = dof_by_name.get(short)
            if _is_gripper(short):
                if dof is not None:
                    grip_map[jnt_id] = dof
                continue
            if dof is None:
                unmapped.append(short or f"<unnamed joint {jnt_id}>")
                continue
            arm_map[jnt_id] = dof
        if unmapped:
            return (
                {},
                {},
                _err(
                    f"move_to: the IK model for '{robot.name}' (data_config '{robot.data_config}') "
                    f"has joints {unmapped} with no articulation counterpart. Joint targets are "
                    "written per NAME, never by flat index, so a solved joint that cannot be "
                    f"mapped is refused. Articulation joints: {short_names}."
                ),
            )
        if not arm_map:
            return (
                {},
                {},
                _err(
                    f"move_to: the IK model for '{robot.name}' (data_config '{robot.data_config}') "
                    "has no non-gripper hinge/slide joints to solve - nothing can move the "
                    "end-effector."
                ),
            )
        return arm_map, grip_map, None

    @staticmethod
    def _quat_wxyz_to_mat(mj: Any, quat: np.ndarray) -> np.ndarray:
        """``(3, 3)`` rotation matrix for a wxyz quaternion (MuJoCo convention)."""
        rot = np.zeros(9, dtype=np.float64)
        mj.mju_quat2Mat(rot, np.asarray(quat, dtype=np.float64))
        return rot.reshape(3, 3)

    # -- primitives --------------------------------------------------------------

    def move_to(
        self,
        robot_name: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        tol: float = 0.01,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Move the end-effector to a world-frame Cartesian target via IK.

        Composite analytic primitive (the staging/transport verb), same public
        API and result semantics as the MuJoCo backend's
        :meth:`~strands_robots.simulation.mujoco.motion_primitives.MotionPrimitivesMixin.move_to`:
        solves inverse kinematics to the target with the shared mink
        damped-least-squares bridge
        (:class:`strands_robots.simulation.ik.MinkIKBridge`; position-only
        when no orientation is given - important for 5-DOF arms like SO-101),
        then drives PD position targets on the Kit loop until the
        end-effector is within ``tol`` meters of the target or ``max_steps``
        control ticks elapse. Deterministic restart seeds
        (:data:`~strands_robots.simulation.motion_primitives_base._IK_RESTART_SEEDS`)
        retry a stalled direct solve; an unreachable target returns the IK
        residual in a structured error, never a raise.

        The bridge operates on the MuJoCo model of the robot, resolved from
        the robot's ``data_config`` (Isaac registry robots carry MJCF
        sources; see :meth:`_load_ik_mjcf`). The solve is seeded from the
        articulation's CURRENT joint positions and its result is reconciled
        with the articulation through an explicit NAME-KEYED joint map
        (:meth:`_mjcf_articulation_joint_map`) - a solved joint that cannot
        be mapped by name is a structured refusal, never a positional write.
        The world-frame target is mapped into the model frame through the
        articulation's live base pose (``get_world_pose``), which is read
        once at setup: the arm's base is assumed fixed for the duration of
        the move.

        CONVERGENCE MEASUREMENT: the EE pose is computed per tick by forward
        kinematics of the live joint readback through the SAME bridge frame
        the solver optimized (then mapped back to world through the base
        pose). Measuring a different frame (e.g. the USD gripper-link
        heuristic) could leave the solver and the convergence check watching
        points that never agree - the same trap the MuJoCo mixin's
        ``_frame_world_pose`` documents.

        GRASP PRESERVATION (contract): gripper DOFs (resolved by the same
        registry-metadata-first classification ``set_gripper`` uses, see
        :meth:`_resolve_gripper_dofs`) are excluded from the IK solve and its
        restart seeding, and are HELD at their live position for the whole
        servo descent - ``set_gripper("close") -> move_to(...)`` carries the
        held object rather than releasing it.

        REFUSES WHILE A POLICY RUNS on the same robot: ``policy_running`` is
        the per-robot flag every Isaac policy-driving loop sets
        (:meth:`IsaacSimulation.run_multi_policy`, the recording rollout
        hook), so it is this backend's counterpart of the MuJoCo
        ``_require_no_running_policy`` guard - checked up front and per
        control tick (a policy starting mid-run aborts the primitive).

        COMMANDED-DOF SOLVE (contract): the IK solve is restricted to the
        arm joints this primitive drives, so ``ik_residual_m`` is the error the
        servo descent is actually left with. mink optimizes over every degree
        of freedom in the IK model, and an unrestricted solve borrows whatever
        is cheapest - a floating/mobile base, the gripper this primitive holds
        - neither of which ``move_to`` commands; the borrowed solve then
        reports a near-zero residual for a pose the arm cannot hold. Same rule
        as the MuJoCo backend, so the two judge reachability identically. When
        the restricted solve cannot reach the target, the refusal re-solves
        unrestricted and names the degrees of freedom that would have to move
        first.

        NOT collision-aware: the straight servo descent can sweep through
        obstacles - the same contract as the MuJoCo backend, which
        deliberately hides the solver so a collision-aware upgrade cannot
        change this surface.

        Args:
            robot_name: Robot to move; defaults to the single robot in the
                world (errors if ambiguous).
            position: World-frame target ``[x, y, z]`` in meters (required).
                Validated by the same rule the scene-construction calls use
                (:func:`strands_robots.utils.coerce_pose_vector`): three finite
                real components, a NumPy array accepted, a ``bool`` refused.
            orientation: Optional target orientation quaternion ``[w, x, y, z]``
                in the world frame, validated the same way and normalized
                before it enters the solve. When omitted the solve is
                position-only - the right choice for arms with fewer than
                6 DOF (e.g. SO-100/SO-101).
            tol: Position convergence tolerance in meters (> 0).
            max_steps: Max control ticks before returning a not-reached error
                (1..10000).

        Returns:
            ``{"status": "success", ...}`` with a json block
            ``{reached, steps, position_error_m, ik_residual_m, ee_position,
            ee_orientation_wxyz, frame, frame_type}`` on arrival;
            ``{"status": "error", ...}`` with the same json block (including
            the residual) when servo convergence times out. The unreachable
            refusal (restricted IK residual > tol) carries
            ``{reached, steps, ik_residual_m, unrestricted_ik_residual_m,
            uncommanded_joints_moved, frame, frame_type}`` - the last two
            reporting what a solve over the whole model could have reached and
            which uncommanded joints it needed, so the caller can tell an
            out-of-workspace target from one needing base motion. Never
            raises.
        """
        # ---- parameter validation (before touching the world) ----
        # Shared with the MuJoCo adapter (motion_primitives_base): same
        # pose-vector rule the scene-construction calls use, same tol /
        # max_steps domains, same wording.
        target, target_quat, max_steps, arg_err = self._validate_move_to_args(position, orientation, tol, max_steps)
        if arg_err is not None:
            return arg_err
        assert target is not None  # no error implies a coerced target
        # Rebound under a plain-ndarray name for the closure below (mypy does
        # not carry an Optional narrowing across a closure boundary).
        target_world: np.ndarray = target

        def _move() -> dict[str, Any]:
            # ---- setup under the lock: guards, IK model, joint map, solve ----
            # Runs on the Kit-owning thread (see _run_primitive_on_kit), so the
            # articulation reads that seed the solve happen on the Kit loop.
            with self._lock:
                robot_name_resolved, robot, error = self._primitive_resolve_robot("move_to", robot_name)
                if error is not None:
                    return error
                assert robot_name_resolved is not None and robot is not None
                name: str = robot_name_resolved
                articulation = robot.articulation
                short_names = [self._short_joint_name(n) for n in robot.joint_names]

                base = self._articulation_base_pose(articulation)
                if base is None:
                    return _err(
                        f"move_to: could not read the articulation base pose for '{name}', so the "
                        "world-frame target cannot be mapped into the robot's model frame."
                    )
                base_pos, base_quat = base
                sanity_err = self._workspace_sanity_error(name, target_world, base_pos)
                if sanity_err is not None:
                    return sanity_err

                mj, model, load_err = self._load_ik_mjcf(robot)
                if load_err is not None:
                    return load_err

                from strands_robots.simulation.ik import discover_ee_frame

                # The registry MJCF is a standalone robot model, so discovery
                # runs un-namespaced (unlike the MuJoCo backend's shared
                # multi-robot world model).
                frame = discover_ee_frame(model, None)
                if frame is None:
                    return _err(
                        f"move_to: could not auto-discover an end-effector frame in the IK model "
                        f"for '{name}' (data_config '{robot.data_config}'). The model has no "
                        "TCP-like site or hand/tool body to track."
                    )
                frame_name, frame_type = frame

                arm_map, grip_map, map_err = self._mjcf_articulation_joint_map(mj, model, robot)
                if map_err is not None:
                    return map_err
                full_map = {**arm_map, **grip_map}

                # Articulation-side gripper DOFs, HELD at their live position
                # below (grasp preservation). Same shared classification as
                # the MJCF-side split, so the two cannot disagree.
                grip_dofs, _, grip_err = self._resolve_gripper_dofs(robot)
                if grip_err is not None:
                    return grip_err

                q_live = self._read_joint_positions(articulation)
                if q_live is None or q_live.size < len(short_names):
                    return _err(
                        f"move_to: could not read joint positions from '{name}' - the "
                        "articulation did not report a usable joint-position vector."
                    )

                try:
                    from strands_robots.simulation.ik import MinkIKBridge

                    # max_iters is raised well above the bridge default (20)
                    # for the same reason as the MuJoCo mixin: move_to jumps
                    # from the current pose to an arbitrary workspace point in
                    # one solve and needs the extra integration budget.
                    # commanded_dofs restricts the solve to the arm joints
                    # the articulation targets below actually drive. mink
                    # optimizes over every DOF of the IK model, so an
                    # unrestricted solve can satisfy the Cartesian task with a
                    # floating base or the held gripper and then report a
                    # residual for a configuration that is never commanded.
                    # Same restriction as the MuJoCo mixin, so the two backends
                    # judge reachability by the same rule.
                    bridge = MinkIKBridge(
                        model,
                        frame_name,
                        frame_type,
                        orientation_cost=1.0 if target_quat is not None else 0.0,
                        max_iters=200,
                        commanded_dofs=self._commanded_dof_indices(model, arm_map),
                    )
                except (ImportError, RuntimeError, ValueError) as e:
                    return _err(f"move_to: IK bridge unavailable: {e}")

                # World -> model-frame transform through the live base pose.
                base_rot = self._quat_wxyz_to_mat(mj, base_quat)
                target_local = base_rot.T @ (target_world - base_pos)

                # Seed the MJCF configuration from the LIVE articulation
                # state, scattered per NAME (never flat-index).
                q0 = np.array(model.qpos0, dtype=np.float64).reshape(-1).copy()
                for jnt_id, dof in full_map.items():
                    q0[int(model.jnt_qposadr[jnt_id])] = float(q_live[dof])

                target_pose = np.eye(4, dtype=np.float64)
                target_pose[:3, 3] = target_local
                if target_quat is not None:
                    quat = target_quat / np.linalg.norm(target_quat)
                    # The requested orientation is world-frame; express it in
                    # the model frame the bridge solves in.
                    target_pose[:3, :3] = base_rot.T @ self._quat_wxyz_to_mat(mj, quat)
                else:
                    # Position-only: keep the current EE orientation in the
                    # target pose (the zero orientation cost makes it a soft
                    # no-op).
                    target_pose[:3, :3] = bridge.ee_pose(q0)[:3, :3]

                q_star = bridge.solve(target_pose, q0)
                ik_residual = float(np.linalg.norm(bridge.ee_pose(q_star)[:3, 3] - target_local))

                # Damped-least-squares IK is a local method: from a distant
                # seed it can stall in a joint-limit / elbow-branch local
                # minimum even for a reachable target. Same deterministic
                # restart schedule as the MuJoCo mixin: the model's home
                # keyframe first when one exists, then uniform draws over the
                # ARM joints' ranges from a per-call fixed-seed RNG (identical
                # calls draw identical seeds - reproducibility, not a bug).
                # Gripper DOFs and everything else stay at the live state.
                if ik_residual > float(tol):
                    rng = np.random.default_rng(0)
                    settable_qadr = [int(model.jnt_qposadr[jnt_id]) for jnt_id in arm_map]
                    ranges = [
                        (float(model.jnt_range[jnt_id][0]), float(model.jnt_range[jnt_id][1]))
                        if bool(model.jnt_limited[jnt_id])
                        else (-np.pi, np.pi)
                        for jnt_id in arm_map
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
                        residual_try = float(np.linalg.norm(bridge.ee_pose(q_try)[:3, 3] - target_local))
                        if residual_try < ik_residual:
                            q_star, ik_residual = q_try, residual_try
                        if ik_residual <= float(tol):
                            break

                if ik_residual > float(tol):
                    # Refusal-path diagnosis only (mirrors the MuJoCo mixin):
                    # re-solve with every model DOF free so the refusal can
                    # tell a point outside the robot's workspace from one that
                    # needs motion this primitive does not command.
                    unrestricted_residual = math.inf
                    uncommanded: list[str] = []
                    try:
                        reference = MinkIKBridge(
                            model,
                            frame_name,
                            frame_type,
                            orientation_cost=1.0 if target_quat is not None else 0.0,
                            max_iters=200,
                        )
                    except (ImportError, RuntimeError, ValueError):  # pragma: no cover - restricted build worked
                        pass
                    else:
                        q_free = reference.solve(target_pose, q0)
                        unrestricted_residual = float(np.linalg.norm(reference.ee_pose(q_free)[:3, 3] - target_local))
                        uncommanded = self._uncommanded_joints_moved(mj, model, arm_map, q0, q_free)
                    return self._move_to_unreachable_error(
                        name,
                        target_world,
                        float(tol),
                        ik_residual=ik_residual,
                        frame_name=frame_name,
                        frame_type=frame_type,
                        unrestricted_residual=unrestricted_residual,
                        uncommanded_joints=uncommanded,
                    )

                # Command ARM DOFs to the solve, per name; HOLD gripper DOFs
                # at their live position (grasp preservation - and every
                # commanded channel is re-asserted per tick, mirroring the
                # MuJoCo mixin).
                targets: dict[int, float] = {
                    dof: float(q_star[int(model.jnt_qposadr[jnt_id])]) for jnt_id, dof in arm_map.items()
                }
                for dof in grip_dofs:
                    targets[dof] = float(q_live[dof])

            # ---- servo loop: self-locking per control tick ----
            steps_used = 0
            reached = False
            position_error = math.inf
            ee_pos_world: np.ndarray = np.array(target_world, dtype=np.float64, copy=True)
            ee_quat_world = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            q_fk = q0.copy()
            for _ in range(max_steps):
                with self._lock:
                    abort = self._primitive_abort_reason("move_to", name)
                    if abort is not None:
                        return abort
                    apply_err = self._apply_position_targets("move_to", name, articulation, targets)
                    if apply_err is not None:
                        return apply_err
                    self._primitive_tick()
                    q = self._read_joint_positions(articulation)
                if q is None or q.size < len(short_names):
                    return _err(f"move_to: could not read joint positions from '{name}' mid-run; aborting.")
                steps_used += 1
                # FK of the live joint readback through the SAME bridge frame
                # the solver optimized, scattered per NAME, mapped to world
                # through the base pose (see the docstring's convergence-
                # measurement contract).
                for jnt_id, dof in full_map.items():
                    q_fk[int(model.jnt_qposadr[jnt_id])] = float(q[dof])
                ee_local = bridge.ee_pose(q_fk)
                ee_pos_world = base_pos + base_rot @ ee_local[:3, 3]
                quat_out = np.zeros(4, dtype=np.float64)
                mj.mju_mat2Quat(quat_out, np.ascontiguousarray(base_rot @ ee_local[:3, :3]).reshape(9))
                ee_quat_world = quat_out
                position_error = float(np.linalg.norm(ee_pos_world - target_world))
                if position_error <= float(tol):
                    reached = True
                    break

            return self._move_to_result(
                name,
                target_world,
                float(tol),
                max_steps,
                reached=reached,
                steps_used=steps_used,
                position_error=position_error,
                ik_residual=ik_residual,
                ee_pos=ee_pos_world,
                ee_quat=ee_quat_world,
                frame_name=frame_name,
                frame_type=frame_type,
            )

        return self._run_primitive_on_kit("move_to", _move)

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

        REFUSES WHILE A POLICY RUNS on the same robot: ``policy_running`` is
        the per-robot flag every Isaac policy-driving loop sets
        (:meth:`IsaacSimulation.run_multi_policy`, the recording rollout
        hook), so it is this backend's counterpart of the MuJoCo
        ``_require_no_running_policy`` guard - checked up front and per
        control tick (a policy starting mid-run aborts the primitive).

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
            robot_name_resolved, robot, error = self._primitive_resolve_robot("set_gripper", robot_name)
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

        REFUSES WHILE A POLICY RUNS on the same robot: ``policy_running`` is
        the per-robot flag every Isaac policy-driving loop sets
        (:meth:`IsaacSimulation.run_multi_policy`, the recording rollout
        hook), so it is this backend's counterpart of the MuJoCo
        ``_require_no_running_policy`` guard - checked up front and per
        control tick (a policy starting mid-run aborts the primitive).

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
            robot_name_resolved, robot, error = self._primitive_resolve_robot("rotate_wrist", robot_name)
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
