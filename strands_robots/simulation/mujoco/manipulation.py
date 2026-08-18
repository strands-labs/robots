"""Manipulation mixin - runtime attach/detach, robot actuation, dynamics reset.

Public, supported forms of the plan-execute-record primitives the
``so101_curobo`` example used to hand-roll against private surfaces
(GH #1533, PR 1 of the lift-the-engine plan):

- :meth:`ManipulationMixin.attach_bodies` / :meth:`ManipulationMixin.detach_bodies`
  - runtime grasp-assist. ``mode="weld"`` adds an equality constraint on the
  live spec holding the CURRENT relative pose; ``mode="kinematic"`` teleports
  the child body to follow the parent every physics step (the example's
  per-waypoint ``move_object`` + qvel-zero carry, locked down and public).
- :meth:`ManipulationMixin.actuate_robot` - add position-servo actuators to an
  actuator-less (URDF-loaded) arm so ``send_action`` / ``run_policy`` can
  drive it (replaces the example's ``_backend_state["spec"]`` surgery).
- :meth:`ManipulationMixin.zero_dynamics` - clear qvel/qacc/warmstart (the
  anti-explosion reset for kinematically teleported states).

Data-honesty caveat (documented per method): welded or kinematically carried
"grasps" are NOT physical grasps - no contact forces hold the object. Datasets
recorded with them contain idealized transport segments; label or gate such
episodes accordingly.

Concurrency contract: every method that writes MuJoCo ``model``/``data`` or
recompiles the spec takes ``self._lock`` and refuses to run while a policy is
active (``_require_no_running_policy``), matching the other scene mutations.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.models import registry_entry
from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _ensure_mujoco, mj_name_to_id
from strands_robots.simulation.mujoco.scene_ops import (
    actuate_robot_in_scene,
    actuator_driven_joint_ids,
    actuator_joint_id,
    add_weld_constraint,
    remove_equality_constraint,
)
from strands_robots.utils import boolean_flag_error

if TYPE_CHECKING:
    from strands_robots.simulation.models import SimRobot

logger = logging.getLogger(__name__)

# How many "joint <- actuator" pairs a refusal spells out before summarizing the
# rest. A dexterous hand carries dozens, and a message that lists every one
# buries the verdict it is there to deliver.
_MAX_NAMED_JOINTS = 4

# Keys inside world._backend_state. "attachments" is the registry of active
# attachments (both modes) keyed by child body name; "kinematic_attachments"
# holds only the per-step-follow entries consumed by the step hooks.
_ATTACH_REGISTRY_KEY = "attachments"
_KINEMATIC_ATTACH_KEY = "kinematic_attachments"


def _finite_positive(value: Any) -> bool:
    """True when ``value`` is a real, finite, strictly positive scalar."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _finite_non_negative(value: Any) -> bool:
    """True when ``value`` is a real, finite, >= 0 scalar."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def _relative_pose(mj: Any, data: Any, parent_id: int, child_id: int) -> tuple[list[float], list[float]]:
    """Pose of ``child_id`` expressed in ``parent_id``'s frame, from live data.

    Returns ``(relpos, relquat)`` with relquat in wxyz order. Callers must have
    run a forward pass so ``xpos`` / ``xquat`` are current.
    """
    parent_neg_quat = np.zeros(4)
    mj.mju_negQuat(parent_neg_quat, data.xquat[parent_id])
    relquat = np.zeros(4)
    mj.mju_mulQuat(relquat, parent_neg_quat, data.xquat[child_id])
    relpos = np.zeros(3)
    mj.mju_rotVecQuat(relpos, data.xpos[child_id] - data.xpos[parent_id], parent_neg_quat)
    return [float(v) for v in relpos], [float(v) for v in relquat]


def _find_free_joint(mj: Any, model: Any, body_id: int) -> int:
    """Return the id of the FREE joint carried by ``body_id``, or -1."""
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[jid]) == body_id:
            return jid
    return -1


class ManipulationMixin:
    """Runtime attach/actuate/zero primitives mixed into ``Simulation``.

    **Coupling** (see the :mod:`simulation` top-level docstring): reaches into
    ``self._world``, ``self._lock``, and the guard helpers. ``TYPE_CHECKING``
    stubs below exist so mypy accepts those lookups; they are a documentary
    contract, not an enforceable protocol.
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: SimWorld | None
        _lock: Any  # threading.RLock from Simulation

        def _require_no_running_policy(self, action_name: str, robot_name: str | None = None) -> dict[str, Any] | None:
            """Provided by ``Simulation``; declared here for type-checkers."""

        def _unknown_robot_msg(self, requested: object) -> str:
            """Provided by ``Simulation``; declared here for type-checkers."""

    # -- shared guards -----------------------------------------------------

    def _attachments(self) -> dict[str, dict[str, Any]]:
        """The attachment registry (child body name -> record), creating it lazily."""
        assert self._world is not None  # callers must check
        reg = self._world._backend_state.setdefault(_ATTACH_REGISTRY_KEY, {})
        assert isinstance(reg, dict)
        return reg

    def _kinematic_attachments(self) -> dict[str, dict[str, Any]]:
        """The per-step-follow subset of the registry, creating it lazily."""
        assert self._world is not None  # callers must check
        reg = self._world._backend_state.setdefault(_KINEMATIC_ATTACH_KEY, {})
        assert isinstance(reg, dict)
        return reg

    def attachment_involving(self, body_name: str) -> str | None:
        """Return the child key of an active attachment involving ``body_name``.

        Matches ``body_name`` as parent or child, exactly or as a robot
        namespace prefix (``"arm"`` matches an attachment whose parent is
        ``"arm/gripper_link"``). Used by ``remove_object`` / ``remove_robot``
        to refuse removing a body that an active attachment still references
        (a dangling weld would fail the next recompile; a dangling kinematic
        follow would silently detach).
        """
        if self._world is None:
            return None
        registry = self._world._backend_state.get(_ATTACH_REGISTRY_KEY) or {}
        prefix = f"{body_name}/"
        for child, record in registry.items():
            parent = str(record.get("parent", ""))
            for candidate in (parent, child):
                if candidate == body_name or candidate.startswith(prefix):
                    return child
        return None

    # -- attach / detach ----------------------------------------------------

    def attach_bodies(
        self,
        parent: str,
        child: str,
        mode: str = "weld",
        torquescale: float = 1.0,
    ) -> dict[str, Any]:
        """Rigidly attach ``child`` to ``parent`` at their CURRENT relative pose.

        The supported grasp-assist primitive for synthetic-data pipelines
        (GH #1533): after a gripper closes around an object, attach the object
        to the gripper body so it rides along for the transport segment, then
        :meth:`detach_bodies` at the place point.

        Modes:

        * ``"weld"`` (default): adds a weld equality constraint on the live
          spec (survives later scene recompiles) holding the current relative
          pose. The constraint is enforced by the solver, so external contacts
          still interact with both bodies.
        * ``"kinematic"``: every physics step, the child's freejoint pose is
          overwritten to follow the parent and its velocity zeroed - the
          example's teleport-carry, made public. Requires ``child`` to be a
          dynamic body with a freejoint (e.g. ``add_object(is_static=False)``).

        Data-honesty caveat: neither mode is a physical grasp - no friction or
        contact force holds the object. Downstream consumers of recorded
        episodes should treat the attached segment as idealized transport.

        Both bodies must exist in the live model; one attachment per child at
        a time. Attachments persist across ``reset()`` (a weld re-enforces the
        captured relative pose against the reset qpos; a kinematic follow
        re-teleports the child on the next step) - call :meth:`detach_bodies`
        first if the post-reset scene must be free. ``remove_object`` /
        ``remove_robot`` refuse to remove a body an attachment references.

        Args:
            parent: Body name the child follows (robot bodies are namespaced
                ``<robot>/<body>``; call ``list_bodies`` to discover names).
            child: Body name to carry. For ``mode="kinematic"`` it must own a
                freejoint.
            mode: ``"weld"`` or ``"kinematic"``.
            torquescale: Weld-only: torque-to-force ratio of the constraint
                (MuJoCo ``torquescale``). Must be finite and > 0.

        Returns:
            ``{status, content}`` tool result; ``status="error"`` when no world
            exists, a policy is running, a name does not resolve, the child is
            already attached, the mode is unknown, or the recompile fails.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("attach_bodies"):
            return err
        if mode not in ("weld", "kinematic"):
            return {
                "status": "error",
                "content": [{"text": f"attach_bodies: unknown mode {mode!r}. Use 'weld' or 'kinematic'."}],
            }
        if parent == child:
            return {
                "status": "error",
                "content": [{"text": f"attach_bodies: parent and child are the same body ({parent!r})."}],
            }
        if not _finite_positive(torquescale):
            return {
                "status": "error",
                "content": [
                    {"text": f"attach_bodies: 'torquescale' must be a finite number > 0, got {torquescale!r}."}
                ],
            }

        mj = _ensure_mujoco()
        with self._lock:
            model, data = self._world._model, self._world._data
            parent_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, parent)
            child_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, child)
            for label, body_id in (("parent", parent_id), ("child", child_id)):
                if body_id < 0:
                    name = parent if label == "parent" else child
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"attach_bodies: {label} body '{name}' not found. Robot bodies are "
                                    "namespaced '<robot>/<body>'; call list_bodies to discover names."
                                )
                            }
                        ],
                    }
            registry = self._attachments()
            if child in registry:
                held = registry[child]
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"attach_bodies: '{child}' is already attached to "
                                f"'{held['parent']}' (mode={held['mode']}). Detach it first."
                            )
                        }
                    ],
                }

            # Capture the CURRENT relative pose from live kinematics.
            mj.mj_forward(model, data)
            relpos, relquat = _relative_pose(mj, data, parent_id, child_id)

            if mode == "weld":
                eq_name = f"attach_weld_{parent}__{child}"
                if not add_weld_constraint(
                    self._world,
                    name=eq_name,
                    parent=parent,
                    child=child,
                    relpos=relpos,
                    relquat=relquat,
                    torquescale=float(torquescale),
                ):
                    return {
                        "status": "error",
                        "content": [{"text": f"attach_bodies: weld recompile failed for '{parent}' <- '{child}'."}],
                    }
                registry[child] = {"parent": parent, "mode": "weld", "eq_name": eq_name}
            else:
                free_jid = _find_free_joint(mj, model, child_id)
                if free_jid < 0:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"attach_bodies: mode='kinematic' requires child '{child}' to be a "
                                    "dynamic body with a freejoint (add_object(is_static=False)). "
                                    "Use mode='weld' for bodies without one."
                                )
                            }
                        ],
                    }
                record = {"parent": parent, "mode": "kinematic", "relpos": relpos, "relquat": relquat}
                registry[child] = record
                self._kinematic_attachments()[child] = record
                # Park the child at rest immediately so it doesn't drift/fall
                # between now and the first step-hook application.
                dof_adr = int(model.jnt_dofadr[free_jid])
                data.qvel[dof_adr : dof_adr + 6] = 0.0
                mj.mj_forward(model, data)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"'{child}' attached to '{parent}' (mode={mode}, relpos="
                        f"{[round(v, 4) for v in relpos]}). Note: not a physical grasp - "
                        "detach_bodies releases it."
                    )
                }
            ],
        }

    def detach_bodies(self, parent: str, child: str) -> dict[str, Any]:
        """Release an attachment created by :meth:`attach_bodies`.

        Weld mode deletes the equality constraint from the live spec and
        recompiles (preserving state); kinematic mode stops the per-step follow
        and leaves the child at rest at its current pose (its velocity was
        zeroed every step while carried), so it falls/settles physically from
        the release point.

        Returns ``status="error"`` when no world exists, a policy is running,
        no attachment exists for ``child``, or ``parent`` does not match the
        recorded attachment.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("detach_bodies"):
            return err

        with self._lock:
            registry = self._attachments()
            record = registry.get(child)
            if record is None:
                active = ", ".join(f"'{c}'<-'{r['parent']}'" for c, r in registry.items()) or "(none)"
                return {
                    "status": "error",
                    "content": [{"text": f"detach_bodies: no attachment found for child '{child}'. Active: {active}."}],
                }
            if record["parent"] != parent:
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"detach_bodies: '{child}' is attached to '{record['parent']}', not '{parent}'.")}
                    ],
                }

            if record["mode"] == "weld":
                if not remove_equality_constraint(self._world, record["eq_name"]):
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"detach_bodies: failed to remove weld constraint for '{child}' (see logs)."}
                        ],
                    }
            else:
                self._kinematic_attachments().pop(child, None)
            registry.pop(child, None)

        return {"status": "success", "content": [{"text": f"'{child}' detached from '{parent}'."}]}

    def _apply_kinematic_attachments(self) -> None:
        """Teleport every kinematically attached child to follow its parent.

        Called after EVERY ``mj_step`` the backend issues - the ``step()``
        batch loop, the single-robot ``_apply_sim_action`` substep loop, the
        motion-primitive tick, and the synchronized ``run_multi_policy`` loop -
        because ``attach_bodies(mode="kinematic")`` promises the child follows
        every physics step, and a stepping path that skips this leaves the
        carried body behind while reporting success. Callers MUST hold
        ``self._lock``. Uses the parent's ``xpos``/``xquat`` from the step's
        forward pass (one integration step of latency at the physics timestep,
        matching the example's carry). Entries whose bodies or freejoint no
        longer resolve (e.g. after a scene rebuild) are dropped with a warning
        so a stale name can't silently corrupt an unrelated joint.
        """
        world = self._world
        if world is None or world._model is None or world._data is None:
            return
        attachments = world._backend_state.get(_KINEMATIC_ATTACH_KEY)
        if not attachments:
            return
        mj = _ensure_mujoco()
        model, data = world._model, world._data
        stale: list[str] = []
        for child, record in attachments.items():
            parent_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, str(record["parent"]))
            child_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, child)
            free_jid = _find_free_joint(mj, model, child_id) if child_id >= 0 else -1
            if parent_id < 0 or child_id < 0 or free_jid < 0:
                stale.append(child)
                continue
            world_pos = np.zeros(3)
            mj.mju_rotVecQuat(world_pos, np.asarray(record["relpos"], dtype=float), data.xquat[parent_id])
            world_pos += data.xpos[parent_id]
            world_quat = np.zeros(4)
            mj.mju_mulQuat(world_quat, data.xquat[parent_id], np.asarray(record["relquat"], dtype=float))
            qpos_adr = int(model.jnt_qposadr[free_jid])
            dof_adr = int(model.jnt_dofadr[free_jid])
            data.qpos[qpos_adr : qpos_adr + 3] = world_pos
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = world_quat
            data.qvel[dof_adr : dof_adr + 6] = 0.0
        for child in stale:
            attachments.pop(child, None)
            registry = world._backend_state.get(_ATTACH_REGISTRY_KEY)
            if isinstance(registry, dict):
                registry.pop(child, None)
            logger.warning(
                "kinematic attachment for '%s' dropped: a referenced body/freejoint no longer exists",
                child,
            )

    # -- actuate ------------------------------------------------------------

    @staticmethod
    def _already_actuated_msg(
        model: Any,
        robot: SimRobot,
        driven: dict[int, int],
        mj: Any,
    ) -> str:
        """Word the refusal for a robot whose joints an actuator already drives.

        Names the joints and the actuators driving them, because the caller's
        next move differs by which it is: a robot driven throughout needs no
        actuators, while one driven in part has a mechanism reason for the rest
        (a mimic joint or a closed linkage member) that adding a servo would
        fight. The sibling refusals in this method already answer at that
        resolution - the unknown-``kp`` one lists the valid joints - so a bare
        "already has actuators" was the one verdict here that named neither what
        it found nor what it left.

        Args:
            model: The compiled ``MjModel``.
            robot: The robot being refused.
            driven: Driven joint id -> the id of an actuator driving it.
            mj: The ``mujoco`` module.

        Returns:
            The refusal text.
        """
        prefix = robot.namespace or ""

        def short(obj: Any, ident: int) -> str:
            full = mj.mj_id2name(model, obj, ident) or f"#{ident}"
            return full[len(prefix) :] if prefix and full.startswith(prefix) else full

        pairs = sorted(
            (short(mj.mjtObj.mjOBJ_JOINT, jnt), short(mj.mjtObj.mjOBJ_ACTUATOR, act)) for jnt, act in driven.items()
        )
        shown = ", ".join(f"{jnt} <- {act}" for jnt, act in pairs[:_MAX_NAMED_JOINTS])
        if len(pairs) > _MAX_NAMED_JOINTS:
            shown += f" and {len(pairs) - _MAX_NAMED_JOINTS} more"
        hinge_slide = sum(
            1
            for jnt_id in robot.joint_ids
            if int(model.jnt_type[jnt_id]) in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE))
        )
        free = hinge_slide - len(driven)
        tail = (
            "The robot is already drivable through those actuators, so it needs no added ones."
            if free <= 0
            else (
                f"The remaining {free} are positioned by the mechanism rather than by a ctrl of "
                "their own - a mimic joint or a closed linkage member - so a position servo there "
                "would fight the constraint that already holds them."
            )
        )
        return (
            f"actuate_robot: {len(driven)} of the {hinge_slide} hinge/slide joint(s) on robot "
            f"'{robot.name}' are already driven by an existing actuator ({shown}); refusing to "
            f"double-actuate, because a second drive on the same joint fights the first. {tail}"
        )

    def actuate_robot(
        self,
        robot_name: str,
        kp: float | dict[str, float] = 100.0,
        damping: float = 2.0,
        armature: float = 0.01,
        gravity_compensation: bool = True,
        disable_self_collision: bool = False,
    ) -> dict[str, Any]:
        """Convert an actuator-less (URDF-loaded) robot into a position-servo arm.

        URDF-loaded arms compile with no actuators (``nu == 0``), so
        ``send_action`` / ``run_policy`` cannot drive them - the reason the
        ``so101_curobo`` example fell back to kinematic qpos teleports
        (GH #1533). This adds a position actuator per hinge/slide joint
        (fixed gain ``kp``, ~critically damped, ctrlrange inherited from the
        joint's limits when it declares any), floors joint ``damping`` /
        ``armature`` (bare URDFs ship none, which blows up explicit
        integration), and switches the scene to the stable ``implicitfast``
        integrator. The change lives on the spec, so it survives later scene
        recompiles.

        After the recompile every new actuator's ``ctrl`` is initialized to
        the joint's CURRENT position so the arm holds its pose instead of
        snapping to zero.

        Scene-wide side effect (documented, not optional): the integrator is
        set to ``implicitfast`` - stiff position servos on undamped URDF
        chains diverge under the default Euler integrator.

        Args:
            robot_name: Robot to actuate. Must exist, and none of its joints
                may already be driven by an actuator - including through a
                fixed tendon, which couples several joints to one ``ctrl`` and
                is the standard MJCF idiom for a coupled gripper.
            kp: Position gain. A single number applies to every hinge/slide
                joint; a ``{short_joint_name: kp}`` dict actuates ONLY the
                listed joints (unknown names are rejected with the valid
                list). Values must be finite and > 0.
            damping: Per-joint damping floor (Ns/m or Nms/rad); existing
                larger values are kept. Must be finite and >= 0.
            armature: Per-joint armature (rotor inertia) floor; existing
                larger values are kept. Must be finite and >= 0.
            gravity_compensation: Apply ``gravcomp=1`` to the robot's bodies
                so modest gains track tightly. Must be a boolean - a truthy
                string such as ``"no"`` is refused rather than read as True.
            disable_self_collision: Zero contype/conaffinity on the robot's
                own geoms. Planners like cuRobo ignore adjacent-link contacts,
                which otherwise block planned motion in MuJoCo. NOTE: this
                disables ALL collision on the robot's geoms (not just
                link-vs-link), so add contact geoms that must keep colliding
                (e.g. fingertip pads) AFTER this call via ``patch_scene_mjcf``.
                Must be a boolean, for the same reason and more sharply: read by
                truthiness, ``"false"`` would disable every collision on the
                robot for a caller spelling the opt-out.

        Returns:
            ``{status, content}`` tool result; ``status="error"`` when no world
            exists, a policy is running, the robot is unknown or already driven,
            ``kp``/``damping``/``armature`` are invalid, either posture flag was
            not supplied as a boolean, or the recompile fails.
            The already-driven refusal names the joints it found and the
            actuators driving them, because what a caller does next differs by
            whether the robot is driven throughout or only in part.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("actuate_robot", robot_name):
            return err
        robot = registry_entry(self._world.robots, robot_name)
        if robot is None:
            return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
        if not _finite_non_negative(damping):
            return {
                "status": "error",
                "content": [{"text": f"actuate_robot: 'damping' must be a finite number >= 0, got {damping!r}."}],
            }
        if not _finite_non_negative(armature):
            return {
                "status": "error",
                "content": [{"text": f"actuate_robot: 'armature' must be a finite number >= 0, got {armature!r}."}],
            }

        # Read by truthiness these two would invert for exactly the spellings a
        # caller opting out reaches for: "false", "no", "off" and "0" are all
        # truthy, and disable_self_collision then zeroes contype/conaffinity on
        # every one of the robot's geoms - on the spec, so it outlives the
        # recompile - while the caller asked to keep them colliding. Their
        # numeric siblings above have been on a shared domain all along.
        for flag_name, flag_value in (
            ("gravity_compensation", gravity_compensation),
            ("disable_self_collision", disable_self_collision),
        ):
            if text := boolean_flag_error(flag_value, flag_name, "actuate_robot"):
                return {"status": "error", "content": [{"text": text}]}

        mj = _ensure_mujoco()
        with self._lock:
            model, data = self._world._model, self._world._data

            robot_joint_ids = set(robot.joint_ids)
            # Resolve the transmission rather than comparing ``actuator_trnid``
            # raw: a site or body transmission carries an id from a different
            # space, so a raw compare refuses a robot whose joint merely shares
            # a number with one. A TENDON id is from another space too, but the
            # joints that tendon wraps ARE driven by it, so the shared rule
            # reports them instead of reporting none.
            driven: dict[int, int] = {}
            for act_id in range(model.nu):
                for jnt_id in actuator_driven_joint_ids(model, act_id, mj) & robot_joint_ids:
                    driven.setdefault(jnt_id, act_id)
            if driven:
                return {
                    "status": "error",
                    "content": [{"text": self._already_actuated_msg(model, robot, driven, mj)}],
                }

            # Eligible joints: the robot's hinge/slide joints, by SHORT name.
            prefix = robot.namespace or ""
            eligible: dict[str, int] = {}
            for jid in robot.joint_ids:
                if int(model.jnt_type[jid]) not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
                    continue
                full = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or ""
                short = full[len(prefix) :] if prefix and full.startswith(prefix) else full
                if short:
                    eligible[short] = jid
            if not eligible:
                return {
                    "status": "error",
                    "content": [{"text": f"actuate_robot: robot '{robot_name}' has no hinge/slide joints to actuate."}],
                }

            if isinstance(kp, dict):
                unknown = sorted(set(kp) - set(eligible))
                if unknown:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"actuate_robot: unknown joint(s) in kp: {unknown}. "
                                    f"Valid joints for '{robot_name}': {sorted(eligible)}."
                                )
                            }
                        ],
                    }
                if not kp:
                    return {
                        "status": "error",
                        "content": [{"text": "actuate_robot: kp dict is empty - nothing to actuate."}],
                    }
                bad = {name: value for name, value in kp.items() if not _finite_positive(value)}
                if bad:
                    return {
                        "status": "error",
                        "content": [{"text": f"actuate_robot: kp values must be finite numbers > 0, got {bad!r}."}],
                    }
                kp_by_joint = {name: float(value) for name, value in kp.items()}
            else:
                if not _finite_positive(kp):
                    return {
                        "status": "error",
                        "content": [{"text": f"actuate_robot: 'kp' must be a finite number > 0, got {kp!r}."}],
                    }
                kp_by_joint = {short: float(kp) for short in eligible}

            if not actuate_robot_in_scene(
                self._world,
                robot,
                kp_by_joint,
                damping=float(damping),
                armature=float(armature),
                gravity_compensation=gravity_compensation,
                disable_self_collision=disable_self_collision,
            ):
                return {
                    "status": "error",
                    "content": [
                        {"text": f"actuate_robot: spec recompile failed for '{robot_name}' (spec restored, see logs)."}
                    ],
                }

            # Hold the current pose: initialize every NEW actuator's ctrl to
            # its joint's current qpos so the arm doesn't snap to zero on the
            # next step. _recompile_preserving_state already re-discovered
            # robot.actuator_ids.
            model, data = self._world._model, self._world._data
            for act_id in robot.actuator_ids:
                # A non-joint transmission has no joint position to hold, and its
                # trnid indexes another entity table - seeding ctrl from it would
                # command a gripper from an unrelated joint's angle.
                jnt_id = actuator_joint_id(model, act_id, mj)
                if jnt_id >= 0:
                    data.ctrl[act_id] = data.qpos[int(model.jnt_qposadr[jnt_id])]
            mj.mj_forward(model, data)
            n_added = len(kp_by_joint)
            nu = int(model.nu)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Robot '{robot_name}' actuated: {n_added} position actuator(s) added "
                        f"({sorted(kp_by_joint)}), nu={nu}, integrator=implicitfast, "
                        f"gravcomp={'on' if gravity_compensation else 'off'}, "
                        f"self_collision={'off' if disable_self_collision else 'on'}. "
                        "ctrl initialized to the current pose."
                    )
                }
            ],
        }

    # -- zero dynamics --------------------------------------------------------

    def zero_dynamics(self, robot_name: str | None = None) -> dict[str, Any]:
        """Zero velocities and accelerations, world-wide or for one robot.

        The anti-explosion reset for kinematically written states: writing
        ``qpos`` directly (teleports, trajectory replay) leaves ``qvel`` /
        ``qacc`` holding pre-teleport values, and integrating explicit
        dynamics from that discontinuity can diverge ("QACC nan"). This clears
        ``qvel``, ``qacc``, and ``qacc_warmstart`` - for every DOF, or only
        the named robot's joints - then re-forwards derived state.

        Args:
            robot_name: When given, only that robot's joint DOFs are zeroed
                (object freejoints and other robots keep their momentum).
                ``None`` zeroes every DOF in the world.

        Returns:
            ``{status, content}`` tool result; ``status="error"`` when no
            world exists, a policy is running, or the robot is unknown.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("zero_dynamics", robot_name):
            return err

        mj = _ensure_mujoco()
        with self._lock:
            model, data = self._world._model, self._world._data
            if robot_name is None:
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
                data.qacc_warmstart[:] = 0.0
                scope = f"all {int(model.nv)} DOFs"
            else:
                robot = registry_entry(self._world.robots, robot_name)
                if robot is None:
                    return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
                n_zeroed = 0
                for jid in robot.joint_ids:
                    jnt_type = int(model.jnt_type[jid])
                    if jnt_type == int(mj.mjtJoint.mjJNT_FREE):
                        width = 6
                    elif jnt_type == int(mj.mjtJoint.mjJNT_BALL):
                        width = 3
                    else:
                        width = 1
                    dof_adr = int(model.jnt_dofadr[jid])
                    data.qvel[dof_adr : dof_adr + width] = 0.0
                    data.qacc[dof_adr : dof_adr + width] = 0.0
                    data.qacc_warmstart[dof_adr : dof_adr + width] = 0.0
                    n_zeroed += width
                scope = f"{n_zeroed} DOFs of robot '{robot_name}'"
            mj.mj_forward(model, data)

        return {"status": "success", "content": [{"text": f"Dynamics zeroed ({scope}): qvel, qacc, qacc_warmstart."}]}
