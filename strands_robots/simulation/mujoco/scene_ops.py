"""Scene mutation via the MuJoCo ``MjSpec`` AST.

This module used to contain ~980 lines of XML-round-trip machinery (tmpdir +
``mj_saveLastXML`` + ElementTree parse + name-mangling + regex path patching).
All of that is replaced by ``spec.recompile(model, data)`` which:

* preserves joint state on unchanged joints automatically,
* initializes new joints to body ``pos``/``quat`` (removing the need to
  delete keyframes on freejoint insertion),
* namespaces robot bodies/joints/geoms/actuators/sensors via ``spec.attach()``
  without us walking the tree manually.

Public API:

* :func:`inject_robot_into_scene` - ``spec.attach(robot_spec, prefix=...)``.
* :func:`inject_object_into_scene` - ``SpecBuilder.add_object(spec, obj)`` + recompile.
* :func:`inject_camera_into_scene` - ``SpecBuilder.add_camera(spec, cam)`` + recompile.
* :func:`install_compiled_model` - install a compiled model/data pair as the
  live scene state; the single point every model swap goes through.
* :func:`eject_body_from_scene` - ``SpecBuilder.remove_body(spec, name)`` + recompile.
* :func:`reposition_body_in_scene` - edit a body's spec ``pos``/``quat`` + recompile.
* :func:`eject_robot_from_scene` - walk the spec, delete everything namespaced
  under ``{robot_name}/``, then recompile.
* :func:`refresh_body_inertial_from_geometry` - re-derive a body's mass /
  center of mass / inertia after one of its geoms was resized at runtime.
* :func:`fromto_fixed_size_components` - which ``geom_size`` components a geom's
  ``<fromto>`` fixes, so a resize of one can be refused rather than reported.

Every function takes a ``SimWorld`` whose ``_backend_state["spec"]`` holds the
live ``MjSpec``. They return ``True`` on success, ``False`` on failure (matching
the legacy API) so call sites in :mod:`simulation` don't need to change.
"""

from __future__ import annotations

import difflib
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco, filter_mujoco_attach_noise, mj_name_to_id
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder
from strands_robots.utils import coerce_rgba, finite_vector_error, pose_vector_error

logger = logging.getLogger(__name__)


def _get_spec(world: SimWorld) -> Any | None:
    """Fetch the live MjSpec from ``world._backend_state``.

    Callers MUST have run ``_compile_world`` at least once before any scene
    mutation - without a spec we can't recompile. Returns ``None`` if missing
    so callers can return a clean error dict rather than crashing mid-op.
    """
    return world._backend_state.get("spec")


def _sync_cached_xml(world: SimWorld, spec: Any) -> None:
    """Refresh the legacy XML cache in ``world._backend_state["xml"]`` from ``spec``.

    Some readers (the ``load_scene`` + ``add_robot`` round-trip) consume the
    cached XML string rather than the live ``MjSpec``, so it must be refreshed
    after every recompile. ``spec.to_xml()`` can fail on specs MuJoCo cannot
    serialise; that is never fatal to the mutation (the live model/spec are
    already updated), so we leave the previous cache in place - but we always
    log the reason at debug so a silently stale cache stays diagnosable rather
    than being swallowed.
    """
    try:
        with filter_mujoco_attach_noise():
            world._backend_state["xml"] = spec.to_xml()
    except Exception as xml_err:
        logger.debug("spec.to_xml() failed; cached XML left stale: %s", xml_err)


def _raise_spec_joint_damping(joint: Any, floor: float) -> None:
    """Floor a spec joint's damping, on either MuJoCo layout of that field.

    ``MjsJoint.damping`` is a per-DOF sequence on MuJoCo builds from 3.10 and a
    plain ``float`` on the older builds this package still supports
    (``mujoco>=3.2.0``). Reading or writing through the wrong one of those two
    layouts raises ``TypeError``, which
    :func:`actuate_robot_in_scene` reports as a refused spec surgery - so a
    robot that needs a damping floor cannot be actuated at all. Both layouts are
    handled here, once, rather than at the call site.

    Existing damping larger than ``floor`` is kept, which is the floor contract
    :meth:`strands_robots.simulation.mujoco.manipulation.ManipulationMixin.actuate_robot`
    documents for its ``damping`` argument.

    Args:
        joint: The ``MjsJoint`` (or any object exposing a ``damping`` field in
            one of the two layouts) to floor in place.
        floor: Minimum damping to leave on the joint.
    """
    current = joint.damping
    try:
        first = float(current[0])
    except TypeError:
        # Scalar layout (mujoco < 3.10): assign the field itself.
        joint.damping = max(float(current), floor)
    else:
        # Per-DOF layout: write element 0, leaving any further DOFs alone.
        joint.damping[0] = max(first, floor)


def _snapshot_spec(spec: Any, *, context: str) -> Any | None:
    """Deep-copy ``spec`` so a refused mutation can be rolled back losslessly.

    A spec mutation that is only validated by the recompile it precedes needs a
    way back. Rebuilding the spec from the registered objects / cameras / robots
    is not it: the live spec can carry mutations that exist ONLY on the spec and
    are absent from that registry - weld equalities from ``attach_bodies``,
    actuators from :func:`actuate_robot_in_scene`, bodies from
    :func:`patch_scene_mjcf`, whole scenes from :func:`replace_scene_mjcf`. A
    rebuild silently drops every one of them, which is why
    :meth:`Simulation.remove_robot` refuses outright while an attachment is
    live rather than rebuilding over it.

    ``MjSpec.copy`` is used rather than a ``spec.to_xml()`` round-trip because
    the round-trip is not faithful for a scene holding attached robots: the
    emitted MJCF loses the asset search paths its mesh references were resolved
    against and re-declares the attached model's keyframes, so the restored spec
    no longer compiles ("Error opening file 'link2.stl'", "repeated name
    'panda/home' in key") - a rollback that leaves the scene as broken as the
    orphan it removed.

    Args:
        spec: live scene spec to snapshot.
        context: short description of the caller, used in log messages.

    Returns:
        An independent copy of ``spec``, or ``None`` when the copy failed (the
        reason is logged). A caller that cannot snapshot must refuse its
        mutation rather than proceed with no way back.
    """
    try:
        return spec.copy()
    except (ValueError, RuntimeError) as e:
        logger.error("%s: cannot snapshot the scene spec: %s", context, e)
        return None


def actuator_joint_id(model: Any, act_id: int, mj: Any) -> int:
    """Return the joint id actuator ``act_id`` drives, or ``-1`` if it drives none.

    ``actuator_trnid[act_id, 0]`` holds a JOINT id only when the actuator's
    transmission is ``mjTRN_JOINT`` or ``mjTRN_JOINTINPARENT``. For the other
    transmissions it holds a tendon, site, slider-crank or body id instead --
    separate id spaces that each start at 0, so a raw comparison against joint
    ids matches an unrelated entity that merely shares a number. A fixed tendon
    coupling a gripper's fingers is the standard MJCF idiom for that (the
    Menagerie Panda hand and the Robotiq 2F-85 both use one), so the collision
    is reachable with stock assets rather than only with exotic models.

    Reading the transmission through this function is what keeps "which joint
    does this actuator drive" a single rule, rather than one that each call site
    re-derives and can omit.

    Args:
        model: The compiled ``MjModel``.
        act_id: Actuator index in ``range(model.nu)``.
        mj: The ``mujoco`` module.

    Returns:
        The driven joint id, or ``-1`` when the transmission is not a joint.
    """
    joint_trn = {int(mj.mjtTrn.mjTRN_JOINT)}
    if hasattr(mj.mjtTrn, "mjTRN_JOINTINPARENT"):
        joint_trn.add(int(mj.mjtTrn.mjTRN_JOINTINPARENT))
    if int(model.actuator_trntype[act_id]) not in joint_trn:
        return -1
    return int(model.actuator_trnid[act_id, 0])


def joint_drive_map(model: Any, mj: Any) -> tuple[dict[int, int], dict[int, int]]:
    """Split the joint-driving actuators into position servos and other drives.

    A *position servo* is an actuator whose ``ctrl`` IS the joint target in the
    joint's own units, so a caller may write a pose into it. That takes three
    terms, because MuJoCo's actuator shortcuts overlap on any one of them
    (measured values below are what ``mujoco`` compiles each shortcut to):

    * ``biastype == mjBIAS_AFFINE`` - the force carries a bias computed from the
      joint's own state rather than from ``ctrl`` alone. Necessary but far from
      sufficient: ``<velocity kv>`` (``biasprm = [0, 0, -kv]``) and
      ``<intvelocity kp>`` (``biasprm = [0, -kp, 0]``) are both affine-bias too.
    * ``biasprm[1] < 0`` - that slot is ``-kp``, the *position* feedback gain, so
      a negative value is what makes the bias restore toward ``ctrl`` as a pose.
      It is zero for ``<velocity>``, whose feedback is on the rate, and positive
      bias here is not a restoring term at all (``<cylinder bias>`` is free to
      set it).
    * ``dyntype == mjDYN_NONE`` - ``ctrl`` reaches the force law directly. A
      stateful drive puts an ``act`` state in between: ``<intvelocity>`` carries
      ``-kp`` yet integrates ``ctrl`` as a *rate*, so it clears the first two
      terms while its command is not a pose.
    * the transmission is the joint itself, which is why the joint is resolved
      through :func:`actuator_joint_id` (``-1`` for a tendon) rather than through
      :func:`actuator_driven_joint_ids`. No gain inspection can supply this term:
      every stock tendon gripper measured clears all three terms above
      (``panda/actuator8`` and ``robotiq_2f85/fingers_actuator`` both compile to
      ``biasprm = [0, -100, 0]``, ``shadow_hand/lh_A_FFJ0`` to ``[0, -0.5, 0]``),
      so they are position servos *on their tendon* and read as one here unless
      the transmission is checked. Two independent facts disqualify them:
      ``ctrl`` is in the tendon's units rather than the joint's (``[0, 255]`` for
      those two grippers, ``[0, 0.52]`` metres for ``stretch3/arm``), and one
      ``ctrl`` drives several joints at once (2 for a gripper, 4 for the
      ``stretch3`` telescoping arm), so no single joint angle can be written into
      it at all.

    Getting this wrong in the permissive direction is what a pose write cannot
    afford: a joint angle written into a velocity drive's ``ctrl`` is commanded
    as a *rate*, which drives the joint away from the pose that was just written
    (measured on a single ``<velocity kv="5">`` scene: the joint spins off the
    written pose, and under gravity ``mj_step`` reports
    ``Nan, Inf or huge value in QACC``). The reverse error only declines to
    help - the joint is reported as left alone - so every term above is required
    rather than assumed.

    Stock assets reach this path: Menagerie's ``pal_tiago`` ships a
    ``tiago_velocity.xml`` variant whose arm is driven by 9 ``<velocity>``
    actuators, and any caller MJCF loaded through ``replace_scene_mjcf`` /
    ``patch_scene_mjcf`` may carry one.

    The classification is not a per-robot property - ``openarm`` ships 2 position
    servos beside 16 motors - so it is per actuator.

    Note:
        :func:`~strands_robots.policies.wbc.sim_control.wbc_uses_position_servo`
        tests the bias type alone and so reads a velocity drive as a servo too.
        That is a pre-existing read-only heuristic - it only decides whether to
        install the whole-body torque shim, and never writes a setpoint - so it
        is left to its own change rather than widened into this one.

    Args:
        model: The compiled ``MjModel``.
        mj: The ``mujoco`` module.

    Returns:
        ``(servos, other_drives)`` - two disjoint ``{joint id: actuator id}``
        maps covering every joint some actuator drives, including the joints a
        tendon couples to one ``ctrl``; those land in *other_drives*, whose
        ``ctrl`` a caller must not write a joint angle into. A joint no actuator
        drives appears in neither. A joint driven by both a position servo and
        another drive appears only in *servos*, because a setpoint written there
        does govern the pose it settles to. Where several actuators of one kind
        drive a joint the last in model order is reported.
    """
    servos: dict[int, int] = {}
    other: dict[int, int] = {}
    affine = int(mj.mjtBias.mjBIAS_AFFINE)
    stateless = int(mj.mjtDyn.mjDYN_NONE)
    for act_id in range(int(model.nu)):
        driven = actuator_driven_joint_ids(model, act_id, mj)
        if not driven:
            continue
        target = actuator_joint_id(model, act_id, mj)
        commands_a_pose = target >= 0 and (
            int(model.actuator_biastype[act_id]) == affine
            and float(model.actuator_biasprm[act_id, 1]) < 0.0
            and int(model.actuator_dyntype[act_id]) == stateless
        )
        if commands_a_pose:
            servos[target] = act_id
        else:
            for jnt_id in driven:
                other[jnt_id] = act_id
    # Disjoint: a servo's setpoint governs the pose even when another drive also
    # pulls on the joint, so the servo is the drive a pose write can move.
    for jnt_id in servos:
        other.pop(jnt_id, None)
    return servos, other


def actuator_driven_joint_ids(model: Any, act_id: int, mj: Any) -> frozenset[int]:
    """Return every joint id actuator ``act_id`` drives.

    :func:`actuator_joint_id` answers the narrower question "which single joint
    is this actuator's transmission target", which is what a caller needs when
    it has one ``ctrl`` slot to seed from one joint's position. It reports
    ``-1`` for a tendon, because a tendon is not a joint.

    A joint can still be driven *through* that tendon, though: a fixed tendon
    that wraps joints couples them to one ``ctrl``, which is the standard MJCF
    gripper idiom. So a caller asking "is this joint already driven" - rather
    than "which joint is this actuator's target" - must resolve the tendon's
    wrap list too, or it reads an actuated joint as free. That distinction is
    the whole reason this is a second function and not a flag on the first: the
    two questions have different answers for the same actuator, and collapsing
    them would make one of the two call sites wrong.

    Transmissions that drive a site, body or slider-crank contribute nothing:
    they move a frame rather than command a joint coordinate, so a joint they
    happen to move is still free for an actuator to claim.

    Args:
        model: The compiled ``MjModel``.
        act_id: Actuator index in ``range(model.nu)``.
        mj: The ``mujoco`` module.

    Returns:
        The driven joint ids, empty when the transmission commands no joint.
    """
    joint_id = actuator_joint_id(model, act_id, mj)
    if joint_id >= 0:
        return frozenset({joint_id})
    if int(model.actuator_trntype[act_id]) != int(mj.mjtTrn.mjTRN_TENDON):
        return frozenset()
    return tendon_joint_ids(model, int(model.actuator_trnid[act_id, 0]), mj)


def tendon_joint_ids(model: Any, tendon_id: int, mj: Any) -> frozenset[int]:
    """Return the joint ids wired into tendon ``tendon_id`` by its wrap list.

    A tendon's wrap entries live in one flat table shared by every tendon, so a
    reader has to slice its own span (``tendon_adr`` / ``tendon_num``) and keep
    only the ``mjWRAP_JOINT`` entries - site and pulley wraps carry ids from
    other spaces. Walking that table is the part both "which actuator drives
    this joint" and "which joints does this actuator drive" need, so it lives
    here once rather than in each direction.

    Args:
        model: The compiled ``MjModel``.
        tendon_id: Tendon index in ``range(model.ntendon)``.
        mj: The ``mujoco`` module.

    Returns:
        The wrapped joint ids, empty when the tendon wraps no joint.
    """
    if tendon_id < 0 or tendon_id >= int(model.ntendon):
        return frozenset()
    wrap_joint = int(mj.mjtWrap.mjWRAP_JOINT)
    adr = int(model.tendon_adr[tendon_id])
    num = int(model.tendon_num[tendon_id])
    return frozenset(int(model.wrap_objid[w]) for w in range(adr, adr + num) if int(model.wrap_type[w]) == wrap_joint)


def robot_owned_actuator_ids(model: Any, robot: SimRobot, mj: Any) -> list[int]:
    """Return the actuator ids ``robot`` owns, in model order.

    Ownership is the union of two rules, because neither alone covers every
    actuator a robot can carry:

    * **Namespace.** ``spec.attach(other, prefix=...)`` prefixes every actuator
      name it merges in, so a prefixed name settles ownership whatever the
      transmission drives -- the only rule that reaches a tendon, site or body
      transmission.
    * **Driven joint.** :func:`actuate_robot_in_scene` names the position
      actuators it injects ``"<robot>_act_<joint>"``, which carries no namespace
      prefix, so those are recognized by the joint they drive.

    The second rule resolves the joint through :func:`actuator_joint_id`, so an
    actuator whose transmission is not a joint is never assigned by an id-space
    collision. Ungated, a tendon-driven gripper is claimed by whichever robot
    owns the joint whose id equals the tendon's and is missing from the robot
    that actually carries it -- the gripper then has no owner to operate it, and
    the other robot advertises an actuator that moves a different machine.

    Args:
        model: The compiled ``MjModel``.
        robot: The robot to scope. Reads its ``namespace`` and its already
            resolved ``joint_ids``.
        mj: The ``mujoco`` module.

    Returns:
        Owned actuator ids, ascending. Ascending model order is the contract:
        :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.robot_action_keys`
        and the recorded dataset action columns are ordered by actuator index,
        so the sequence is stable and only membership is corrected here.
    """
    pfx = robot.namespace or ""
    joint_ids = set(robot.joint_ids)
    owned: list[int] = []
    for act_id in range(int(model.nu)):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
        if pfx and name.startswith(pfx):
            owned.append(act_id)
        elif actuator_joint_id(model, act_id, mj) in joint_ids:
            owned.append(act_id)
    return owned


def install_compiled_model(world: SimWorld, model: Any, data: Any) -> None:
    """Install a compiled ``model``/``data`` pair as ``world``'s live scene state.

    Every path that swaps the live model goes through here, because the swap has
    a consequence that is easy to leave behind: ``world._recompile_generation``
    is the only part of the ``save_state`` / ``load_state`` fingerprint that
    changes when the new model happens to keep the previous ``nq``/``nv``/``na``/
    ``nu``. A site that rebinds ``world._model`` without bumping it lets a
    checkpoint taken against the previous model be applied to this one - the
    state vector is the right length, so it is written index by index into
    whatever those indices now mean.

    Concentrating the rebind here makes that impossible to forget: a new swap
    site cannot install a model without also invalidating the checkpoints taken
    against the old one.

    Args:
        world: The scene whose live model/data are replaced.
        model: The freshly compiled ``MjModel``.
        data: An ``MjData`` matching ``model``.
    """
    world._model = model
    world._data = data
    # Any swap invalidates every outstanding checkpoint, including one whose
    # state vector still has the right length: only this counter distinguishes
    # a same-shape model from the one the checkpoint was taken against.
    world._recompile_generation += 1


def _snapshot_body_wrenches(model: Any, data: Any, mj: Any) -> dict[str, list[float]]:
    """Capture every latched ``apply_force`` wrench, keyed by body name.

    ``xfrc_applied`` is indexed by body, and a scene rebuild renumbers bodies,
    so the wrench is carried by NAME for the same reason every other field of
    :class:`_SceneState` is. Only non-zero rows are captured: an all-zero row is
    "no wrench latched here", which is exactly what a fresh allocation already
    holds, so a scene with no wrench in it costs an empty dict.

    Args:
        model: The compiled model the rows are being read against.
        data: The ``MjData`` holding the latched wrenches.
        mj: The resolved ``mujoco`` module.

    Returns:
        ``body name -> [fx, fy, fz, tx, ty, tz]`` for each body with a wrench.
    """
    wrenches: dict[str, list[float]] = {}
    for bid in range(int(model.nbody)):
        row = data.xfrc_applied[bid]
        if not row.any():
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if not name:
            # No namespace names it, so it cannot be matched across the rebuild.
            # ``apply_force`` resolves its target by name and so cannot latch
            # one here; a wrench on an unnamed body came from elsewhere.
            logger.debug("snapshot_body_wrenches: body id %d has no name, wrench not carried over", bid)
            continue
        wrenches[name] = [float(x) for x in row]
    return wrenches


def _snapshot_joint_forces(model: Any, data: Any, mj: Any) -> dict[_JointKey, list[float]]:
    """Capture every latched generalized force, keyed by joint.

    The joint-space sibling of :func:`_snapshot_body_wrenches`. ``spec.recompile``
    drops ``qfrc_applied`` for the same reason it drops ``xfrc_applied`` -- it
    carries neither applied-force buffer -- so a grow needs this pass just as the
    eject does. Keyed the same way as :class:`_SceneState`'s joints, because a
    rebuild renumbers dofs. Only non-zero slices are captured: all-zero is "no
    force latched here", which a fresh allocation already holds.

    Args:
        model: The compiled model the slices are being read against.
        data: The ``MjData`` holding the latched forces.
        mj: The resolved ``mujoco`` module.

    Returns:
        ``joint key -> qfrc_applied`` slice, at the dof width the joint uses.
    """
    forces: dict[_JointKey, list[float]] = {}
    for jid in range(int(model.njnt)):
        key = _joint_key(model, jid, mj)
        if key is None:
            continue
        dof_adr = int(model.jnt_dofadr[jid])
        _, dof_w = _joint_state_widths(int(model.jnt_type[jid]), mj)
        row = data.qfrc_applied[dof_adr : dof_adr + dof_w]
        if not row.any():
            continue
        forces[key] = [float(x) for x in row]
    return forces


def _restore_joint_forces(model: Any, data: Any, forces: dict[_JointKey, list[float]], mj: Any) -> None:
    """Re-apply ``forces`` to the joints that still answer to those keys.

    A joint the rebuilt model no longer has is skipped: its absence is the point
    of a rebuild that removed it. A joint whose type changed is skipped rather
    than written with a mismatched slice, which would put one dof's force on
    another.

    Args:
        model: The rebuilt compiled model.
        data: The rebuilt ``MjData`` receiving the forces.
        forces: A snapshot from :func:`_snapshot_joint_forces`.
        mj: The resolved ``mujoco`` module.
    """
    for key, vals in forces.items():
        jid = _resolve_joint_key(model, key, mj)
        if jid < 0:
            continue
        _, dof_w = _joint_state_widths(int(model.jnt_type[jid]), mj)
        if len(vals) != dof_w:
            logger.warning(
                "_restore_joint_forces: dof width mismatch for %r (%d!=%d), skipping",
                key,
                len(vals),
                dof_w,
            )
            continue
        dof_adr = int(model.jnt_dofadr[jid])
        for i, v in enumerate(vals):
            data.qfrc_applied[dof_adr + i] = v


def _restore_body_wrenches(model: Any, data: Any, wrenches: dict[str, list[float]], mj: Any) -> None:
    """Re-latch ``wrenches`` onto the bodies that still carry those names.

    A body that no longer exists is skipped: its absence is the point of a
    rebuild that removed it. Bodies absent from ``wrenches`` keep their
    fresh-compile zero row, which is the same "no wrench" the snapshot read.

    Args:
        model: The freshly compiled model to resolve names against.
        data: The ``MjData`` to write the wrenches into.
        wrenches: A snapshot from :func:`_snapshot_body_wrenches`.
        mj: The resolved ``mujoco`` module.
    """
    for name, row in wrenches.items():
        bid = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue  # body no longer exists (expected for a removed robot)
        for i, v in enumerate(row):
            data.xfrc_applied[bid, i] = v


def _recompile_preserving_state(world: SimWorld, spec: Any, *, raise_on_refusal: bool = False) -> bool:
    """Recompile ``spec`` in place, replacing ``world._model`` and ``_data``.

    Uses ``spec.recompile(model, data)`` which auto-preserves qpos/qvel for
    existing joints and initializes new joints to their body's pos/quat. No
    manual state-copy loop is required.

    That transfer is POSITIONAL, not by name: the new buffers receive the old
    values for the indices both models share, and the entries past the old size
    are whatever the fresh allocation happened to contain. The compiler does not
    define that tail, so every buffer this recompile grows is defined here --
    ``qpos`` from ``qpos0`` and ``qvel``/``ctrl``/``act`` as zero, which is what
    a reset writes and the only state a caller who has not touched the new
    entries can mean -- before the forward pass below reads it.

    The tail is not a harmless nonsense number in either buffer:

    * ``ctrl`` -- MuJoCo's ``mj_checkCtrl`` disables actuation for the WHOLE
      model on any step where a single entry is non-finite, so one uninitialized
      entry can silently release every held pose in the scene, only on the runs
      where the leftover memory happens to be NaN.
    * ``qfrc_applied`` and ``xfrc_applied`` -- ``spec.recompile`` transfers
      neither applied-force buffer at all, so unlike the buffers above these are
      not a tail problem: every entry is lost, including the slices of joints and
      bodies that never moved. Both are therefore snapshotted by name before the
      recompile and re-applied after it, rather than merely having a tail
      defined. A force a caller latched otherwise stopped acting the moment
      anything entered the scene, under a ``"status": "success"``.
    * ``qpos`` -- a new joint that no name-keyed pass reaches keeps the tail as
      its pose. The robot-scoped reset in
      :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine._reset_robot_to_reference`
      walks ``robot.joint_ids``, which is resolved by joint NAME, so an
      UNNAMED ``<freejoint/>`` -- the standard MJCF idiom for a floating base,
      used by the Unitree Go2 and by LeKiwi -- is invisible to it and to the
      keyframe apply beside it. Measured on ``go2`` spawned with
      ``keyframe="home"``: with one robot already in the world the tail happened
      to hold ``qpos0`` and the base stood at ``z=0.445``, and with two it held
      zeros -- including an all-zero quaternion, which is not a pose any
      compiler or reset writes -- dropping the base to the floor. The hinges
      were applied either way, so the only symptom was a quadruped lying down
      under a ``"status": "success"``.

    TWO buffers are not transferred at all rather than only in their tail --
    both applied-force buffers come back entirely zero, so for these the tail
    initialization above is beside the point: every entry is lost, including the
    rows of elements that were there all along. Measured by growing a scene by
    one body on mujoco 3.5.0 (the floor this package declares), 3.10.0 (the
    locked version) and 3.11.0, identically on all three -- ``qpos``, ``qvel``,
    ``ctrl`` and the clock carried, ``qfrc_applied`` and ``xfrc_applied``
    zeroed.

    * ``xfrc_applied``, the per-body row
      :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.apply_force`
      latches a wrench in, IS carried here. That wrench is documented to hold
      until the next ``apply_force`` on the same body or a ``reset()``, and a
      scene rebuild is neither, so it is snapshotted by body name before the
      recompile and re-latched after.
    * ``qfrc_applied``, its joint-space sibling, is deliberately NOT carried
      here: nothing in this package ever writes a non-zero value into it.
      ``apply_force`` documents why it latches the per-body buffer instead (one
      world-wide generalized-force vector has no slice that belongs to one
      body), and no other caller touches it, so there is no value for the
      recompile to lose and a carry here could only be exercised by writing the
      buffer from outside the public API. A joint-force API added later must add
      the carry with it -- ``_SceneState`` already carries this buffer on the
      eject path, where it is free because that path snapshots joints anyway.

    Also re-discovers per-robot joint and actuator IDs (they may have shifted
    as new bodies were inserted earlier in the body tree). Returns True on
    success, False on compile failure (logged).

    Args:
        world: The scene whose model/data are replaced on success.
        spec: The ``MjSpec`` to compile.
        raise_on_refusal: Re-raise the compiler's exception instead of folding
            it into a ``False`` return. A refusal message names what the
            compiler could not honor, and a ``bool`` cannot carry it: a caller
            that reports the reason to a user needs it, whereas a caller that
            only rolls back does not. Off by default so existing callers keep
            the ``bool`` contract.
    """
    mj = _ensure_mujoco()
    # Sizes of the buffers being handed to the compiler: everything at or past
    # these indices in the new data is an entry the old model had no value for.
    old_nq = int(world._model.nq) if world._model is not None else 0
    old_nv = int(world._model.nv) if world._model is not None else 0
    old_nu = int(world._model.nu) if world._model is not None else 0
    old_na = int(world._model.na) if world._model is not None else 0
    # ``spec.recompile`` carries no part of EITHER applied-force buffer -- both
    # ``xfrc_applied`` and ``qfrc_applied`` come back zero -- so the latched
    # forces are read off the outgoing data here and re-applied by name below.
    _have_state = world._model is not None and world._data is not None
    wrenches = _snapshot_body_wrenches(world._model, world._data, mj) if _have_state else {}
    joint_forces = _snapshot_joint_forces(world._model, world._data, mj) if _have_state else {}
    try:
        with filter_mujoco_attach_noise():
            new_model, new_data = spec.recompile(world._model, world._data)
    except (ValueError, RuntimeError) as e:
        logger.error("spec.recompile failed: %s", e)
        if raise_on_refusal:
            raise
        return False

    install_compiled_model(world, new_model, new_data)
    # Define every entry the positional transfer left untouched (see the
    # docstring). ``qvel`` and ``act`` are measurably zero-filled today, but they
    # come out of the same allocation as ``qpos`` and ``ctrl`` and carry a new
    # joint's velocity and an actuator's activation -- its effective command --
    # so they are defined here too rather than by luck.
    if new_model.nq > old_nq:
        new_data.qpos[old_nq:] = new_model.qpos0[old_nq:]
    if new_model.nv > old_nv:
        new_data.qvel[old_nv:] = 0.0
    if new_model.nu > old_nu:
        new_data.ctrl[old_nu:] = 0.0
    if new_model.na > old_na:
        new_data.act[old_na:] = 0.0
    # Re-apply the latched external forces, before the forward pass reads them.
    _restore_body_wrenches(new_model, new_data, wrenches, mj)
    _restore_joint_forces(new_model, new_data, joint_forces, mj)
    # Forward pass so newly-injected bodies have valid xpos/xquat and any
    # camera xforms are populated. Without this, the next render() call
    # after add_object / add_robot / add_camera returns a 100% black frame
    # because the MjData arrays still hold their initialization zeros.
    mj.mj_forward(new_model, new_data)

    # Keep the cached XML in sync with the spec for legacy readers (e.g.
    # load_scene + add_robot round-trip).
    _sync_cached_xml(world, spec)

    # Re-discover per-robot IDs. Names inside MuJoCo are namespaced under
    # robot.namespace (e.g. "arm1/shoulder_pan") when robots were attached
    # via SpecBuilder.attach_robot; fall back to the raw name otherwise.
    for robot in world.robots.values():
        pfx = robot.namespace or ""
        robot.joint_ids = []
        for jnt_name in robot.joint_names:
            jid = -1
            if pfx:
                jid = mj_name_to_id(new_model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jid < 0:
                jid = mj_name_to_id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        robot.actuator_ids = robot_owned_actuator_ids(new_model, robot, mj)
        # Single-robot fallback. Ownership above is settled by namespace or by
        # driven joint; a lone robot whose actuators are neither prefixed nor
        # joint-driven (a tendon or site transmission in a scene loaded whole,
        # so no attach prefix) matches on neither, and in a one-robot scene
        # every actuator is unambiguously that robot's.
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(new_model.nu))

    return True


# Persist


_NO_SPEC_REASON = "the scene has no live spec, so the change cannot be recorded durably"


def _spec_element_by_id(elements: Any, entity_id: int, kind: str) -> tuple[Any | None, str | None]:
    """Return the spec element a compiled entity id was built from.

    The compiler emits one model entity per spec element in declaration order and
    records the resulting index on the element, so a compiled id indexes the
    spec's element list directly. That mapping is what lets an UNNAMED element be
    addressed at all: most geoms in a robot scene carry no name, so resolving by
    name would silently cover almost none of them.

    The recorded ``id`` is verified rather than assumed. Writing a property onto
    the wrong entity is a worse outcome than not recording it, so a spec that no
    longer agrees with the compiled model is reported instead of written to.

    Args:
        elements: The spec's element list (``spec.geoms`` / ``spec.bodies``).
        entity_id: Compiled index of the entity, as resolved by the caller.
        kind: Entity name used in the reason text (``"geom"`` / ``"body"``).

    Returns:
        ``(element, None)`` when the element was located, otherwise
        ``(None, reason)`` naming why it was not.
    """
    count = len(elements)
    if entity_id < 0 or entity_id >= count:
        return None, f"{kind} id {entity_id} is outside the scene spec's {count} {kind}(s)"
    element = elements[entity_id]
    if element.id != entity_id:
        return None, (
            f"the scene spec no longer agrees with the compiled model: the {kind} at index "
            f"{entity_id} reports id {element.id}"
        )
    return element, None


def persist_geom_properties(
    world: SimWorld,
    geom_id: int,
    *,
    color: list[float] | None = None,
    friction: list[float] | None = None,
    size: list[float] | None = None,
) -> str | None:
    """Record a runtime geom property write in the spec the model is compiled from.

    ``world._model`` is DERIVED state: every scene mutation recompiles the spec
    over it (see :func:`_recompile_preserving_state`), so a value written only
    into the model is discarded by the next ``add_object`` / ``add_camera`` /
    ``add_robot`` call and the geom reverts to whatever it was compiled with -
    after the setter already reported the new value. Writing the same value into
    the spec is what makes that reported result durable.

    Nothing is written when a reason is returned, so a caller can refuse before it
    touches the model and keep the two representations in step.

    Args:
        world: The scene holding the live spec.
        geom_id: Compiled geom index, already resolved by the caller.
        color: RGBA components, already validated.
        friction: The three friction coefficients, already validated.
        size: Half-extents, already validated. Only the components the caller
            supplied are written, matching the model write: the unused tail of
            the spec's 3-wide row keeps its declared value.

    A ``size`` component the geom's ``<fromto>`` fixes cannot be made durable by
    writing this row at all; :func:`fromto_fixed_size_components` reports those
    and the caller refuses such a change before reaching here, so what is written
    below is what the next compile reproduces.

    Returns:
        ``None`` once the value is recorded, otherwise the reason it could not be.
    """
    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON
    spec_geom, reason = _spec_element_by_id(spec.geoms, geom_id, "geom")
    if spec_geom is None:
        return reason

    if color is not None:
        spec_geom.rgba = list(color)
    if friction is not None:
        spec_geom.friction[:] = friction
    if size is not None:
        spec_geom.size[: len(size)] = size
    return None


def fromto_fixed_size_components(world: SimWorld, geom_id: int) -> dict[int, tuple[str, int | None]]:
    """Return the ``geom_size`` components a geom's ``<fromto>`` fixes.

    ``fromto`` gives a geom's extent along its own axis as two endpoints, and the
    compiler then FIXES part of its ``geom_size`` row rather than reading it from
    ``size``: the axis extent comes from the endpoints, and for a box or ellipsoid
    the cross-section is additionally made square by copying the first component.
    Those components are re-derived on every compile, so a value written into the
    spec's ``size`` row never reaches the model - the resize is reported and then
    discarded by the next scene recompile, and an inertial row re-derived from the
    spec (see :func:`refresh_body_inertial_from_geometry`) describes the extent the
    endpoints still declare rather than the requested one.

    Only the fixed components are affected. A capsule's or cylinder's radius still
    comes from ``size``, so a caller can resize it and have the change recorded
    durably; callers use this mapping to refuse a change to the rest instead of
    reporting it.

    Args:
        world: The scene holding the live spec.
        geom_id: Compiled geom index, already resolved by the caller.

    Returns:
        ``{index: (name, follows)}`` for each fixed component, where ``follows``
        is the ``size`` index whose value that component copies, or ``None`` when
        the segment endpoints fix it. Empty when nothing is fixed - which covers a
        geom with no ``fromto``, a geom type ``fromto`` cannot be used with, and a
        spec that cannot resolve ``geom_id``.
    """
    mj = _ensure_mujoco()
    spec = _get_spec(world)
    if spec is None:
        return {}
    spec_geom, _reason = _spec_element_by_id(spec.geoms, geom_id, "geom")
    if spec_geom is None:
        return {}
    # MuJoCo marks an unset spec field with NaN in its first element, so a
    # ``fromto`` only describes this geom when it was actually declared.
    fromto = spec_geom.fromto
    if fromto is None or math.isnan(float(fromto[0])):
        return {}
    geom = mj.mjtGeom
    fixed: dict[int, dict[int, tuple[str, int | None]]] = {
        int(geom.mjGEOM_CAPSULE): {1: ("half-length", None)},
        int(geom.mjGEOM_CYLINDER): {1: ("half-length", None)},
        int(geom.mjGEOM_BOX): {1: ("y half-extent", 0), 2: ("z half-extent", None)},
        int(geom.mjGEOM_ELLIPSOID): {1: ("y semi-axis", 0), 2: ("z semi-axis", None)},
    }
    return fixed.get(int(spec_geom.type), {})


def refresh_body_inertial_from_geometry(world: SimWorld, geom_id: int) -> str | None:
    """Re-derive the inertial row of the body owning ``geom_id`` from its geometry.

    A body that does not declare an ``<inertial>`` takes its mass, center of mass
    and inertia tensor from the shapes it owns: the compiler integrates them over
    the geoms once, at compile time. ``model.body_mass`` / ``body_ipos`` /
    ``body_iquat`` / ``body_inertia`` are therefore DERIVED from ``geom_size``, and
    no ``mj_forward`` / ``mj_step`` recomputes them. Resizing a geom in the model
    alone leaves the body describing the shape it used to have, so it resists
    rotation as the old geometry did while colliding as the new one - and for a
    geom whose mass comes from a density, it also keeps the old mass and the old
    balance point.

    The values are read from a compile of a *copy* of the live spec, which already
    carries the new geometry (see :func:`persist_geom_properties`). That makes the
    refreshed row equal, by construction, to the one the next scene recompile will
    produce - so the same resize no longer means two different things depending on
    whether an unrelated ``add_object`` happens afterwards. Deriving it from
    MuJoCo's own integrator rather than from per-primitive formulas is also what
    makes a multi-geom body and a shifted center of mass come out right, with no
    shape-by-shape special cases to keep correct.

    The live model is not swapped and the scene's own ``mjData`` is never passed to
    MuJoCo, so entity ids, joint state and the recompile generation are untouched -
    the refresh reads geometry and writes constants, and a resize therefore leaves
    the scene exactly where it was. The cost is one spec compile plus one scratch
    ``mjData``, paid only for a resize of a geom whose body derives its inertia
    from geometry.

    Args:
        world: The scene holding the live spec and compiled model.
        geom_id: Compiled geom index whose size has already been recorded in the
            spec, as resolved by the caller.

    Returns:
        ``None`` once the row is refreshed - including when the owning body
        declares its own ``<inertial>``, which takes nothing from geometry and so
        needs no refresh - otherwise the reason it could not be, leaving both the
        model and the spec untouched so the caller can restore them.
    """
    mj = _ensure_mujoco()
    model = world._model
    if model is None:
        return "the scene has no compiled model whose inertia could be re-derived"

    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON

    body_id = int(model.geom_bodyid[geom_id])
    spec_body, reason = _spec_element_by_id(spec.bodies, body_id, "body")
    if spec_body is None:
        return reason
    if spec_body.explicitinertial:
        # The compiler ignores a body's geoms when the body declares its own
        # <inertial>, so that row does not describe the geometry and a resize
        # cannot make it stale. Returning early also keeps resizing a robot
        # link's collision geom free, since those links declare their inertials.
        return None

    try:
        with filter_mujoco_attach_noise():
            candidate = spec.copy().compile()
    except (ValueError, RuntimeError) as e:
        return f"the resized geometry does not compile, so the body's inertia cannot be re-derived: {e}"

    # Writing a row read from a model with a different body ordering would move
    # one body's inertia onto another, which is a worse outcome than leaving the
    # row stale and saying so.
    if candidate.nbody != model.nbody:
        return (
            "the scene spec no longer agrees with the compiled model: it compiles to "
            f"{candidate.nbody} bodies against the model's {model.nbody}"
        )
    if mj.mj_id2name(candidate, mj.mjtObj.mjOBJ_BODY, body_id) != mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id):
        return (
            "the scene spec no longer agrees with the compiled model: body "
            f"{body_id} is named differently in a fresh compile"
        )

    model.body_mass[body_id] = candidate.body_mass[body_id]
    model.body_ipos[body_id] = candidate.body_ipos[body_id]
    model.body_iquat[body_id] = candidate.body_iquat[body_id]
    model.body_inertia[body_id] = candidate.body_inertia[body_id]
    # body_subtreemass and the invweight / M0 reference constants the constraint
    # solver scales with are themselves derived from the inertial rows and are not
    # refreshed by a step. mj_setConst recomputes them, which is what makes the
    # whole inertial state of the live model equal to that of a fresh compile.
    #
    # It is handed a scratch mjData, never the scene's own. Those constants are by
    # definition evaluated at the model's reference configuration, so mj_setConst
    # writes qpos0 into whatever data it is given and does not put back what was
    # there. Passing the live data would rewind every joint and body pose to its
    # declared value while leaving qvel as it was, so a resize issued after any
    # stepping would teleport the scene into a state it never occupied - the
    # order-dependent silent damage this refresh exists to remove.
    mj.mj_setConst(model, mj.MjData(model))
    return None


def persist_body_mass(world: SimWorld, body_id: int, *, mass_ratio: float) -> str | None:
    """Record a runtime body mass change in the spec the model is compiled from.

    Recording the change as a scale is what keeps the two representations equal:
    ``set_body_properties`` documents a mass change as a uniform density change at
    fixed geometry, and both mass and inertia are linear in density, so applying
    one ratio reproduces exactly the inertial the setter reported.

    A body's compiled inertial comes from one of two places, and only the one in
    force is writable. A body that declares an explicit ``<inertial>`` carries its
    own mass and inertia (``explicitinertial``); every other body has both
    integrated from its geoms, and there assigning ``mass`` on the body element is
    silently ignored by the compiler - the geoms' mass or density is what has to
    move.

    Args:
        world: The scene holding the live spec.
        body_id: Compiled body index, already resolved by the caller.
        mass_ratio: The new mass divided by the compiled mass. Finite and ``> 0``,
            which the caller guarantees by refusing a body with no mass to scale.

    Returns:
        ``None`` once the change is recorded, otherwise the reason it could not be.
    """
    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON
    spec_body, reason = _spec_element_by_id(spec.bodies, body_id, "body")
    if spec_body is None:
        return reason

    if spec_body.explicitinertial:
        spec_body.mass *= mass_ratio
        # A body states its inertia as either the principal diagonal or the six
        # unique components of the full tensor; whichever form is not in use is
        # all zeros, so scaling both is exact and needs no branch.
        spec_body.inertia[:] = [value * mass_ratio for value in spec_body.inertia]
        spec_body.fullinertia[:] = [value * mass_ratio for value in spec_body.fullinertia]
        return None

    spec_geoms = list(spec_body.geoms)
    if not spec_geoms:
        return (
            "the body declares no explicit inertial and owns no geom, so it holds "
            "nothing whose mass the change could scale"
        )
    for spec_geom in spec_geoms:
        # A geom states either an explicit mass or a density, and the compiler
        # uses the mass only when it is set (an unset mass reads as nan, which
        # fails every comparison). Scaling whichever one is in force scales that
        # geom's contribution, so the body total scales by the same ratio.
        if spec_geom.mass > 0:
            spec_geom.mass *= mass_ratio
        else:
            spec_geom.density *= mass_ratio
    return None


def persist_world_option(
    world: SimWorld,
    *,
    gravity: list[float] | None = None,
    timestep: float | None = None,
) -> str | None:
    """Record a runtime physics-option write in the spec the model is compiled from.

    ``model.opt`` is compiled from ``spec.option``, so a gravity or timestep
    written only into the model is restored to the scene's declared value by the
    next recompile - putting a lunar-gravity world back to 9.81 m/s^2 on the next
    ``add_object``.

    Args:
        world: The scene holding the live spec.
        gravity: The three gravity components, already validated.
        timestep: The integration step in seconds, already validated.

    Returns:
        ``None`` once the value is recorded, otherwise the reason it could not be.
    """
    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON
    if gravity is not None:
        spec.option.gravity[:] = gravity
    if timestep is not None:
        spec.option.timestep = timestep
    return None


# Inject


def inject_robot_into_scene(
    world: SimWorld,
    robot: SimRobot,
    robot_xml_path: str,
) -> bool:
    """Attach a robot to the scene via ``spec.attach(other, prefix=..., frame=...)``.

    MuJoCo handles name prefixing (bodies, joints, geoms, actuators, sensors,
    sites), asset deduplication (meshes, textures, materials), and default-
    class namespacing. No manual tree-walking required.

    Registers the robot's source joint names on ``robot.joint_names`` so
    downstream observation/policy code can resolve them via
    ``{robot.namespace}{joint_name}``.

    ``spec.attach`` mutates the scene spec before the recompile that validates
    the result, so a refused recompile - e.g. the robot model references a mesh
    file that cannot be opened - leaves the robot's whole namespaced subtree in
    the live spec. Every later scene mutation then recompiles that same broken
    spec and fails too, so one unloadable robot bricked the world: subsequent
    ``add_object`` / ``add_camera`` / ``add_robot`` calls all reported "spec
    recompile refused" with nothing wrong with them, and each failed retry left
    another orphan subtree behind. The attach is therefore rolled back out
    before returning ``False``, mirroring :func:`inject_object_into_scene` and
    :func:`inject_camera_into_scene`, so a refused robot costs exactly the add
    that was refused.

    The rollback reinstalls a snapshot taken before the attach
    (:func:`_snapshot_spec`) rather than rebuilding the scene from ``world``: a
    rebuild would drop the spec-only mutations the scene may already hold (weld
    equalities, added actuators, agent-authored bodies), turning a correctly
    refused add into corruption of a scene that was healthy before it. The
    refused ``spec.recompile`` leaves ``world._model`` and ``_data`` untouched,
    so putting the spec back is the whole rollback - no state restore, forward
    pass or ID rediscovery is needed.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_robot: no spec or model in world")
        return False

    # Snapshot BEFORE mutating. Without a way back the add is refused here,
    # while the scene is still exactly as it was found.
    context = f"inject_robot {robot.name!r}"
    backup_spec = _snapshot_spec(spec, context=context)
    if backup_spec is None:
        return False

    try:
        with filter_mujoco_attach_noise():
            joint_names = SpecBuilder.attach_robot(spec, robot, robot_xml_path)
        robot.joint_names = joint_names
    except (ValueError, RuntimeError, OSError) as e:
        # attach_robot can insert its worldbody frame before the call that
        # raised. That leftover compiles, so it never broke the scene, but the
        # snapshot is already in hand - restoring it costs nothing and puts
        # every failure path on one rule: the spec is left as it was found.
        logger.error("Robot attach failed for '%s': %s", robot.name, e)
        world._backend_state["spec"] = backup_spec
        return False

    if _recompile_preserving_state(world, spec):
        return True

    # The attach landed in the spec but the model it produced was refused, so
    # the robot's whole namespaced subtree is sitting in the live spec while the
    # caller is about to report a failed add. Put the pre-attach spec back.
    world._backend_state["spec"] = backup_spec
    return False


def inject_object_into_scene(world: SimWorld, obj: SimObject) -> bool:
    """Add a ``SimObject`` to the scene and recompile in place.

    A ``shape="mesh"`` object references a mesh asset that must be registered
    on the spec before the geom that names it can compile. The full-scene
    ``SpecBuilder.build`` registers those assets in its own pass, but the
    incremental path (``SpecBuilder.add_object``) does not, so this function
    registers the mesh (``spec.add_mesh(name=f"mesh_{obj.name}", ...)``) itself
    before adding the body. Without this, ``add_object(shape="mesh")`` at
    runtime always failed to recompile even for a valid mesh file.

    ``SpecBuilder.add_object`` mutates the spec (adds the body + geom) before
    the recompile that validates it. If that recompile is refused - e.g. the
    mesh file cannot be loaded - the just-added body AND its mesh asset are
    deleted again before returning ``False`` so the spec stays compilable.
    Without the rollback the orphan lingers and every later scene mutation,
    including a corrected retry under the same name, keeps failing to recompile
    (``repeated name`` collisions), bricking the whole scene after one bad add.

    The same rollback applies when ``SpecBuilder.add_object`` itself raises
    part-way through, which it can do *after* inserting the body (the geom's
    type lookup rejects an unsupported shape; the mass write rejects a
    non-numeric value). That error is then re-raised rather than folded into a
    ``False`` return, so the caller can report the actual reason - a swallowed
    ``ValueError`` left the caller with nothing but "spec recompile refused"
    while the actionable message went to the log.

    Every rollback here deletes only the bodies THIS call appended, counted
    before the insert (``SpecBuilder.count_bodies_named`` /
    ``remove_surplus_bodies``). A delete by name is wrong on the collision path:
    ``load_scene`` replaces the world registry, so a body declared by the scene
    MJCF is invisible to ``add_object``'s registry check and the insert reaches
    MuJoCo, leaving two bodies under one name. ``SpecBuilder.remove_body``
    resolves the name to the body present at the last compile - the ORIGINAL -
    so rolling back with it deleted the healthy scene body and left the rejected
    one holding its name, and the next mutation recompiled cleanly with the
    original geometry gone.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_object: no spec or model in world")
        return False

    # How many bodies already carry this name, taken BEFORE the insert. Every
    # rollback below deletes only the bodies beyond this count - the ones this
    # call appended - because a delete by name resolves the pre-existing body on
    # a collision and would remove the healthy scene body instead of the orphan.
    pre_bodies = SpecBuilder.count_bodies_named(spec, obj.name)

    try:
        # Meshes need their asset registered before the geom references it.
        # build() registers meshes in a separate pass, so add_object does not;
        # the incremental path must register it here.
        if obj.shape == "mesh" and obj.mesh_path:
            spec.add_mesh(name=f"mesh_{obj.name}", file=obj.mesh_path)
        SpecBuilder.add_object(spec, obj)
    except (ValueError, RuntimeError):
        # add_object is atomic over its own body mutation: a raise there (an
        # unsupported shape, or a name that collides with an existing scene
        # body) rolls the half-built body back out itself, so only the mesh
        # asset registered here still needs undoing. Removing it keeps the spec
        # compilable, then the error propagates: the caller turns it into a
        # structured result, and the reason - e.g. the exact unsupported shape
        # and the supported list - is what a caller needs instead of a generic
        # recompile refusal.
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        raise

    # Roll the just-added body (and any mesh asset) back out so the spec
    # returns to its last good, compilable state (a worldbody body delete is
    # safe - the attach/delete segfault only affects spec.attach() child specs).
    #
    # Ask for the compiler's own reason rather than a bare False. Because the
    # object's mass is declared on its geom, MuJoCo integrates the inertia from
    # the shape, so its "mass and inertia of moving bodies must be larger than
    # mjMINVAL" floor is shape-dependent: a mass above mjMINVAL can still
    # integrate to an inertia below it on a small geom. add_object's numeric
    # pre-check cannot express that floor without duplicating the compiler's
    # per-shape integration, so the residual case has to arrive as the reason
    # the compiler gives. Folded into a False it became "spec recompile
    # refused." with the actionable text left in the log - the same dead end
    # the unsupported-shape path was fixed to stop producing.
    try:
        recompiled = _recompile_preserving_state(world, spec, raise_on_refusal=True)
    except (ValueError, RuntimeError):
        SpecBuilder.remove_surplus_bodies(spec, obj.name, pre_bodies)
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        raise
    if not recompiled:
        SpecBuilder.remove_surplus_bodies(spec, obj.name, pre_bodies)
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        return False
    return True


def inject_camera_into_scene(world: SimWorld, cam: SimCamera) -> bool:
    """Add a camera to the scene and recompile in place.

    Mirrors :func:`inject_object_into_scene`: ``SpecBuilder.add_camera`` mutates
    the spec before the validating recompile, so a refused recompile rolls the
    just-added camera back out to keep the spec compilable for later edits.

    That rollback removes only the cameras THIS call appended, counted before the
    insert. It cannot be a delete by name: when the name collides with a camera
    the loaded scene already declares - which ``add_camera``'s registry check
    cannot see, because ``load_scene`` replaces the registry while the MJCF keeps
    its cameras - ``SpecBuilder.remove_camera`` deletes the FIRST camera carrying
    the name, i.e. the scene's own. The refused camera then inherited the name and
    every later render of it answered with the pose the caller was told had been
    rejected.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_camera: no spec or model in world")
        return False

    pre_cameras = SpecBuilder.count_cameras_named(spec, cam.name)

    try:
        SpecBuilder.add_camera(spec, cam)
    except (ValueError, RuntimeError) as e:
        logger.error("Camera add failed for '%s': %s", cam.name, e)
        return False

    if not _recompile_preserving_state(world, spec):
        SpecBuilder.remove_surplus_cameras(spec, cam.name, pre_cameras)
        return False
    return True


# Eject


def eject_body_from_scene(world: SimWorld, body_name: str) -> bool:
    """Remove a body (by short name) and recompile.

    A camera mounted on that body (``SimCamera.parent_body == body_name``) goes
    with it. The recompile drops the camera element -- it is a child of the body
    being deleted -- so a surviving registry entry would advertise a camera no
    consumer can resolve: ``list_cameras`` offers it while ``render`` refuses it
    as unknown and names it in the same breath as an available alternative. Such
    entries are dropped with a warning naming the camera and its parent, the same
    treatment :func:`eject_robot_from_scene` gives a camera whose parent belonged
    to the robot being removed.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_body: no spec or model in world")
        return False

    if not SpecBuilder.remove_body(spec, body_name):
        logger.warning("Body '%s' not found in spec - nothing ejected", body_name)
        # Matching legacy behaviour: return True so scene state stays consistent
        # (caller has already popped the Python-side dict entry).
        return True

    # Cameras mounted on this body lose the frame their pose is expressed in.
    # The recompile below drops the camera element with its parent, so the
    # registry entry has to go too: a stale entry lingers and confuses
    # observation code, which is the same reason eject_robot_from_scene drops
    # the cameras of the robot it is ejecting. Keyed by registry name, not
    # ``cam.name``, because a URDF-discovered camera stores its namespaced
    # MuJoCo name there.
    for cam_key in [key for key, cam in world.cameras.items() if cam.parent_body == body_name]:
        logger.warning(
            "eject_body: dropping camera %r - it was mounted on the removed body %r.",
            cam_key,
            body_name,
        )
        del world.cameras[cam_key]

    # Objects added at runtime register a mesh asset named f"mesh_{name}".
    # Delete it too so the name is fully reusable and unused assets do not
    # accumulate across remove/re-add cycles (safe no-op for primitives).
    SpecBuilder.remove_mesh(spec, f"mesh_{body_name}")

    return _recompile_preserving_state(world, spec)


def eject_camera_from_scene(world: SimWorld, mj_name: str) -> bool:
    """Remove a camera element from the scene spec and recompile in place.

    The inverse of :func:`inject_camera_into_scene`, and it needs the same way
    back. ``SpecBuilder.remove_camera`` mutates the live spec before the
    recompile that validates the result, and a refused ``spec.recompile`` leaves
    ``world._model`` untouched -- so with no rollback the spec stops declaring
    the camera while the compiled model still has it. Consumers then disagree
    about whether the camera exists: ``render`` and ``get_camera_params``
    resolve it from the model and succeed, while the delete lands later, applied
    by whichever unrelated mutation next recompiles successfully. Restoring the
    pre-delete spec keeps a refused removal costing exactly the removal that was
    refused.

    The rollback reinstalls a snapshot taken before the delete
    (:func:`_snapshot_spec`) rather than re-adding the camera from its
    ``SimCamera`` registry entry: a camera discovered inside a robot's URDF can
    carry element attributes that entry does not model, so re-adding it would
    restore a different camera. A caller that cannot snapshot refuses the
    removal rather than proceeding with no way back.

    Args:
        world: The scene to mutate.
        mj_name: The camera's name in the MjSpec -- namespaced for a camera
            discovered inside a robot's URDF, otherwise the registry name.

    Returns:
        ``True`` when the camera is gone from the spec and the recompiled model
        is installed, or when the spec never declared it (nothing to eject).
        ``False`` when there is no spec, the snapshot failed, or the recompile
        was refused; in the last two cases the scene is left as it was found.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_camera: no spec or model in world")
        return False

    # Snapshot BEFORE mutating. Without a way back the removal is refused here,
    # while the scene is still exactly as it was found.
    backup_spec = _snapshot_spec(spec, context=f"eject_camera {mj_name!r}")
    if backup_spec is None:
        return False

    if not SpecBuilder.remove_camera(spec, mj_name):
        # Nothing was mutated, so there is nothing to roll back. Mirrors
        # :func:`eject_body_from_scene`: the spec already agrees with where the
        # caller's registry is heading, so the removal is not an error.
        logger.warning("Camera '%s' not found in spec - nothing ejected", mj_name)
        return True

    if _recompile_preserving_state(world, spec):
        return True

    # The delete landed in the spec but the model it produced was refused, so
    # the spec is now missing a camera the live model still has. Put the
    # pre-delete spec back.
    world._backend_state["spec"] = backup_spec
    return False


def reposition_body_in_scene(
    world: SimWorld,
    body_name: str,
    position: list[float] | None = None,
    orientation: list[float] | None = None,
) -> bool:
    """Reposition a body (by short name) by editing its spec pose and recompiling.

    Used for STATIC objects, which have no freejoint and therefore cannot be
    moved through ``data.qpos`` at runtime - MuJoCo welds a static body to the
    worldbody with no DOF. Editing the spec body ``pos``/``quat`` and
    recompiling (preserving other joints' state) is the only way to move a
    welded fixture, mirroring :func:`inject_object_into_scene` /
    :func:`eject_body_from_scene`.

    ``position`` / ``orientation`` are applied only when provided (a ``None``
    leaves that component untouched). Returns ``True`` on success, ``False`` if
    the spec/body is missing or the recompile fails (both logged), so the
    caller can surface a clean error instead of a silent no-op.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("reposition_body: no spec or model in world")
        return False

    try:
        body = spec.body(body_name)
    except (KeyError, ValueError):
        body = None
    if body is None:
        logger.warning("Body '%s' not found in spec - nothing repositioned", body_name)
        return False

    if position is not None:
        body.pos = list(position)
    if orientation is not None:
        body.quat = list(orientation)

    return _recompile_preserving_state(world, spec)


# A joint is identified across a scene rebuild by the name of whichever
# namespace can name it. ``("joint", <joint name>)`` covers every named joint.
# ``("body", <body name>)`` covers an UNNAMED ``<freejoint/>`` -- the standard
# MJCF floating-base idiom, used by the Unitree Go2 and by LeKiwi -- which no
# joint-name lookup can reach. The owning body identifies it exactly because
# the compiler refuses a free joint alongside any other joint on the same body
# ("more than 6 dofs in body"), so such a body carries exactly one joint.
#
# The two namespaces are kept apart in the key because MuJoCo's joint and body
# names are independent: a joint may be named exactly like some body, so a
# single flat string key could collide.
_JointKey = tuple[str, str]


@dataclass(frozen=True)
class _SceneState:
    """The dynamic state of a scene, keyed by name so a rebuild can restore it.

    :func:`eject_robot_from_scene` compiles a fresh model and allocates a fresh
    ``MjData``, so every buffer starts at its reset value. What is captured here
    is what has to be carried over that gap for the scene to continue as it was:
    the same state :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.save_state`
    checkpoints, minus the entries a rebuilt scene cannot have a surviving
    instance of.

    Flat-index slicing cannot do this: removing a robot shifts every joint, dof
    and actuator index that came after it, so a positional copy would re-assign
    one element's state to another. Every field below is therefore keyed by name.

    Attributes:
        joints: ``key -> (qpos, qvel, qfrc_applied)`` slices, each at the width
            its joint type uses (free 7/6/6, ball 4/3/3, hinge or slide 1/1/1).
        actuators: ``actuator name -> (ctrl, act)``. ``ctrl`` is the servo
            setpoint holding a robot's pose; ``act`` is the internal activation
            of a stateful actuator, its effective command.
        body_wrenches: ``body name -> [fx, fy, fz, tx, ty, tz]``, the external
            wrench :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.apply_force`
            latches in a body's own ``xfrc_applied`` row. ``qfrc_applied``
            above is its joint-space sibling; both are part of the state
            ``save_state`` checkpoints, so both are carried -- on the eject path
            by this snapshot, and on the grow path by
            :func:`_snapshot_joint_forces` and :func:`_snapshot_body_wrenches`,
            since ``spec.recompile`` carries neither.
        time: ``data.time``, the clock the physics reads.
    """

    joints: dict[_JointKey, tuple[list[float], list[float], list[float]]]
    actuators: dict[str, tuple[float, list[float]]]
    body_wrenches: dict[str, list[float]]
    time: float


def _joint_state_widths(jtype: int, mj: Any) -> tuple[int, int]:
    """Return the ``(qpos, dof)`` width MuJoCo uses for a joint of ``jtype``."""
    if jtype == mj.mjtJoint.mjJNT_FREE:
        return 7, 6
    if jtype == mj.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _joint_key(model: Any, jid: int, mj: Any) -> _JointKey | None:
    """Identify joint ``jid`` by name, or by its owning body when it has none.

    Returns ``None`` for a joint no namespace can name -- an unnamed non-free
    joint (its body may carry several, so the body does not single it out), or
    an unnamed free joint on an unnamed body. Such a joint cannot be matched
    across a rebuild, so it is reported rather than guessed at.
    """
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
    if name:
        return ("joint", name)
    if int(model.jnt_type[jid]) != mj.mjtJoint.mjJNT_FREE:
        return None
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid]))
    return ("body", body_name) if body_name else None


def _snapshot_scene_state(world: SimWorld) -> _SceneState:
    """Capture ``world``'s dynamic state keyed by name, for a scene rebuild.

    Used by :func:`eject_robot_from_scene`. See :class:`_SceneState` for what is
    captured and why it is keyed by name rather than by index.
    """
    if world._model is None or world._data is None:
        return _SceneState(joints={}, actuators={}, body_wrenches={}, time=0.0)
    mj = _ensure_mujoco()
    model = world._model
    data = world._data

    joints: dict[_JointKey, tuple[list[float], list[float], list[float]]] = {}
    for jid in range(model.njnt):
        key = _joint_key(model, jid, mj)
        if key is None:
            logger.debug(
                "snapshot_scene_state: joint id %d has no name and no single owning body, state not carried over",
                jid,
            )
            continue
        qpos_adr = int(model.jnt_qposadr[jid])
        dof_adr = int(model.jnt_dofadr[jid])
        qpos_w, dof_w = _joint_state_widths(int(model.jnt_type[jid]), mj)
        joints[key] = (
            [float(x) for x in data.qpos[qpos_adr : qpos_adr + qpos_w]],
            [float(x) for x in data.qvel[dof_adr : dof_adr + dof_w]],
            [float(x) for x in data.qfrc_applied[dof_adr : dof_adr + dof_w]],
        )

    actuators: dict[str, tuple[float, list[float]]] = {}
    for aid in range(int(model.nu)):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, aid)
        if not name:
            # No namespace names it, and its transmission target may drive
            # several actuators, so it cannot be matched across the rebuild.
            logger.debug("snapshot_scene_state: actuator id %d has no name, ctrl/act not carried over", aid)
            continue
        act_adr = int(model.actuator_actadr[aid])
        act_num = int(model.actuator_actnum[aid])
        act_vals = [float(x) for x in data.act[act_adr : act_adr + act_num]] if act_adr >= 0 else []
        actuators[name] = (float(data.ctrl[aid]), act_vals)

    return _SceneState(
        joints=joints,
        actuators=actuators,
        body_wrenches=_snapshot_body_wrenches(model, data, mj),
        time=float(data.time),
    )


def _restore_scene_state(world: SimWorld, snapshot: _SceneState) -> int:
    """Write ``snapshot`` back into ``world._data`` by name.

    Elements that no longer exist in the compiled model (those belonging to the
    ejected robot) are silently skipped -- their absence is the point of the
    rebuild. Elements present in the new model but absent from the snapshot keep
    their fresh-compile values.

    Returns the number of joints restored, for logging.
    """
    if world._model is None or world._data is None:
        return 0
    mj = _ensure_mujoco()
    model = world._model
    data = world._data

    restored = 0
    for key, (qpos_vals, qvel_vals, qfrc_vals) in snapshot.joints.items():
        jid = _resolve_joint_key(model, key, mj)
        if jid < 0:
            continue  # element no longer exists (expected for the ejected robot)
        qpos_adr = int(model.jnt_qposadr[jid])
        dof_adr = int(model.jnt_dofadr[jid])
        # Width sanity check: if the joint type changed (should not happen for a
        # same-key joint across an eject), skip rather than corrupt the DOFs.
        expect_qp, expect_dof = _joint_state_widths(int(model.jnt_type[jid]), mj)
        if len(qpos_vals) != expect_qp or len(qvel_vals) != expect_dof or len(qfrc_vals) != expect_dof:
            logger.warning(
                "_restore_scene_state: width mismatch for %r (qpos %d!=%d or dof %d/%d!=%d), skipping",
                key,
                len(qpos_vals),
                expect_qp,
                len(qvel_vals),
                len(qfrc_vals),
                expect_dof,
            )
            continue
        for i, v in enumerate(qpos_vals):
            data.qpos[qpos_adr + i] = v
        for i, v in enumerate(qvel_vals):
            data.qvel[dof_adr + i] = v
        for i, v in enumerate(qfrc_vals):
            data.qfrc_applied[dof_adr + i] = v
        restored += 1

    for name, (ctrl_val, act_vals) in snapshot.actuators.items():
        aid = mj_name_to_id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue  # actuator belonged to the ejected robot
        data.ctrl[aid] = ctrl_val
        act_adr = int(model.actuator_actadr[aid])
        act_num = int(model.actuator_actnum[aid])
        if act_adr < 0 or len(act_vals) != act_num:
            # An actuator whose activation width changed across the rebuild:
            # leave the fresh zero rather than write a mismatched slice.
            if act_vals or act_num:
                logger.warning(
                    "_restore_scene_state: act width mismatch for actuator %r (%d!=%d), skipping activation",
                    name,
                    len(act_vals),
                    act_num,
                )
            continue
        for i, v in enumerate(act_vals):
            data.act[act_adr + i] = v

    _restore_body_wrenches(model, data, snapshot.body_wrenches, mj)
    data.time = snapshot.time
    return restored


def _resolve_joint_key(model: Any, key: _JointKey, mj: Any) -> int:
    """Resolve a :data:`_JointKey` to a joint id in ``model``, or ``-1``."""
    kind, name = key
    if kind == "joint":
        return mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, name)
    bid = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, name)
    if bid < 0 or int(model.body_jntnum[bid]) != 1:
        # The body is gone, or no longer carries exactly the one joint that made
        # it an unambiguous handle for its unnamed free joint.
        return -1
    jid = int(model.body_jntadr[bid])
    return jid if int(model.jnt_type[jid]) == mj.mjtJoint.mjJNT_FREE else -1


def eject_robot_from_scene(world: SimWorld, robot_name: str) -> bool:
    """Remove every spec element namespaced under ``{robot_name}/``.

    Implementation note: deleting a body that was added via ``spec.attach()``
    triggers a known MuJoCo 3.8 segfault at interpreter shutdown (the
    attached child spec's memory gets freed twice). To sidestep that bug
    we REBUILD the scene spec from scratch using the post-remove
    ``world.robots`` / ``world.objects`` / ``world.cameras`` state, then
    re-attach the remaining robots.

    State preservation: the fresh compile below allocates a fresh ``MjData``,
    so every buffer starts at its reset value. Before the rebuild we snapshot
    the scene's dynamic state keyed by name (:func:`_snapshot_scene_state`) and
    restore it afterwards, so a surviving element continues exactly as it was --
    which is what :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.add_robot`
    already guarantees for the composition in the other direction ("an arm
    already in the world keeps the pose it is in and the actuator setpoints
    holding it there ... and the clock keeps counting"). Elements belonging to
    the ejected robot are naturally dropped: their names no longer resolve.

    Restoring the pose alone is not enough for the scene to continue:

    * ``ctrl`` -- the setpoint a position servo holds a pose with. Dropped, it
      reads as zero, and the next ``mj_step`` drives every actuator of every
      SURVIVING robot toward its zero configuration: an arm parked mid-air sags
      to the floor while ``remove_robot`` reported ``"success"`` and the pose
      read back correct, because nothing had stepped yet. ``act`` carries the
      same command for a stateful actuator.
    * an UNNAMED ``<freejoint/>`` -- the standard MJCF floating-base idiom, used
      by the Unitree Go2 and by LeKiwi -- is invisible to a joint-name lookup,
      so a surviving mobile base was re-seated at its spawn pose with zero
      velocity: measured on ``go2`` driven to ``(0.8, -0.4, 0.30)``, removing a
      DIFFERENT robot teleported it back to ``(0, 0, 0.445)``. The recompile
      path names this same hazard (see :func:`_recompile_preserving_state`).
    * ``data.time`` -- reset to 0 while ``world.sim_time`` kept counting, so the
      clock the physics reads and the clock ``get_state`` reports disagreed.

    Flat-index slicing is **not** safe here: removing a robot shifts every
    body/joint/dof/actuator index that comes after it, so ``data.qpos[:]``-style
    copies across compiles would mis-assign one element's state to another.
    Per-name lookup is the only correct approach (see AGENTS.md).
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_robot: no spec or model in world")
        return False

    mj = _ensure_mujoco()

    # Snapshot joint state BEFORE we rebuild. Keyed by the fully-qualified
    # MuJoCo joint name (prefix/joint for attached robots, bare name for
    # object freejoints).
    state_snapshot = _snapshot_scene_state(world)

    # First drop cameras that originated from the robot being ejected.
    # They're in world.cameras with origin_robot == robot_name. Without this,
    # SpecBuilder.build would skip them (via origin_robot), but stale entries
    # would linger in the registry and confuse observation code.
    stale_cam_names = [cname for cname, cam in world.cameras.items() if getattr(cam, "origin_robot", "") == robot_name]
    for cname in stale_cam_names:
        del world.cameras[cname]

    # Step 1: rebuild the base spec from world (objects + cameras +
    # lights + ground).
    new_spec = SpecBuilder.build(world)

    # Step 2: re-attach every remaining robot (the one being ejected is
    # already popped from ``world.robots`` by the caller).
    for robot in world.robots.values():
        # Re-discover joint names via the attach - they're stable per URDF.
        with filter_mujoco_attach_noise():
            joint_names = SpecBuilder.attach_robot(new_spec, robot, robot.urdf_path)
        robot.joint_names = joint_names

    # Step 2b: mount the body-mounted cameras ``SpecBuilder.build`` deferred,
    # now that every surviving robot's bodies exist in the spec. A camera whose
    # parent belonged to the robot being ejected has no parent left, so it is
    # dropped from the registry with a warning rather than aborting the eject -
    # the same treatment the robot's own URDF cameras get above. Without this
    # step the rebuild raised ``ValueError`` straight out of ``remove_robot``,
    # so a scene with a wrist camera could not remove any robot at all.
    for cname in SpecBuilder.add_deferred_cameras(new_spec, world):
        logger.warning(
            "eject_robot: dropping camera %r - its parent body %r belonged to the removed robot %r.",
            cname,
            getattr(world.cameras[cname], "parent_body", ""),
            robot_name,
        )
        del world.cameras[cname]

    # Step 3: compile fresh and install. No spec.recompile(model, data)
    # here - recompile implicitly preserves qpos state which doesn't
    # make sense across a scene rebuild, and forcing a fresh compile
    # avoids the attach/delete bug.
    try:
        with filter_mujoco_attach_noise():
            new_model = new_spec.compile()
        new_data = mj.MjData(new_model)
    except (ValueError, RuntimeError) as e:
        logger.error("eject_robot: fresh compile failed: %s", e)
        return False

    install_compiled_model(world, new_model, new_data)
    world._backend_state["spec"] = new_spec
    _sync_cached_xml(world, new_spec)

    # Step 4: restore state for every joint that survived the rebuild. Joints
    # belonging to the ejected robot simply don't resolve and get skipped.
    restored = _restore_scene_state(world, state_snapshot)

    # Step 5: run a forward pass so derived quantities (xpos, cam xforms)
    # reflect the restored state. Without this, the next render() call can
    # produce stale frames because MjData was freshly allocated in Step 3.
    mj.mj_forward(new_model, new_data)

    # Re-discover joint/actuator IDs for remaining robots.
    for robot in world.robots.values():
        pfx = robot.namespace or ""
        robot.joint_ids = []
        for jnt_name in robot.joint_names:
            jid = -1
            if pfx:
                jid = mj_name_to_id(new_model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jid < 0:
                jid = mj_name_to_id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        robot.actuator_ids = robot_owned_actuator_ids(new_model, robot, mj)
        # See _recompile_preserving_state for why the lone-robot case falls back.
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(new_model.nu))

    logger.debug(
        "eject_robot %r: scene rebuilt, restored state for %d/%d joints",
        robot_name,
        restored,
        len(state_snapshot.joints),
    )
    return True


# Runtime attach / actuate primitives (GH #1533, PR 1)


def add_weld_constraint(
    world: SimWorld,
    *,
    name: str,
    parent: str,
    child: str,
    relpos: list[float],
    relquat: list[float],
    torquescale: float = 1.0,
) -> bool:
    """Add a named weld equality constraint between two bodies and recompile.

    ``relpos`` / ``relquat`` are the pose of ``child`` expressed in ``parent``'s
    frame - callers capture the CURRENT runtime relative pose so the weld holds
    the bodies exactly where they are (MuJoCo's all-zero ``relpose`` quat would
    instead bake in the compile-time ``qpos0`` pose, which is wrong for a
    runtime grasp-attach). The equality is stored on the live spec so it
    survives later recompiles (``add_object`` etc.).

    Returns ``True`` on success. On recompile failure the just-added equality
    is deleted again so the spec stays compilable, and ``False`` is returned.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("add_weld_constraint: no spec or model in world")
        return False
    mj = _ensure_mujoco()

    eq = spec.add_equality()
    eq.name = name
    eq.type = mj.mjtEq.mjEQ_WELD
    eq.objtype = mj.mjtObj.mjOBJ_BODY
    eq.name1 = parent
    eq.name2 = child
    # mjEQ_WELD data layout (mjNEQDATA=11): [anchor(3), relpose pos(3),
    # relpose quat(4), torquescale(1)]. anchor stays zero - relpose fully
    # determines the held configuration.
    data = [0.0] * 11
    data[3:6] = [float(v) for v in relpos]
    data[6:10] = [float(v) for v in relquat]
    data[10] = float(torquescale)
    eq.data = data

    if not _recompile_preserving_state(world, spec):
        spec.delete(eq)
        return False
    return True


def remove_equality_constraint(world: SimWorld, name: str) -> bool:
    """Delete a named equality constraint from the live spec and recompile.

    Returns ``False`` (logged) when the constraint is missing or the recompile
    fails, so callers can surface a clean error instead of a silent no-op.

    On recompile failure the deletion is rolled back from a pre-delete
    snapshot, mirroring the way :func:`add_weld_constraint` deletes the
    equality it had just added. Without that restore a refused recompile leaves
    the constraint gone from the live spec while the compiled model still holds
    it, so the caller is told the removal failed, the identical retry is then
    refused as "not found", and the next unrelated scene mutation recompiles
    the spec and silently drops the constraint the caller was told still stood.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("remove_equality_constraint: no spec or model in world")
        return False
    for eq in spec.equalities:
        if eq.name == name:
            # Snapshot before the delete, and only once the constraint is known
            # to exist: a lookup that finds nothing mutates nothing and so needs
            # no way back.
            backup_spec = _snapshot_spec(spec, context="remove_equality_constraint")
            if backup_spec is None:
                return False
            spec.delete(eq)
            if not _recompile_preserving_state(world, spec):
                world._backend_state["spec"] = backup_spec
                return False
            return True
    logger.warning("Equality constraint '%s' not found in spec - nothing removed", name)
    return False


def actuate_robot_in_scene(
    world: SimWorld,
    robot: SimRobot,
    kp_by_joint: dict[str, float],
    *,
    damping: float,
    armature: float,
    gravity_compensation: bool,
    disable_self_collision: bool,
) -> bool:
    """Add position-servo actuators to a robot's joints and recompile.

    The supported form of the private-spec surgery the ``so101_curobo`` example
    performed by hand (GH #1533): converts an actuator-less (URDF-loaded) arm
    into a position-controlled one so ``send_action`` / ``run_policy`` can
    drive it. Per joint in ``kp_by_joint`` (SHORT joint names, values = kp):

    * a position actuator (``set_to_position``: fixed gain kp, affine bias with
      ``dampratio=1.0`` for ~critical damping, ctrlrange inherited from the
      joint's range when it declares one),
    * joint ``damping`` / ``armature`` floors (bare URDFs ship none, which
      blows up explicit integration),

    plus, scene-wide, the stable ``implicitfast`` integrator, and optionally
    gravity compensation on the robot's bodies and self-collision disable on
    the robot's own geoms (cuRobo-style planners ignore adjacent-link
    contacts, which otherwise block planned motion).

    Atomicity: the spec is snapshotted (XML round-trip) before surgery; any
    failure restores the snapshot so no partial edit (integrator flip, gravcomp,
    half the actuators) lingers on the live spec. Returns ``True`` on success.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("actuate_robot_in_scene: no spec or model in world")
        return False
    mj = _ensure_mujoco()
    pfx = robot.namespace or ""

    # Snapshot for atomic rollback (mirrors patch_scene_mjcf): the surgery
    # touches option/bodies/joints/actuators/geoms, too many objects to undo
    # piecewise.
    backup_spec = _snapshot_spec(spec, context="actuate_robot_in_scene")
    if backup_spec is None:
        return False

    try:
        # Bare URDF chains (no damping/armature) diverge under the default
        # Euler integrator once stiff position servos are added; implicitfast
        # integrates joint damping implicitly and stays stable.
        spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

        for body in spec.bodies:
            body_name = body.name or ""
            if pfx and body_name.startswith(pfx):
                if gravity_compensation:
                    body.gravcomp = 1.0
                if disable_self_collision:
                    for geom in body.geoms:
                        geom.contype = 0
                        geom.conaffinity = 0

        for joint in spec.joints:
            joint_name = joint.name or ""
            if not (pfx and joint_name.startswith(pfx)):
                continue
            short = joint_name[len(pfx) :]
            if short not in kp_by_joint:
                continue
            _raise_spec_joint_damping(joint, damping)
            joint.armature = max(float(joint.armature), armature)

        for short, kp in kp_by_joint.items():
            act = spec.add_actuator()
            act.name = f"{robot.name}_act_{short}"
            act.target = f"{pfx}{short}"
            act.trntype = mj.mjtTrn.mjTRN_JOINT
            # Find the joint's spec range: inheritrange clamps ctrl to the
            # joint's limits when it declares any (URDF limits map here);
            # unlimited joints keep an unlimited ctrlrange.
            jnt_range_defined = False
            for joint in spec.joints:
                if (joint.name or "") == f"{pfx}{short}":
                    jnt_range_defined = bool(float(joint.range[0]) < float(joint.range[1]))
                    break
            act.set_to_position(kp=float(kp), dampratio=1.0, inheritrange=jnt_range_defined)
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error("actuate_robot_in_scene: spec surgery failed for '%s': %s", robot.name, e)
        world._backend_state["spec"] = backup_spec
        return False

    if not _recompile_preserving_state(world, spec):
        # Restore the pre-surgery spec so the failed edit doesn't poison
        # later scene mutations.
        world._backend_state["spec"] = backup_spec
        return False
    return True


# Agent-authored raw MJCF (Stage 6)


def replace_scene_mjcf(world: SimWorld, xml: str) -> bool:
    """Atomically swap the whole scene for agent-written MJCF.

    Validated by actually compiling it. On failure raises ``ValueError`` with
    MuJoCo's compiler error verbatim. On success, the old spec/model/data are
    replaced and all per-robot joint/actuator IDs re-discovered (but since
    the agent may have changed the whole scene, the ``world.robots`` dict
    is NOT touched - that's the caller's responsibility).
    """
    mj = _ensure_mujoco()
    new_spec = SpecBuilder.from_mjcf_string(xml)
    # Compile eagerly so malformed XML fails here rather than on the next
    # mj_step.
    with filter_mujoco_attach_noise():
        new_model = new_spec.compile()
    new_data = mj.MjData(new_model)

    world._backend_state["spec"] = new_spec
    install_compiled_model(world, new_model, new_data)

    # Run a single forward pass so geom positions / camera xforms are
    # populated. Without this, the very first sim.render() call after
    # replace_scene_mjcf hits `data.xpos == 0 for all bodies` and the
    # renderer dumps a 100% black frame. Matches the semantics of
    # _compile_world() which also calls mj_forward after MjData construction.
    mj.mj_forward(new_model, new_data)

    _sync_cached_xml(world, new_spec)
    return True


# Structured-op patching of the live spec (Stage 6, part 2 - GH #125)

# Supported ops for patch_scene_mjcf, each mapped to the complete set of keys
# that op reads. Kept narrow on purpose - adding unchecked attribute setters
# would make the tool an arbitrary-code hole. Agents that need exotic MJCF
# should go through replace_scene_mjcf with a full XML.
#
# This mapping is the single source of truth for the op vocabulary: it names
# the supported ops AND, per op, every key that reaches MuJoCo. Any other key
# is refused, because each field below is read with a fallback default - a
# misspelled key would otherwise apply that default and report success, so
# ``{"op": "set_body_pos", "name": "crate", "position": [...]}`` would move the
# body to the origin rather than to the requested pose.
_PATCH_OP_KEYS: dict[str, frozenset[str]] = {
    "add_body": frozenset({"op", "parent", "name", "pos", "quat"}),
    "add_geom": frozenset({"op", "body", "type", "size", "rgba", "name", "pos", "quat"}),
    "add_site": frozenset({"op", "body", "name", "pos", "size", "rgba"}),
    "set_body_pos": frozenset({"op", "name", "pos"}),
    "set_body_quat": frozenset({"op", "name", "quat"}),
    "delete_body": frozenset({"op", "name"}),
}


def _unknown_op_keys_error(kind: str, op: Mapping[str, Any]) -> str | None:
    """Reject keys a patch op does not read, before its defaults are applied.

    Every field of every op is read with a fallback default (``pos`` defaults
    to the origin, ``quat`` to identity, ``type`` to ``"box"``, ``parent`` to
    ``"world"``), so an unrecognised key is not an inert extra: the op runs
    with that default and reports success. A misspelled ``pos`` silently moves
    a body to the world origin, a misspelled ``parent`` re-parents it to the
    worldbody, and a misspelled ``type`` compiles a box where a sphere was
    asked for.

    Args:
        kind: The op name, already known to be in :data:`_PATCH_OP_KEYS`.
        op: The caller-supplied op dict.

    Returns:
        An error message naming the unknown key(s), a close-match suggestion
        where one exists, and the keys this op accepts - or ``None`` when
        every key is honored.
    """
    accepted = _PATCH_OP_KEYS[kind]
    unknown = [key for key in op if key not in accepted]
    if not unknown:
        return None
    known = sorted(accepted)
    # Suggest field names only. "op" selects the op and is already validated
    # above, so it is never what a misspelled field meant - offering it turns a
    # real MJCF attribute like "group" into a nonsense hint.
    candidates = [key for key in known if key != "op"]
    described = []
    for key in sorted(unknown, key=str):
        close = difflib.get_close_matches(str(key).lower(), candidates, n=1, cutoff=0.5)
        described.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    return f"{kind}: unknown op key(s): {', '.join(described)}. Accepted keys: {', '.join(known)}."


# The numeric vector fields each op writes into the compiled MJCF. Every name
# here is also an accepted key of the same op in :data:`_PATCH_OP_KEYS`; a
# parity test pins that, so the two tables cannot drift into disagreement about
# which fields an op reads.
_PATCH_OP_VECTOR_FIELDS: dict[str, frozenset[str]] = {
    "add_body": frozenset({"pos", "quat"}),
    "add_geom": frozenset({"pos", "quat", "size", "rgba"}),
    "add_site": frozenset({"pos", "size", "rgba"}),
    "set_body_pos": frozenset({"pos"}),
    "set_body_quat": frozenset({"quat"}),
    "delete_body": frozenset(),
}


def _rgba_field_domain(kind: str, field: str, value: Any) -> tuple[Any, str | None]:
    """Validate an op's colour and complete a 3-component RGB with opaque alpha.

    Thin adapter so :func:`~strands_robots.utils.coerce_rgba` - the single
    definition of the library's colour contract - can sit in
    :data:`_OP_FIELD_DOMAINS` beside the other field domains.

    It differs from that helper in one respect, because an op dict differs from a
    keyword argument: a key that is PRESENT carries a supplied value, so ``None``
    is a colour this op cannot apply rather than an omission asking for the
    default. Reading it as omitted would paint the fallback grey under a success
    result, which is the silent default the op-key check exists to prevent.

    Args:
        kind: The op name, for the message.
        field: The field name, always ``"rgba"``.
        value: The caller-supplied colour.

    Returns:
        ``(rgba, None)`` with exactly four components, or ``(None, message)``.
    """
    if value is None:
        return None, f"{kind}: '{field}' must be a sequence of numbers, got None"
    return coerce_rgba(kind, field, value)


# The domain each numeric op field is held to, keyed by field name. Every field
# any op declares in :data:`_PATCH_OP_VECTOR_FIELDS` has an entry here and a
# parity test pins that, so a field added to an op cannot reach the compiled
# model without a decided domain.
#
# The widths are the library's, not MuJoCo's: a ``pos`` is three components and a
# ``quat`` four whichever op writes it, and a colour is the contract
# :func:`~strands_robots.utils.coerce_rgba` defines for every backend. ``size``
# is the one field whose count is shape-dependent, so only its components are
# checked here and the count is left to ``_validate_size``, which knows the shape.
_OP_FIELD_DOMAINS: dict[str, Callable[[str, str, Any], tuple[Any, str | None]]] = {
    "pos": lambda kind, field, value: (value, pose_vector_error(kind, field, value, 3)),
    "quat": lambda kind, field, value: (value, pose_vector_error(kind, field, value, 4)),
    "rgba": _rgba_field_domain,
    "size": lambda kind, field, value: (value, finite_vector_error(kind, field, value)),
}


def _normalized_op_vector_fields(kind: str, op: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Hold every numeric op field to its domain, before the spec is touched.

    MuJoCo's compiler does not reject a ``nan``/``inf`` pose, extent or colour
    (its one exception is a ``nan`` geom size), so an unchecked component is
    written verbatim into the model and the patch reports success. A non-finite
    ``pos`` on a body that owns a freejoint then poisons ``qpos``/``qvel`` on the
    next step, and every call in that chain - the patch, the step, the
    observation read - keeps reporting success, so a caller has no way to learn
    the scene is dead.

    Component count is part of the same contract. These ops write the very
    buffers ``add_object``, ``add_camera`` and ``move_object`` write, so a value
    either surface refuses has to be refused by the other - and MuJoCo alone
    cannot deliver that:

    * ``set_body_pos`` / ``set_body_quat`` assign the field as a spec ATTRIBUTE
      (``body.pos = ...``) rather than passing it as a constructor keyword, and
      pybind11 reports a width mismatch there by dumping its C++ overload table
      and the receiving object's address. That message names neither the op nor
      the field, so the one thing the caller needs is the one thing absent from
      it, while the sibling ops writing the same two fields report cleanly.
    * A three-component ``rgba`` is refused outright, though it is the RGB that
      ``add_object(color=...)`` accepts and completes with an opaque alpha. One
      backend, two surfaces, one ``geom_rgba`` buffer, opposite verdicts.

    Args:
        kind: The op name, already known to be in :data:`_PATCH_OP_KEYS`.
        op: The caller-supplied op dict.

    Returns:
        ``(op, None)`` with every numeric field present normalized to what the
        spec write consumes, or ``(None, message)`` naming the op, the field and
        the offending value.
    """
    normalized = dict(op)
    for field in sorted(_PATCH_OP_VECTOR_FIELDS[kind]):
        if field not in op:
            continue
        value, msg = _OP_FIELD_DOMAINS[field](kind, field, op[field])
        if msg is not None:
            return None, msg
        normalized[field] = value
    return normalized, None


def _find_body(spec: Any, name: str, new_bodies: dict[str, Any]) -> Any:
    """Locate a body by name in a live spec, checking batch-local additions.

    MuJoCo 3.8 ``spec.body(name)`` only resolves bodies that existed at the
    last ``compile()`` / ``recompile()`` call. Bodies added mid-batch are
    not visible through that lookup but ARE present on the spec - we track
    their handles in ``new_bodies`` so ``add_geom`` / ``add_site`` /
    ``set_body_pos`` etc. can reference them within the same patch.
    """
    if name == "world":
        return spec.worldbody
    if name in new_bodies:
        return new_bodies[name]
    b = spec.body(name)
    if b is not None:
        return b
    # Fallback: scan all bodies. Catches bodies introduced via spec.attach()
    # (e.g. robots composed into the scene) that aren't in new_bodies because
    # we didn't create them in this batch.
    for body in spec.bodies:
        if body.name == name:
            return body
    return None


def _apply_patch_op(spec: Any, op: dict[str, Any], new_bodies: dict[str, Any]) -> None:
    """Apply a single structured op to a live MjSpec.

    Raises ``ValueError`` with a human-readable message on bad input;
    MuJoCo compile errors surface on the enclosing ``recompile`` call.
    ``new_bodies`` is a batch-local cache of body handles added earlier
    in the same patch (see ``_find_body`` for why this is needed).
    """
    if not isinstance(op, dict):
        raise ValueError(f"each op must be a dict, got {type(op).__name__}")

    kind = op.get("op")
    if kind not in _PATCH_OP_KEYS:
        raise ValueError(f"unknown op '{kind}'. Supported: {sorted(_PATCH_OP_KEYS)}")
    if err := _unknown_op_keys_error(str(kind), op):
        raise ValueError(err)
    normalized, err = _normalized_op_vector_fields(str(kind), op)
    if normalized is None:
        raise ValueError(err)
    # Every numeric field below now holds the value the spec write consumes -
    # notably a three-component 'rgba' completed to RGBA.
    op = normalized

    if kind == "add_body":
        parent = op.get("parent", "world")
        name = op.get("name")
        if not name:
            raise ValueError("add_body requires 'name'")
        pos = op.get("pos", [0.0, 0.0, 0.0])
        quat = op.get("quat", [1.0, 0.0, 0.0, 0.0])
        parent_body = _find_body(spec, parent, new_bodies)
        if parent_body is None:
            raise ValueError(f"add_body: parent '{parent}' not found")
        new_body = parent_body.add_body(name=name, pos=pos, quat=quat)
        new_bodies[name] = new_body
        return

    if kind == "add_geom":
        body_name = op.get("body")
        if not body_name:
            raise ValueError("add_geom requires 'body'")
        body = _find_body(spec, body_name, new_bodies)
        if body is None:
            raise ValueError(f"add_geom: body '{body_name}' not found")

        shape = op.get("type", "box")
        from strands_robots.simulation.mujoco.spec_builder import (
            _geom_type,
            _normalize_size,
        )

        geom_kwargs: dict[str, Any] = {
            "type": _geom_type(shape),
            "size": _normalize_size(shape, op.get("size", [0.1, 0.1, 0.1])),
            "rgba": op.get("rgba", [0.5, 0.5, 0.5, 1.0]),
        }
        if "name" in op:
            geom_kwargs["name"] = op["name"]
        if "pos" in op:
            geom_kwargs["pos"] = op["pos"]
        if "quat" in op:
            geom_kwargs["quat"] = op["quat"]
        body.add_geom(**geom_kwargs)
        return

    if kind == "add_site":
        body_name = op.get("body", "world")
        body = _find_body(spec, body_name, new_bodies)
        if body is None:
            raise ValueError(f"add_site: body '{body_name}' not found")
        name = op.get("name")
        if not name:
            raise ValueError("add_site requires 'name'")
        site_kwargs: dict[str, Any] = {
            "name": name,
            "pos": op.get("pos", [0.0, 0.0, 0.0]),
        }
        if "size" in op:
            site_kwargs["size"] = op["size"]
        if "rgba" in op:
            site_kwargs["rgba"] = op["rgba"]
        body.add_site(**site_kwargs)
        return

    if kind == "set_body_pos":
        name = op.get("name")
        if not name:
            raise ValueError("set_body_pos requires 'name'")
        body = _find_body(spec, name, new_bodies)
        if body is None:
            raise ValueError(f"set_body_pos: body '{name}' not found")
        body.pos = op.get("pos", [0.0, 0.0, 0.0])
        return

    if kind == "set_body_quat":
        name = op.get("name")
        if not name:
            raise ValueError("set_body_quat requires 'name'")
        body = _find_body(spec, name, new_bodies)
        if body is None:
            raise ValueError(f"set_body_quat: body '{name}' not found")
        body.quat = op.get("quat", [1.0, 0.0, 0.0, 0.0])
        return

    if kind == "delete_body":
        name = op.get("name")
        if not name:
            raise ValueError("delete_body requires 'name'")
        body = _find_body(spec, name, new_bodies)
        if body is None:
            raise ValueError(f"delete_body: body '{name}' not found")
        spec.delete(body)
        new_bodies.pop(name, None)
        return


def patch_scene_mjcf(world: SimWorld, ops: list[dict[str, Any]]) -> int:
    """Apply a sequence of structured ops to the live spec in order.

    Each op is a small dict like::

        {"op": "add_body", "parent": "world", "name": "foo", "pos": [0,0,1]}
        {"op": "add_geom", "body": "foo", "type": "sphere", "size": [0.1]}
        {"op": "set_body_pos", "name": "foo", "pos": [1,0,1]}
        {"op": "delete_body", "name": "foo"}

    Each op accepts exactly the keys listed for it in :data:`_PATCH_OP_KEYS`;
    anything else is rejected, because an unread key would leave the op running
    on its fallback default (see :func:`_unknown_op_keys_error`). Every numeric
    field an op writes is held to its domain in :data:`_OP_FIELD_DOMAINS` - a
    ``pos`` of exactly 3 finite components, a ``quat`` of 4, a ``rgba`` of 3
    (RGB, completed with an opaque alpha) or 4 - because MuJoCo bakes a
    ``nan``/``inf`` component into the model without complaint, and reports a
    width mismatch on the attribute writes by dumping a C++ overload table that
    names neither the op nor the field.

    The list is applied atomically: if any op raises, the whole patch is
    rejected and the world is left in its original state. After all ops
    succeed, ``spec.recompile(model, data)`` is called once, so joint
    qpos/qvel for unchanged joints are preserved automatically.

    Returns the number of ops applied (same as ``len(ops)`` on success).
    """
    if not isinstance(ops, list):
        raise ValueError(f"ops must be a list, got {type(ops).__name__}")
    if not ops:
        return 0

    spec = world._backend_state.get("spec")
    if spec is None:
        raise RuntimeError("world has no spec; patch_scene_mjcf requires a compiled world")

    # Snapshot the spec so a failed op can be atomically rejected: the batch is
    # applied to the live spec and the snapshot put back if any op fails.
    backup_spec = _snapshot_spec(spec, context="patch_scene_mjcf")
    if backup_spec is None:
        raise RuntimeError("failed to snapshot spec before patch (see logs)")

    applied = 0
    new_bodies: dict[str, Any] = {}
    try:
        for op in ops:
            _apply_patch_op(spec, op, new_bodies)
            applied += 1
    except Exception as err:
        world._backend_state["spec"] = backup_spec
        raise ValueError(f"patch op #{applied + 1} failed: {err}") from err

    # One recompile for the whole batch - preserves qpos/qvel for unchanged joints.
    with filter_mujoco_attach_noise():
        new_model, new_data = spec.recompile(world._model, world._data)
    install_compiled_model(world, new_model, new_data)

    # Forward pass so new bodies' xpos / xquat / cam_xmat are populated for
    # the very next render() or get_body_state() call. Same reasoning as
    # replace_scene_mjcf.
    mj = _ensure_mujoco()
    mj.mj_forward(world._model, world._data)

    _sync_cached_xml(world, spec)
    return applied
