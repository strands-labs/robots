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
* :func:`eject_body_from_scene` - ``SpecBuilder.remove_body(spec, name)`` + recompile.
* :func:`reposition_body_in_scene` - edit a body's spec ``pos``/``quat`` + recompile.
* :func:`eject_robot_from_scene` - walk the spec, delete everything namespaced
  under ``{robot_name}/``, then recompile.

Every function takes a ``SimWorld`` whose ``_backend_state["spec"]`` holds the
live ``MjSpec``. They return ``True`` on success, ``False`` on failure (matching
the legacy API) so call sites in :mod:`simulation` don't need to change.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import (
    _ensure_mujoco,
    filter_mujoco_attach_noise,
)
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder

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


def _recompile_preserving_state(world: SimWorld, spec: Any) -> bool:
    """Recompile ``spec`` in place, replacing ``world._model`` and ``_data``.

    Uses ``spec.recompile(model, data)`` which auto-preserves qpos/qvel for
    existing joints and initializes new joints to their body's pos/quat. No
    manual state-copy loop is required.

    Also re-discovers per-robot joint and actuator IDs (they may have shifted
    as new bodies were inserted earlier in the body tree). Returns True on
    success, False on compile failure (logged).
    """
    mj = _ensure_mujoco()
    try:
        with filter_mujoco_attach_noise():
            new_model, new_data = spec.recompile(world._model, world._data)
    except (ValueError, RuntimeError) as e:
        logger.error("spec.recompile failed: %s", e)
        return False

    world._model = new_model
    world._data = new_data

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
        robot.actuator_ids = []
        for jnt_name in robot.joint_names:
            jid = -1
            if pfx:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jid < 0:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        for i in range(new_model.nu):
            jnt_id = new_model.actuator_trnid[i, 0]
            if jnt_id in robot.joint_ids:
                robot.actuator_ids.append(i)
        # Single-robot fallback: if no actuators matched by joint, assume
        # all actuators belong to this robot. Matches the legacy behaviour.
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(new_model.nu))

    return True


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
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_robot: no spec or model in world")
        return False

    try:
        with filter_mujoco_attach_noise():
            joint_names = SpecBuilder.attach_robot(spec, robot, robot_xml_path)
        robot.joint_names = joint_names
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Robot attach failed for '%s': %s", robot.name, e)
        return False

    return _recompile_preserving_state(world, spec)


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
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_object: no spec or model in world")
        return False

    try:
        # Meshes need their asset registered before the geom references it.
        # build() registers meshes in a separate pass, so add_object does not;
        # the incremental path must register it here.
        if obj.shape == "mesh" and obj.mesh_path:
            spec.add_mesh(name=f"mesh_{obj.name}", file=obj.mesh_path)
        SpecBuilder.add_object(spec, obj)
    except (ValueError, RuntimeError) as e:
        logger.error("Object add failed for '%s': %s", obj.name, e)
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        return False

    if not _recompile_preserving_state(world, spec):
        # Roll the just-added body (and any mesh asset) back out so the spec
        # returns to its last good, compilable state (a worldbody body delete
        # is safe - the attach/delete segfault only affects spec.attach()
        # child specs).
        SpecBuilder.remove_body(spec, obj.name)
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        return False
    return True


def inject_camera_into_scene(world: SimWorld, cam: SimCamera) -> bool:
    """Add a camera to the scene and recompile in place.

    Mirrors :func:`inject_object_into_scene`: ``SpecBuilder.add_camera`` mutates
    the spec before the validating recompile, so a refused recompile rolls the
    just-added camera back out to keep the spec compilable for later edits.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_camera: no spec or model in world")
        return False

    try:
        SpecBuilder.add_camera(spec, cam)
    except (ValueError, RuntimeError) as e:
        logger.error("Camera add failed for '%s': %s", cam.name, e)
        return False

    if not _recompile_preserving_state(world, spec):
        SpecBuilder.remove_camera(spec, cam.name)
        return False
    return True


# Eject


def eject_body_from_scene(world: SimWorld, body_name: str) -> bool:
    """Remove a body (by short name) and recompile."""
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_body: no spec or model in world")
        return False

    if not SpecBuilder.remove_body(spec, body_name):
        logger.warning("Body '%s' not found in spec - nothing ejected", body_name)
        # Matching legacy behaviour: return True so scene state stays consistent
        # (caller has already popped the Python-side dict entry).
        return True

    # Objects added at runtime register a mesh asset named f"mesh_{name}".
    # Delete it too so the name is fully reusable and unused assets do not
    # accumulate across remove/re-add cycles (safe no-op for primitives).
    SpecBuilder.remove_mesh(spec, f"mesh_{body_name}")

    return _recompile_preserving_state(world, spec)


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


def _snapshot_joint_state(world: SimWorld) -> dict[str, tuple[list[float], list[float]]]:
    """Snapshot per-joint ``(qpos, qvel)`` slices keyed by fully-qualified
    MuJoCo joint name.

    Used by :func:`eject_robot_from_scene` to preserve the state of surviving
    robots and object freejoints across a scene rebuild. Flat-index slicing
    is unsafe here because the body-tree order may shift when a robot is
    removed (see AGENTS.md "Per-name state copy" rule).

    Returns a dict mapping ``<joint_name> -> (qpos_slice, qvel_slice)`` where
    each slice has the appropriate width for the joint type (1 for hinge/
    slide, 4 for ball, 7 for free).
    """
    if world._model is None or world._data is None:
        return {}
    mj = _ensure_mujoco()
    model = world._model
    data = world._data
    snap: dict[str, tuple[list[float], list[float]]] = {}
    for jid in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])
        jtype = int(model.jnt_type[jid])
        # qpos width: free=7, ball=4, hinge/slide=1
        # qvel width: free=6, ball=3, hinge/slide=1
        if jtype == mj.mjtJoint.mjJNT_FREE:
            qpos_w, qvel_w = 7, 6
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            qpos_w, qvel_w = 4, 3
        else:
            qpos_w, qvel_w = 1, 1
        snap[name] = (
            [float(x) for x in data.qpos[qpos_adr : qpos_adr + qpos_w]],
            [float(x) for x in data.qvel[qvel_adr : qvel_adr + qvel_w]],
        )
    return snap


def _restore_joint_state(
    world: SimWorld,
    snapshot: dict[str, tuple[list[float], list[float]]],
) -> int:
    """Restore per-joint state from a snapshot into ``world._data`` by name.

    Joints that no longer exist in the compiled model (e.g. those belonging
    to the ejected robot) are silently skipped. Joints that exist in the
    new model but were not in the snapshot keep their fresh-compile defaults
    (body pos/quat for freejoints, 0 for hinge/slide).

    Returns the number of joints actually restored, for logging.
    """
    if world._model is None or world._data is None:
        return 0
    mj = _ensure_mujoco()
    model = world._model
    data = world._data
    restored = 0
    for name, (qpos_vals, qvel_vals) in snapshot.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue  # joint no longer exists (expected for ejected robot)
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])
        # Width sanity check: if joint type changed (should not happen for
        # same-name joints across an eject), skip to avoid corrupting state.
        jtype = int(model.jnt_type[jid])
        if jtype == mj.mjtJoint.mjJNT_FREE:
            expect_qp, expect_qv = 7, 6
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            expect_qp, expect_qv = 4, 3
        else:
            expect_qp, expect_qv = 1, 1
        if len(qpos_vals) != expect_qp or len(qvel_vals) != expect_qv:
            logger.warning(
                "_restore_joint_state: width mismatch for %r (qpos %d!=%d or qvel %d!=%d), skipping",
                name,
                len(qpos_vals),
                expect_qp,
                len(qvel_vals),
                expect_qv,
            )
            continue
        for i, v in enumerate(qpos_vals):
            data.qpos[qpos_adr + i] = v
        for i, v in enumerate(qvel_vals):
            data.qvel[qvel_adr + i] = v
        restored += 1
    return restored


def eject_robot_from_scene(world: SimWorld, robot_name: str) -> bool:
    """Remove every spec element namespaced under ``{robot_name}/``.

    Implementation note: deleting a body that was added via ``spec.attach()``
    triggers a known MuJoCo 3.8 segfault at interpreter shutdown (the
    attached child spec's memory gets freed twice). To sidestep that bug
    we REBUILD the scene spec from scratch using the post-remove
    ``world.robots`` / ``world.objects`` / ``world.cameras`` state, then
    re-attach the remaining robots.

    Joint state preservation: before the rebuild we snapshot every joint's
    ``(qpos, qvel)`` keyed by fully-qualified name; after the fresh compile
    we restore state for every joint that still exists in the new model.
    Joints belonging to the ejected robot are naturally dropped (their name
    no longer resolves). This keeps surviving robots at their current pose
    and object freejoints at their current world pose - the behaviour the
    agent expects when calling ``remove_robot`` mid-scene.

    Flat-index slicing is **not** safe here: removing a robot shifts every
    body/joint index that comes after it in the kinematic tree, so
    ``data.qpos[:]`` copies across compiles would mis-assign DOFs. Per-name
    lookup is the only correct approach (see AGENTS.md).
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_robot: no spec or model in world")
        return False

    mj = _ensure_mujoco()

    # Snapshot joint state BEFORE we rebuild. Keyed by the fully-qualified
    # MuJoCo joint name (prefix/joint for attached robots, bare name for
    # object freejoints).
    state_snapshot = _snapshot_joint_state(world)

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

    world._model = new_model
    world._data = new_data
    world._backend_state["spec"] = new_spec
    _sync_cached_xml(world, new_spec)

    # Step 4: restore state for every joint that survived the rebuild. Joints
    # belonging to the ejected robot simply don't resolve and get skipped.
    restored = _restore_joint_state(world, state_snapshot)

    # Step 5: run a forward pass so derived quantities (xpos, cam xforms)
    # reflect the restored state. Without this, the next render() call can
    # produce stale frames because MjData was freshly allocated in Step 3.
    mj.mj_forward(new_model, new_data)

    # Re-discover joint/actuator IDs for remaining robots.
    for robot in world.robots.values():
        pfx = robot.namespace or ""
        robot.joint_ids = []
        robot.actuator_ids = []
        for jnt_name in robot.joint_names:
            jid = -1
            if pfx:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jid < 0:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        for i in range(new_model.nu):
            jnt_id = new_model.actuator_trnid[i, 0]
            if jnt_id in robot.joint_ids:
                robot.actuator_ids.append(i)
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(new_model.nu))

    logger.debug(
        "eject_robot %r: scene rebuilt, restored state for %d/%d joints",
        robot_name,
        restored,
        len(state_snapshot),
    )
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
    world._model = new_model
    world._data = new_data

    # Run a single forward pass so geom positions / camera xforms are
    # populated. Without this, the very first sim.render() call after
    # replace_scene_mjcf hits `data.xpos == 0 for all bodies` and the
    # renderer dumps a 100% black frame. Matches the semantics of
    # _compile_world() which also calls mj_forward after MjData construction.
    mj.mj_forward(new_model, new_data)

    _sync_cached_xml(world, new_spec)
    return True


# Structured-op patching of the live spec (Stage 6, part 2 - GH #125)

# Supported ops for patch_scene_mjcf. Kept narrow on purpose - adding unchecked
# attribute setters would make the tool an arbitrary-code hole. Agents that
# need exotic MJCF should go through replace_scene_mjcf with a full XML.
_PATCH_OPS = {
    "add_body",
    "add_geom",
    "add_site",
    "set_body_pos",
    "set_body_quat",
    "delete_body",
}


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
    if kind not in _PATCH_OPS:
        raise ValueError(f"unknown op '{kind}'. Supported: {sorted(_PATCH_OPS)}")

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

    # Apply ops on a *clone* of the spec so we can atomically reject on failure.
    # MjSpec doesn't expose a cheap deep-copy, but round-tripping through XML
    # is safe: it's the same canonical form used by the compiler.
    try:
        with filter_mujoco_attach_noise():
            backup_xml = spec.to_xml()
    except Exception as e:  # pragma: no cover - spec.to_xml on brand-new specs is fine
        raise RuntimeError(f"failed to snapshot spec before patch: {e}") from e

    applied = 0
    new_bodies: dict[str, Any] = {}
    try:
        for op in ops:
            _apply_patch_op(spec, op, new_bodies)
            applied += 1
    except Exception as err:
        # Restore from backup.
        try:
            restored = SpecBuilder.from_mjcf_string(backup_xml)
            world._backend_state["spec"] = restored
        except Exception:
            # If even the backup round-trip fails, the world is in a bad state
            # and the caller should rebuild. Propagate the original error
            # either way so the user sees what went wrong.
            pass
        raise ValueError(f"patch op #{applied + 1} failed: {err}") from err

    # One recompile for the whole batch - preserves qpos/qvel for unchanged joints.
    with filter_mujoco_attach_noise():
        world._model, world._data = spec.recompile(world._model, world._data)

    # Forward pass so new bodies' xpos / xquat / cam_xmat are populated for
    # the very next render() or get_body_state() call. Same reasoning as
    # replace_scene_mjcf.
    mj = _ensure_mujoco()
    mj.mj_forward(world._model, world._data)

    _sync_cached_xml(world, spec)
    return applied
