"""MjSpec-based MJCF builder - programmatic scene construction via the MuJoCo AST.

This is the ONLY path for building / mutating MuJoCo scenes in strands-robots.
It replaces the string-concat ``MJCFBuilder`` (deleted) and the XML-round-trip
helpers in ``scene_ops.py``:

- ``SpecBuilder.build(world)``: build a fresh ``MjSpec`` from a ``SimWorld``.
- ``add_object`` / ``remove_body`` / ``add_camera``: mutate an existing spec.
- ``attach_robot``: compose a URDF/MJCF file into a scene with a name prefix.
- ``replace_scene``: load an agent-authored MJCF string as the new scene.

All builders return a ``MjSpec`` that the caller compiles via ``spec.compile()``
or re-compiles in-place via ``spec.recompile(model, data)`` (which preserves
existing joint state automatically).

This module does NOT import any XML / ElementTree / regex machinery - every
transformation goes through MuJoCo's own AST.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


# MuJoCo geom-type enum mapping. Populated lazily on first call so module
# import doesn't require mujoco to be installed (backend _ensure_mujoco gates).
_GEOM_TYPE_CACHE: dict[str, int] | None = None


def _geom_type(shape: str) -> int:
    """Map our shape-name vocabulary to MuJoCo's ``mjtGeom`` enum.

    Raises ValueError for shapes unsupported by the current pipeline. New
    shapes (``ellipsoid``, ``hfield``) can be added here without touching
    the rest of the builder.
    """
    global _GEOM_TYPE_CACHE
    if _GEOM_TYPE_CACHE is None:
        mujoco = _ensure_mujoco()
        _GEOM_TYPE_CACHE = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "mesh": mujoco.mjtGeom.mjGEOM_MESH,
            "plane": mujoco.mjtGeom.mjGEOM_PLANE,
        }
    try:
        return _GEOM_TYPE_CACHE[shape]
    except KeyError as e:
        supported = ", ".join(sorted(_GEOM_TYPE_CACHE.keys()))
        raise ValueError(f"Unsupported shape {shape!r}. Supported: {supported}.") from e


def _normalize_size(shape: str, size: list[float]) -> list[float]:
    """Convert SimObject ``size`` convention to MuJoCo's per-geom size vector.

    MuJoCo's geom-size conventions (all in the LOCAL frame):

    * ``box``:       half-extents ``[hx, hy, hz]``
    * ``sphere``:    ``[radius]``      (MuJoCo uses size[0] as radius)
    * ``cylinder``:  ``[radius, half-height]``
    * ``capsule``:   ``[radius, half-height]``  (cap hemisphere radius = radius)
    * ``ellipsoid``: ``[rx, ry, rz]``
    * ``plane``:     ``[hx, hy, grid_spacing]`` (hx/hy are half-sizes)
    * ``mesh``:      ``[]``            (mesh asset dictates extent; size ignored)

    ``SimObject.size`` is always 3 floats. Box/ellipsoid use all 3 as full
    extents, sphere uses ``size[0]`` as diameter (MuJoCo halves it to radius),
    cylinder/capsule use ``size[0]`` as diameter and ``size[2]`` as full height
    (both halved), plane uses ``size[0]``/``size[1]`` as full extents (halved).
    """
    if shape == "box":
        sx, sy, sz = size if len(size) >= 3 else (0.1, 0.1, 0.1)
        return [sx / 2, sy / 2, sz / 2]
    if shape == "sphere":
        # Legacy builder used size[0]/2 as radius - preserve that.
        radius = size[0] / 2 if size else 0.025
        return [radius, 0.0, 0.0]
    if shape in ("cylinder", "capsule"):
        radius = size[0] / 2 if size else 0.025
        half_h = size[2] / 2 if len(size) > 2 else 0.05
        return [radius, half_h, 0.0]
    if shape == "ellipsoid":
        sx, sy, sz = size if len(size) >= 3 else (0.05, 0.05, 0.05)
        return [sx / 2, sy / 2, sz / 2]
    if shape == "plane":
        sx = size[0] if size else 1.0
        sy = size[1] if len(size) > 1 else sx
        return [sx, sy, 0.01]
    if shape == "mesh":
        return [0.0, 0.0, 0.0]
    raise ValueError(f"Cannot normalize size for shape {shape!r}.")


def _target_quat(position: list[float], target: list[float]) -> list[float] | None:
    """Compute the camera orientation quaternion that makes ``position`` look
    at ``target`` with world +Z as the up vector.

    Camera convention:

    * Forward (cam local -Z) = normalize(target - position)
    * Right   (cam local +X) = normalize(forward x up)
    * Image-up (cam local +Y) = normalize(right x forward)

    Returns ``None`` for degenerate cases (target == position, or forward
    parallel to up). Callers handle the degenerate case upstream.

    Uses MuJoCo's ``mju_mat2Quat`` so no hand-rolled quaternion math.
    """
    mujoco = _ensure_mujoco()

    fwd = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    flen = float(np.linalg.norm(fwd))
    if flen < 1e-9:
        return None
    fwd /= flen

    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    rlen = float(np.linalg.norm(right))
    if rlen < 1e-9:
        return None
    right /= rlen
    image_up = np.cross(right, fwd)
    image_up /= float(np.linalg.norm(image_up))

    # Columns of R are [right, image_up, -forward] - the camera's +X, +Y, +Z
    # basis vectors expressed in world frame. Row-major layout for MuJoCo.
    rot = np.column_stack([right, image_up, -fwd])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return quat.tolist()


# SpecBuilder - the public API


class SpecBuilder:
    """Builds and mutates ``mujoco.MjSpec`` trees from ``SimWorld`` state.

    Three distinct operations:

    * :meth:`build(world)` - fresh spec from all world contents. Called by
      ``Simulation._compile_world`` when first creating a world.
    * :meth:`add_object` / :meth:`remove_body` / :meth:`add_camera` - mutate
      an existing spec in-place. Caller calls ``spec.recompile(model, data)``
      afterwards to propagate changes. State of unchanged joints is preserved
      automatically by MuJoCo.
    * :meth:`attach_robot` - compose a robot MJCF/URDF from disk into the
      scene spec via ``spec.attach(other, prefix=..., frame=...)``. MuJoCo
      handles name prefixing, asset deduplication, and default-class
      namespacing natively.
    """

    # full build
    @staticmethod
    def build(world: SimWorld) -> Any:
        """Build a fresh ``mujoco.MjSpec`` reflecting the current ``SimWorld``.

        Produces:
          * option (timestep, gravity)
          * visual + offscreen framebuffer size
          * grid texture/material (for the ground plane)
          * mesh assets for any objects with ``shape == "mesh"``
          * lights (``main_light``, ``fill_light``)
          * ground plane (if ``world.ground_plane``)
          * cameras
          * objects

        Robots are NOT included here - they're attached separately via
        :meth:`attach_robot` because each attach consumes a fresh MjSpec
        loaded from the URDF/MJCF file on disk.

        Caller is responsible for ``spec.compile()`` to produce an MjModel.
        """
        mujoco = _ensure_mujoco()

        spec = mujoco.MjSpec()
        spec.modelname = "strands_sim"

        # Compiler + simulation options.
        spec.compiler.degree = False  # radians
        spec.compiler.autolimits = True

        spec.option.timestep = float(world.timestep)
        spec.option.gravity = list(world.gravity)

        # Offscreen framebuffer - the default 640x480 is too small for common
        # camera res. 1280x960 matches what the legacy builder used.
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        spec.visual.quality.shadowsize = 4096

        # Headlight. MuJoCo's default headlight is a camera-tracking light
        # that is ALWAYS on (active=1, diffuse 0.4, specular 0.5). It stacks
        # additively on top of our ``main_light`` (1.0) + ``fill_light`` (0.5)
        # below, so the scene renders washed-out / over-bright and flat (the
        # head-on fill kills the shadow contrast that makes geometry legible).
        # Real robot camera footage is NOT lit by a head-mounted light, so a
        # bright headlight also makes sim renders look unlike the real data we
        # want to collect. Dim it to a low, shadow-free ambient term and let
        # the explicit scene lights do the directional work -- this mirrors
        # the upstream SO-ARM ``scene.xml`` (headlight diffuse 0.6, the only
        # other light a single directional). We go slightly lower (0.2)
        # because we ship TWO explicit lights, not one.
        spec.visual.headlight.active = 1
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]

        # Ground texture + material - MuJoCo's built-in checkerboard.
        grid_tex = spec.add_texture(
            name="grid_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=512,
            height=512,
            rgb1=[0.9, 0.9, 0.9],
            rgb2=[0.7, 0.7, 0.7],
        )
        grid_mat = spec.add_material(name="grid_mat", texrepeat=[8, 8], reflectance=0.1)
        grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = grid_tex.name

        # Mesh assets for objects that declare ``shape == "mesh"``.
        for obj in world.objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                spec.add_mesh(name=f"mesh_{obj.name}", file=obj.mesh_path)

        # Lights.
        spec.worldbody.add_light(
            name="main_light",
            pos=[0.0, 0.0, 3.0],
            dir=[0.0, 0.0, -1.0],
            diffuse=[1.0, 1.0, 1.0],
            specular=[0.3, 0.3, 0.3],
        )
        spec.worldbody.add_light(
            name="fill_light",
            pos=[1.0, 1.0, 2.0],
            dir=[-0.5, -0.5, -1.0],
            diffuse=[0.5, 0.5, 0.5],
        )

        # Ground plane.
        if world.ground_plane:
            spec.worldbody.add_geom(
                name="ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[5.0, 5.0, 0.01],
                material="grid_mat",
                conaffinity=1,
                condim=3,
            )

        # Cameras. Skip cameras that were discovered inside a robot's URDF -
        # they'll come back automatically via ``spec.attach(robot_spec)``.
        # Re-adding them at the top level would collide with the attached
        # namespaced copy at compile time.
        for cam in world.cameras.values():
            if getattr(cam, "origin_robot", ""):
                continue
            SpecBuilder.add_camera(spec, cam)

        # Objects.
        for obj in world.objects.values():
            SpecBuilder.add_object(spec, obj)

        return spec

    # from_mjcf
    @staticmethod
    def from_mjcf_string(xml: str) -> Any:
        """Load an MJCF XML string as a fresh spec. Used by ``replace_scene``.

        Raises ``ValueError`` on malformed XML via MuJoCo's compiler.
        """
        mujoco = _ensure_mujoco()
        return mujoco.MjSpec.from_string(xml)

    @staticmethod
    def from_file(path: str) -> Any:
        """Load an MJCF/URDF file as a fresh spec.

        MuJoCo 3.2+ reads URDF as well as MJCF via the same entry point - the
        file extension + XML root determines the path. Raises ``ValueError``
        on invalid files.
        """
        mujoco = _ensure_mujoco()
        return mujoco.MjSpec.from_file(str(path))

    # object add
    @staticmethod
    def add_object(spec: Any, obj: SimObject) -> None:
        """Add a ``SimObject`` to ``spec.worldbody`` in-place.

        * Dynamic objects (``is_static=False``) get a freejoint + explicit
          inertial block (diag 0.001, user-supplied mass) matching the
          legacy builder.
        * Static objects skip the freejoint and inertial.
        * Meshes require a matching ``spec.add_mesh(...)`` to have been
          registered (usually by :meth:`build`); this method does NOT
          register mesh assets.
        """
        body = spec.worldbody.add_body(
            name=obj.name,
            pos=list(obj.position),
            quat=list(obj.orientation),
        )

        if not obj.is_static:
            body.add_freejoint(name=f"{obj.name}_joint")
            body.mass = float(obj.mass)
            body.inertia = [0.001, 0.001, 0.001]
            body.ipos = [0.0, 0.0, 0.0]
            body.explicitinertial = True

        geom_kwargs: dict[str, Any] = {
            "name": f"{obj.name}_geom",
            "type": _geom_type(obj.shape),
            "rgba": list(obj.color),
            "condim": 3,
        }
        if obj.shape == "mesh":
            geom_kwargs["meshname"] = f"mesh_{obj.name}"
        else:
            geom_kwargs["size"] = _normalize_size(obj.shape, list(obj.size))

        # Legacy code only set explicit friction on boxes; preserve parity.
        if obj.shape == "box":
            geom_kwargs["friction"] = [1.0, 0.5, 0.001]

        body.add_geom(**geom_kwargs)

    # camera add
    @staticmethod
    def add_camera(spec: Any, cam: SimCamera) -> None:
        """Add a camera to the scene.

        Two modes:

        * **World-fixed** (default): the camera is added under ``worldbody``
          at ``cam.position`` looking at ``cam.target`` (both in world
          coordinates).
        * **Body-mounted** (``cam.parent_body`` set): the camera is added as
          a child of that body, so ``cam.position``/``cam.target`` are in the
          body's LOCAL frame and the camera tracks the body as it moves. This
          is how a realistic wrist/gripper camera is modelled -- it rides
          along with the gripper exactly like the physical camera on a real
          SO101/SO100.

        If ``cam.target`` is set, the look-at direction is converted to a
        quaternion via :func:`_target_quat`.
        """
        mujoco = _ensure_mujoco()
        pos = list(cam.position)
        kwargs: dict[str, Any] = {
            "name": cam.name,
            "pos": pos,
            "fovy": float(cam.fov),
            "mode": mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        }
        target = getattr(cam, "target", None)
        if target is not None:
            quat = _target_quat(pos, list(target))
            if quat is not None:
                kwargs["quat"] = quat

        parent_name = getattr(cam, "parent_body", "") or ""
        if parent_name:
            # Mount on the named body. spec.body() only resolves bodies that
            # existed at the last compile; robot bodies introduced via
            # spec.attach() may not be visible that way, so fall back to a
            # full scan (mirrors scene_ops._find_body's robustness).
            parent = None
            try:
                parent = spec.body(parent_name)
            except (KeyError, ValueError):
                parent = None
            if parent is None:
                for body in spec.bodies:
                    if body.name == parent_name:
                        parent = body
                        break
            if parent is None:
                raise ValueError(
                    f"add_camera: parent_body {parent_name!r} not found in scene. "
                    "Pass the fully-qualified body name (e.g. 'so101/gripper')."
                )
            parent.add_camera(**kwargs)
        else:
            spec.worldbody.add_camera(**kwargs)

    # body remove
    @staticmethod
    def remove_body(spec: Any, name: str) -> bool:
        """Remove a body by name from the spec.

        Uses ``spec.delete(body)`` which walks the spec's typed registry.
        Returns ``True`` if the body existed and was removed, ``False``
        otherwise (to match the legacy scene_ops API).

        Note: this removes ONLY the body; any actuators/sensors referencing
        its joints must be cleaned up separately via :meth:`remove_refs_by_prefix`.
        That's only needed for robots - for plain object bodies there are
        no actuators/sensors tied to them.
        """
        try:
            body = spec.body(name)
        except (KeyError, ValueError):
            return False
        if body is None:
            return False
        spec.delete(body)
        return True

    # camera remove
    @staticmethod
    def remove_camera(spec: Any, name: str) -> bool:
        """Remove a camera by name from the spec."""
        # spec.cameras returns the list; find by name
        cameras = getattr(spec, "cameras", None)
        if cameras is None:
            return False
        for cam in cameras:
            if cam.name == name:
                spec.delete(cam)
                return True
        return False

    # -attach
    @staticmethod
    def attach_robot(
        scene_spec: Any,
        robot: SimRobot,
        robot_file_path: str,
    ) -> list[str]:
        """Attach a URDF/MJCF file into the scene spec with a name prefix.

        Uses ``spec.attach(other, prefix=..., frame=...)`` which handles
        body/joint/geom/actuator/sensor name prefixing automatically, dedups
        shared assets (meshes, textures, materials), and namespaces default
        classes - replacing ~400 lines of hand-rolled tree-walking from the
        legacy ``scene_ops._prefix_robot_names`` +
        ``_namespace_robot_default_classes``.

        Args:
            scene_spec: the scene spec to mutate.
            robot: ``SimRobot`` carrying ``name`` (used as prefix) and
                ``position`` / ``orientation`` (used as attach frame).
            robot_file_path: absolute or relative path to an MJCF/URDF file.

        Returns:
            List of joint names belonging to the attached robot, in the order
            MuJoCo discovered them (no prefix - caller namespaces via
            ``robot.namespace`` when it resolves IDs post-compile).
        """
        mujoco = _ensure_mujoco()

        robot_spec = mujoco.MjSpec.from_file(str(robot_file_path))

        # Strip the robot scene's own ground/floor plane(s) before attaching.
        # Many menagerie scenes (e.g. franka_emika_panda/scene.xml) ship a
        # ``floor`` plane at z=0; merged in alongside the world's own ``ground``
        # plane (also z=0) it produces two coplanar infinite planes with
        # different checker materials -> depth-buffer Z-fighting and a broken
        # floor render. The world ``ground`` plane (configurable via
        # ``create_world(ground_plane=...)``) is the single source of truth;
        # robots contribute only their own bodies/joints/actuators. See #320.
        #
        # Three guards from the #360 review (#363):
        #   1. Conditional strip -- only remove the robot's floor when the world
        #      actually owns a ground plane to replace it. Under
        #      ``create_world(ground_plane=False)`` the world has no ground, so
        #      stripping the robot's plane would leave the scene floorless; in
        #      that case we keep the robot's plane (it is the only floor source).
        #   2. Narrow predicate -- only strip planes that are plausibly the z=0
        #      axis-aligned ground (a robot MJCF may intentionally ship an
        #      angled/elevated plane, e.g. a ramp or wall, which must survive).
        #   3. Debug log -- record which geoms were stripped so a disappearing
        #      (or surviving) robot floor is diagnosable.
        world_has_ground = any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in scene_spec.geoms)
        stripped: list[str] = []
        if world_has_ground:
            for plane in [
                g for g in robot_spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE and _is_z0_ground_plane(g)
            ]:
                stripped.append(plane.name or "<unnamed>")
                robot_spec.delete(plane)
            if stripped:
                logger.debug(
                    "attach_robot: stripped %d robot-scene z=0 ground plane geom(s) "
                    "for %r (world owns the ground plane): %r",
                    len(stripped),
                    robot.name,
                    stripped,
                )
        else:
            # ground_plane=False opt-out: keep the robot's own floor (if any) so
            # the scene is not left without any ground.
            kept = [g for g in robot_spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
            if kept:
                logger.debug(
                    "attach_robot: world has no ground plane (ground_plane=False); "
                    "keeping %d robot-scene plane geom(s) for %r as the floor source",
                    len(kept),
                    robot.name,
                )

        # Collect source joint names BEFORE attach - attach mutates the child
        # spec in-place (the child gets reparented).
        source_joint_names: list[str] = []

        def _walk(body: Any) -> None:
            for j in body.joints:
                jname = j.name or ""
                if jname and jname not in source_joint_names:
                    source_joint_names.append(jname)
            for sub in body.bodies:
                _walk(sub)

        for top_body in robot_spec.worldbody.bodies:
            _walk(top_body)

        frame = scene_spec.worldbody.add_frame(
            pos=list(robot.position),
            quat=list(robot.orientation),
        )
        scene_spec.attach(robot_spec, prefix=f"{robot.name}/", frame=frame)

        return source_joint_names


def _is_z0_ground_plane(geom: Any) -> bool:
    """True if a plane geom is plausibly the z=0 axis-aligned ground.

    MuJoCo planes default to a +Z normal at the body origin. We treat a plane
    as "ground" when its body-frame position z is ~0 and its orientation is
    axis-aligned (quat ~ identity, so the normal stays +Z). A robot MJCF that
    ships an intentional ramp/wall plane (rotated or elevated) is NOT matched
    and survives the attach. See #363.
    """
    pos = getattr(geom, "pos", None)
    if pos is not None and abs(float(pos[2])) > 1e-6:
        return False
    quat = getattr(geom, "quat", None)
    if quat is not None:
        # Identity quat is (1, 0, 0, 0); allow small FP noise.
        w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        if abs(w - 1.0) > 1e-6 or abs(x) > 1e-6 or abs(y) > 1e-6 or abs(z) > 1e-6:
            return False
    return True


__all__ = [
    "SpecBuilder",
    "_is_z0_ground_plane",
    "_geom_type",
    "_normalize_size",
    "_target_quat",
]
