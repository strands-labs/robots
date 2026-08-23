"""MjSpec-based MJCF builder - programmatic scene construction via the MuJoCo AST.

This is the ONLY path for building / mutating MuJoCo scenes in strands-robots.
It replaces the string-concat ``MJCFBuilder`` (deleted) and the XML-round-trip
helpers in :mod:`scene_ops`:

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

import difflib
import logging
import os
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco
from strands_robots.simulation.terrain import (
    TERRAIN_BASE,
    TERRAIN_RADIUS,
    TERRAIN_RESOLUTION,
    TERRAIN_SEED,
    generate_heightfield,
    terrain_elevation,
)

logger = logging.getLogger(__name__)


def _absolutize_asset_paths(spec: Any) -> int:
    """Rewrite a spec's file-backed asset references to absolute paths.

    MuJoCo resolves a ``<mesh>`` / ``<hfield>`` / ``<texture>`` ``file=`` against
    a base directory (the model file's own directory, plus ``meshdir`` /
    ``texturedir``). That base is tracked on the spec as ``modelfiledir`` and is
    NOT emitted by ``spec.to_xml()``, and ``spec.attach()`` does not carry it
    onto the parent. So a spec whose assets are named relatively compiles fine
    in the session that loaded it while serialising to an XML that resolves
    those names against wherever the XML happens to be written - i.e. nowhere.
    Rewriting each reference to an absolute path makes the spec self-describing,
    so every consumer of ``to_xml()`` gets asset locations instead of an
    implicit dependency on a directory the text never mentions.

    MuJoCo honours an absolute ``file=`` regardless of any ``meshdir`` /
    ``texturedir`` still declared, so those attributes are left untouched.

    A reference is rewritten only when it is relative AND the resolved path
    exists. An unresolvable reference is left exactly as authored: MuJoCo's own
    "Error opening file" names the reference the model actually declares, which
    is more useful than an invented absolute path that was never on disk.

    Args:
        spec: the ``MjSpec`` to repair in place.

    Returns:
        Number of asset references rewritten.
    """
    base = getattr(spec, "modelfiledir", "") or ""
    rewritten = 0
    # meshdir covers meshes AND height fields; textures have their own dir.
    for assets, subdir in (
        (spec.meshes, getattr(spec, "meshdir", "") or ""),
        (spec.hfields, getattr(spec, "meshdir", "") or ""),
        (spec.textures, getattr(spec, "texturedir", "") or ""),
    ):
        for asset in assets:
            ref = asset.file or ""
            if not ref or os.path.isabs(ref):
                continue
            resolved = os.path.abspath(os.path.join(base, subdir, ref))
            if not os.path.isfile(resolved):
                continue
            asset.file = resolved
            rewritten += 1
    if rewritten:
        logger.debug(
            "absolutized %d asset reference(s) against %r so spec.to_xml() stays reloadable",
            rewritten,
            base,
        )
    return rewritten


# Accepted keys of the ``add_object(material=...)`` spec. Single source of truth
# shared by :func:`material_spec_error` and :meth:`SpecBuilder._build_material`.
MATERIAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "reflectance",
        "specular",
        "shininess",
        "texrepeat",
        "texture",
        "builtin",
        "rgb1",
        "rgb2",
        "texdim",
    }
)

# Keys that only mean something for a procedural (``builtin``) texture.
_BUILTIN_ONLY_KEYS: Final[tuple[str, ...]] = ("rgb1", "rgb2", "texdim")


def material_spec_error(obj_name: str, material: Any) -> str | None:
    """Validate an ``add_object(material=...)`` spec's shape.

    Every key of the material spec is optional, so a key the builder does not
    recognise (a typo such as ``"rgb_1"``, or a field borrowed from another
    renderer such as ``"roughness"``) would otherwise be dropped and the
    surface would compile with MuJoCo's glossy defaults while the call still
    reported success. Reject those keys instead.

    Args:
        obj_name: Object name, used to make the message actionable.
        material: The caller-supplied material spec.

    Returns:
        An error message, or ``None`` when the spec's shape is honorable.
        Value-level checks (texture file exists, ``builtin`` name known,
        ``texture``/``builtin`` mutually exclusive) live in
        :meth:`SpecBuilder._build_material`.
    """
    accepted = ", ".join(sorted(MATERIAL_KEYS))
    prefix = f"add_object material for {obj_name!r}: "
    if not isinstance(material, Mapping):
        return f"{prefix}expected a dict of material options, got {type(material).__name__}. Accepted keys: {accepted}."
    if not material:
        return (
            f"{prefix}material={{}} has no effect - omit material= to keep the "
            f"flat 'color' rgba, or set at least one of: {accepted}."
        )
    unknown = [key for key in material if key not in MATERIAL_KEYS]
    if unknown:
        described = []
        for key in sorted(unknown, key=str):
            close = difflib.get_close_matches(str(key).lower(), sorted(MATERIAL_KEYS), n=1, cutoff=0.7)
            described.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        return f"{prefix}unknown material key(s): {', '.join(described)}. Accepted keys: {accepted}."
    if material.get("builtin") is None:
        orphans = [key for key in _BUILTIN_ONLY_KEYS if material.get(key) is not None]
        if orphans:
            return (
                f"{prefix}{', '.join(orphans)} only colour/size a procedural "
                "texture and are ignored without it - set builtin='checker'|'gradient'|'flat', "
                "or drop them (an image 'texture' is coloured by the file itself, tinted by "
                "the geom 'color' rgba)."
            )
    return None


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


# Per-shape ``size`` contract: how many leading components the shape actually
# consumes, plus the layout to quote back at a caller who supplied the wrong
# count. A shape that consumes component i needs a vector of at least i + 1
# components -- a shorter one cannot express the request at all, so it is
# rejected instead of being padded from a backend default (which would compile
# a differently-sized object while reporting success). ``plane`` needs only its
# x half-width (y mirrors x when omitted); ``mesh`` takes its extent from the
# asset and consumes nothing.
_SIZE_LAYOUT: dict[str, tuple[int, str]] = {
    "box": (3, "[x, y, z] full edge lengths"),
    "ellipsoid": (3, "[x, y, z] full diameters"),
    "sphere": (1, "[diameter]"),
    "cylinder": (3, "[diameter, unused, full height]"),
    "capsule": (3, "[diameter, unused, full height]"),
    "plane": (1, "[x] or [x, y] visual half-widths"),
    "mesh": (0, "unused - the asset's own units define the extent"),
}

# ``SimObject.size`` is a 3-vector, and every shape above consumes at most its
# first three components, so a longer vector carries values no shape can honor.
_MAX_SIZE_COMPONENTS = 3


def _validate_size(shape: str, size: list[float]) -> str | None:
    """Return an error message if ``size`` cannot be honored for ``shape``.

    ``size`` follows the full-extent convention used by
    :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.add_object`
    (meters along each axis, halved internally to MuJoCo's half-extents). Two
    classes are rejected:

    * A **component count** the shape cannot consume -- fewer components than
      :data:`_SIZE_LAYOUT` requires (a box needs all three), or more than three
      (nothing consumes a fourth). A short vector used to be replaced wholesale
      by a hardcoded default, so ``size=[0.5]`` on a box compiled a 10 cm cube
      while the call reported success and echoed the requested ``[0.5]``.
    * A **non-positive** value in a component the shape consumes. Only consumed
      components are checked, so a cylinder may legitimately pass
      ``size[1] == 0`` (it is ignored).

    Returns ``None`` when the size is acceptable.
    """
    required, layout = _SIZE_LAYOUT.get(shape, (0, ""))
    if len(size) > _MAX_SIZE_COMPONENTS:
        return (
            f"add_object: 'size' takes at most {_MAX_SIZE_COMPONENTS} components, got "
            f"{len(size)} (size={list(size)}). 'size' is the full extent in meters "
            "along each axis (not MuJoCo's half-extent)."
        )
    if len(size) < required:
        return (
            f"add_object: {shape} needs {required} 'size' component(s) {layout}, got "
            f"{len(size)} (size={list(size)}). 'size' is the full extent in meters "
            "along each axis (not MuJoCo's half-extent); pass every component the "
            "shape consumes rather than a partial vector."
        )
    if shape == "mesh":
        return None
    if shape in ("box", "ellipsoid"):
        used: tuple[tuple[int, str], ...] = ((0, "x"), (1, "y"), (2, "z"))
    elif shape == "sphere":
        used = ((0, "diameter"),)
    elif shape in ("cylinder", "capsule"):
        used = ((0, "diameter"), (2, "height"))
    elif shape == "plane":
        used = ((0, "x"),) if len(size) < 2 else ((0, "x"), (1, "y"))
    else:
        # Unknown shapes are rejected by _geom_type; nothing to validate here.
        return None
    for idx, axis in used:
        if size[idx] <= 0:
            return (
                f"add_object: {shape} {axis} extent must be > 0, got "
                f"{size[idx]} (size={list(size)}). 'size' is the full extent "
                "in meters along each axis (not MuJoCo's half-extent)."
            )
    return None


def _normalize_size(shape: str, size: list[float]) -> list[float]:
    """Convert SimObject ``size`` convention to MuJoCo's per-geom size vector.

    MuJoCo's geom-size conventions (all in the LOCAL frame):

    * ``box``:       half-extents ``[hx, hy, hz]``
    * ``sphere``:    ``[radius]``      (MuJoCo uses size[0] as radius)
    * ``cylinder``:  ``[radius, half-height]``
    * ``capsule``:   ``[radius, half-height]``  (cap hemisphere radius = radius)
    * ``ellipsoid``: ``[rx, ry, rz]``
    * ``plane``:     ``[hx, hy, grid_spacing]`` (hx/hy are half-sizes)
    * ``mesh``:      refused           (see below - no caller normalizes one)

    Box/ellipsoid use all 3 components as full extents, sphere uses ``size[0]``
    as diameter (MuJoCo halves it to radius), cylinder/capsule use ``size[0]``
    as diameter and ``size[2]`` as full height (both halved). Plane is the one
    exception: ``size[0]``/``size[1]`` are passed through unchanged as MuJoCo's
    visual half-widths (a plane is infinite for collision, so only its rendered
    grid extent matters) and ``size[1]`` mirrors ``size[0]`` when omitted.

    ``mesh`` is deliberately absent from the branches below and falls through to
    the trailing ``raise``. A mesh geom takes its extent from the asset, so it
    carries no ``size`` at all: ``SpecBuilder.build`` passes ``meshname``
    instead of calling this function, and the ``add_geom`` patch op refuses
    ``type="mesh"`` outright (:func:`~strands_robots.simulation.mujoco.scene_ops._geom_shape_error`)
    because it has no key that could name the asset. So nothing should be
    asking, and the branch that used to answer ``[0.0, 0.0, 0.0]`` -- a third
    statement of the contract, and one the docstring above contradicted -- was
    unreachable on both paths.

    Raises:
        ValueError: When :func:`_validate_size` rejects ``size`` -- too few
            components for the shape to consume, more than three, or a
            non-positive consumed extent. Every component this function reads
            is guaranteed present by that check, so a partial vector is never
            silently completed from a default. Also when ``shape`` is one no
            caller normalizes, ``mesh`` included.
    """
    if (msg := _validate_size(shape, size)) is not None:
        raise ValueError(msg)
    if shape in ("box", "ellipsoid"):
        return [size[0] / 2, size[1] / 2, size[2] / 2]
    if shape == "sphere":
        # Legacy builder used size[0]/2 as radius - preserve that.
        return [size[0] / 2, 0.0, 0.0]
    if shape in ("cylinder", "capsule"):
        return [size[0] / 2, size[2] / 2, 0.0]
    if shape == "plane":
        sx = size[0]
        sy = size[1] if len(size) > 1 else sx
        return [sx, sy, 0.01]
    raise ValueError(f"Cannot normalize size for shape {shape!r}.")


def _target_quat(position: list[float], target: list[float]) -> list[float] | None:
    """Compute the camera orientation quaternion that makes ``position`` look
    at ``target`` with world +Z as the up vector.

    Camera convention:

    * Forward (cam local -Z) = normalize(target - position)
    * Right   (cam local +X) = normalize(forward x up)
    * Image-up (cam local +Y) = normalize(right x forward)

    The up vector is world +Z, except when the view direction is parallel to
    +Z (a vertical / top-down camera), in which case world +Y is used as the
    reference up so the look-at still resolves to a valid orientation.

    Returns ``None`` only for the truly degenerate case where
    ``target == position`` (zero-length view direction). Callers handle the
    degenerate case upstream.

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
        # Forward is parallel to world +Z (a vertical camera, e.g. the
        # top-down overhead view). The world-up reference is degenerate, so
        # fall back to world +Y as the reference up. This still yields a
        # well-defined orientation instead of silently dropping the quat and
        # leaving the camera at its default (mis-oriented) pose.
        up = np.array([0.0, 1.0, 0.0])
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


def _find_spec_body(spec: Any, name: str) -> Any | None:
    """Resolve a body by name in ``spec``, or ``None`` when it is not there.

    ``spec.body(name)`` only resolves bodies that existed at the last compile,
    and it RETURNS ``None`` for a body added since (rather than raising), so a
    lookup that trusts it alone misses every body introduced by
    ``spec.attach()`` or ``add_body`` in the current editing session. Falling
    back to a scan of ``spec.bodies`` covers those, mirroring the robustness
    of ``scene_ops._find_body``.
    """
    try:
        body = spec.body(name)
    except (KeyError, ValueError):
        body = None
    if body is not None:
        return body
    for candidate in spec.bodies:
        if candidate.name == name:
            return candidate
    return None


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
          * world-fixed cameras
          * objects

        Robots are NOT included here - they're attached separately via
        :meth:`attach_robot` because each attach consumes a fresh MjSpec
        loaded from the URDF/MJCF file on disk.

        Because no robot is attached yet, a BODY-MOUNTED camera (one with
        ``parent_body`` set, e.g. a wrist camera) has no parent to mount on
        and is deferred: the caller must invoke :meth:`add_deferred_cameras`
        after attaching robots and before compiling, or those cameras are
        absent from the model.

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
        # Dark blue-grey checker matching the Menagerie SO101 scene.xml
        # groundplane (rgb1=0.2 0.3 0.4 / rgb2=0.1 0.2 0.3). The previous
        # near-white grid (0.9/0.7) saturated to pure white under the scene
        # lights below (measured 250/255 in rendered frames), washing out the
        # floor and killing shadow contrast.
        grid_tex = spec.add_texture(
            name="grid_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=512,
            height=512,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
        )
        grid_mat = spec.add_material(name="grid_mat", texrepeat=[8, 8], reflectance=0.2)
        grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = grid_tex.name

        # Mesh assets for objects that declare ``shape == "mesh"``.
        for obj in world.objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                spec.add_mesh(name=f"mesh_{obj.name}", file=obj.mesh_path)

        # Lights. A single directional light matching the Menagerie SO101
        # scene.xml (``<light pos="0 0 3.5" dir="0 0 -1" directional="true"/>``).
        # The previous setup stacked TWO bright point lights (main diffuse 1.0
        # + fill 0.5) firing straight down, which over-lit the scene, blew out
        # the floor to white, and flattened shadow contrast. A directional
        # light is also more physically representative of overhead room
        # lighting than a point light 3m above the origin.
        main_light = spec.worldbody.add_light(
            name="main_light",
            pos=[0.0, 0.0, 3.5],
            dir=[0.0, 0.0, -1.0],
            diffuse=[0.6, 0.6, 0.6],
            specular=[0.2, 0.2, 0.2],
        )
        main_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

        # Ground. A rough-terrain heightfield (``create_world(terrain=...)``)
        # replaces the flat plane so a locomotion policy walks over bumps; both
        # share the checker material and the "ground" geom name so downstream
        # ground-plane detection (attach_robot floor-strip) and any name lookup
        # are terrain-agnostic. The hfield surface spans z in
        # [0, terrain_elevation(difficulty)] on a solid base slab -> flush with z=0 at its
        # lowest point (no hole under the robot), same +/-5 m footprint as the
        # flat plane so the reachable workspace is unchanged.
        if world.ground_plane:
            if world.terrain:
                heights = generate_heightfield(world.terrain, resolution=TERRAIN_RESOLUTION, seed=TERRAIN_SEED)
                # difficulty scales the hfield PEAK elevation (kind-agnostic): the
                # normalized [0, 1] heights are the same, only the metre scale changes.
                elevation = terrain_elevation(world.terrain_difficulty)
                spec.add_hfield(
                    name="terrain_hfield",
                    size=[TERRAIN_RADIUS, TERRAIN_RADIUS, elevation, TERRAIN_BASE],
                    nrow=TERRAIN_RESOLUTION,
                    ncol=TERRAIN_RESOLUTION,
                    userdata=heights,
                )
                spec.worldbody.add_geom(
                    name="ground",
                    type=mujoco.mjtGeom.mjGEOM_HFIELD,
                    hfieldname="terrain_hfield",
                    material="grid_mat",
                    conaffinity=1,
                    condim=3,
                )
            else:
                spec.worldbody.add_geom(
                    name="ground",
                    type=mujoco.mjtGeom.mjGEOM_PLANE,
                    size=[5.0, 5.0, 0.01],
                    material="grid_mat",
                    conaffinity=1,
                    condim=3,
                )

        # Cameras. Two kinds are skipped here.
        #
        # Cameras discovered inside a robot's URDF come back automatically via
        # ``spec.attach(robot_spec)``; re-adding them at the top level would
        # collide with the attached namespaced copy at compile time.
        #
        # Body-mounted cameras are DEFERRED: ``build`` does not attach robots
        # (see this method's own contract), so a camera whose ``parent_body``
        # is a robot body has no parent in the spec yet. Adding it here raised
        # ``ValueError`` and aborted the whole rebuild. The caller mounts them
        # with :meth:`add_deferred_cameras` once every robot is attached.
        for cam in world.cameras.values():
            if getattr(cam, "origin_robot", "") or getattr(cam, "parent_body", ""):
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
        """Load an MJCF/URDF file as a fresh spec, with asset refs absolutized.

        MuJoCo 3.2+ reads URDF as well as MJCF via the same entry point - the
        file extension + XML root determines the path. Raises ``ValueError``
        on invalid files.

        Every file-backed asset reference is rewritten to an absolute path via
        :func:`_absolutize_asset_paths` before the spec is handed back, so the
        spec carries its own asset locations instead of depending on a base
        directory MuJoCo tracks separately and does not serialise. This is what
        makes ``spec.to_xml()`` (and therefore ``export_xml``) reloadable from
        anywhere; see that helper for why the repair belongs at load time.
        """
        mujoco = _ensure_mujoco()
        spec = mujoco.MjSpec.from_file(str(path))
        _absolutize_asset_paths(spec)
        return spec

    # object add
    @staticmethod
    def add_object(spec: Any, obj: SimObject) -> None:
        """Add a ``SimObject`` to ``spec.worldbody`` in-place.

        * Dynamic objects (``is_static=False``) get a freejoint and declare
          ``obj.mass`` on the GEOM, so MuJoCo's compiler integrates the
          inertia tensor from the shape the caller actually asked for.
        * Static objects skip the freejoint and the mass declaration; their
          compiled mass comes from the geom's default density.
        * Meshes require a matching ``spec.add_mesh(...)`` to have been
          registered (usually by :meth:`build`); this method does NOT
          register mesh assets.
        """
        # Build the visual material/texture FIRST (gated on ``obj.material is
        # not None`` so the rgba-only path is byte-for-byte unchanged). Doing
        # it before ``add_body`` means an invalid material raises ValueError
        # before any body/geom is added to the spec, so a rejected add never
        # leaves an orphan body behind.
        material_name = SpecBuilder._build_material(spec, obj) if obj.material is not None else None

        # ``add_body(name=...)`` inserts the duplicate even when the name
        # collides with an existing scene body, and the steps after it (the geom
        # type lookup, ``add_geom``) can raise as well. Any raise in this block
        # must undo only what THIS call inserted, then re-raise so the caller
        # reports the real reason - hence the body count taken before the insert
        # and :meth:`remove_surplus_bodies` after it, never a delete by name.
        pre_count = SpecBuilder.count_bodies_named(spec, obj.name)
        try:
            body = spec.worldbody.add_body(
                name=obj.name,
                pos=list(obj.position),
                quat=list(obj.orientation),
            )

            if not obj.is_static:
                body.add_freejoint(name=f"{obj.name}_joint")

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

            if material_name is not None:
                geom_kwargs["material"] = material_name

            if not obj.is_static:
                # Declare the mass on the GEOM rather than on the body. A geom
                # mass makes MuJoCo's compiler integrate the inertia tensor
                # over the shape the caller asked for; a body-level explicit
                # inertial block requires an inertia tensor to be supplied with
                # it, and there is no tensor to invent for an arbitrary shape.
                # The constant diagonal this replaced was wrong by orders of
                # magnitude in BOTH directions and the direction flipped with
                # size: for a 1 cm 100 g cube 0.001 is 600x the true value, so
                # the cube resisted rotation like a flywheel and refused to
                # tumble or spin out of a gripper; for a 30 cm 1 kg crate it is
                # 15x too SMALL, so the crate spun as if hollow. Translation
                # was unaffected (``body_mass`` was right), which is exactly
                # what kept it silent - the object fell correctly and only its
                # rotation was wrong.
                #
                # Consequence worth knowing: the compiled inertia now scales
                # with the shape, so MuJoCo's "mass and inertia of moving
                # bodies must be larger than mjMINVAL" floor becomes
                # shape-dependent - a mass that clears mjMINVAL itself can
                # still integrate to an inertia below it on a small geom.
                # ``_validate_mass`` deliberately stays a mass-only pre-check
                # (a fixed numeric bound cannot express a shape-dependent
                # floor); the residual case surfaces MuJoCo's own reason, which
                # names both mass and inertia.
                geom_kwargs["mass"] = float(obj.mass)

            body.add_geom(**geom_kwargs)
        except (ValueError, RuntimeError):
            SpecBuilder.remove_surplus_bodies(spec, obj.name, pre_count)
            raise

    # material build
    @staticmethod
    def _build_material(spec: Any, obj: SimObject) -> str:
        """Create a ``mjMaterial`` (+ optional texture) for ``obj.material``.

        Returns the material name to assign to the geom's ``material`` field.
        Called only when ``obj.material is not None``.

        Schema of the ``obj.material`` dict (all keys optional):

        * ``reflectance`` / ``specular`` / ``shininess`` (float 0..1): surface
          response. ``specular=0, shininess=0`` gives a matte (non-plastic)
          look; the MuJoCo defaults read as glossy plastic.
        * ``texrepeat`` (``[u, v]``): texture tiling across the surface.
        * exactly one texture source, OR neither (matte solid colour):
            - ``texture`` (str): absolute path to an image file (PNG/etc.),
              applied as an ``mjTEXTURE_2D`` RGB texture.
            - ``builtin`` (``"checker" | "gradient" | "flat"``): procedural
              texture, coloured by ``rgb1`` / ``rgb2`` (each ``[r, g, b]``)
              and sized ``texdim`` (default 512) per side.

        Fails loudly (``ValueError``) on a key outside :data:`MATERIAL_KEYS`,
        an empty spec, ``rgb1``/``rgb2``/``texdim`` without ``builtin``, a
        missing texture file, an unknown ``builtin`` name, or both ``texture``
        and ``builtin`` set -- there is no silent fallback to the flat-plastic
        default. The geom keeps its ``rgba`` (which tints an image/solid
        material), so callers can still colour a procedural/solid surface via
        ``color=``.
        """
        if (shape_err := material_spec_error(obj.name, obj.material)) is not None:
            raise ValueError(shape_err)
        mujoco = _ensure_mujoco()
        mat_spec: Mapping[str, Any] = obj.material or {}
        mat_name = f"{obj.name}_mat"
        tex_name = f"{obj.name}_tex"

        # 1) Validate the whole spec BEFORE mutating ``spec`` so an invalid
        #    material never leaves a half-built (orphan) asset behind.
        texture = mat_spec.get("texture")
        builtin = mat_spec.get("builtin")
        if texture is not None and builtin is not None:
            raise ValueError(
                f"add_object material for {obj.name!r}: specify either 'texture' "
                "(image file) or 'builtin' (procedural), not both."
            )
        builtin_enum = None
        if texture is not None:
            path = str(texture)
            if not os.path.isfile(path):
                raise ValueError(f"add_object material for {obj.name!r}: texture file not found: {path!r}")
        elif builtin is not None:
            builtin_map = {
                "checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            }
            key = str(builtin).lower()
            if key not in builtin_map:
                raise ValueError(
                    f"add_object material for {obj.name!r}: unknown builtin "
                    f"{builtin!r}; supported: {', '.join(sorted(builtin_map))}."
                )
            builtin_enum = builtin_map[key]

        # 2) Defensive cleanup: a prior add under this object name (then a
        #    remove_object that deletes only the body) can leave a stale
        #    material/texture asset behind. Re-adding would collide at compile
        #    ("repeated name"), so drop any existing asset under our names.
        for existing in list(getattr(spec, "materials", []) or []):
            if existing.name == mat_name:
                spec.delete(existing)
        for existing in list(getattr(spec, "textures", []) or []):
            if existing.name == tex_name:
                spec.delete(existing)

        # 3) Build the texture (if any), then the material, then bind them.
        tex = None
        if texture is not None:
            tex = spec.add_texture(name=tex_name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture))
        elif builtin_enum is not None:
            texdim = int(mat_spec.get("texdim", 512))
            tex_kwargs: dict[str, Any] = {
                "name": tex_name,
                "type": mujoco.mjtTexture.mjTEXTURE_2D,
                "builtin": builtin_enum,
                "width": texdim,
                "height": texdim,
            }
            if mat_spec.get("rgb1") is not None:
                tex_kwargs["rgb1"] = [float(c) for c in mat_spec["rgb1"]]
            if mat_spec.get("rgb2") is not None:
                tex_kwargs["rgb2"] = [float(c) for c in mat_spec["rgb2"]]
            tex = spec.add_texture(**tex_kwargs)

        mat_kwargs: dict[str, Any] = {}
        for mat_key in ("reflectance", "specular", "shininess"):
            if mat_spec.get(mat_key) is not None:
                mat_kwargs[mat_key] = float(mat_spec[mat_key])
        if mat_spec.get("texrepeat") is not None:
            tr = mat_spec["texrepeat"]
            mat_kwargs["texrepeat"] = [float(tr[0]), float(tr[1])]

        mat = spec.add_material(name=mat_name, **mat_kwargs)
        if tex is not None:
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        return mat_name

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

        ``add_camera(name=...)`` inserts the duplicate even when the name
        collides with a camera the scene already declares, so - exactly as in
        :meth:`add_object` - a raise from the insert rolls only the cameras THIS
        call appended back out (:meth:`remove_surplus_cameras`) before
        re-raising. Without that, a refused camera left an orphan in the spec and
        every later scene mutation kept failing to recompile on the duplicate
        name, bricking the world after one bad add.
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
            # Mount on the named body, so cam.position/target are read in that
            # body's LOCAL frame and the camera rides along with it.
            parent = _find_spec_body(spec, parent_name)
            if parent is None:
                raise ValueError(
                    f"add_camera: parent_body {parent_name!r} not found in scene. "
                    "Pass the fully-qualified body name (e.g. 'so101/gripper')."
                )
            attach_to = parent
        else:
            attach_to = spec.worldbody

        pre_count = SpecBuilder.count_cameras_named(spec, cam.name)
        try:
            attach_to.add_camera(**kwargs)
        except (ValueError, RuntimeError):
            SpecBuilder.remove_surplus_cameras(spec, cam.name, pre_count)
            raise

    # deferred (body-mounted) cameras
    @staticmethod
    def add_deferred_cameras(spec: Any, world: SimWorld) -> list[str]:
        """Mount the body-mounted cameras :meth:`build` deferred.

        :meth:`build` deliberately does not attach robots, so a camera whose
        ``parent_body`` names a robot body cannot be added while the base spec
        is being assembled. Call this once every robot has been attached and
        before ``spec.compile()``.

        Returns:
            The names of cameras that could NOT be mounted because their
            ``parent_body`` is absent from the rebuilt spec -- which happens
            when the body belonged to a robot that is being removed. They are
            reported rather than raised so removing a robot stays possible;
            the caller decides what to do with the now-parentless entries.
        """
        unmounted: list[str] = []
        for cam in world.cameras.values():
            # Both fields are declared on SimCamera, so read them directly.
            parent_name = cam.parent_body
            if not parent_name or cam.origin_robot:
                continue
            if _find_spec_body(spec, parent_name) is None:
                unmounted.append(cam.name)
                continue
            SpecBuilder.add_camera(spec, cam)
        return unmounted

    # body remove
    @staticmethod
    def remove_body(spec: Any, name: str) -> bool:
        """Remove a body by name from the spec.

        Uses ``spec.delete(body)`` which walks the spec's typed registry.
        Returns ``True`` if the body existed and was removed, ``False``
        otherwise (to match the legacy scene_ops API).

        ``spec.body(name)`` only resolves bodies that existed at the last
        ``compile()``/``recompile()``, so a body added since then - including
        one this class inserted moments ago, before the validating recompile -
        is invisible to it and enumerated only by ``spec.bodies``. Both lookups
        are therefore tried: without the enumeration fallback a rollback of a
        just-added body silently removed nothing, leaving an orphan that made
        every later recompile fail on the duplicate name. This mirrors
        :func:`~strands_robots.simulation.mujoco.scene_ops._find_body` and the
        parent-body lookup in :meth:`add_camera`.

        Note: this removes ONLY the body; any actuators/sensors referencing
        its joints must be cleaned up separately via :meth:`remove_refs_by_prefix`.
        That's only needed for robots - for plain object bodies there are
        no actuators/sensors tied to them.
        """
        body = None
        try:
            body = spec.body(name)
        except (KeyError, ValueError):
            body = None
        if body is None:
            for candidate in getattr(spec, "bodies", ()):
                if candidate.name == name:
                    body = candidate
                    break
        if body is None:
            return False
        spec.delete(body)
        return True

    # surplus rollback (identify what THIS call inserted, never by name)
    @staticmethod
    def count_bodies_named(spec: Any, name: str) -> int:
        """Count the bodies in ``spec`` that carry ``name``.

        Take this BEFORE an insert that may have to be rolled back, and pass it
        as the ``keep`` argument of :meth:`remove_surplus_bodies`. A plain count
        rather than a membership test because a spec can legitimately hold two
        bodies under one name between an insert and the compile that refuses it.

        Args:
            spec: The ``mjSpec`` to enumerate.
            name: The body name to count.

        Returns:
            How many bodies currently carry ``name`` (0 when none do).
        """
        return sum(1 for body in getattr(spec, "bodies", ()) if body.name == name)

    @staticmethod
    def count_cameras_named(spec: Any, name: str) -> int:
        """Count the cameras in ``spec`` that carry ``name``.

        The camera-side counterpart of :meth:`count_bodies_named`; pair it with
        :meth:`remove_surplus_cameras`.

        Args:
            spec: The ``mjSpec`` to enumerate.
            name: The camera name to count.

        Returns:
            How many cameras currently carry ``name`` (0 when none do).
        """
        return sum(1 for camera in getattr(spec, "cameras", ()) if camera.name == name)

    @staticmethod
    def remove_surplus_bodies(spec: Any, name: str, keep: int) -> int:
        """Delete the bodies named ``name`` beyond the first ``keep`` of them.

        This is the rollback a refused insert needs, and it is deliberately NOT
        :meth:`remove_body`. A scene injection mutates the live spec before the
        compile that validates it, so at rollback time a colliding name is
        carried by TWO bodies: the healthy pre-existing one and the orphan the
        refused call appended. ``remove_body`` resolves the name through
        ``spec.body(name)``, which answers with the body present at the last
        compile - the ORIGINAL - so using it to roll back deleted the healthy
        body and left the orphan holding its name. The scene then recompiled
        successfully with the original geometry gone: a rejected add silently
        rewrote the scene.

        Identifying the surplus by position instead can never touch a body this
        call did not create. MuJoCo appends new elements, so the bodies to delete
        are the tail of the run carrying ``name``; ``keep`` is the count taken
        before the insert (:meth:`count_bodies_named`). ``keep`` at or above the
        current count is a no-op, so a rollback is safe to attempt on a path that
        may not have inserted anything.

        Args:
            spec: The ``mjSpec`` to mutate.
            name: The body name whose surplus copies to delete.
            keep: How many bodies with that name to leave in place.

        Returns:
            The number of bodies deleted.
        """
        surplus = [body for body in getattr(spec, "bodies", ()) if body.name == name][keep:]
        for body in surplus:
            spec.delete(body)
        return len(surplus)

    @staticmethod
    def remove_surplus_cameras(spec: Any, name: str, keep: int) -> int:
        """Delete the cameras named ``name`` beyond the first ``keep`` of them.

        The camera-side counterpart of :meth:`remove_surplus_bodies`, and for the
        same reason: :meth:`remove_camera` deletes the FIRST camera carrying the
        name, which on a collision is the one the scene already declared, so
        rolling a refused camera back with it moved the scene's camera to the
        rejected pose. Every later render from that name then answered with a
        view the caller was told had been refused.

        Args:
            spec: The ``mjSpec`` to mutate.
            name: The camera name whose surplus copies to delete.
            keep: How many cameras with that name to leave in place.

        Returns:
            The number of cameras deleted.
        """
        surplus = [camera for camera in getattr(spec, "cameras", ()) if camera.name == name][keep:]
        for camera in surplus:
            spec.delete(camera)
        return len(surplus)

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

    # mesh remove
    @staticmethod
    def remove_mesh(spec: Any, name: str) -> bool:
        """Remove a mesh asset by name from the spec.

        Objects added at runtime through ``inject_object_into_scene`` register
        a mesh asset named ``f"mesh_{obj.name}"``. Removing the body alone
        leaves that asset orphaned in the spec, so a later add under the same
        name would collide on a duplicate mesh at recompile. This deletes the
        asset so the name stays fully reusable. Returns ``True`` if the mesh
        existed and was removed, ``False`` otherwise (safe no-op for
        primitive-shape objects that never registered a mesh).
        """
        try:
            mesh = spec.mesh(name)
        except (KeyError, ValueError):
            return False
        if mesh is None:
            return False
        spec.delete(mesh)
        return True

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

        # Loaded through SpecBuilder.from_file (not mujoco.MjSpec.from_file) so
        # the child's file-backed asset references are absolutized while the
        # child still knows its own base directory. ``spec.attach`` does not
        # carry that directory onto the parent, so a child left with bare
        # filenames would compile here and then serialise to an XML no one can
        # reload - see _absolutize_asset_paths.
        robot_spec = SpecBuilder.from_file(str(robot_file_path))

        # Strip the robot scene's own ground/floor plane(s) before attaching.
        # Many menagerie scenes (e.g. franka_emika_panda/scene.xml) ship a
        # ``floor`` plane at z=0; merged in alongside the world's own ``ground``
        # plane (also z=0) it produces two coplanar infinite planes with
        # different checker materials -> depth-buffer Z-fighting and a broken
        # floor render. The world ``ground`` plane (configurable via
        # ``create_world(ground_plane=...)``) is the single source of truth;
        # robots contribute only their own bodies/joints/actuators. See #320.
        #
        # Three guards keep this strip safe:
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
        ground_types = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
        world_has_ground = any(int(g.type) in ground_types for g in scene_spec.geoms)
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

        # Read the solver settings the robot model declares for itself. The read
        # has to happen here, before ``attach`` consumes the child spec, but the
        # scene is only written once the attach below has succeeded: everything
        # from ``add_frame`` on can raise (``inject_robot_into_scene`` catches
        # exactly that and reports a failed add), and the live spec outlives the
        # failed call, so a scene mutated on the way to an error would keep
        # solver settings for a robot that never entered the world.
        declared_options = SpecBuilder.declared_options(robot_spec)

        frame = scene_spec.worldbody.add_frame(
            pos=list(robot.position),
            quat=list(robot.orientation),
        )
        scene_spec.attach(robot_spec, prefix=f"{robot.name}/", frame=frame)

        SpecBuilder.adopt_declared_options(scene_spec, declared_options, robot.name)

        return source_joint_names

    @staticmethod
    def declared_options(robot_spec: Any) -> dict[str, Any]:
        """Read the ``<option>`` fields a robot model declares for itself.

        Call this before ``spec.attach()``, which consumes the child spec. The
        result is a plain mapping and reading it mutates nothing, so a caller
        can hold it across the attach and only commit it to the scene once the
        attach has actually succeeded - see :meth:`adopt_declared_options`.

        Args:
            robot_spec: the robot's freshly loaded spec.

        Returns:
            Mapping of field name to declared value, for every field in
            :data:`_ADOPTED_OPTION_FIELDS` this model sets away from MuJoCo's
            default. Empty when the model declares nothing beyond the defaults.
        """
        mujoco = _ensure_mujoco()

        defaults = mujoco.MjSpec().option
        declared: dict[str, Any] = {}
        for field in _ADOPTED_OPTION_FIELDS:
            value = getattr(robot_spec.option, field)
            # A field restating MuJoCo's default is indistinguishable from one
            # the model does not mention, and adopting it would claim the field
            # against the next robot for no gain.
            if value != getattr(defaults, field):
                declared[field] = value
        return declared

    @staticmethod
    def adopt_declared_options(
        scene_spec: Any,
        declared_options: Mapping[str, Any],
        robot_name: str,
    ) -> dict[str, Any]:
        """Apply a robot's declared ``<option>`` fields onto the scene.

        ``<option>`` is model-global and does not survive ``spec.attach()``, so
        without this a robot is simulated under solver settings its own model
        rejects. See :data:`_ADOPTED_OPTION_FIELDS` for the adopted set and the
        fields deliberately left to the world.

        This writes to ``scene_spec``, so call it only once the robot is
        actually in the scene - i.e. after a successful ``attach``. The values
        themselves come from :meth:`declared_options`, which has to run before
        the attach; splitting the read from the write is what keeps a failed
        attach from leaving the scene holding this robot's settings.

        Precedence, in order:

        1. A field the scene already sets to a non-default value belongs to
           whoever set it - the caller's own scene MJCF, or a robot attached
           earlier. It is kept.
        2. Otherwise the robot's declared value is adopted.

        A model-global field holds exactly one value, so when an already-set
        field disagrees with this robot's declaration the scene value wins and
        the discarded request is logged by field, value and robot: the caller
        can force the other value by declaring it in their own scene MJCF or by
        adding that robot first.

        Args:
            scene_spec: the scene spec to mutate.
            declared_options: this robot's declared fields, from
                :meth:`declared_options`.
            robot_name: robot name, used in the conflict log.

        Returns:
            Mapping of field name to the value adopted from this robot. Empty
            when the model declared nothing, or when the scene already owned
            every field it declared.
        """
        mujoco = _ensure_mujoco()

        defaults = mujoco.MjSpec().option
        adopted: dict[str, Any] = {}
        for field, declared in declared_options.items():
            default = getattr(defaults, field)
            current = getattr(scene_spec.option, field)
            if current != default:
                if current != declared:
                    logger.warning(
                        "add_robot(%r): scene already sets option %s=%s, so the "
                        "%s=%s this model declares is not applied. A model-global "
                        "option holds one value: declare it in the scene MJCF, or "
                        "attach this robot first, to make it win.",
                        robot_name,
                        field,
                        current,
                        field,
                        declared,
                    )
                continue
            setattr(scene_spec.option, field, declared)
            adopted[field] = declared

        if adopted:
            logger.debug(
                "add_robot(%r): adopted declared physics option(s) %r",
                robot_name,
                adopted,
            )
        return adopted


# Model-global ``<option>`` fields adopted from an attached robot model.
#
# ``<option>`` is model-global, so it does not come across ``spec.attach()``:
# a robot MJCF that declares the solver settings its own contacts and actuators
# were tuned for loses them the moment it is composed into a generated scene.
# The consequences are physical, not cosmetic - a Franka Panda declares
# ``integrator="implicitfast"``, and under the default Euler integrator its
# position servos diverge enough that a top-down grasp pushes the object away on
# approach and squeezes through it on the lift. ``actuate_robot`` already flips
# the integrator to ``implicitfast`` scene-wide for the robots it actuates
# itself, for exactly that reason; a robot that ships the same declaration in
# its own model deserves the same treatment.
#
# Excluded on purpose:
#   * ``timestep`` / ``gravity`` - owned by ``create_world(timestep=, gravity=)``.
#     The world always writes both, so a robot's declaration cannot be told
#     apart from the caller's and adopting it would move the world's dt (and the
#     rollout duration math built on it) without the caller asking.
#   * ``wind`` / ``magnetic`` / ``o_solref`` / ``o_solimp`` / ``o_friction`` -
#     vector-valued environment and contact-override fields that describe the
#     world a robot is placed in, not the robot.
#   * ``disableflags`` / ``enableflags`` / ``disableactuator`` - bitfields. One
#     value per field is a resolvable arbitration; merging bitmasks from several
#     models is a different decision and is not made here.
_ADOPTED_OPTION_FIELDS: tuple[str, ...] = (
    "integrator",
    "cone",
    "jacobian",
    "solver",
    "iterations",
    "ls_iterations",
    "noslip_iterations",
    "ccd_iterations",
    "sdf_iterations",
    "sdf_initpoints",
    "impratio",
    "tolerance",
    "ls_tolerance",
    "noslip_tolerance",
    "ccd_tolerance",
    "density",
    "viscosity",
    "o_margin",
)


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
]
