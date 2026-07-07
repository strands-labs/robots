"""Robot description file loaders -> :class:`ProceduralRobot`.

Follow-up to the R7 Phase 1 procedural-builder slice (PR #46): instead of
hardcoding ``_build_so100`` / ``_build_panda`` / ``_build_unitree_g1`` in
``procedural.py``, drive the same ``ProceduralRobot`` dataclass from existing
robot description files (URDF, MJCF, USD) so the code path becomes a generic
loader rather than a per-robot Python builder.

Supported formats:
    * **URDF** - ``load_urdf(path)``. Parsed with stdlib
      ``xml.etree.ElementTree``. No external deps.
    * **MJCF** - ``load_mjcf(path)``. Parsed with stdlib
      ``xml.etree.ElementTree``. Handles ``<worldbody>`` / nested ``<body>``
      / ``<joint>`` for LIBERO-style scenes. No mujoco-Python dep needed for
      definition extraction.
    * **USD** - ``load_usd(path)``. Walks the USD prim hierarchy via
      ``pxr.Usd`` / ``pxr.UsdPhysics`` to extract ``PhysicsRevoluteJoint`` /
      ``PhysicsPrismaticJoint`` + body inertia. Gated behind the ``[isaac]``
      extra (``usd-core>=24.5``); raises :class:`ImportError` with an
      install hint when ``pxr`` is unavailable.

Failure semantics (closes the #33 class of bugs - silent ``joint_count=0``
on parse failure):

    * Missing path -> :class:`FileNotFoundError`.
    * Malformed XML / unparseable document -> :class:`ValueError` with the
      file path and the offending element / parser message.
    * Empty document (zero links / zero joints / zero bodies) ->
      :class:`ValueError`. Loaders never silently return a phantom robot.

The procedural builders in :mod:`strands_robots.simulation.isaac.procedural` are
intentionally retained as the zero-dep, testable fallback used when no
description file is configured. The loaders here layer on top.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.isaac.procedural import (
    BodyDef,
    JointDef,
    ProceduralRobot,
    _validate_kinematic_tree,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "load_urdf",
    "load_mjcf",
    "load_usd",
    "SceneObject",
    "load_mjcf_scene_objects",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_existing_file(path: str, fmt: str) -> None:
    """Raise FileNotFoundError if path doesn't exist or isn't a file.

    Parameters
    ----------
    path : str
        Filesystem path to check.
    fmt : str
        Format label for the error message ("URDF", "MJCF", "USD").
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{fmt} loader: file not found: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{fmt} loader: path is not a regular file: {path}")


def _parse_xml(path: str, fmt: str) -> ET.Element:
    """Parse an XML file, converting parser errors into ValueError.

    Returns the root element. The :class:`xml.etree.ElementTree.ParseError`
    is wrapped in a :class:`ValueError` carrying the file path so the
    failure mode is explicit (not a silent zero-joint robot - see #33).
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise ValueError(f"{fmt} loader: malformed XML in {path}: {e}") from e
    return tree.getroot()


def _parse_axis(
    axis_str: str | None, default: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> tuple[float, float, float]:
    """Parse a whitespace-separated 3-vector. Returns ``default`` if empty / malformed."""
    if not axis_str:
        return default
    try:
        parts = axis_str.replace(",", " ").split()
        if len(parts) != 3:
            return default
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except (ValueError, TypeError):
        return default


def _parse_xyz(
    xyz_str: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[float, float, float]:
    """Parse a whitespace-separated 3-vector position. Returns ``default`` on failure."""
    if not xyz_str:
        return default
    try:
        parts = xyz_str.replace(",", " ").split()
        if len(parts) < 3:
            return default
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except (ValueError, TypeError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    """Parse a float, returning ``default`` on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# URDF
# ---------------------------------------------------------------------------


# Map URDF joint type -> ProceduralRobot joint type.
# URDF spec types: revolute, continuous, prismatic, fixed, floating, planar.
# We collapse "continuous" -> "revolute" (continuous is a revolute with no
# limits; we surface unbounded +/-pi as the limit). "floating" / "planar" are
# rare and don't have a clean 1-DOF axis; we surface them as "fixed" with a
# warning-via-comment in the joint name (callers can refine if needed).
_URDF_JOINT_TYPE_MAP = {
    "revolute": "revolute",
    "continuous": "revolute",
    "prismatic": "prismatic",
    "fixed": "fixed",
    "floating": "fixed",
    "planar": "fixed",
}


def load_urdf(path: str) -> ProceduralRobot:
    """Load a URDF file and return a :class:`ProceduralRobot`.

    Parses ``<link>`` and ``<joint>`` elements via stdlib
    :mod:`xml.etree.ElementTree`. Joint axes, limits, parent / child link
    references, and per-link inertial mass are extracted; geometry is
    surfaced as a best-effort ``shape`` / ``shape_size`` (defaulting to a
    unit box when absent - Phase 1 doesn't render, only the kinematic
    structure matters).

    Parameters
    ----------
    path : str
        Filesystem path to a URDF (XML) file.

    Returns
    -------
    ProceduralRobot
        Robot definition mirroring the file's link / joint topology.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If the XML is malformed, the root tag isn't ``<robot>``, or the
        document declares zero links (a #33-style "phantom robot" guard).
    """
    _require_existing_file(path, "URDF")
    root = _parse_xml(path, "URDF")

    if root.tag != "robot":
        raise ValueError(f"URDF loader: root element must be <robot>, got <{root.tag}> in {path}")

    name = root.get("name", os.path.splitext(os.path.basename(path))[0])

    # Pass 1: collect links -> bodies (preserving file order so joint
    # parent/child name lookups become a stable index).
    bodies: list[BodyDef] = []
    link_index: dict[str, int] = {}
    for link_el in root.findall("link"):
        link_name = link_el.get("name")
        if not link_name:
            raise ValueError(f"URDF loader: <link> without name attribute in {path}")
        if link_name in link_index:
            raise ValueError(f"URDF loader: duplicate <link name='{link_name}'> in {path}")

        # Inertial mass (defaults to 1.0 for renderable / 0.0 would suggest
        # massless - but URDF mass is required for non-fixed children, so
        # default 1.0 is the safer guess for procedural builders).
        mass = 1.0
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                mass = _safe_float(mass_el.get("value"), 1.0)

        # Geometry - best effort; URDF lets multiple <visual>/<collision>
        # blocks coexist and arbitrary mesh references. We extract the
        # first <collision><geometry> we find, falling back to <visual>.
        shape, shape_size = _extract_urdf_shape(link_el)

        bodies.append(
            BodyDef(
                name=link_name,
                position=(0.0, 0.0, 0.0),  # absolute pose computed by joint chain at instantiation time
                mass=mass,
                shape=shape,
                shape_size=shape_size,
            )
        )
        link_index[link_name] = len(bodies) - 1

    if not bodies:
        raise ValueError(f"URDF loader: {path} declares zero <link> elements (phantom robot guard)")

    # Pass 2: collect joints. For each joint, look up parent / child link
    # by name and resolve to body indices.
    joints: list[JointDef] = []
    for joint_el in root.findall("joint"):
        jname = joint_el.get("name")
        if not jname:
            raise ValueError(f"URDF loader: <joint> without name attribute in {path}")

        urdf_type = joint_el.get("type", "fixed")
        jtype = _URDF_JOINT_TYPE_MAP.get(urdf_type)
        if jtype is None:
            raise ValueError(
                f"URDF loader: <joint name='{jname}' type='{urdf_type}'> in {path}: "
                f"unknown joint type (expected one of {sorted(_URDF_JOINT_TYPE_MAP)})"
            )

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            raise ValueError(f"URDF loader: <joint name='{jname}'> in {path} missing <parent> or <child>")
        parent_name = parent_el.get("link")
        child_name = child_el.get("link")
        if not parent_name or not child_name:
            raise ValueError(
                f"URDF loader: <joint name='{jname}'> in {path}: <parent> / <child> missing 'link' attribute"
            )
        if parent_name not in link_index:
            raise ValueError(
                f"URDF loader: <joint name='{jname}'> references unknown parent link '{parent_name}' in {path}"
            )
        if child_name not in link_index:
            raise ValueError(
                f"URDF loader: <joint name='{jname}'> references unknown child link '{child_name}' in {path}"
            )

        axis_el = joint_el.find("axis")
        axis = _parse_axis(axis_el.get("xyz") if axis_el is not None else None)

        # Limits - URDF requires <limit> for revolute/prismatic, optional
        # for continuous. Defaults below match the dataclass defaults.
        lower = -3.14159
        upper = 3.14159
        damping = 0.1
        limit_el = joint_el.find("limit")
        if limit_el is not None:
            lower = _safe_float(limit_el.get("lower"), lower)
            upper = _safe_float(limit_el.get("upper"), upper)
        dynamics_el = joint_el.find("dynamics")
        if dynamics_el is not None:
            damping = _safe_float(dynamics_el.get("damping"), damping)

        joints.append(
            JointDef(
                name=jname,
                joint_type=jtype,
                parent_body=link_index[parent_name],
                child_body=link_index[child_name],
                axis=axis,
                limit_lower=lower,
                limit_upper=upper,
                damping=damping,
            )
        )

    robot = ProceduralRobot(name=name, bodies=bodies, joints=joints)
    _validate_kinematic_tree(robot)
    return robot


def _extract_urdf_shape(link_el: ET.Element) -> tuple[str, tuple[float, ...]]:
    """Best-effort URDF link -> (shape, shape_size) extraction.

    Falls back to a small unit box when no <geometry> primitive is found.
    Mesh-only links surface as ``shape="box"`` with an estimated size - the
    loader is for kinematic structure, not visual fidelity.
    """
    for parent_tag in ("collision", "visual"):
        parent = link_el.find(parent_tag)
        if parent is None:
            continue
        geom = parent.find("geometry")
        if geom is None:
            continue
        for prim_tag, parser in (
            ("box", _parse_box_size),
            ("cylinder", _parse_cylinder_size),
            ("sphere", _parse_sphere_size),
            ("capsule", _parse_cylinder_size),  # uncommon, treat like cylinder
        ):
            prim = geom.find(prim_tag)
            if prim is not None:
                return prim_tag, parser(prim)
        # Mesh - no primitive size; default to small box.
        if geom.find("mesh") is not None:
            return "box", (0.05, 0.05, 0.05)
    return "box", (0.05, 0.05, 0.05)


def _parse_box_size(el: ET.Element) -> tuple[float, ...]:
    size = _parse_xyz(el.get("size"), default=(0.05, 0.05, 0.05))
    return size


def _parse_cylinder_size(el: ET.Element) -> tuple[float, ...]:
    radius = _safe_float(el.get("radius"), 0.05)
    length = _safe_float(el.get("length"), 0.1)
    return (radius, length)


def _parse_sphere_size(el: ET.Element) -> tuple[float, ...]:
    radius = _safe_float(el.get("radius"), 0.05)
    return (radius,)


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------


# Map MJCF joint type -> ProceduralRobot joint type.
# MJCF spec types: free, ball, slide, hinge.
# - hinge -> revolute (1-DOF rotational)
# - slide -> prismatic
# - ball  -> not 1-DOF; no clean mapping - surface as "fixed" so the body
#           index is preserved without claiming actuated DOF.
# - free  -> 6-DOF root joint; not part of the actuated chain - "fixed".
_MJCF_JOINT_TYPE_MAP = {
    "hinge": "revolute",
    "slide": "prismatic",
    "ball": "fixed",
    "free": "fixed",
}


def load_mjcf(path: str) -> ProceduralRobot:
    """Load an MJCF file and return a :class:`ProceduralRobot`.

    Parses MuJoCo's MJCF format with stdlib
    :mod:`xml.etree.ElementTree`. Walks ``<worldbody>`` / nested ``<body>``
    elements depth-first to assign body indices in tree order, then emits a
    :class:`JointDef` for each ``<joint>`` connecting that body to its
    parent. Useful for LIBERO scenes (the matrix's main consumer ships
    MJCF).

    Parameters
    ----------
    path : str
        Filesystem path to an MJCF (XML) file.

    Returns
    -------
    ProceduralRobot
        Robot definition mirroring the body / joint topology.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If the XML is malformed, the root tag isn't ``<mujoco>``, or no
        ``<worldbody>`` / no descendant ``<body>`` is present.
    """
    _require_existing_file(path, "MJCF")
    root = _parse_xml(path, "MJCF")

    if root.tag != "mujoco":
        raise ValueError(f"MJCF loader: root element must be <mujoco>, got <{root.tag}> in {path}")

    model_name = root.get("model", os.path.splitext(os.path.basename(path))[0])

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF loader: {path} has no <worldbody>")

    bodies: list[BodyDef] = []
    joints: list[JointDef] = []

    # Synthetic root body so MJCF top-level <body>s under <worldbody>
    # always have a valid parent index. MJCF's "world" is implicit.
    bodies.append(
        BodyDef(
            name="world",
            position=(0.0, 0.0, 0.0),
            mass=0.0,
            shape="box",
            shape_size=(0.0, 0.0, 0.0),
        )
    )

    def _walk(body_el: ET.Element, parent_idx: int) -> None:
        body_name = body_el.get("name") or f"body_{len(bodies)}"
        position = _parse_xyz(body_el.get("pos"))

        # MJCF mass - usually inferred via <inertial mass=...> or via
        # <geom> density; default to 1.0 if absent.
        mass = 1.0
        inertial = body_el.find("inertial")
        if inertial is not None:
            mass = _safe_float(inertial.get("mass"), 1.0)

        # Geometry - first <geom> primitive type.
        shape, shape_size = _extract_mjcf_shape(body_el)

        bodies.append(
            BodyDef(
                name=body_name,
                position=position,
                mass=mass,
                shape=shape,
                shape_size=shape_size,
            )
        )
        body_idx = len(bodies) - 1

        # Each <joint> child connects this body to its parent.
        for joint_el in body_el.findall("joint"):
            jname = joint_el.get("name") or f"{body_name}_joint_{len(joints)}"
            mjcf_type = joint_el.get("type", "hinge")
            jtype = _MJCF_JOINT_TYPE_MAP.get(mjcf_type)
            if jtype is None:
                raise ValueError(
                    f"MJCF loader: <joint name='{jname}' type='{mjcf_type}'> in {path}: "
                    f"unknown joint type (expected one of {sorted(_MJCF_JOINT_TYPE_MAP)})"
                )

            axis = _parse_axis(joint_el.get("axis"))
            range_str = joint_el.get("range")
            lower, upper = -3.14159, 3.14159
            if range_str:
                try:
                    parts = range_str.replace(",", " ").split()
                    if len(parts) >= 2:
                        lower = float(parts[0])
                        upper = float(parts[1])
                except (ValueError, TypeError):
                    pass

            damping = _safe_float(joint_el.get("damping"), 0.1)
            armature = _safe_float(joint_el.get("armature"), 0.01)

            joints.append(
                JointDef(
                    name=jname,
                    joint_type=jtype,
                    parent_body=parent_idx,
                    child_body=body_idx,
                    axis=axis,
                    limit_lower=lower,
                    limit_upper=upper,
                    damping=damping,
                    armature=armature,
                )
            )

        for child in body_el.findall("body"):
            _walk(child, body_idx)

    top_bodies = list(worldbody.findall("body"))
    if not top_bodies:
        raise ValueError(f"MJCF loader: {path} <worldbody> has no <body> children (phantom robot guard)")

    for body_el in top_bodies:
        _walk(body_el, parent_idx=0)

    robot = ProceduralRobot(name=model_name, bodies=bodies, joints=joints)
    _validate_kinematic_tree(robot)
    return robot


def _extract_mjcf_shape(body_el: ET.Element) -> tuple[str, tuple[float, ...]]:
    """Best-effort MJCF body -> (shape, shape_size) extraction from first <geom>."""
    geom = body_el.find("geom")
    if geom is None:
        return "box", (0.05, 0.05, 0.05)
    gtype = geom.get("type", "box")
    size_str = geom.get("size", "")
    sizes: list[float] = []
    if size_str:
        try:
            sizes = [float(p) for p in size_str.replace(",", " ").split()]
        except (ValueError, TypeError):
            sizes = []
    if gtype == "box":
        if len(sizes) >= 3:
            return "box", (sizes[0], sizes[1], sizes[2])
        return "box", (0.05, 0.05, 0.05)
    if gtype == "sphere":
        if sizes:
            return "sphere", (sizes[0],)
        return "sphere", (0.05,)
    if gtype in ("cylinder", "capsule"):
        # MJCF size for capsule/cylinder is (radius, half-length).
        if len(sizes) >= 2:
            return gtype, (sizes[0], sizes[1])
        if len(sizes) == 1:
            return gtype, (sizes[0], 0.05)
        return gtype, (0.05, 0.05)
    # Mesh, plane, ellipsoid, hfield etc. - treat as a small box for kinematic-only purposes.
    return "box", (0.05, 0.05, 0.05)


# ---------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------


def _lazy_import_usd() -> tuple[Any, Any, Any]:
    """Lazy-import pxr.Usd / Sdf / UsdPhysics. Mirrors the pattern from PR #44.

    Returns (Usd, Sdf, UsdPhysics) tuple. Raises ImportError with an install
    hint when the modules are unavailable (Pixar USD ships only via the
    ``[isaac]`` extra).
    """
    try:
        from pxr import Sdf, Usd, UsdPhysics  # type: ignore[import-not-found]

        return Usd, Sdf, UsdPhysics
    except ImportError as e:
        raise ImportError(
            "USD loader requires Pixar USD (pxr.Usd / pxr.UsdPhysics). "
            "Install via: pip install 'strands-robots-sim[isaac]' "
            "or directly: pip install 'usd-core>=24.5'"
        ) from e


# Map UsdPhysics joint API -> ProceduralRobot joint type.
_USD_JOINT_TYPE_MAP = {
    "PhysicsRevoluteJoint": "revolute",
    "PhysicsPrismaticJoint": "prismatic",
    "PhysicsFixedJoint": "fixed",
    "PhysicsSphericalJoint": "fixed",  # 3-DOF; not 1-DOF, surface as fixed
    "PhysicsDistanceJoint": "fixed",
}


# Map USD physics joint axis token -> ProceduralRobot axis tuple.
_USD_AXIS_MAP = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def load_usd(path: str) -> ProceduralRobot:
    """Load a USD file and return a :class:`ProceduralRobot`.

    Walks the USD prim hierarchy via ``pxr.Usd`` / ``pxr.UsdPhysics`` to
    extract physics joint prims (``PhysicsRevoluteJoint`` /
    ``PhysicsPrismaticJoint`` / ``PhysicsFixedJoint``) plus rigid-body
    prims with ``UsdPhysicsRigidBodyAPI``.

    Gated behind the ``[isaac]`` extra (``usd-core``); raises
    :class:`ImportError` with an install hint when ``pxr`` is unavailable.

    Parameters
    ----------
    path : str
        Filesystem path to a USD file (.usd / .usda / .usdc / .usdz).

    Returns
    -------
    ProceduralRobot
        Robot definition mirroring the rigid-body / physics-joint graph.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ImportError
        If ``pxr`` is not importable (install via ``[isaac]`` extra).
    ValueError
        If the stage fails to open, declares zero rigid bodies, or has a
        joint with an unresolved body0 / body1 reference.
    """
    _require_existing_file(path, "USD")
    Usd, _Sdf, UsdPhysics = _lazy_import_usd()

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise ValueError(f"USD loader: failed to open stage at {path}")

    name = os.path.splitext(os.path.basename(path))[0]

    # Pass 1: collect rigid bodies. We treat any prim with
    # UsdPhysicsRigidBodyAPI as a body; ordering follows depth-first
    # traversal of the stage's pseudo-root.
    bodies: list[BodyDef] = []
    body_index: dict[str, int] = {}

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        prim_path = str(prim.GetPath())
        if prim_path in body_index:
            continue
        body_name = prim.GetName() or prim_path.replace("/", "_").lstrip("_")

        # Mass - UsdPhysicsMassAPI
        mass = 1.0
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasAuthoredValue():
                mass = float(mass_attr.Get() or 1.0)

        bodies.append(
            BodyDef(
                name=body_name,
                position=(0.0, 0.0, 0.0),
                mass=mass,
                shape="box",
                shape_size=(0.05, 0.05, 0.05),
            )
        )
        body_index[prim_path] = len(bodies) - 1

    if not bodies:
        raise ValueError(
            f"USD loader: {path} declares zero rigid bodies (no prims with UsdPhysicsRigidBodyAPI); phantom robot guard"
        )

    # Pass 2: collect physics joints. Any UsdPhysics.Joint subclass
    # (Revolute / Prismatic / Fixed / Spherical / Distance) shows up here
    # via prim.IsA(UsdPhysics.Joint).
    joints: list[JointDef] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        type_name = prim.GetTypeName()
        jtype = _USD_JOINT_TYPE_MAP.get(str(type_name))
        if jtype is None:
            # Unknown subclass - preserve the prim as a fixed joint to
            # keep body indexing consistent. Surfacing via name pattern.
            jtype = "fixed"

        joint_api = UsdPhysics.Joint(prim)
        body0_rel = joint_api.GetBody0Rel()
        body1_rel = joint_api.GetBody1Rel()
        body0_targets = list(body0_rel.GetTargets()) if body0_rel else []
        body1_targets = list(body1_rel.GetTargets()) if body1_rel else []
        if not body0_targets or not body1_targets:
            raise ValueError(
                f"USD loader: joint {prim.GetPath()} in {path} has unresolved "
                f"body0/body1 relationship (Phase-1 phantom-robot guard)"
            )
        body0_path = str(body0_targets[0])
        body1_path = str(body1_targets[0])
        if body0_path not in body_index:
            raise ValueError(
                f"USD loader: joint {prim.GetPath()} body0 references {body0_path} which is not a rigid body in {path}"
            )
        if body1_path not in body_index:
            raise ValueError(
                f"USD loader: joint {prim.GetPath()} body1 references {body1_path} which is not a rigid body in {path}"
            )

        # Axis: UsdPhysicsRevoluteJoint / PrismaticJoint expose an "axis"
        # token attribute valued "X" / "Y" / "Z".
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
        if str(type_name) in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            schema_cls = (
                UsdPhysics.RevoluteJoint if str(type_name) == "PhysicsRevoluteJoint" else UsdPhysics.PrismaticJoint
            )
            schema = schema_cls(prim)
            axis_attr = schema.GetAxisAttr()
            if axis_attr and axis_attr.HasAuthoredValue():
                axis = _USD_AXIS_MAP.get(str(axis_attr.Get()), axis)

            lower_attr = schema.GetLowerLimitAttr()
            upper_attr = schema.GetUpperLimitAttr()
            lower = -3.14159
            upper = 3.14159
            if lower_attr and lower_attr.HasAuthoredValue():
                lower = float(lower_attr.Get())
            if upper_attr and upper_attr.HasAuthoredValue():
                upper = float(upper_attr.Get())
        else:
            lower = -3.14159
            upper = 3.14159

        jname = prim.GetName() or str(prim.GetPath()).replace("/", "_").lstrip("_")

        joints.append(
            JointDef(
                name=jname,
                joint_type=jtype,
                parent_body=body_index[body0_path],
                child_body=body_index[body1_path],
                axis=axis,
                limit_lower=lower,
                limit_upper=upper,
            )
        )

    robot = ProceduralRobot(name=name, bodies=bodies, joints=joints)
    _validate_kinematic_tree(robot)
    return robot


# ---------------------------------------------------------------------------
# MJCF scene-object extraction (LIBERO/BDDL -> Isaac stage prims)
# ---------------------------------------------------------------------------
#
# ``load_mjcf`` above models a *single robot's* body/joint topology. LIBERO
# task scenes are different: a robosuite-compiled MJCF carrying a ground
# plane, the Panda robot, one or more table/fixture bodies, and the task's
# movable objects (mugs, plates, bowls ...). ``IsaacSimulation.load_scene``
# needs to realize those *objects* (not the robot - the LiberoAdapter loads
# the Panda separately via ``add_robot``) as USD prims on the stage so the
# Isaac LIBERO eval renders a populated scene instead of an empty one.
#
# LIBERO object meshes are not portable to the Isaac stage (their asset
# paths live inside the upstream ``libero`` package and aren't resolvable
# as USD references here). So instead of meshes we approximate each object
# with a single box primitive sized to the axis-aligned bounding box (AABB)
# of its *collision* geoms (MuJoCo ``group="0"``), which robosuite always
# emits as analytic primitives (boxes / spheres / cylinders) even when the
# visual geom is a mesh. That gives a faithful-enough footprint for
# rollout-video parity with the MuJoCo driver without needing the meshes.

# MJCF body-name prefixes/exact-names that are NOT task objects and must be
# skipped when realizing a LIBERO scene as Isaac prims:
#   * ``floor`` / planes  -> ground plane is created by ``create_world``.
#   * ``robot0`` / robot  -> the Panda is loaded separately by the adapter.
_MJCF_SCENE_SKIP_EXACT = frozenset({"floor", "ground", "world"})
_MJCF_SCENE_SKIP_PREFIXES = ("robot0", "robot_", "gripper0", "mount0")


@dataclass
class SceneObject:
    """A single object extracted from a LIBERO/BDDL MJCF scene.

    Carries just enough geometry for ``IsaacSimulation.load_scene`` to call
    ``add_object(...)``: a box-AABB approximation of the object's collision
    geometry, its world position (body ``pos`` + AABB centre), and whether
    it is a static fixture (no free joint) or a dynamic, physics-driven
    object (has a ``<freejoint>`` / ``<joint type="free">``).

    Attributes
    ----------
    name : str
        Object body name from the MJCF (e.g. ``"porcelain_mug_1_main"``).
    position : tuple[float, float, float]
        World-space ``[x, y, z]`` of the object's AABB centre.
    size : tuple[float, float, float]
        Full box extents ``[sx, sy, sz]`` (NOT half-extents) of the AABB.
    is_static : bool
        ``True`` for fixtures (tables, cabinets) pinned in space; ``False``
        for movable objects that participate in physics.
    quat : tuple[float, float, float, float]
        Orientation quaternion ``[w, x, y, z]`` from the body's ``quat``
        attribute (identity when absent).
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    is_static: bool
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _parse_quat(quat_str: str | None) -> tuple[float, float, float, float]:
    """Parse an MJCF ``quat="w x y z"`` string. Identity on failure."""
    if not quat_str:
        return (1.0, 0.0, 0.0, 0.0)
    try:
        parts = [float(p) for p in quat_str.replace(",", " ").split()]
    except (ValueError, TypeError):
        return (1.0, 0.0, 0.0, 0.0)
    if len(parts) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    return (parts[0], parts[1], parts[2], parts[3])


def _is_skipped_scene_body(name: str) -> bool:
    """True when an MJCF top-level body is the floor or the robot (not an object)."""
    lname = name.lower()
    if lname in _MJCF_SCENE_SKIP_EXACT:
        return True
    return any(lname.startswith(p) for p in _MJCF_SCENE_SKIP_PREFIXES)


def _geom_aabb(geom: ET.Element) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return ``(center, half_extent)`` AABB for a collision ``<geom>``.

    Handles MuJoCo's analytic primitives (box / sphere / cylinder /
    capsule / ellipsoid). Mesh / plane / unknown geoms return ``None`` so
    the caller can fall back to other geoms. The geom-local ``pos`` is the
    AABB centre relative to the owning body's frame.
    """
    gtype = geom.get("type", "sphere")
    pos = _parse_xyz(geom.get("pos"))
    size_str = geom.get("size", "")
    try:
        sizes = [float(p) for p in size_str.replace(",", " ").split()] if size_str else []
    except (ValueError, TypeError):
        sizes = []

    if gtype == "box":
        if len(sizes) >= 3:
            half = (sizes[0], sizes[1], sizes[2])
        else:
            return None
    elif gtype == "sphere":
        if sizes:
            r = sizes[0]
            half = (r, r, r)
        else:
            return None
    elif gtype in ("cylinder", "capsule"):
        # MJCF (radius, half-length) along local z.
        if len(sizes) >= 2:
            r, hl = sizes[0], sizes[1]
            ext = hl + (r if gtype == "capsule" else 0.0)
            half = (r, r, ext)
        elif len(sizes) == 1:
            r = sizes[0]
            half = (r, r, r)
        else:
            return None
    elif gtype == "ellipsoid":
        if len(sizes) >= 3:
            half = (sizes[0], sizes[1], sizes[2])
        else:
            return None
    else:
        # mesh / plane / hfield / sdf -> no analytic AABB.
        return None
    return pos, half


def _body_collision_aabb(
    body_el: ET.Element,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute the AABB (center, full-size) over a body's own geoms.

    Prefers MuJoCo collision geoms (``group="0"``); if a body has only
    analytic geoms in another group those are used as a fallback. Geom
    positions are taken relative to the body frame, so the returned centre
    is a body-frame offset. Returns ``None`` when no analytic geom is found
    (e.g. a mesh-only visual body).
    """
    for group_filter in ("0", None):
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        found = False
        for geom in body_el.findall("geom"):
            if group_filter is not None and geom.get("group") != group_filter:
                continue
            aabb = _geom_aabb(geom)
            if aabb is None:
                continue
            center, half = aabb
            for i in range(3):
                mins[i] = min(mins[i], center[i] - half[i])
                maxs[i] = max(maxs[i], center[i] + half[i])
            found = True
        if found:
            center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))  # type: ignore[assignment]
            size = tuple(max(maxs[i] - mins[i], 1e-4) for i in range(3))
            return center, size  # type: ignore[return-value]
    return None


def _recursive_collision_aabb(
    body_el: ET.Element,
    offset: tuple[float, float, float],
    bounds: list[list[float]],
) -> bool:
    """Fold this body's (and nested bodies') collision AABBs into ``bounds``.

    ``bounds`` is ``[mins, maxs]`` accumulated in place; ``offset`` is the
    running body-frame offset from the top-level object body. Returns
    ``True`` if any analytic geometry was found in this subtree.
    """
    found = False
    aabb = _body_collision_aabb(body_el)
    if aabb is not None:
        center, size = aabb
        for i in range(3):
            lo = offset[i] + center[i] - size[i] / 2.0
            hi = offset[i] + center[i] + size[i] / 2.0
            bounds[0][i] = min(bounds[0][i], lo)
            bounds[1][i] = max(bounds[1][i], hi)
        found = True
    for child in body_el.findall("body"):
        child_off = _parse_xyz(child.get("pos"))
        new_off = (offset[0] + child_off[0], offset[1] + child_off[1], offset[2] + child_off[2])
        found = _recursive_collision_aabb(child, new_off, bounds) or found
    return found


def load_mjcf_scene_objects(path: str) -> list[SceneObject]:
    """Extract LIBERO/BDDL task objects from a compiled MJCF scene.

    Walks the MJCF ``<worldbody>`` top-level bodies, skips the floor and
    the robot, and emits one :class:`SceneObject` per remaining body (table
    fixtures and movable task objects). Each object's geometry is the
    axis-aligned bounding box of its collision geoms (recursing into nested
    bodies so multi-link fixtures like tables are captured), approximated as
    a single box primitive.

    This is the parse half of ``IsaacSimulation.load_scene``: pure stdlib
    (no ``mujoco`` / ``pxr`` dependency), so it is unit-testable on CPU-only
    CI without Isaac Sim installed.

    Parameters
    ----------
    path : str
        Filesystem path to a robosuite-compiled LIBERO MJCF (``.xml``).

    Returns
    -------
    list[SceneObject]
        One entry per task object / fixture. Empty list only if the scene
        genuinely has no objects beyond the floor + robot (rare).

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If the XML is malformed, the root isn't ``<mujoco>``, or there is no
        ``<worldbody>``.
    """
    _require_existing_file(path, "MJCF scene")
    root = _parse_xml(path, "MJCF scene")
    if root.tag != "mujoco":
        raise ValueError(f"MJCF scene loader: root element must be <mujoco>, got <{root.tag}> in {path}")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF scene loader: {path} has no <worldbody>")

    objects: list[SceneObject] = []
    for body_el in worldbody.findall("body"):
        name = body_el.get("name") or ""
        if not name or _is_skipped_scene_body(name):
            continue

        body_pos = _parse_xyz(body_el.get("pos"))
        body_quat = _parse_quat(body_el.get("quat"))

        # Movable object? -> has a free joint (``<freejoint>`` or
        # ``<joint type="free">``). Otherwise treat as a static fixture.
        has_freejoint = body_el.find("freejoint") is not None or any(
            j.get("type") == "free" for j in body_el.findall("joint")
        )

        # Gather collision geometry from this body and any nested bodies
        # (e.g. ``living_room_table`` -> ``living_room_table_col``), folding
        # nested-body offsets into the AABB.
        bounds = [[float("inf")] * 3, [float("-inf")] * 3]
        found = _recursive_collision_aabb(body_el, (0.0, 0.0, 0.0), bounds)
        mins, maxs = bounds[0], bounds[1]

        if not found:
            # No analytic collision geometry (mesh-only with no convex
            # decomposition). Fall back to a small default box so the
            # object still appears on the stage.
            center = (0.0, 0.0, 0.0)
            size = (0.05, 0.05, 0.05)
        else:
            center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))  # type: ignore[assignment]
            size = tuple(max(maxs[i] - mins[i], 1e-3) for i in range(3))  # type: ignore[assignment]

        world_pos = tuple(body_pos[i] + center[i] for i in range(3))
        objects.append(
            SceneObject(
                name=name,
                position=world_pos,  # type: ignore[arg-type]
                size=size,  # type: ignore[arg-type]
                is_static=not has_freejoint,
                quat=body_quat,
            )
        )

    return objects
