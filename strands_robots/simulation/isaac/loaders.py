"""Robot description file loaders -> :class:`ProceduralRobot`.

Follow-up to the R7 Phase 1 procedural-builder slice (robots-sim#46): instead of
hardcoding one Python builder per robot in
:mod:`strands_robots.simulation.isaac.procedural`, drive the same
``ProceduralRobot`` dataclass from existing robot description files (URDF, MJCF,
USD) so the code path is a generic loader. Those three builders have since been
deleted outright - they described no real robot and nothing turned them into one
- so this module is now the only producer of a ``ProceduralRobot``.

Scope - who these three loaders are FOR (adjudicated deliberately, so a
dead-code audit does not re-litigate it): ``load_urdf`` / ``load_mjcf`` /
``load_usd`` are the *description-introspection* API - parse a robot file into
a joints/bodies report - published in ``__all__`` and documented on the Isaac
docs page for library users, and consumed in-repo by the cross-backend parity
suites (~17 test modules grade joint vocabulary, axis defaults, free-joint
spellings and pose conventions against them). They are NOT the load path:
``IsaacSimulation.add_robot`` builds articulations through Isaac's own
URDF/MJCF importers and never calls them, which is why a caller-grep of the
package finds none - the callers are outside the package and inside ``tests/``.
The rest of this module (``load_mjcf_scene_objects`` / ``SceneObject``) is the
``load_scene`` path's parser and is called from
:mod:`strands_robots.simulation.isaac.simulation` directly.

Supported formats:
    * **URDF** - ``load_urdf(path)``. Parsed with stdlib
      ``xml.etree.ElementTree``. No external deps.
    * **MJCF** - ``load_mjcf(path)``. Parsed with stdlib
      ``xml.etree.ElementTree``. Handles ``<worldbody>`` / nested ``<body>``
      / ``<joint>`` for LIBERO-style scenes. No mujoco-Python dep needed for
      definition extraction.
    * **USD** - ``load_usd(path)``. Walks the USD prim hierarchy via
      ``pxr.Usd`` / ``pxr.UsdPhysics`` to extract ``PhysicsRevoluteJoint`` /
      ``PhysicsPrismaticJoint`` + body inertia. Gated behind the ``sim-isaac``
      extra (``usd-core>=25.5``); raises :class:`ImportError` with an
      install hint when ``pxr`` is unavailable.

Failure semantics (closes the #33 class of bugs - silent ``joint_count=0``
on parse failure):

    * Missing path -> :class:`FileNotFoundError`.
    * Malformed XML / unparseable document -> :class:`ValueError` with the
      file path and the offending element / parser message.
    * A required attribute the document omits -> :class:`ValueError` naming the
      element and the attribute. Never a default standing in for it: a URDF
      ``<joint>`` must declare ``name`` and ``type``, and reading an absent
      ``type`` as ``fixed`` welded a joint the file never described. An
      attribute the format itself documents a default for (an MJCF ``<joint>``
      ``type``, a URDF ``<axis>``) is a declaration of that default and is read
      as one.
    * Empty document (zero links / zero joints / zero bodies) ->
      :class:`ValueError`. Loaders never silently return a phantom robot.

The procedural builders in :mod:`strands_robots.simulation.isaac.procedural` are
intentionally retained as the zero-dep, testable fallback used when no
description file is configured. The loaders here layer on top.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
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
    "mjcf_declares_floating_base",
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


def _parse_axis(axis_str: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Parse a whitespace-separated 3-vector. Returns ``default`` if empty / malformed.

    ``default`` is required rather than carrying a value of its own: the two
    description formats this module reads declare different defaults for the
    attributes parsed here (a URDF joint axis defaults to +X, an MJCF joint axis
    to +Z, an MJCF mesh scale to unit scale), so a call site that omitted it
    would silently inherit whichever format the signature happened to name.
    """
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


def _parse_urdf_rpy(rpy_str: str | None) -> tuple[float, float, float, float]:
    """wxyz quaternion for a URDF ``rpy`` triple.

    URDF fixes both conventions the MJCF reader has to look up: ``rpy`` is
    always radians - there is no ``<compiler angle>`` to consult - and always
    names rotations about the *fixed* axes, applied roll then pitch then yaw,
    so the composed rotation is ``Rz(yaw) * Ry(pitch) * Rx(roll)``. An absent
    or malformed triple reads as identity, matching :func:`_parse_xyz`'s
    tolerant reading of the sibling ``xyz`` attribute on the same element.
    """
    identity = (1.0, 0.0, 0.0, 0.0)
    if not rpy_str:
        return identity
    parts = rpy_str.replace(",", " ").split()
    if len(parts) != 3:
        return identity
    # Only the float conversion can raise, so only it is guarded: a wider try
    # would also swallow a refusal raised deliberately by the checks above.
    try:
        roll, pitch, yaw = (float(parts[0]), float(parts[1]), float(parts[2]))
    except (ValueError, TypeError):
        return identity
    return _quat_mul(
        _quat_about_axis((0.0, 0.0, 1.0), yaw),
        _quat_mul(_quat_about_axis((0.0, 1.0, 0.0), pitch), _quat_about_axis((1.0, 0.0, 0.0), roll)),
    )


# Map URDF joint type -> ProceduralRobot joint type.
# URDF spec types: revolute, continuous, prismatic, fixed, floating, planar.
# We collapse "continuous" -> "revolute" (continuous is a revolute with no
# limits; we surface unbounded +/-pi as the limit). "floating" / "planar" are
# rare and don't have a clean 1-DOF axis; we surface them as "fixed" with a
# warning-via-comment in the joint name (callers can refine if needed).
# URDF's default joint axis. <axis> is optional, and a joint that omits it
# rotates or slides about +X - not about the +Z an MJCF joint defaults to.
# MuJoCo, which parses URDF, reads an absent <axis> as +X for revolute,
# continuous and prismatic alike, so this is a property of the format rather
# than of any one joint type.
_URDF_DEFAULT_JOINT_AXIS = (1.0, 0.0, 0.0)

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

    A joint's axis comes from ``<axis xyz>``, and from URDF's own default of
    ``+X`` when the optional ``<axis>`` element is absent or states no vector
    this parser can read.

    Each link's pose is reported in its parent's frame, as URDF declares it:
    both halves come from the ``<origin>`` of the joint that reaches the link -
    ``xyz`` into ``position`` and ``rpy`` into ``orientation`` - because URDF
    places a link on that joint rather than on the ``<link>`` element itself. A
    root link, reached by no joint, keeps the identity pose.

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
                position=(0.0, 0.0, 0.0),  # replaced below by the reaching joint's <origin>
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
    # child body index -> (position, orientation) from the reaching joint's
    # <origin>, applied to the bodies once the tree guard below has run.
    child_pose: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for joint_el in root.findall("joint"):
        jname = joint_el.get("name")
        if not jname:
            raise ValueError(f"URDF loader: <joint> without name attribute in {path}")

        # ``type`` is a REQUIRED attribute of a URDF ``<joint>``, so an absent one
        # is missing information rather than a declaration of a default - unlike
        # the optional ``<axis>`` below, and unlike MJCF's ``type``, which the
        # format documents as defaulting to ``hinge``. Defaulting it to ``fixed``
        # here produced a JointDef byte-identical to the one a deliberate
        # ``type="fixed"`` produces, so a joint the file failed to declare was
        # indistinguishable from one the author welded on purpose: the robot came
        # back with fewer actuated DOFs than the file names, and the load
        # reported success - the silent ``joint_count`` this module's failure
        # semantics exist to convert into a message. The empty spelling
        # (``type=""``) was already refused by name below, so one file was
        # refused and the other welded for the same missing declaration. Both are
        # refused now, which is also what MuJoCo - this loader's reference for
        # the ``<axis>`` default - does with either spelling.
        urdf_type = joint_el.get("type")
        if urdf_type is None:
            raise ValueError(
                f"URDF loader: <joint name='{jname}'> in {path} states no 'type' attribute; "
                f"URDF requires it (one of {sorted(_URDF_JOINT_TYPE_MAP)})"
            )
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

        # <axis> is optional in URDF, so an absent one is a declaration of the
        # format's default rather than missing information. That default is +X;
        # reading it as the +Z an MJCF joint defaults to turns a joint the file
        # declares in one plane into one acting in the perpendicular plane, and
        # both are valid axes, so no caller can tell the two apart.
        axis_el = joint_el.find("axis")
        axis = _parse_axis(
            axis_el.get("xyz") if axis_el is not None else None,
            default=_URDF_DEFAULT_JOINT_AXIS,
        )

        # URDF states a link's placement once, on the joint that reaches it:
        # <origin> is the child link's frame expressed in the parent link's.
        origin_el = joint_el.find("origin")
        child_pose[link_index[child_name]] = (
            _parse_xyz(origin_el.get("xyz") if origin_el is not None else None),
            _parse_urdf_rpy(origin_el.get("rpy") if origin_el is not None else None),
        )

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
    # Applied after the tree guard, which establishes that at most one joint
    # reaches each body: that is what makes "the joint that reaches this link"
    # a single origin rather than a choice between two. A root link, reached by
    # no joint, keeps the identity pose - its placement in the model frame.
    for child_idx, (child_position, child_orientation) in child_pose.items():
        robot.bodies[child_idx].position = child_position
        robot.bodies[child_idx].orientation = child_orientation
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
# MJCF's default joint axis: a <joint> that omits ``axis`` acts about +Z.
_MJCF_DEFAULT_JOINT_AXIS = (0.0, 0.0, 1.0)

# MJCF's default geom type: a <geom> that omits ``type`` is a sphere, and
# ``size`` then holds its radius. Both readers of the attribute resolve the
# default through this name, because the two answering one element differently
# is how a link declared as a ball came to be reported as a box.
_MJCF_DEFAULT_GEOM_TYPE = "sphere"

_MJCF_JOINT_TYPE_MAP = {
    "hinge": "revolute",
    "slide": "prismatic",
    "ball": "fixed",
    "free": "fixed",
}

# MJCF spells a free joint two ways, and MuJoCo compiles both to the same
# ``mjJNT_FREE``: ``<joint type="free">`` and the dedicated ``<freejoint>``
# element. A reader that consults only the first reports no joint at all for a
# floating base, which is how every quadruped and humanoid in the shipped asset
# corpus declares its root. Both tags are read here so the two spellings of one
# model produce one report.
_MJCF_JOINT_TAGS = ("joint", "freejoint")

# The tag whose type MJCF fixes rather than declaring: ``<freejoint>`` accepts
# only ``name``, ``group`` and ``align`` - MuJoCo refuses ``type`` on it - and
# MJCF has no ``<default><freejoint>`` block, so it resolves no default class.
_MJCF_FREEJOINT_TAG = "freejoint"


def mjcf_declares_floating_base(path: str) -> bool:
    """Whether the MJCF's root body is attached to the world by a FREE joint.

    The question ``add_robot`` has to answer about a converted MJCF and cannot ask
    the flag it uses for URDF: URDF cannot declare a floating base, so that path
    takes ``fix_base`` from the caller - but MJCF *can*, with ``<freejoint/>`` or
    ``<joint type="free">``, and every quadruped and humanoid in the shipped asset
    corpus uses one. So for an MJCF the answer is in the file, and asking the
    caller would be asking them to restate it.

    Read from the whole spliced model via :func:`_mjcf_model_worldbody_bodies`,
    because a root body can live in an ``<include>``d fragment - the same reason
    that helper exists. Both spellings are read through
    :data:`_MJCF_JOINT_TAGS` and :data:`_MJCF_FREEJOINT_TAG`, so the vocabulary has
    one owner and a reader consulting only ``<joint type="free">`` cannot drift
    back in.

    Only a TOP-LEVEL body counts. A free joint deeper in the tree is a free-flying
    child (a MuJoCo idiom for a detached payload), not the robot's base, and
    reporting that as a floating base would put ``base_pos`` on a robot bolted to
    a table.

    Returns ``False`` for a file that cannot be read or parsed rather than
    raising: the caller is deciding whether to *report* four observation keys, and
    a robot that loads with no ``base_*`` is a smaller error than an ``add_robot``
    that refuses over a file MuJoCo itself may accept. The load path parses the
    same file immediately afterwards and reports any real problem there.
    """
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return False
    root = tree.getroot()
    base_dir = os.path.dirname(os.path.abspath(path))
    _declares, top_bodies = _mjcf_model_worldbody_bodies(root, base_dir)
    for body_el in top_bodies:
        for child in body_el:
            if child.tag == _MJCF_FREEJOINT_TAG:
                return True
            if child.tag in _MJCF_JOINT_TAGS and (child.get("type") or "").strip() == "free":
                return True
    return False


def load_mjcf(path: str) -> ProceduralRobot:
    """Load an MJCF file and return a :class:`ProceduralRobot`.

    Parses MuJoCo's MJCF format with stdlib
    :mod:`xml.etree.ElementTree`. Walks ``<worldbody>`` / nested ``<body>``
    elements depth-first to assign body indices in tree order, then emits a
    :class:`JointDef` for each joint connecting that body to its parent -
    ``<joint>`` and the dedicated ``<freejoint>``, MJCF's two spellings of a
    free joint, which MuJoCo compiles to the same joint. Useful for LIBERO
    scenes (the matrix's main consumer ships MJCF).

    Each body's pose is reported in its parent's frame, as MJCF declares it:
    ``position`` from ``pos``, and ``orientation`` from whichever of MJCF's five
    mutually exclusive orientation spellings the element uses (``quat``,
    ``euler``, ``axisangle``, ``xyaxes`` or ``zaxis``). ``<compiler angle>`` and
    ``<compiler eulerseq>`` are model-global and read from the spliced model, so
    a robot inheriting ``angle="radian"`` from an ``<include>`` is not read as
    degrees.

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
        ``<worldbody>`` / no descendant ``<body>`` is present, or a body
        declares more than one orientation spelling (MuJoCo refuses such a
        model outright, so there is no rotation to pick).
    """
    _require_existing_file(path, "MJCF")
    root = _parse_xml(path, "MJCF")

    if root.tag != "mujoco":
        raise ValueError(f"MJCF loader: root element must be <mujoco>, got <{root.tag}> in {path}")

    model_name = root.get("model", os.path.splitext(os.path.basename(path))[0])

    mjcf_dir = os.path.dirname(os.path.abspath(path))
    declares_worldbody, top_bodies = _mjcf_model_worldbody_bodies(root, mjcf_dir)
    if not declares_worldbody:
        raise ValueError(f"MJCF loader: {path} has no <worldbody>")

    geom_defaults = _mjcf_class_defaults(root, mjcf_dir, "geom")
    joint_defaults = _mjcf_class_defaults(root, mjcf_dir, "joint")
    # Model-global, so read from the spliced model: a robot whose own
    # <compiler> sets only meshdir still inherits angle="radian" from an
    # <include>d fragment, and an angle read in the wrong units is a
    # rotation wrong by a factor of 57.
    angle_scale, eulerseq = _mjcf_angle_units(_mjcf_model_toplevel(root, mjcf_dir))
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

    def _walk(body_el: ET.Element, parent_idx: int, childclass: str) -> None:
        # ``childclass`` is MJCF's per-subtree default class: it applies to every
        # descendant that does not name a class of its own, and a nested body
        # overrides it for its own subtree.
        childclass = body_el.get("childclass") or childclass
        body_name = body_el.get("name") or f"body_{len(bodies)}"
        position = _parse_xyz(body_el.get("pos"))

        # MJCF mass - usually inferred via <inertial mass=...> or via
        # <geom> density; default to 1.0 if absent.
        mass = 1.0
        inertial = body_el.find("inertial")
        if inertial is not None:
            mass = _safe_float(inertial.get("mass"), 1.0)

        # Geometry - first <geom> primitive type.
        shape, shape_size = _extract_mjcf_shape(body_el, geom_defaults, childclass)

        # MJCF spells a body's rotation five mutually exclusive ways. The
        # position above is read from the element, so reporting identity for
        # the rotation half places a rotated link unrotated -- and identity
        # is a valid orientation, so no caller can tell a link the model
        # never rotates from one whose rotation was dropped.
        orientation = _parse_orientation(body_el.attrib, angle_scale=angle_scale, eulerseq=eulerseq)

        bodies.append(
            BodyDef(
                name=body_name,
                position=position,
                orientation=orientation,
                mass=mass,
                shape=shape,
                shape_size=shape_size,
            )
        )
        body_idx = len(bodies) - 1

        # Each joint-bearing child connects this body to its parent, in
        # document order. MuJoCo refuses a free joint beside any other joint on
        # one body ("more than 6 dofs in body"), so a body carrying a
        # <freejoint> carries nothing else and the order is never ambiguous.
        for joint_el in (child for child in body_el if child.tag in _MJCF_JOINT_TAGS):
            if joint_el.tag == _MJCF_FREEJOINT_TAG:
                # <freejoint> declares no type, axis, range, damping or
                # armature, and MJCF has no <default><freejoint> block, so it
                # resolves no class: MuJoCo gives such a joint the built-in
                # values even where a <default><joint> class is in force. Its
                # own attributes are therefore all it has.
                jattrs: dict[str, str] = {}
                mjcf_type = "free"
            else:
                # A joint's own attributes, with its default class's underneath:
                # a class may declare ``type``, ``axis``, ``range``, ``damping``
                # and ``armature``, so a joint that spells none of them still
                # has all five. ``name`` stays an attribute of the element - a
                # class names a kind of joint, never an instance of one.
                jattrs = _class_attrs(joint_el, joint_defaults, childclass)
                mjcf_type = jattrs.get("type", "hinge")
            jname = joint_el.get("name") or f"{body_name}_joint_{len(joints)}"
            jtype = _MJCF_JOINT_TYPE_MAP.get(mjcf_type)
            if jtype is None:
                raise ValueError(
                    f"MJCF loader: <joint name='{jname}' type='{mjcf_type}'> in {path}: "
                    f"unknown joint type (expected one of {sorted(_MJCF_JOINT_TYPE_MAP)})"
                )

            axis = _parse_axis(jattrs.get("axis"), default=_MJCF_DEFAULT_JOINT_AXIS)
            range_str = jattrs.get("range")
            lower, upper = -3.14159, 3.14159
            if range_str:
                try:
                    parts = range_str.replace(",", " ").split()
                    if len(parts) >= 2:
                        lower = float(parts[0])
                        upper = float(parts[1])
                except (ValueError, TypeError):
                    pass

            damping = _safe_float(jattrs.get("damping"), 0.1)
            armature = _safe_float(jattrs.get("armature"), 0.01)

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
            _walk(child, body_idx, childclass)

    if not top_bodies:
        raise ValueError(f"MJCF loader: {path} <worldbody> has no <body> children (phantom robot guard)")

    # Each top-level body inherits its OWN <worldbody>'s childclass: a spliced
    # model may carry several, and they need not name the same class.
    body_childclass = {
        id(body): wb.get("childclass") or ""
        for wb in _mjcf_model_toplevel(root, mjcf_dir)
        if wb.tag == "worldbody"
        for body in wb.findall("body")
    }
    for body_el in top_bodies:
        _walk(body_el, parent_idx=0, childclass=body_childclass.get(id(body_el), ""))

    robot = ProceduralRobot(name=model_name, bodies=bodies, joints=joints)
    _validate_kinematic_tree(robot)
    return robot


def _extract_mjcf_shape(
    body_el: ET.Element,
    defaults: dict[str, dict[str, str]],
    childclass: str,
) -> tuple[str, tuple[float, ...]]:
    """Best-effort MJCF body -> (shape, shape_size) extraction from first <geom>.

    A capsule or cylinder may spell its axis extent with ``fromto`` instead of
    ``pos`` + ``size``, in which case ``size`` holds only the radius and the
    endpoints carry the length - the same two spellings
    :func:`_geom_aabb` consults :func:`_parse_fromto` for on the scene-object
    side, and the fixed-component rule
    :func:`strands_robots.simulation.mujoco.scene_ops.fromto_fixed_size_components`
    states for the MuJoCo backend. Read as ``size`` alone, such a link reports
    the 0.05 m default half-length however long the segment is, so the
    endpoints are consulted first.

    ``fromto`` on a box or ellipsoid squares the cross-section and needs the
    rotated-box bound; those keep their existing default-box reading, matching
    the exclusion :func:`_geom_aabb` documents rather than asserting an
    approximation this does not compute.

    All three attributes are read through :func:`_class_attrs`, because a geom
    need not spell any of them itself - ``<default>`` inheritance may supply
    them, and a link whose class declares ``type="capsule"`` reads as the
    typeless default if the element is asked directly.

    A ``<geom>`` that states no ``type`` at all takes MJCF's documented default
    (:data:`_MJCF_DEFAULT_GEOM_TYPE`, a sphere), which is also what
    :func:`_geom_aabb` reads it as - the two answering one element differently
    is a link reported as a shape its own file does not describe. Read as a box,
    such a geom loses its stated ``size`` as well as its shape, because the box
    branch needs three components and a ball declares one.

    A body with no ``<geom>`` at all is the separate case: there is no geometry
    to name, so it keeps the module's no-geometry box proxy, the same one
    :func:`_extract_urdf_shape` returns for a link with no ``<visual>`` or
    ``<collision>``.
    """
    geom = body_el.find("geom")
    if geom is None:
        return "box", (0.05, 0.05, 0.05)
    attrs = _class_attrs(geom, defaults, childclass)
    gtype = attrs.get("type", _MJCF_DEFAULT_GEOM_TYPE)
    size_str = attrs.get("size", "")
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
        # MJCF size for capsule/cylinder is (radius, half-length) - but only
        # for the ``pos`` spelling. With ``fromto`` the endpoints carry the
        # length and ``size`` holds only the radius, so read them first.
        segment = _parse_fromto(attrs.get("fromto"))
        if segment is not None:
            length = _segment_length(segment[0], segment[1])
            if length > 0.0:
                return gtype, (sizes[0] if sizes else 0.05, length / 2.0)
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
    """Lazy-import pxr.Usd / Sdf / UsdPhysics. Mirrors the pattern from robots-sim#44.

    Returns (Usd, Sdf, UsdPhysics) tuple. Raises ImportError with an install
    hint when the modules are unavailable (Pixar USD ships only via the
    ``sim-isaac`` extra).
    """
    try:
        from pxr import Sdf, Usd, UsdPhysics  # type: ignore[import-not-found]

        return Usd, Sdf, UsdPhysics
    except ImportError as e:
        raise ImportError(
            "USD loader requires Pixar USD (pxr.Usd / pxr.UsdPhysics). "
            "Install via: pip install 'strands-robots[sim-isaac]' "
            "or directly: pip install 'usd-core>=25.5,<27.0.0'"
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

    Gated behind the ``sim-isaac`` extra (``usd-core``); raises
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
        If ``pxr`` is not importable (install via ``sim-isaac`` extra).
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
# COLLISION stays a box proxy: each object's physics footprint is the
# axis-aligned bounding box (AABB) of its collision geoms (MuJoCo
# ``group="0"``), which robosuite always emits as analytic primitives
# (boxes / spheres / cylinders) even when the visual geom is a mesh. That
# preserves the physics behaviour the MuJoCo-parity evals were validated
# against. The VISUAL is the real mesh where the MJCF declares one (#2459):
# each SceneObject additionally carries the resolved mesh asset path +
# scale of its first visual mesh geom, so the Isaac realization can render
# the bowl/plate a pixel-conditioned policy was trained on instead of a
# gray box. A body whose declared mesh asset file is MISSING on disk is a
# hard ValueError - never a silent box (the loaders' fail-loud contract).

# MJCF body-name prefixes/exact-names that are NOT task objects and must be
# skipped when realizing a LIBERO scene as Isaac prims:
#   * ``floor`` / planes  -> ground plane is created by ``create_world``.
#   * ``robot0`` / robot  -> the Panda is loaded separately by the adapter.
_MJCF_SCENE_SKIP_EXACT = frozenset({"floor", "ground", "world"})
_MJCF_SCENE_SKIP_PREFIXES = ("robot0", "robot_", "gripper0", "mount0")


@dataclass
class SceneObject:
    """A single object extracted from a LIBERO/BDDL MJCF scene.

    Carries just enough geometry for ``IsaacSimulation.load_scene`` to
    realize the object: a box-AABB approximation of the object's collision
    geometry, its world position (body ``pos`` + AABB centre), whether it
    is a static fixture (no free joint) or a dynamic, physics-driven
    object (has a ``<freejoint>`` / ``<joint type="free">``), and - when
    the MJCF declares one - the object's visual mesh asset (path + scale +
    body-frame pose), so the realization can render the real mesh while
    keeping the AABB box as the collision proxy (#2459).

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
    offset : tuple[float, float, float]
        Body-frame offset of the AABB centre from the MJCF body origin
        (``position = body_pos + offset`` at load time). Needed by pose
        appliers (#1820): LIBERO init states carry per-episode *body*
        poses, but the realized box prim sits at the AABB centre, so a
        new body pose maps to prim pose ``body_pos + R(body_quat) @
        offset``. Without this field the offset is unrecoverable from
        ``position`` alone once the body moves.
    mesh_path : str or None
        Resolved absolute path of the object's visual mesh asset (the
        first mesh geom in the body subtree, preferring non-collision
        groups), or ``None`` when the body declares no mesh. The Isaac
        realization renders this mesh as the visual while keeping the
        AABB box as the collision proxy (#2459); the file is verified to
        exist at parse time - a declared-but-missing asset raises rather
        than degrading to a silent box.
    mesh_scale : tuple[float, float, float]
        Per-axis scale from the MJCF ``<mesh scale=...>`` asset attribute
        (unit scale when absent). Applied on the visual prim's xform.
    mesh_pos : tuple[float, float, float]
        Body-frame position of the visual mesh geom (nested-body offsets
        folded in, matching how ``offset`` accumulates collision AABBs).
    mesh_quat : tuple[float, float, float, float]
        Body-frame orientation ``[w, x, y, z]`` of the visual mesh geom
        (identity when absent).
    """

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    is_static: bool
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_path: str | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _parse_quat(quat_str: str | None) -> tuple[float, float, float, float]:
    """Parse an MJCF ``quat="w x y z"`` string, normalized. Identity on failure.

    A quaternion describes a rotation only at unit norm, and MuJoCo normalizes
    every ``quat`` its compiler reads. Reporting the four components as written
    hands the caller a value that is not a rotation: MuJoCo's own
    ``mju_quat2Mat`` builds the matrix from the components as given, so
    ``quat="1 -1 0 0"`` -- the idiomatic spelling of a quarter turn, which the
    shipped asset corpus uses on hundreds of robot links -- yields that quarter
    turn composed with a uniform scale of ``|q|**2``, twice size.

    A zero quaternion has no direction to normalize onto, so it is read as
    malformed: identity, this function's reading for every ``quat`` it cannot
    use. MuJoCo refuses such a model outright ("zero quaternion is not
    allowed"), so no model that compiles reaches here with one.
    """
    if not quat_str:
        return (1.0, 0.0, 0.0, 0.0)
    try:
        parts = [float(p) for p in quat_str.replace(",", " ").split()]
    except (ValueError, TypeError):
        return (1.0, 0.0, 0.0, 0.0)
    if len(parts) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    norm = math.sqrt(sum(p * p for p in parts))
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (parts[0] / norm, parts[1] / norm, parts[2] / norm, parts[3] / norm)


#: MJCF's orientation spellings other than ``quat``. MuJoCo keeps these four in
#: one "alternative orientation" slot carrying which of them was used, separate
#: from the ``quat`` attribute, and resolves that slot in preference to ``quat``
#: when it is set. Two consequences, both measured on mujoco 3.5.0 and 3.12.0:
#: a ``<default>`` class supplying any of these beats a ``quat`` the element
#: itself declares, and an inner declaration of one of these replaces an outer
#: declaration of another rather than sitting beside it.
_ORIENTATION_ALT_SPELLINGS = ("euler", "axisangle", "xyaxes", "zaxis")

#: MJCF's mutually exclusive orientation spellings, in the order this module
#: reports them when ONE element declares more than one (MuJoCo refuses that).
_ORIENTATION_SPELLINGS = ("quat", *_ORIENTATION_ALT_SPELLINGS)


def _merge_mjcf_attrs(outer: Mapping[str, str], inner: Mapping[str, str]) -> dict[str, str]:
    """``inner``'s attributes over ``outer``'s, with MJCF's orientation precedence.

    Every attribute merges by name -- an inner declaration replaces an outer one
    -- except that the orientation spellings do not share a name. MuJoCo keeps
    :data:`_ORIENTATION_ALT_SPELLINGS` in one slot, so an inner ``euler``
    REPLACES an outer ``zaxis`` instead of joining it, and reading both would
    leave the effective rotation ambiguous. ``quat`` is a plain attribute and
    merges by name; whether it or a surviving alternative spelling is in effect
    is :func:`_parse_orientation`'s decision, not this one's.

    Args:
        outer: The enclosing level's attributes -- a ``<default>`` class's, or
            the parent class's when flattening a nested one.
        inner: The narrower level's own attributes.

    Returns:
        The merged mapping, carrying at most one of the alternative spellings.
    """
    merged = {**outer, **inner}
    if any(inner.get(spelling) for spelling in _ORIENTATION_ALT_SPELLINGS):
        for spelling in _ORIENTATION_ALT_SPELLINGS:
            if not inner.get(spelling):
                merged.pop(spelling, None)
    return merged


def _mjcf_angle_units(elements: list[ET.Element]) -> tuple[float, str]:
    """The model's angle scale and Euler sequence, from its ``<compiler>`` elements.

    Read from the whole spliced model (the caller passes
    :func:`_mjcf_model_toplevel`'s result) because MuJoCo treats ``<compiler>``
    as model-global: a scene whose only ``<compiler>`` sets ``meshdir`` still
    inherits ``angle="radian"`` from an ``<include>``d robot fragment.

    Where several ``<compiler>`` elements carry the same attribute the last in
    document order wins, which is MuJoCo's own precedence and the same rule
    :func:`_parse_mjcf_mesh_assets` applies to ``meshdir``.

    Returns:
        ``(angle_scale, eulerseq)`` -- the factor converting a declared angle to
        radians (``1.0`` for ``angle="radian"``, else degrees-to-radians, since
        MJCF's default is ``degree``) and the Euler axis sequence as declared,
        case intact - the case selects the composition order for ``euler``.
        (default ``"xyz"``).
    """
    angle = "degree"
    eulerseq = "xyz"
    for compiler_el in (el for el in elements if el.tag == "compiler"):
        declared_angle = compiler_el.get("angle")
        if declared_angle:
            angle = declared_angle.strip().lower()
        declared_seq = compiler_el.get("eulerseq")
        if declared_seq:
            eulerseq = declared_seq.strip()
    return (1.0 if angle == "radian" else math.pi / 180.0, eulerseq)


def _quat_about_axis(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    """Unit quaternion (wxyz) for a rotation of ``angle`` radians about ``axis``."""
    norm = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    half = angle / 2.0
    scale = math.sin(half) / norm
    return (math.cos(half), axis[0] * scale, axis[1] * scale, axis[2] * scale)


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_from_basis(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """wxyz quaternion for the rotation whose columns are the given unit axes.

    Shepperd's method, branching on the largest diagonal term: the trace-only
    form degenerates for a rotation near 180 degrees, which is exactly what an
    ``xyaxes`` flipping two axes declares. This module keeps its own copy rather
    than importing the Isaac backend's -- ``loaders`` is stdlib-only and
    :mod:`strands_robots.simulation.isaac.simulation` imports *it*, so the
    dependency cannot run the other way. The Newton backend and the mesh sensor
    reader keep their own copies for the same reason.
    """
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((s / 4.0), (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return ((m21 - m12) / s, s / 4.0, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m02 - m20) / s, (m01 + m10) / s, s / 4.0, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4.0)


def _parse_orientation(
    attrs: Mapping[str, str],
    *,
    own: Mapping[str, str] | None = None,
    angle_scale: float = math.pi / 180.0,
    eulerseq: str = "xyz",
) -> tuple[float, float, float, float]:
    """The wxyz orientation an MJCF element declares, whichever spelling it uses.

    MJCF gives a body, geom or site five mutually exclusive ways to state one
    rotation -- ``quat``, ``euler``, ``axisangle``, ``xyaxes`` and ``zaxis`` --
    and reading only ``quat`` reports identity for the other four, so an object
    a model rotates is placed unrotated. Each is resolved the way MuJoCo's
    compiler does:

    * ``quat="w x y z"`` -- normalized (see the note below).
    * ``euler="a b c"`` -- three rotations about the axes ``eulerseq`` names,
      composed about the *moving* axes (intrinsic), with the angles in the
      units ``<compiler angle>`` declares.
    * ``axisangle="x y z a"`` -- ``a`` (in the declared units) about the axis,
      which need not be normalized.
    * ``xyaxes="x1 y1 z1 x2 y2 z2"`` -- the x axis, then the y axis
      orthogonalized against it, then z as their cross product.
    * ``zaxis="x y z"`` -- the minimal rotation taking ``+z`` onto that axis.

    They are mutually exclusive on ONE element, but not across MJCF's
    ``<default>`` precedence: a class may supply one spelling and the element
    declare another, and MuJoCo compiles that model. It resolves it by keeping
    :data:`_ORIENTATION_ALT_SPELLINGS` in a slot separate from ``quat`` and
    preferring that slot, so an alternative spelling from ANY level beats a
    ``quat`` from any level -- including a class ``euler`` over the element's own
    ``quat``. :func:`_merge_mjcf_attrs` has already resolved which alternative
    spelling survives, so at most one reaches here.

    Every orientation this returns is a unit quaternion. The four alternative
    spellings are constructed at unit norm, and a ``quat`` is normalized the way
    MuJoCo's compiler normalizes it -- see :func:`_parse_quat` for why the
    components as written are not a rotation.

    Args:
        attrs: The element's effective attributes -- ``el.attrib``, or
            :func:`_class_attrs`' result where a ``<default>`` class may supply
            the orientation.
        own: The element's OWN attributes when ``attrs`` merges a class's in.
            Only the spellings an element declares itself are mutually
            exclusive, so this is what the refusal below is measured against;
            defaults to ``attrs``, which is correct for a ``<body>`` (MJCF has
            no ``<body>`` default class, so nothing is merged in).
        angle_scale: Factor converting a declared angle to radians, from
            :func:`_mjcf_angle_units`.
        eulerseq: Euler axis sequence with its case intact, from
            :func:`_mjcf_angle_units`.

    Returns:
        The orientation as a wxyz tuple; identity when the element declares no
        orientation, or when the one it declares is malformed (the historical
        reading for a malformed ``quat``).

    Raises:
        ValueError: If ``own`` declares more than one orientation spelling.
            MuJoCo refuses such a model outright ("multiple orientation
            specifiers are not allowed"), so there is no rotation to pick. A
            class supplying one spelling while the element declares another is
            NOT that case -- MuJoCo compiles it, and it is resolved above.
    """
    declared_here = [s for s in _ORIENTATION_SPELLINGS if (attrs if own is None else own).get(s)]
    if len(declared_here) > 1:
        raise ValueError(
            f"MJCF orientation: {declared_here} were all declared on one element, but they are mutually "
            "exclusive - MuJoCo refuses a model with multiple orientation specifiers. Keep one."
        )
    # An alternative spelling beats ``quat`` whichever level declared it, and
    # ``_merge_mjcf_attrs`` leaves at most one of them, so the first is the one.
    alternatives = [s for s in _ORIENTATION_ALT_SPELLINGS if attrs.get(s)]
    if alternatives:
        spelling = alternatives[0]
    elif attrs.get("quat"):
        spelling = "quat"
    else:
        return (1.0, 0.0, 0.0, 0.0)
    if spelling == "quat":
        return _parse_quat(attrs.get("quat"))
    try:
        nums = [float(p) for p in str(attrs[spelling]).replace(",", " ").split()]
    except (ValueError, TypeError):
        return (1.0, 0.0, 0.0, 0.0)

    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    if spelling == "euler":
        sequence = eulerseq.lower()
        if len(nums) != 3 or len(sequence) != 3 or any(c not in axes for c in sequence):
            return (1.0, 0.0, 0.0, 0.0)
        # An upper-case sequence rotates about the moving axes, which is the same
        # three rotations composed in the opposite order (measured over all six
        # permutations). A mixed-case sequence is read as the fixed-axis form; no
        # shipped model declares one.
        steps = list(zip(sequence, nums, strict=True))
        if eulerseq.isupper():
            steps.reverse()
        quat = (1.0, 0.0, 0.0, 0.0)
        for axis_name, angle in steps:
            quat = _quat_mul(quat, _quat_about_axis(axes[axis_name], angle * angle_scale))
        return quat
    if spelling == "axisangle":
        if len(nums) != 4:
            return (1.0, 0.0, 0.0, 0.0)
        return _quat_about_axis((nums[0], nums[1], nums[2]), nums[3] * angle_scale)
    if spelling == "zaxis":
        if len(nums) != 3:
            return (1.0, 0.0, 0.0, 0.0)
        norm = math.sqrt(sum(n * n for n in nums))
        if norm == 0.0:
            return (1.0, 0.0, 0.0, 0.0)
        zx, zy, zz = (n / norm for n in nums)
        # Minimal rotation from +z onto the declared axis. Antipodal (-z) has no
        # unique minimal rotation; MuJoCo answers a half turn about x.
        if zz <= -1.0 + 1e-12:
            return (0.0, 1.0, 0.0, 0.0)
        return _quat_about_axis((-zy, zx, 0.0), math.acos(max(-1.0, min(1.0, zz))))
    # xyaxes
    if len(nums) != 6:
        return (1.0, 0.0, 0.0, 0.0)
    xn = math.sqrt(nums[0] ** 2 + nums[1] ** 2 + nums[2] ** 2)
    if xn == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    x_axis = (nums[0] / xn, nums[1] / xn, nums[2] / xn)
    dot = nums[3] * x_axis[0] + nums[4] * x_axis[1] + nums[5] * x_axis[2]
    ortho = (nums[3] - dot * x_axis[0], nums[4] - dot * x_axis[1], nums[5] - dot * x_axis[2])
    yn = math.sqrt(sum(c * c for c in ortho))
    if yn == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    y_axis = (ortho[0] / yn, ortho[1] / yn, ortho[2] / yn)
    z_axis = (
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    )
    return _quat_from_basis(x_axis, y_axis, z_axis)


def _is_skipped_scene_body(name: str) -> bool:
    """True when an MJCF top-level body is the floor or the robot (not an object)."""
    lname = name.lower()
    if lname in _MJCF_SCENE_SKIP_EXACT:
        return True
    return any(lname.startswith(p) for p in _MJCF_SCENE_SKIP_PREFIXES)


def _mjcf_model_toplevel(root: ET.Element, base_dir: str, _seen: frozenset[str] = frozenset()) -> list[ET.Element]:
    """Top-level elements of the whole model, with ``<include>`` spliced in document order.

    MuJoCo treats ``<include file=...>`` as a textual splice: the referenced
    file's children take the ``<include>`` element's place. ``<compiler>`` and
    ``<asset>`` are therefore model-global - the fragment declaring the mesh
    search directory need not be the fragment declaring the mesh, and neither
    need be the top file. Reading only the top file's direct children reports a
    mesh that is present as absent, which is the harm
    :func:`strands_robots.assets.download._mjcf_mesh_subdir` names for the same
    rule on the robot-asset path.

    An include path is resolved against the *including* file's directory, so
    nested includes chain. A missing, unreadable, malformed or cyclic include
    contributes nothing rather than failing the scene: MuJoCo names the
    offending file itself on the load that follows, which is a better report
    than anything this reader could invent.

    Element order is preserved because it is load-bearing: within one
    ``<compiler>`` element ``meshdir`` overrides ``assetdir``, but a later
    ``<compiler>`` element overrides an earlier one, so the last declaration in
    document order wins.
    """
    out: list[ET.Element] = []
    for child in root:
        if child.tag != "include":
            out.append(child)
            continue
        rel = child.get("file")
        if not rel:
            continue
        inc_path = os.path.normpath(os.path.abspath(rel if os.path.isabs(rel) else os.path.join(base_dir, rel)))
        if inc_path in _seen or not os.path.isfile(inc_path):
            continue
        try:
            frag = ET.parse(inc_path).getroot()
        except (ET.ParseError, OSError):
            continue
        out.extend(_mjcf_model_toplevel(frag, os.path.dirname(inc_path), _seen | {inc_path}))
    return out


# MJCF's implicit root default class. MuJoCo names it and will not let a model
# rename it, so this is the one spelling a file can use for the root besides
# leaving the top-level ``<default>`` unnamed.
_MJCF_ROOT_DEFAULT_CLASS = "main"


def _mjcf_class_defaults(root: ET.Element, base_dir: str, tag: str) -> dict[str, dict[str, str]]:
    """Every ``<default>`` class's effective ``<tag>`` attributes, flattened.

    MJCF's ``<default>`` elements form a tree: the top-level element is the root
    class whether or not it names itself, a ``<default class="X">`` inherits its
    enclosing element's attributes, and an element takes its class's attributes
    for every attribute it does not spell itself. So the attributes that decide
    what a ``<geom>`` is - ``type``, ``size``, ``fromto`` - and the ones that
    decide what a ``<joint>`` is - ``type``, ``axis``, ``range`` - need not
    appear on the element at all. Read as the element's own attributes alone, a
    link whose class declares ``type="capsule" size="0.05"`` reports the default
    box however long its ``fromto`` segment is, and a finger whose class declares
    ``type="slide"`` reports a revolute joint turning about the default axis.

    ``tag`` selects the ``<default>`` child whose attributes to collect. One
    class carries a separate attribute set per element kind, so ``geom`` and
    ``joint`` are collected separately and never merged: they share attribute
    names (``type``) that mean different things.

    ``<default>`` is a top-level element, so it is model-global: the fragment
    declaring a class need not be the fragment declaring the element, and
    neither need be the top file - the same splice
    :func:`_mjcf_model_toplevel` resolves for ``<compiler>`` and ``<asset>``.

    A model may carry SEVERAL top-level ``<default>`` elements, and MuJoCo
    merges them into the one root class in document order - the same treatment
    it gives ``<compiler>``, ``<asset>`` and ``<worldbody>``. They are therefore
    accumulated here, not read independently. The merge is per attribute, so a
    later element overriding one attribute ``tag`` selects does not discard
    another that an earlier one declared.

    A nested class is flattened where it appears, which is what MuJoCo does: it
    inherits the root class as accumulated *up to its own position*, so an
    attribute declared by a top-level element AFTER the one enclosing it does
    not reach it. Measured on mujoco 3.12.0 - a nested class in the first of two
    top-level elements compiles with MuJoCo's own sphere default, not the
    ``type="capsule"`` the second element declares.

    Returns a mapping from class name to that class's merged ``<tag>``
    attributes, with the root class under both of the spellings that reach it -
    ``""`` and ``"main"``. A class an element names but no ``<default>``
    declares contributes nothing rather than failing the load: MuJoCo refuses
    such a model itself, and naming the offending class is its report to make.
    """

    def _attrs_of(default_el: ET.Element) -> dict[str, str]:
        child = default_el.find(tag)
        if child is None:
            return {}
        return {k: v for k, v in child.attrib.items() if k != "class"}

    classes: dict[str, dict[str, str]] = {"": {}, _MJCF_ROOT_DEFAULT_CLASS: {}}

    def _flatten(default_el: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
        merged = _merge_mjcf_attrs(inherited, _attrs_of(default_el))
        classes[default_el.get("class", "")] = merged
        for nested in default_el.findall("default"):
            _flatten(nested, merged)
        return merged

    # The root class accumulates across the model's top-level ``<default>``
    # elements rather than restarting at each one, because MuJoCo merges them.
    # Starting from ``{}`` per element lets the last one REPLACE the others, so a
    # ``type`` only the first declares is dropped and every geom resolving
    # against the root reports the 0.05 m fallback proxy under a successful load.
    root_attrs: dict[str, str] = {}
    for el in _mjcf_model_toplevel(root, base_dir):
        if el.tag == "default":
            # A top-level ``<default>`` is the root class under either of two
            # spellings: MuJoCo names it ``main`` and refuses to let it be
            # renamed ("top-level default class 'main' cannot be renamed"), so a
            # file may leave it unnamed or write ``class="main"`` and mean the
            # same class. A geom reaches it by the same two names, plus by
            # naming no class at all. Keying it on whichever spelling the file
            # used loses every geom that arrives by the other one - which is the
            # whole model when the two disagree, and they disagree in shipped
            # assets: Menagerie's ``pal_tiago_dual`` writes ``class="main"`` and
            # gives none of its 46 geoms a class, so 34 of them report the
            # 0.05 m fallback proxy for the ``type="mesh"`` that class declares.
            #
            # Registering both spellings cannot shadow a different class,
            # because neither name is available to one: MuJoCo refuses a nested
            # ``class="main"`` ("repeated default class name") and a nested
            # unnamed ``<default>`` ("empty class name").
            root_attrs = _flatten(el, root_attrs)
            classes[""] = root_attrs
            classes[_MJCF_ROOT_DEFAULT_CLASS] = root_attrs
    return classes


def _class_attrs(el: ET.Element, defaults: dict[str, dict[str, str]], childclass: str) -> dict[str, str]:
    """An element's effective attributes: its default class's, overridden by its own.

    The class is the element's own ``class``, else the nearest enclosing body's
    ``childclass``, else the root class - MJCF's own precedence, and the same
    rule for a ``<joint>`` as for a ``<geom>``. Every attribute this module
    reads off either kind goes through here so one rule answers "what did this
    element declare", rather than each reader asking the element directly and
    seeing only half of it.

    ``defaults`` must be the map :func:`_mjcf_class_defaults` collected for
    ``el``'s own tag: a joint resolved against geom defaults would inherit a
    geom's ``type``, which names a shape rather than a degree of freedom.

    An ``<asset><mesh>`` reaches this with an empty ``childclass``, which is the
    format's own rule rather than a simplification: ``<asset>`` is a top-level
    element, so no body encloses it and its only class is the one it names -
    else the root class.
    """
    cls = el.get("class") or childclass
    return _merge_mjcf_attrs(defaults.get(cls, {}), el.attrib)


def _mjcf_model_worldbody_bodies(root: ET.Element, base_dir: str) -> tuple[bool, list[ET.Element]]:
    """Whether the model declares a ``<worldbody>``, and its top-level bodies.

    Read from the whole model - the MJCF plus every ``<include>``d fragment, via
    :func:`_mjcf_model_toplevel` - because MuJoCo splices includes and MERGES
    every ``<worldbody>`` the spliced model carries, exactly as it treats
    ``<compiler>`` and ``<asset>`` as model-global. So a model's bodies need not
    live in the top file, and a model may declare several ``<worldbody>``
    elements across its fragments.

    Reading only the top file's direct children breaks in two ways, and they
    fail differently. A model whose ``<worldbody>`` lives entirely in a fragment
    reads as having none, so the loader refuses a model MuJoCo compiles. A model
    that keeps some bodies in the top file and includes the rest reads only the
    ones it can see, silently dropping the others - and their whole subtrees and
    joints with them - under a successful load.

    Both elements and bodies keep document order, which is the order MuJoCo
    assigns body indices in.

    Args:
        root: Parsed ``<mujoco>`` root element.
        base_dir: Directory the model's own ``<include>`` paths resolve against.

    Returns:
        ``(declares_a_worldbody, top_level_body_elements)``. The flag separates
        "no ``<worldbody>`` anywhere in the model" from "a ``<worldbody>`` that
        carries no bodies", which the two callers report differently.
    """
    worldbodies = [el for el in _mjcf_model_toplevel(root, base_dir) if el.tag == "worldbody"]
    return bool(worldbodies), [body for wb in worldbodies for body in wb.findall("body")]


def _parse_mjcf_mesh_assets(root: ET.Element, mjcf_dir: str) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Collect the model's ``<asset><mesh>`` registry: name -> (file path, scale).

    The registry and its search directory are read from the whole model - the
    MJCF plus every ``<include>``d fragment, via
    :func:`_mjcf_model_toplevel` - because MuJoCo splices includes and treats
    ``<compiler>`` and ``<asset>`` as model-global.

    File paths resolve the way MuJoCo's compiler does: an absolute ``file`` is
    used as-is; a relative one is joined against ``<compiler meshdir=...>`` (or
    ``assetdir``), itself resolved against the *model* file's directory even
    when an included fragment in a subdirectory declared it; with no meshdir the
    model directory is the base. Where one ``<compiler>`` element carries both
    attributes ``meshdir`` wins; where several elements carry one, the last in
    document order wins. Paths are resolved but NOT existence-checked here -
    only meshes actually consumed by a task object are checked (at selection
    time in :func:`load_mjcf_scene_objects`), so an unused stale asset entry
    cannot fail a scene that never touches it.

    A ``<mesh>`` without a ``name`` defaults to the file's basename without
    extension, per the MJCF spec.

    ``scale`` resolves through the mesh's ``<default>`` class, because MJCF lets
    a class supply it: ``<default class="X"><mesh scale="0.001 0.001 0.001"/>``
    plus ``<mesh class="X" file="palm.obj"/>`` declares a millimetre asset with
    no ``scale`` on the element at all. Read as the element's own attribute
    alone, such a mesh reports the unit fallback - a plausible value, so nothing
    downstream can tell "authored at unit scale" from "the declared scale was
    not read", and the asset is reported a thousand times too large.

    ``file`` and ``name`` are read from the element only, which is what the
    format allows: MuJoCo's schema refuses either attribute inside a
    ``<default><mesh>`` ("unrecognized attribute"), so a class cannot name an
    asset or its file.
    """
    elements = _mjcf_model_toplevel(root, mjcf_dir)

    mesh_base = mjcf_dir
    for compiler_el in (el for el in elements if el.tag == "compiler"):
        for attr in ("meshdir", "assetdir"):
            d = compiler_el.get(attr)
            if d:
                mesh_base = d if os.path.isabs(d) else os.path.join(mjcf_dir, d)
                break

    mesh_defaults = _mjcf_class_defaults(root, mjcf_dir, "mesh")
    registry: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for asset_el in (el for el in elements if el.tag == "asset"):
        for mesh_el in asset_el.findall("mesh"):
            file_attr = mesh_el.get("file")
            if not file_attr:
                continue
            name = mesh_el.get("name") or os.path.splitext(os.path.basename(file_attr))[0]
            resolved = file_attr if os.path.isabs(file_attr) else os.path.join(mesh_base, file_attr)
            attrs = _class_attrs(mesh_el, mesh_defaults, "")
            scale = _parse_axis(attrs.get("scale"), default=(1.0, 1.0, 1.0))
            registry[name] = (os.path.normpath(resolved), scale)
    return registry


# How strongly a mesh geom's own declaration reads as its body's visual asset,
# best first. Used only to order candidates within one body subtree.
_MESH_VISUAL_RANK_NON_COLLIDING = 0
_MESH_VISUAL_RANK_NON_DEFAULT_GROUP = 1
_MESH_VISUAL_RANK_UNMARKED = 2


def _geom_cannot_collide(attrs: dict[str, str]) -> bool:
    """Whether a geom's own declaration makes it unable to collide with anything.

    MuJoCo lets two geoms touch only when ``contype1 & conaffinity2`` or
    ``contype2 & conaffinity1`` is non-zero, so a geom declaring *both* as
    ``0`` can never collide with any other geom whatever the other declares.
    MuJoCo defaults both to ``1``, so an omitted attribute reads as colliding.

    A value that is not an integer also reads as colliding: MuJoCo refuses such
    a model outright, so there is no intent to infer from it.

    Args:
        attrs: The geom's effective attributes from :func:`_class_attrs`, so a
            geom inheriting the pair from a ``<default class="visual">`` is
            judged as MuJoCo reads it rather than by what it spells itself.

    Returns:
        ``True`` when the geom is contact-free, and therefore decorative.
    """
    for key in ("contype", "conaffinity"):
        try:
            if int(attrs.get(key, "1")) != 0:
                return False
        except ValueError:
            return False
    return True


def _mesh_geom_visual_rank(attrs: dict[str, str]) -> int:
    """How strongly a mesh geom reads as its body's visual asset (lower is stronger).

    A geom MuJoCo cannot collide with (see :func:`_geom_cannot_collide`) exists
    only to be looked at, which is the format's own statement of intent, so it
    ranks first.

    ``group`` carries no meaning in MuJoCo itself - it is a visualiser toggle -
    and the conventions built on it disagree: robosuite emits visual geoms as
    ``group="1"`` while MuJoCo Menagerie spells *collision* as ``group="3"``.
    A non-default group is therefore only a weak hint. It is taken when the
    contact declaration says nothing, and never over a contact-free geom.
    Anything else ranks last.

    Args:
        attrs: The geom's effective attributes from :func:`_class_attrs`.

    Returns:
        One of :data:`_MESH_VISUAL_RANK_NON_COLLIDING`,
        :data:`_MESH_VISUAL_RANK_NON_DEFAULT_GROUP` or
        :data:`_MESH_VISUAL_RANK_UNMARKED`.
    """
    if _geom_cannot_collide(attrs):
        return _MESH_VISUAL_RANK_NON_COLLIDING
    if (attrs.get("group") or "0") != "0":
        return _MESH_VISUAL_RANK_NON_DEFAULT_GROUP
    return _MESH_VISUAL_RANK_UNMARKED


def _find_body_mesh(
    body_el: ET.Element,
    defaults: dict[str, dict[str, str]],
    childclass: str,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    angle_scale: float = math.pi / 180.0,
    eulerseq: str = "xyz",
) -> tuple[str, tuple[float, float, float], tuple[float, float, float, float], int] | None:
    """Best visual mesh geom in a body subtree: ``(mesh name, body-frame pos, quat, rank)``.

    Prefers the geom whose declaration reads most strongly as the body's visual
    asset - :func:`_mesh_geom_visual_rank`, strongest first - because the visual
    asset is the one a pixel-conditioned policy was trained on. Document order
    breaks a tie between geoms of equal rank, so a subtree whose geoms carry no
    visual marking at all still yields its first mesh.

    Ranking rather than returning the first non-default ``group`` matters for the
    dominant convention: MuJoCo Menagerie marks visual geoms with
    ``contype="0" conaffinity="0"`` and *collision* geoms with ``group="3"``, and
    a collision geom is routinely declared first. Taking a non-default group as
    the visual signal therefore handed back the convex collision hull.

    Recurses into nested bodies, folding their ``pos`` offsets into the returned
    body-frame position the same way :func:`_recursive_collision_aabb`
    accumulates AABBs (nested body rotations are not composed - LIBERO's
    compiled object bodies do not rotate their nested visual bodies). A stronger
    candidate in a nested body wins over a weaker one on the body itself.
    Returns ``None`` when the subtree declares no mesh geom.

    ``mesh``, ``pos``, ``quat`` and the ``contype``/``conaffinity``/``group``
    that decide the rank are read through :func:`_class_attrs`, so a geom that
    inherits them from a ``<default class="visual">`` is ranked as MuJoCo reads
    it rather than being taken for a collision geom.

    The geom's own ``pos`` and orientation are refused when they parse to a
    non-finite value (:func:`_refuse_non_finite_geom`), because this reader is
    the only one that sees them. :func:`_geom_aabb` refuses the same quantities,
    but :func:`_body_collision_aabb` calls it for the geoms MuJoCo can collide
    and returns as soon as it has a bound - so a *visual* mesh geom beside a
    collision primitive, which is the Menagerie convention this ranking exists
    to prefer, is never handed to it. The nested-body ``pos`` folded into the
    offset needs no test here: :func:`_recursive_collision_aabb` refuses it on
    the same ``findall("body")`` traversal, so a second one would be
    unreachable.

    Raises:
        ValueError: If a mesh geom's ``pos`` or orientation resolves to a value
            that is not finite (:func:`_refuse_non_finite_geom`).
    """
    best: tuple[str, tuple[float, float, float], tuple[float, float, float, float], int] | None = None
    childclass = body_el.get("childclass") or childclass
    for geom in body_el.findall("geom"):
        attrs = _class_attrs(geom, defaults, childclass)
        mesh_name = attrs.get("mesh")
        if not mesh_name:
            continue
        gpos = _parse_xyz(attrs.get("pos"))
        _refuse_non_finite_geom(attrs, "pos", gpos)
        pos = (offset[0] + gpos[0], offset[1] + gpos[1], offset[2] + gpos[2])
        quat = _parse_orientation(attrs, own=geom.attrib, angle_scale=angle_scale, eulerseq=eulerseq)
        # Name the spelling the file used, so the refusal points at an attribute
        # a reader can go and look at rather than at the resolved quaternion.
        spelling = next((a for a in _ORIENTATION_SPELLINGS if attrs.get(a)), "quat")
        _refuse_non_finite_geom(attrs, spelling, quat)
        rank = _mesh_geom_visual_rank(attrs)
        if rank == _MESH_VISUAL_RANK_NON_COLLIDING:
            return (mesh_name, pos, quat, rank)
        if best is None or rank < best[3]:
            best = (mesh_name, pos, quat, rank)
    for child in body_el.findall("body"):
        child_off = _parse_xyz(child.get("pos"))
        new_off = (offset[0] + child_off[0], offset[1] + child_off[1], offset[2] + child_off[2])
        found = _find_body_mesh(child, defaults, childclass, new_off, angle_scale=angle_scale, eulerseq=eulerseq)
        if found is not None:
            if found[3] == _MESH_VISUAL_RANK_NON_COLLIDING:
                return found
            if found[3] < (best[3] if best is not None else _MESH_VISUAL_RANK_UNMARKED + 1):
                best = found
    return best


def _parse_fromto(
    fromto_str: str | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Parse a geom ``fromto`` into its two endpoints, or ``None``.

    ``fromto`` is MJCF's endpoint spelling for a capsule / cylinder / box /
    ellipsoid: six numbers giving the two ends of the geom's own axis. It is
    mutually exclusive with ``pos`` - MuJoCo refuses a geom declaring both -
    so the endpoint pair is the whole placement, not an offset from one.
    Anything that is not exactly six parseable numbers returns ``None``:
    MuJoCo refuses those files outright, so there is no shape to approximate.
    """
    if not fromto_str:
        return None
    try:
        parts = [float(p) for p in fromto_str.replace(",", " ").split()]
    except (ValueError, TypeError):
        return None
    if len(parts) != 6:
        return None
    return (parts[0], parts[1], parts[2]), (parts[3], parts[4], parts[5])


def _segment_length(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Length of a ``fromto`` segment - the single owner of that arithmetic.

    Both consumers of a ``fromto`` geom need it: :func:`_segment_aabb` for the
    AABB a scene object's collision proxy is built from, and
    :func:`_extract_mjcf_shape` for the half-length a robot link's
    ``shape_size`` reports. A zero length is MuJoCo's own refusal case
    ("fromto points too close"), so callers treat ``0.0`` as "no geometry".
    """
    delta = tuple(p2[i] - p1[i] for i in range(3))
    return math.sqrt(sum(d * d for d in delta))


def _quat_matrix(
    quat: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Rotation matrix rows of a unit ``wxyz`` quaternion.

    Column ``j`` is where the geom's own axis ``j`` points in the frame the
    quaternion is expressed in, which is what a rotated primitive's extent is
    built from. :func:`_parse_orientation` returns a unit quaternion whichever
    spelling the element used, so no renormalization happens here.
    """
    w, x, y, z = quat
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _rotated_box_half(
    half: tuple[float, float, float],
    rot: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float]:
    """Half-extent of a rotated box - exact, not an approximation.

    A box's farthest point along axis ``i`` is the corner whose every local
    component pushes that way, so the extent is ``sum_j |R[i][j]| * half[j]``.
    Read without the rotation the extent is reported on the geom's own axes
    instead of the body's, which for a quarter turn is a permutation rather
    than a small error: a 0.16 m finger pad rotated onto ``x`` keeps reporting
    0.0085 m there.
    """
    return tuple(sum(abs(rot[i][j]) * half[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _rotated_ellipsoid_half(
    half: tuple[float, float, float],
    rot: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float]:
    """Half-extent of a rotated ellipsoid - exact, and tighter than its box.

    The support function of an ellipsoid with semi-axes ``half`` is
    ``sqrt(sum_j (R[i][j] * half[j]) ** 2)``, which is why this cannot reuse
    :func:`_rotated_box_half`: the box bound of the same rotated ellipsoid is
    correct but loose, and a collision proxy standing in for the object should
    not be inflated by the reader's choice of formula.
    """
    return tuple(  # type: ignore[return-value]
        math.sqrt(sum((rot[i][j] * half[j]) ** 2 for j in range(3))) for i in range(3)
    )


def _segment_aabb(
    gtype: str,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    radius: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """``(center, half_extent)`` AABB of a ``fromto`` capsule or cylinder.

    The centre is the segment midpoint - what MuJoCo's compiler resolves
    ``geom_pos`` to for a ``fromto`` geom - and the extent is the exact
    bound rather than the box of a rotated local box: a capsule is the
    segment swept by a ball of ``radius``, so every axis grows by the full
    radius, while a cylinder's cap is a disc perpendicular to the axis and
    grows by ``radius * sqrt(1 - u_i ** 2)``. For an axis-aligned
    ``fromto`` - the spelling a robosuite-compiled scene uses - both agree
    with MuJoCo's own resolved ``geom_pos`` and extent exactly.

    Returns ``None`` for a degenerate (zero-length) segment, which MuJoCo
    also refuses ("fromto points too close").
    """
    delta = tuple(p2[i] - p1[i] for i in range(3))
    length = _segment_length(p1, p2)
    if length <= 0.0:
        return None
    center = tuple((p1[i] + p2[i]) / 2.0 for i in range(3))
    if gtype == "capsule":
        half = tuple(abs(delta[i]) / 2.0 + radius for i in range(3))
    else:
        half = tuple(
            abs(delta[i]) / 2.0 + radius * math.sqrt(max(0.0, 1.0 - (delta[i] / length) ** 2)) for i in range(3)
        )
    return center, half  # type: ignore[return-value]


def _refuse_non_finite_placement(where: str, attribute: str, values: Sequence[float]) -> None:
    """Refuse a placement or extent that parsed but is not finite.

    The single owner of the finiteness test, so the geom paths and the body
    paths cannot drift into tolerating what the other refuses. ``where`` is the
    locator its caller built - :func:`_refuse_non_finite_geom` for a geom,
    :func:`_refuse_non_finite_body` for a body - because the two elements are
    located differently and only the wording below is shared.

    Args:
        where: The element at fault, already rendered for the message.
        attribute: The MJCF attribute whose parsed value is not finite.
        values: The parsed components, reported so the caller sees which.

    Raises:
        ValueError: If any component of ``values`` is not finite.
    """
    if all(map(math.isfinite, values)):
        return
    raise ValueError(
        f"{where}: {attribute} has a component that is not finite ({tuple(values)!r}) - a body "
        f"measured around it reports a collision bound for geometry the scene does not declare, "
        f"so the geom is refused instead of measured around"
    )


def _refuse_non_finite_body(body_el: ET.Element, attribute: str, values: Sequence[float]) -> None:
    """Refuse a body whose own placement parsed but is not finite.

    A body's ``pos`` is folded into the bound exactly as a geom's is - the
    top-level one becomes the object's reported ``position``, and a nested
    one becomes the running offset every geom below it is measured from. A
    non-finite component there is dropped by the same running ``min``/``max``
    :func:`_refuse_non_finite_geom` describes, so the whole subtree disappears
    from the bound while every field the loader reports stays finite: a table
    whose leg hangs off a ``pos="0 0 nan"`` body is measured as its tabletop
    alone, byte-identical to the same fixture with the leg deleted.

    Bodies carry no ``type``, so an unnamed one is located as ``<body>`` rather
    than by the resolved-type fallback the geom locator uses. The element is
    read directly rather than through :func:`_class_attrs`: MJCF ``<default>``
    classes supply geom attributes, not a body's own placement.

    Args:
        body_el: The ``<body>`` whose parsed placement is not finite.
        attribute: The MJCF attribute at fault (``pos`` or the orientation).
        values: The parsed components, reported so the caller sees which.

    Raises:
        ValueError: If any component of ``values`` is not finite.
    """
    name = body_el.get("name")
    _refuse_non_finite_placement(f"body {name!r}" if name else "unnamed <body>", attribute, values)


def _refuse_non_finite_geom(attrs: Mapping[str, str], attribute: str, values: Sequence[float]) -> None:
    """Refuse a geom whose placement or extent parsed but is not finite.

    A NaN or infinite coordinate is not a position, and neither of the two ways
    this module folds a geom into a bound reports that.
    :func:`_body_collision_aabb` unions a body's geoms with a running
    ``min``/``max``, which orders a NaN as neither smaller nor larger than
    anything and so drops it - returning the bounds of the geoms that *are*
    finite, the same numbers a body declaring only those would produce, under
    no error at all. A fixture whose leg declares ``euler="nan 0 0"`` is
    measured as its tabletop alone: a 4 cm slab where the file declares a 77 cm
    table, byte-identical to the same fixture with no leg. An infinite
    coordinate fails the other way and no more usefully: it makes one axis of
    the reported extent unbounded, and the ``inf - inf`` that leaves is a NaN
    which :func:`_recursive_collision_aabb`'s own accumulator drops in turn, so
    the widest geom in a subtree can size the proxy at the ``1e-4`` floor.
    Either way the object :func:`load_mjcf_scene_objects` emits is a collision
    proxy for geometry the scene does not contain, which is the phantom this
    module's failure semantics exist to refuse.

    MuJoCo - the owner of the format - refuses a non-finite geom quantity
    wherever it checks one (``nan size in geom``) and warns about the document
    as a whole (``XML contains a 'NaN'``), so refusing is the format's own
    disposition rather than a policy invented here. It nevertheless compiles a
    non-finite ``pos`` or orientation on a body with no free joint, which is
    why this reader cannot defer the question to a compile step: the scenes
    that reach it are exactly the ones that load.

    One wording for all four parse paths - ``pos``, ``size``, whichever
    orientation spelling the geom used, and ``fromto`` - so no one spelling
    drifts into tolerating what its siblings refuse. Only a value that
    *parsed* is graded, so an unreadable attribute keeps falling back to the
    documented default exactly as before.

    The locator is read from the *resolved* attributes rather than the element,
    which is the rule every other reader in this module follows: a ``<default>``
    class may supply what a geom does not spell itself. ``name`` is the one
    attribute a class cannot supply - MuJoCo rejects it in a ``<default>`` as a
    schema violation - so going through the resolver costs nothing there and
    keeps one rule answering the question for every reader.

    Args:
        attrs: The geom's attributes with ``<default>`` inheritance resolved,
            read for the ``name`` and ``type`` that locate it. MJCF geoms are
            frequently unnamed, so the type is the fallback locator.
        attribute: The MJCF attribute whose parsed value is not finite.
        values: The parsed components, reported so the caller sees which.

    Raises:
        ValueError: If any component of ``values`` is not finite.
    """
    name = attrs.get("name")
    gtype = attrs.get("type", _MJCF_DEFAULT_GEOM_TYPE)
    where = f"geom {name!r}" if name else f'unnamed <geom type="{gtype}">'
    _refuse_non_finite_placement(where, attribute, values)


def _geom_aabb(
    geom: ET.Element,
    defaults: dict[str, dict[str, str]],
    childclass: str,
    *,
    angle_scale: float = math.pi / 180.0,
    eulerseq: str = "xyz",
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return ``(center, half_extent)`` AABB for a collision ``<geom>``.

    Handles MuJoCo's analytic primitives (box / sphere / cylinder /
    capsule / ellipsoid). Mesh / plane / unknown geoms return ``None`` so
    the caller can fall back to other geoms. The geom-local ``pos`` is the
    AABB centre relative to the owning body's frame.

    A geom's own rotation is part of its extent. MJCF lets a geom state one
    with ``quat``, ``euler``, ``axisangle``, ``xyaxes`` or ``zaxis``, and read
    without it the extent is reported on the geom's own axes rather than the
    body's - for a quarter turn a permutation of the answer, not a small error.
    :func:`_parse_orientation` resolves whichever spelling the geom (or the
    ``<default>`` class it inherits from) used, and the bound is then the exact
    one for that primitive: :func:`_rotated_box_half` for a box,
    :func:`_rotated_ellipsoid_half` for an ellipsoid, the segment bound below
    for a capsule or cylinder, and the radius itself for a sphere, which no
    rotation changes.

    A capsule or cylinder may spell its placement and axis extent with
    ``fromto`` instead of ``pos`` + ``size``, in which case the endpoints
    carry both and ``size`` holds only the radius - the same fixed-component
    rule
    :func:`strands_robots.simulation.mujoco.scene_ops.fromto_fixed_size_components`
    states for the MuJoCo backend. Read as ``pos`` + ``size`` alone such a
    geom collapses to a ball of that radius at the body origin, losing the
    length and the offset, so :func:`_parse_fromto` is consulted first. MuJoCo
    derives a ``fromto`` geom's orientation from the endpoints and discards any
    the element declares, so that branch is deliberately rotation-free; the
    ``pos`` + ``size`` spelling builds the same endpoints from the rotation and
    shares :func:`_segment_aabb`, so one geom cannot get two different bounds
    for being spelled two ways. ``fromto`` on a box or ellipsoid additionally
    squares the cross-section by copying the first ``size`` component, a
    separate rule this reader does not derive, so those keep returning ``None``
    (no analytic AABB) and the caller falls back.

    ``type``, ``pos``, ``size``, ``fromto`` and the orientation are read through
    :func:`_class_attrs`: MJCF's ``<default>`` classes may supply any of them, so
    asking the element directly sees only the half the geom spells itself.

    A quantity that parses to a non-finite value is refused by
    :func:`_refuse_non_finite_geom` rather than returned, because ``None``
    here means "no analytic AABB, fall back to another geom" and the caller
    would silently measure the body around the geoms that remain.

    Raises:
        ValueError: If ``pos``, ``size``, ``fromto`` or the orientation
            resolves to a value that is not finite
            (:func:`_refuse_non_finite_geom`).
    """
    attrs = _class_attrs(geom, defaults, childclass)
    gtype = attrs.get("type", _MJCF_DEFAULT_GEOM_TYPE)
    pos = _parse_xyz(attrs.get("pos"))
    _refuse_non_finite_geom(attrs, "pos", pos)
    size_str = attrs.get("size", "")
    try:
        sizes = [float(p) for p in size_str.replace(",", " ").split()] if size_str else []
    except (ValueError, TypeError):
        sizes = []
    _refuse_non_finite_geom(attrs, "size", sizes)
    quat = _parse_orientation(attrs, own=geom.attrib, angle_scale=angle_scale, eulerseq=eulerseq)
    # Name the spelling the file used, so the refusal points at an attribute a
    # reader can go and look at rather than at the resolved quaternion.
    spelling = next((a for a in _ORIENTATION_SPELLINGS if attrs.get(a)), "quat")
    _refuse_non_finite_geom(attrs, spelling, quat)
    rot = _quat_matrix(quat)

    if gtype == "box":
        if len(sizes) >= 3:
            half = _rotated_box_half((sizes[0], sizes[1], sizes[2]), rot)
        else:
            return None
    elif gtype == "sphere":
        if sizes:
            # A ball is its own rotation: the extent is the radius on every axis.
            r = sizes[0]
            half = (r, r, r)
        else:
            return None
    elif gtype in ("cylinder", "capsule"):
        # MJCF spells these either as ``pos`` + ``size="radius half-length"``
        # along the geom's local z, or as ``fromto`` + ``size="radius"`` with
        # the endpoints carrying the placement and the axis extent.
        segment = _parse_fromto(attrs.get("fromto"))
        if segment is not None:
            _refuse_non_finite_geom(attrs, "fromto", (*segment[0], *segment[1]))
            # MuJoCo takes a ``fromto`` geom's axis from the endpoints and
            # discards a declared orientation, so ``rot`` is not applied here.
            return _segment_aabb(gtype, segment[0], segment[1], sizes[0]) if sizes else None
        if len(sizes) >= 2:
            r, hl = sizes[0], sizes[1]
            # The rotated local +z axis is the geom's own axis in the body
            # frame, so the same endpoints a ``fromto`` would have carried are
            # recoverable - and the exact segment bound is then shared with
            # that branch rather than re-derived for this spelling.
            axis = (rot[0][2], rot[1][2], rot[2][2])
            p1 = tuple(pos[i] - axis[i] * hl for i in range(3))
            p2 = tuple(pos[i] + axis[i] * hl for i in range(3))
            return _segment_aabb(gtype, p1, p2, r)  # type: ignore[arg-type]
        else:
            # One size and no ``fromto`` is not a shape MuJoCo compiles
            # ("size 1 must be positive in geom"), so there is no geometry
            # here to approximate.
            return None
    elif gtype == "ellipsoid":
        if len(sizes) >= 3:
            half = _rotated_ellipsoid_half((sizes[0], sizes[1], sizes[2]), rot)
        else:
            return None
    else:
        # mesh / plane / hfield / sdf -> no analytic AABB.
        return None
    return pos, half


def _body_collision_aabb(
    body_el: ET.Element,
    defaults: dict[str, dict[str, str]],
    childclass: str,
    *,
    angle_scale: float = math.pi / 180.0,
    eulerseq: str = "xyz",
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute the AABB (center, full-size) over a body's own collidable geoms.

    Prefers the geoms MuJoCo can actually collide, read off the format's own
    contact declaration by :func:`_geom_cannot_collide`, so a decorative shell
    drawn around a small collision primitive does not widen the proxy that
    stands in for the body. A body whose every analytic geom is contact-free
    falls back to all of them: an approximate bound is still better than none.

    ``group`` is deliberately not consulted. It is a MuJoCo visualiser toggle
    carrying no contact meaning, and the conventions built on it disagree -
    Menagerie spells *collision* as ``group="3"`` while robosuite spells
    *visual* as ``group="1"`` - which is why :func:`_mesh_geom_visual_rank`
    takes it only as a weak hint. Selecting on ``group="0"`` also missed every
    geom that omits the attribute, which is MJCF's spelling of group 0.

    Geom positions are taken relative to the body frame, so the returned centre
    is a body-frame offset, and each geom's own rotation is folded into its
    extent by :func:`_geom_aabb` - a body whose collision primitive is turned
    onto another axis is bounded on the axes it really occupies. Returns
    ``None`` when no analytic geom is found (e.g. a mesh-only visual body).

    The union below is a running ``min``/``max``, which would order a NaN as
    neither smaller nor larger than anything and drop the geom carrying it -
    reporting the bounds of the geoms that are finite, under no error. That
    input never arrives here: :func:`_geom_aabb` refuses a non-finite geom
    placement or extent at the parse (:func:`_refuse_non_finite_geom`), and
    :func:`_recursive_collision_aabb` refuses a non-finite body ``pos`` before
    it becomes the offset a subtree is measured from
    (:func:`_refuse_non_finite_body`), so every AABB folded in is finite by the
    time it reaches this comparison.

    ``angle_scale`` and ``eulerseq`` come from the model's ``<compiler>`` and
    are only forwarded; they default to MJCF's own defaults (degrees, ``xyz``)
    so a caller reading a single body needs neither.
    """
    for collidable_only in (True, False):
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        found = False
        for geom in body_el.findall("geom"):
            attrs = _class_attrs(geom, defaults, childclass)
            if collidable_only and _geom_cannot_collide(attrs):
                continue
            aabb = _geom_aabb(geom, defaults, childclass, angle_scale=angle_scale, eulerseq=eulerseq)
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
    defaults: dict[str, dict[str, str]],
    childclass: str,
    *,
    angle_scale: float = math.pi / 180.0,
    eulerseq: str = "xyz",
) -> bool:
    """Fold this body's (and nested bodies') collision AABBs into ``bounds``.

    ``bounds`` is ``[mins, maxs]`` accumulated in place; ``offset`` is the
    running body-frame offset from the top-level object body. Returns
    ``True`` if any analytic geometry was found in this subtree.
    """
    childclass = body_el.get("childclass") or childclass
    found = False
    aabb = _body_collision_aabb(body_el, defaults, childclass, angle_scale=angle_scale, eulerseq=eulerseq)
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
        # The running offset every geom below this child is measured from, so a
        # non-finite component makes each of their bounds non-finite and the
        # min/max above drops the whole subtree - leaving every field the loader
        # reports finite and the bound describing the geoms that remain.
        _refuse_non_finite_body(child, "pos", child_off)
        new_off = (offset[0] + child_off[0], offset[1] + child_off[1], offset[2] + child_off[2])
        found = (
            _recursive_collision_aabb(
                child, new_off, bounds, defaults, childclass, angle_scale=angle_scale, eulerseq=eulerseq
            )
            or found
        )
    return found


def load_mjcf_scene_objects(path: str) -> list[SceneObject]:
    """Extract LIBERO/BDDL task objects from a compiled MJCF scene.

    Walks the MJCF ``<worldbody>`` top-level bodies, skips the floor and
    the robot, and emits one :class:`SceneObject` per remaining body (table
    fixtures and movable task objects). Each object's *collision* geometry
    is the axis-aligned bounding box of its collision geoms (recursing into
    nested bodies so multi-link fixtures like tables are captured),
    approximated as a single box primitive. Where the body declares a mesh
    geom, the resolved asset path + scale ride along as the object's
    *visual* (#2459) - preferring a visual-group mesh over a collision one
    - so the Isaac realization renders the real bowl/plate instead of a
    gray box while the physics footprint stays the validated AABB proxy.

    Fail-loud contract for meshes: a body whose selected mesh asset is
    declared in ``<asset>`` but MISSING on disk raises :class:`ValueError`
    (never a silent box) - a policy conditioned on that object's pixels
    would otherwise be evaluated against a proxy with nothing in the
    output saying so. A mesh in a format this backend cannot convert
    (outside OBJ/STL/MSH/USD) is surfaced as ``mesh_path=None`` (box proxy),
    as is a mesh reference with no ``<asset>`` entry at all - which cannot
    occur in a robosuite-compiled scene, since MuJoCo refuses to compile
    it. A body with a mesh but NO analytic collision geometry gets the
    mesh's computed bounds as its box proxy instead of the historical
    hardcoded 0.05 m fallback.

    This is the parse half of ``IsaacSimulation.load_scene``: pure stdlib
    (no ``mujoco`` / ``pxr`` dependency; mesh bounds are read with the
    stdlib OBJ/STL/MSH parser in
    :mod:`strands_robots.simulation.isaac.mesh_assets`), so it is
    unit-testable on CPU-only CI without Isaac Sim installed.

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
        If the XML is malformed, the root isn't ``<mujoco>``, there is no
        ``<worldbody>``, a selected mesh asset file is missing on disk, or a
        geom or body declares a placement or extent that is not finite (a body
        measured around one reports a collision bound for geometry the scene
        does not declare, so it is refused rather than approximated).
    """
    from strands_robots.simulation.isaac.mesh_assets import (
        MESH_EXTENSIONS,
        USD_EXTENSIONS,
        mesh_aabb,
        refuse_non_finite_scale,
    )

    _require_existing_file(path, "MJCF scene")
    root = _parse_xml(path, "MJCF scene")
    if root.tag != "mujoco":
        raise ValueError(f"MJCF scene loader: root element must be <mujoco>, got <{root.tag}> in {path}")
    mjcf_dir = os.path.dirname(os.path.abspath(path))
    declares_worldbody, top_bodies = _mjcf_model_worldbody_bodies(root, mjcf_dir)
    if not declares_worldbody:
        raise ValueError(f"MJCF scene loader: {path} has no <worldbody>")

    mesh_registry = _parse_mjcf_mesh_assets(root, mjcf_dir)
    geom_defaults = _mjcf_class_defaults(root, mjcf_dir, "geom")
    # Each top-level body inherits its OWN <worldbody>'s childclass: a spliced
    # model may carry several, and they need not name the same class.
    body_childclass = {
        id(body): wb.get("childclass") or ""
        for wb in _mjcf_model_toplevel(root, mjcf_dir)
        if wb.tag == "worldbody"
        for body in wb.findall("body")
    }
    # Model-global, so read from the spliced model: a scene whose own <compiler>
    # sets only meshdir still inherits angle="radian" from an <include>.
    angle_scale, eulerseq = _mjcf_angle_units(_mjcf_model_toplevel(root, mjcf_dir))

    objects: list[SceneObject] = []
    for body_el in top_bodies:
        name = body_el.get("name") or ""
        if not name or _is_skipped_scene_body(name):
            continue

        body_pos = _parse_xyz(body_el.get("pos"))
        _refuse_non_finite_body(body_el, "pos", body_pos)
        body_quat = _parse_orientation(body_el.attrib, angle_scale=angle_scale, eulerseq=eulerseq)
        # Name the spelling the file used, so the refusal points at an attribute
        # a reader can go and look at rather than at the resolved quaternion -
        # the rule _geom_aabb already follows for a geom's orientation.
        body_spelling = next((a for a in _ORIENTATION_SPELLINGS if body_el.get(a)), "quat")
        _refuse_non_finite_body(body_el, body_spelling, body_quat)

        # Movable object? -> has a free joint (``<freejoint>`` or
        # ``<joint type="free">``). Otherwise treat as a static fixture.
        has_freejoint = body_el.find("freejoint") is not None or any(
            j.get("type") == "free" for j in body_el.findall("joint")
        )

        # Visual mesh passthrough (#2459). Resolved before the AABB so a
        # mesh-only body can fall back to the mesh's own bounds below.
        mesh_path: str | None = None
        mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
        mesh_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        mesh_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        cc = body_childclass.get(id(body_el), "")
        mesh_ref = _find_body_mesh(body_el, geom_defaults, cc, angle_scale=angle_scale, eulerseq=eulerseq)
        if mesh_ref is not None:
            mesh_name, mesh_pos, mesh_quat, _visual_rank = mesh_ref
            asset = mesh_registry.get(mesh_name)
            if asset is not None:
                candidate_path, mesh_scale = asset
                # Refused here rather than at the measurement below, because the
                # scale rides onto the scene object whichever branch supplies the
                # bound: a body with collidable geometry takes the analytic bound
                # and never measures, yet the realization still applies this scale
                # to the visual prim's xform, beside the mesh geom's position and
                # orientation - which are already refused when they are not
                # finite. Placed beside the missing-file contract for the same
                # reason: an asset declaration a body reaches has to be usable.
                refuse_non_finite_scale(candidate_path, mesh_scale)
                ext = os.path.splitext(candidate_path)[1].lower()
                if ext in MESH_EXTENSIONS or ext in USD_EXTENSIONS:
                    if not os.path.isfile(candidate_path):
                        raise ValueError(
                            f"MJCF scene loader: mesh asset {candidate_path!r} for body {name!r} in "
                            f"{path} is missing on disk - refusing to degrade the object to a silent "
                            f"box proxy (#2459 fail-loud contract)"
                        )
                    mesh_path = candidate_path
                # An unconvertible format (e.g. a COLLADA .dae)
                # keeps mesh_path=None: the realization uses the box proxy
                # and load_scene reports the object among the proxies.
            # A mesh reference with no <asset> entry anywhere in the model
            # (the top file or any <include>d fragment) is hand-written test
            # scaffolding: keep the historical degrade-cleanly box fallback
            # rather than failing the scene. The registry is model-global, so
            # a scene that declares its assets in an included fragment - which
            # MuJoCo compiles - resolves here too, instead of reaching this
            # fallback and rendering the 0.05 m proxy #2459 removed.

        # Gather collision geometry from this body and any nested bodies
        # (e.g. ``living_room_table`` -> ``living_room_table_col``), folding
        # nested-body offsets into the AABB.
        bounds = [[float("inf")] * 3, [float("-inf")] * 3]
        found = _recursive_collision_aabb(
            body_el, (0.0, 0.0, 0.0), bounds, geom_defaults, cc, angle_scale=angle_scale, eulerseq=eulerseq
        )
        mins, maxs = bounds[0], bounds[1]

        if found:
            center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))  # type: ignore[assignment]
            size = tuple(max(maxs[i] - mins[i], 1e-3) for i in range(3))  # type: ignore[assignment]
        elif mesh_path is not None and os.path.splitext(mesh_path)[1].lower() in MESH_EXTENSIONS:
            # No analytic collision geometry, but the mesh itself is
            # parseable: use its real bounds as the proxy instead of the
            # historical 0.05 m guess. The mesh geom's own rotation is not
            # composed into the bounds (an AABB of a rotated mesh is still
            # an approximation either way); its body-frame position is.
            mcenter, msize = mesh_aabb(mesh_path, mesh_scale)
            center = tuple(mesh_pos[i] + mcenter[i] for i in range(3))  # type: ignore[assignment]
            size = msize
        else:
            # No analytic collision geometry and no parseable mesh. Fall
            # back to a small default box so the object still appears on
            # the stage.
            center = (0.0, 0.0, 0.0)
            size = (0.05, 0.05, 0.05)

        world_pos = tuple(body_pos[i] + center[i] for i in range(3))
        objects.append(
            SceneObject(
                name=name,
                position=world_pos,  # type: ignore[arg-type]
                size=size,  # type: ignore[arg-type]
                is_static=not has_freejoint,
                quat=body_quat,
                offset=center,  # type: ignore[arg-type]
                mesh_path=mesh_path,
                mesh_scale=mesh_scale,
                mesh_pos=mesh_pos,
                mesh_quat=mesh_quat,
            )
        )

    return objects
