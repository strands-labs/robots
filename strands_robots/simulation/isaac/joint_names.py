"""URDF <-> USD joint-name translation for the Isaac backend.

Isaac Sim's URDF importer writes each joint as a USD prim, and a USD prim
name must be a valid identifier - it cannot start with a digit, contain
spaces, and so on. Joint names that violate the rules get *transcoded*: on
Isaac Sim 6.0.x the importer applies USD's bootstring transcoding, which
wraps the name in a ``tn__`` prefix and a delimiter/suffix (the
``robotstudio_so101`` URDF names its joints literally ``"1"``..``"6"``,
which import as ``tn__1_``..``tn__6_``); older toolchains applied
``TfMakeValidIdentifier``, which substitutes ``_`` for each offending
character. Either way the articulation's ``dof_names`` report the mangled
form, and before #1900 that form leaked through the backend's public API:
``robot_joint_names``, ``get_observation`` keys and ``send_action`` name
resolution all disagreed with the MuJoCo backend - and with any planner
built from the same URDF - about what the robot's joints are called.

This module recovers the URDF vocabulary at import time. Every joint-name
read/write on the articulation handle is positional (index into
``dof_names`` order), so translating the names once - right where
:meth:`~strands_robots.simulation.isaac.simulation.IsaacSimulation.add_robot`
loads the URDF - makes the whole backend speak URDF names without touching
any hot path:

* :func:`urdf_joint_names` parses the movable joints out of the URDF with
  stdlib XML (no kit install required), and
* :func:`demangle_usd_joint_names` maps the importer's ``dof_names`` output
  back onto those URDF names, deterministically, refusing any mapping that
  is ambiguous or would collide.

The resulting ``usd_name -> urdf_name`` map is recorded on the robot's
bookkeeping entry so diagnostics can still correlate the USD prim names
with the public vocabulary.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

__all__ = ["urdf_joint_names", "demangle_usd_joint_names"]

# Characters USD identifiers may contain (ASCII identifier alphabet). The
# bootstring transcoding keeps exactly these and encodes everything else in
# the suffix after the final delimiter.
_BASIC_CHAR_RE = re.compile(r"[A-Za-z0-9_]")

# Prefix USD's bootstring transcoding stamps on every transcoded identifier
# (Isaac Sim 6.0.x URDF importer output, e.g. ``tn__1_`` for joint ``1``).
_TRANSCODING_PREFIX = "tn__"

# URDF joint types that produce articulation DOFs on import. ``fixed`` never
# does; ``floating``/``planar`` are multi-DOF types the Isaac importer does
# not map onto single named DOFs, and the registry URDFs do not use them.
_MOVABLE_URDF_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic", "spherical"})


def urdf_joint_names(urdf_path: str) -> list[str]:
    """Return the movable joint names declared by a URDF, in file order.

    Stdlib-XML parse (no kit / importer dependency), so the mapping side of
    the #1900 fix is testable anywhere. Only joints whose type produces an
    articulation DOF are returned (``revolute``, ``continuous``,
    ``prismatic``, ``spherical``) - ``fixed`` joints never surface in
    ``dof_names``, and including them could only make a mangled-name match
    ambiguous.

    Parameters
    ----------
    urdf_path : str
        Filesystem path to the URDF the Isaac importer just consumed.

    Returns
    -------
    list[str]
        Movable joint names in document order.

    Raises
    ------
    FileNotFoundError
        If ``urdf_path`` does not exist.
    ValueError
        If the XML is malformed, the root element is not ``<robot>``, or a
        joint carries no ``name`` attribute - the same fail-loud contract as
        :func:`strands_robots.simulation.isaac.loaders.load_urdf`.
    """
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError as e:
        raise ValueError(f"URDF joint-name parse: malformed XML in {urdf_path}: {e}") from e
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"URDF joint-name parse: root element must be <robot>, got <{root.tag}> in {urdf_path}")

    names: list[str] = []
    for joint_el in root.findall("joint"):
        jtype = joint_el.get("type", "fixed")
        if jtype not in _MOVABLE_URDF_JOINT_TYPES:
            continue
        jname = joint_el.get("name")
        if not jname:
            raise ValueError(f"URDF joint-name parse: <joint> without name attribute in {urdf_path}")
        names.append(jname)
    return names


def demangle_usd_joint_names(
    dof_names: list[str],
    urdf_names: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Map USD-mangled ``dof_names`` back onto the URDF joint vocabulary.

    For each DOF name the articulation reports, in order:

    * a name the URDF declares verbatim passes through unchanged (the
      common case - a valid identifier is never transcoded);
    * a name exactly one unclaimed URDF joint mangles to (bootstring
      ``tn__`` transcoding or legacy ``TfMakeValidIdentifier``
      substitution, both deterministic - see :func:`_mangles_to`) is
      translated to that URDF name;
    * anything else stays as reported.

    The result can never key two DOFs by one public name: a translated name
    is drawn from ``unclaimed`` - URDF joints *not* reported verbatim by any
    DOF - and is removed from the pool when claimed, so translated names are
    disjoint from pass-through names and from each other. An ambiguous
    decode (two URDF joints that mangle to the same DOF name) is refused
    and the DOF name kept, which leaves the pre-#1900 behaviour for that
    joint: Isaac self-consistent on its own mangled name.

    Parameters
    ----------
    dof_names : list[str]
        Joint names as reported by ``articulation.dof_names`` after import.
    urdf_names : list[str]
        Movable joint names from :func:`urdf_joint_names`.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        ``(public_names, usd_to_urdf)`` - ``public_names`` aligns
        one-to-one with ``dof_names`` (same order, so every positional
        articulation read/write stays valid), and ``usd_to_urdf`` records
        only the entries that were actually translated.
    """
    urdf_set = set(urdf_names)
    verbatim = urdf_set.intersection(dof_names)
    # URDF joints not already claimed verbatim by some DOF name are the only
    # candidates a mangled DOF name may decode to. Drawing translations from
    # this pool (and removing each claim) is what makes the public names
    # collision-free by construction - see the docstring.
    unclaimed = [u for u in urdf_names if u not in verbatim]

    public: list[str] = []
    usd_to_urdf: dict[str, str] = {}
    for dof in dof_names:
        if dof in urdf_set:
            public.append(dof)
            continue
        candidates = [u for u in unclaimed if _mangles_to(u, dof)]
        if len(candidates) == 1:
            urdf_name = candidates[0]
            unclaimed.remove(urdf_name)
            usd_to_urdf[dof] = urdf_name
            public.append(urdf_name)
        else:
            if candidates:
                logger.warning(
                    "USD joint name %r decodes ambiguously to URDF joints %s; keeping the USD name.",
                    dof,
                    sorted(candidates),
                )
            public.append(dof)
    return public, usd_to_urdf


def _mangles_to(urdf_name: str, dof_name: str) -> bool:
    """Whether the USD importer would mangle ``urdf_name`` into ``dof_name``.

    Two deterministic encodings are recognised:

    * **Bootstring transcoding** (Isaac Sim 6.0.x): the output is
      ``tn__<basic>_<suffix>``, where ``<basic>`` is the subsequence of
      ASCII-identifier characters of the input and ``<suffix>`` encodes the
      positions/values of everything else. A name made *entirely* of basic
      characters (a leading digit being its only defect, e.g. ``1``)
      transcodes with an empty suffix - exactly ``tn__1_`` - so that case
      is matched exactly. A name with non-basic characters has a non-empty
      suffix this module does not decode; it is matched on the
      deterministic ``tn__<basic>_`` stem, and
      :func:`demangle_usd_joint_names` only accepts the match when it is
      unique.
    * **Legacy substitution** (``TfMakeValidIdentifier``): each character
      outside the identifier alphabet - and a leading character that is not
      a letter or underscore - is replaced by ``_``. Fully deterministic,
      so matched exactly.
    """
    if dof_name == _tf_make_valid_identifier(urdf_name):
        return True
    if not dof_name.startswith(_TRANSCODING_PREFIX):
        return False
    basic = "".join(ch for ch in urdf_name if _BASIC_CHAR_RE.match(ch))
    stem = f"{_TRANSCODING_PREFIX}{basic}_"
    if basic == urdf_name:
        # Empty bootstring suffix: the only defect was the leading
        # character, so the full transcoded form is known exactly.
        return dof_name == stem
    return dof_name.startswith(stem) and len(dof_name) > len(stem)


def _tf_make_valid_identifier(name: str) -> str:
    """Pure-Python clone of USD's ``TfMakeValidIdentifier``.

    Replaces every character outside ``[A-Za-z0-9_]`` with ``_``, and a
    first character that is not ``[A-Za-z_]`` likewise; an empty input
    yields ``_``. This is the legacy (pre-bootstring) mangle older Isaac
    URDF importers applied to invalid prim names.
    """
    if not name:
        return "_"
    first = name[0]
    out = [first if (first.isascii() and (first.isalpha() or first == "_")) else "_"]
    out.extend(ch if _BASIC_CHAR_RE.match(ch) else "_" for ch in name[1:])
    return "".join(out)
