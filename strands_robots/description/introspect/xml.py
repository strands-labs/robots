"""Read a model's joints, actuators, limits and keyframes from the XML directly.

The MJCF walk computes joint limits the way MuJoCo's compiler does: ``<include>`` files are spliced in where they
appear, joint attributes come from the ``<default>`` class tree (a joint's ``class``, else the nearest enclosing
``childclass``, else ``main``), ``limited`` / ``autolimits`` decide whether a range applies, and hinge ranges are in
degrees unless ``<compiler angle="radian">`` says otherwise (degrees are MJCF's default). An ``<attach>``ed model's
joints sit where it is attached, read with its own ``<compiler>``, names prefixed. An actuator whose joint the walk
cannot find refuses the model rather than leave that joint without limits.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_MAX_INCLUDE_DEPTH = 8
_DIR = "{strands_robots}dir"  # a namespaced key no MJCF attribute can have


def _introspect_urdf(root: ET.Element) -> dict[str, Any]:
    joints: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    floating = False
    for j in root.findall("joint"):  # direct children only: <ros2_control>/<transmission> name joints too
        jt = j.get("type", "revolute")
        if jt == "fixed":
            continue
        if jt == "floating":
            floating = True
            continue
        name = j.get("name", f"joint{len(joints)}")
        if name in joints:
            raise ValueError(f"the URDF names joint {name!r} twice")
        joints.append(name)
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
            limits[name] = (float(lim.get("lower", 0)), float(lim.get("upper", 0)))
    return {
        "joints": joints,
        "actuators": list(joints),
        "action_joints": list(joints),
        "limits": limits,
        "keyframes": [],
        "floating": floating,
    }


def _splice_includes(el: ET.Element, base: Path, depth: int = 0) -> None:
    """Replace every ``<include file=...>`` below ``el`` with the children of the file it names. As in MuJoCo, every
    file, nested or not, is relative to the main model's directory ``base``."""
    for i, child in reversed(list(enumerate(el))):
        if child.tag != "include":
            _splice_includes(child, base, depth)
            continue
        el.remove(child)
        sub = base / child.get("file", "")
        if not sub.is_file() or depth >= _MAX_INCLUDE_DEPTH:
            continue
        inc = ET.parse(sub).getroot()
        _splice_includes(inc, base, depth + 1)
        for m in inc.iter("model"):  # an attached model's file is found next to the file that names it
            m.attrib.setdefault(_DIR, str(sub.parent))
        for k, grand in enumerate(list(inc)):
            el.insert(i + k, grand)


def _default_classes(root: ET.Element) -> dict[str, dict[str, str]]:
    """Joint attributes per default class, each class inheriting from the class it is nested in."""
    classes: dict[str, dict[str, str]] = {"main": {}}

    def walk(el: ET.Element, inherited: dict[str, str]) -> None:
        attrs = dict(inherited)
        for j in el.findall("joint"):
            attrs.update(j.attrib)
        classes[el.get("class", "main")] = attrs
        for child in el.findall("default"):
            walk(child, attrs)

    for top in root.findall("default"):
        walk(top, classes["main"])
    return classes


def _mjcf_joints(root: ET.Element, subtree: str | None = None) -> tuple[list[dict[str, str]], bool]:
    """Every ``<joint>`` in the kinematic tree (or below the body named ``subtree``) with its default-class attributes
    applied, in tree order; and whether one is free. An ``<attach>`` leaves a ``{"__attach__": ...}`` placeholder."""
    classes = _default_classes(root)
    out: list[dict[str, str]] = []
    floating = False

    def walk(el: ET.Element, childclass: str, on: bool) -> None:
        nonlocal floating
        for child in el:
            if on and child.tag == "freejoint":
                floating = True
            elif on and child.tag == "joint":
                cls = child.get("class") or childclass
                out.append({**classes.get(cls, classes["main"]), **child.attrib})
            elif on and child.tag == "attach":
                out.append({"__attach__": "1", **child.attrib})
            elif child.tag in ("body", "frame", "replicate"):
                inside = on or (child.tag == "body" and child.get("name") == subtree)
                walk(child, child.get("childclass") or childclass, inside)

    for wb in root.findall("worldbody"):
        walk(wb, wb.get("childclass") or "main", subtree is None)
    return out, floating


def _limit(attrs: dict[str, str], degrees: bool, autolimits: bool) -> tuple[float, float] | None:
    limited, rng = attrs.get("limited", "auto"), attrs.get("range")
    if limited == "false" or (limited == "auto" and (rng is None or not autolimits)):
        return None
    lo, hi = (float(x) for x in (rng or "0 0").split()[:2])
    if limited == "auto" and lo >= hi:  # MuJoCo: auto means limited only when range[0] < range[1]
        return None
    if attrs.get("type", "hinge") == "hinge" and degrees:
        lo, hi = math.radians(lo), math.radians(hi)
    return lo, hi


def _walk_mjcf(path: Path, subtree: str | None = None, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """One MJCF file (or the subtree an ``<attach>`` takes from it): its own ``<compiler>`` applies to its joints, and
    an attached model's joints sit where it is attached, its actuators after the file's own (as MuJoCo compiles it)."""
    root = ET.parse(path).getroot()
    _splice_includes(root, path.parent)
    compiler: dict[str, str] = {}
    for c in root.iter("compiler"):
        compiler.update(c.attrib)
    degrees = compiler.get("angle", "degree") != "radian"
    autolimits = compiler.get("autolimits", "true") != "false"
    models: dict[str, Path | None] = {}
    for m in (m for a in root.findall("asset") for m in a.findall("model")):
        where = [Path(m.get(_DIR, path.parent)) / m.get("file", ""), path.parent / m.get("file", "")]
        models[m.get("name", "")] = next((w for w in where if w.is_file()), None)
    found, floating = _mjcf_joints(root, subtree)
    joints: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    attached: list[tuple[str, str]] = []
    for attrs in found:
        if "__attach__" in attrs:
            sub = models.get(attrs.get("model", ""))
            if sub is None or depth >= _MAX_INCLUDE_DEPTH:
                raise ValueError(f"{path.name}: cannot read the model attached as {attrs.get('model')!r}")
            inner = _walk_mjcf(sub, attrs.get("body"), prefix + attrs.get("prefix", ""), depth + 1)
            joints += inner["joints"]
            limits.update(inner["limits"])
            floating = floating or inner["floating"]
            attached += zip(inner["actuators"], inner["action_joints"], strict=True)
            continue
        jt = attrs.get("type", "hinge")
        if jt == "free":
            floating = True
            continue
        if jt == "ball":
            continue
        name = prefix + attrs.get("name", f"joint{len(joints)}")
        joints.append(name)
        lim = _limit(attrs, degrees, autolimits)
        if lim is not None:
            limits[name] = lim
    own = set(joints) - {j for _, j in attached}
    actuators: list[str] = []
    action_joints: list[str] = []
    for act in root.findall("actuator"):
        for a in act:
            joint = a.get("joint")
            if subtree is not None and (joint is None or prefix + joint not in own):
                continue
            actuators.append(prefix + a.get("name", f"actuator{len(actuators)}"))
            action_joints.append(prefix + joint if joint is not None else actuators[-1])
    for name, joint in attached:
        actuators.append(name)
        action_joints.append(joint)
    keyframes: list[str] = []
    for kf in root.findall("keyframe"):
        keyframes += [k.get("name", f"key{len(keyframes) + i}") for i, k in enumerate(kf)]
    return {
        "joints": joints,
        "actuators": actuators,
        "action_joints": action_joints,
        "limits": limits,
        "keyframes": keyframes,
        "floating": floating,
    }


def introspect_xml(path: Path, fmt: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if fmt == "urdf":
        return _introspect_urdf(root)
    out = _walk_mjcf(path)
    known = set(out["joints"])
    for name, joint in zip(out["actuators"], out["action_joints"], strict=True):
        if joint != name and joint not in known:
            raise ValueError(
                f"{path.name}: actuator {name!r} drives joint {joint!r}, which the XML walk cannot find; "
                f"its limits would be unknown (install the mujoco extra to read this model)"
            )
    return out
