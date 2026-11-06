"""Read a model's joints, actuators, limits and keyframes from the XML directly.

The MJCF walk computes joint limits the way MuJoCo's compiler does: ``<include>`` files are spliced in where they
appear, joint attributes come from the ``<default>`` class tree (a joint's ``class``, else the nearest enclosing
``childclass``, else ``main``), ``limited`` / ``autolimits`` decide whether a range applies, and hinge ranges are in
degrees unless ``<compiler angle="radian">`` says otherwise (degrees are MJCF's default).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_MAX_INCLUDE_DEPTH = 8


def _introspect_urdf(root: ET.Element) -> dict[str, Any]:
    joints: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    floating = False
    for j in root.iter("joint"):
        jt = j.get("type", "revolute")
        if jt == "fixed":
            continue
        if jt == "floating":
            floating = True
            continue
        name = j.get("name", f"joint{len(joints)}")
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
    """Replace every ``<include file=...>`` below ``el`` with the children of the file it names."""
    for i, child in reversed(list(enumerate(el))):
        if child.tag != "include":
            _splice_includes(child, base, depth)
            continue
        el.remove(child)
        sub = base / child.get("file", "")
        if not sub.is_file() or depth >= _MAX_INCLUDE_DEPTH:
            continue
        inc = ET.parse(sub).getroot()
        _splice_includes(inc, sub.parent, depth + 1)
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


def _mjcf_joints(root: ET.Element) -> tuple[list[dict[str, str]], bool]:
    """Every ``<joint>`` in the kinematic tree with its default-class attributes applied; and whether one is free."""
    classes = _default_classes(root)
    out: list[dict[str, str]] = []
    floating = False

    def walk(el: ET.Element, childclass: str) -> None:
        nonlocal floating
        for child in el:
            if child.tag == "freejoint":
                floating = True
            elif child.tag == "joint":
                cls = child.get("class") or childclass
                out.append({**classes.get(cls, classes["main"]), **child.attrib})
            elif child.tag in ("body", "frame", "replicate"):
                walk(child, child.get("childclass") or childclass)

    for wb in root.findall("worldbody"):
        walk(wb, wb.get("childclass") or "main")
    return out, floating


def introspect_xml(path: Path, fmt: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if fmt == "urdf":
        return _introspect_urdf(root)
    _splice_includes(root, path.parent)
    compiler: dict[str, str] = {}
    for c in root.iter("compiler"):
        compiler.update(c.attrib)
    degrees = compiler.get("angle", "degree") != "radian"
    autolimits = compiler.get("autolimits", "true") != "false"
    found, floating = _mjcf_joints(root)
    joints: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    for attrs in found:
        jt = attrs.get("type", "hinge")
        if jt == "free":
            floating = True
            continue
        if jt == "ball":
            continue
        name = attrs.get("name", f"joint{len(joints)}")
        joints.append(name)
        limited = attrs.get("limited", "auto")
        rng = attrs.get("range")
        if limited == "false" or (limited == "auto" and (rng is None or not autolimits)):
            continue
        lo, hi = (float(x) for x in (rng or "0 0").split()[:2])
        if limited == "auto" and lo >= hi:  # MuJoCo: auto means limited only when range[0] < range[1]
            continue
        if jt == "hinge" and degrees:
            lo, hi = math.radians(lo), math.radians(hi)
        limits[name] = (lo, hi)
    actuators: list[str] = []
    action_joints: list[str] = []
    for act in root.findall("actuator"):
        for a in act:
            actuators.append(a.get("name", f"actuator{len(actuators)}"))
            action_joints.append(a.get("joint", actuators[-1]))
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
