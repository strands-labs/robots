"""Read a model's joints, actuators, limits and keyframes from the XML directly."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def introspect_xml(path: Path, fmt: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    joints: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    floating = False
    if fmt == "urdf":
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
    for inc in root.iter("include"):  # one level of <include file=...>
        sub = path.parent / inc.get("file", "")
        if sub.is_file():
            root.extend(list(ET.parse(sub).getroot()))
    for j in root.iter("joint"):
        jt = j.get("type", "hinge")
        if jt == "free":
            floating = True
            continue
        if jt == "ball":
            continue
        name = j.get("name", f"joint{len(joints)}")
        joints.append(name)
        if j.get("range"):
            lo, hi = (float(x) for x in j.get("range", "0 0").split()[:2])
            limits[name] = (lo, hi)
    if root.find(".//freejoint") is not None:
        floating = True
    actuators, action_joints = [], []
    act = root.find("actuator")
    for i, a in enumerate(list(act) if act is not None else []):
        actuators.append(a.get("name", f"actuator{i}"))
        action_joints.append(a.get("joint", actuators[-1]))
    kf = root.find("keyframe")
    keyframes = [k.get("name", f"key{i}") for i, k in enumerate(list(kf) if kf is not None else [])]
    return {
        "joints": joints,
        "actuators": actuators,
        "action_joints": action_joints,
        "limits": limits,
        "keyframes": keyframes,
        "floating": floating,
    }
