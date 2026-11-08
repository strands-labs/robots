"""A robot's model from its own ROS graph: the URDF on ``/robot_description`` becomes a Description file.

Only joints and limits are needed, so ``<visual>``/``<collision>`` (``package://`` meshes this machine
does not have) are stripped, and a link with no ``<inertial>`` gets a 1 g placeholder so MuJoCo (when it is
installed) compiles the file. Never use this cached file for physics; it is a joint list with limits.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from .bus import RosBus

CACHE = Path(os.environ.get("STRANDS_ROBOTS_CACHE", "~/.cache/strands_robots")).expanduser() / "ros"


def fetch_urdf(
    bus: RosBus,
    topic: str,
    *,
    type: str = "std_msgs/msg/String",
    qos: str = "latched",
    timeout: float = 5.0,
    name: str = "robot",
) -> Path:
    """Wait for the URDF on ``topic`` (latched), strip meshes, cache it, and return the file path."""
    msg = bus.wait_first(topic, type, timeout, qos=qos)
    if msg is None:
        raise ConnectionError(
            f"no robot description on {topic} within {timeout}s; pass model= a URDF/MJCF path instead"
        )
    return cache_urdf(str(msg.data), name=name)


def cache_urdf(text: str, *, name: str = "robot") -> Path:
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError(f"the robot description is not a URDF (root <{root.tag}>)")
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for el in list(link.findall(tag)):
                link.remove(el)
        if link.find("inertial") is None:  # MuJoCo refuses massless moving bodies; this file is only read for joints
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.001")
            ET.SubElement(inertial, "inertia", ixx="1e-6", iyy="1e-6", izz="1e-6", ixy="0", ixz="0", iyz="0")
    body = ET.tostring(root, encoding="unicode")
    digest = hashlib.sha256(body.encode()).hexdigest()[:10]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}-{digest}.urdf"
    if not path.is_file():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body)
        tmp.replace(path)
    return path
