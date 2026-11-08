"""``ros.connect(profile, host=...)`` -> a strands-robots ``Robot`` on any ROS 2 robot.

    g1 = ros.connect("unitree_g1", released=True)            # on the G1's Jetson (eth0 DDS), low-level joints
    arm = ros.connect("ros2_control", host="arm.local")      # any ros2_control robot, model from /robot_description

The model comes from ``model=`` (a path or registry name), else the profile's registry entry, else the
robot's own ``/robot_description``. The Robot is a normal hardware Robot: safety chain, e-stop latch,
watchdog, run loop, mesh mount and dashboard all apply unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import profile as _profile
from .bus import RosBus
from .describe import fetch_urdf
from .discovery import Discovery


def model_source(prof: _profile.Profile, model: str | Path | None, *, host: str | None, timeout: float) -> str | Path:
    """A model file path, a registry name, or (``model="/topic"`` or the profile's ``model.topic``) the
    URDF the robot publishes, fetched and cached as a file."""
    spec = dict(prof.data.get("model") or {})
    if isinstance(model, Path) or (
        isinstance(model, str) and (model.endswith((".urdf", ".xml")) or Path(model).is_file())
    ):
        return Path(model)
    if isinstance(model, str) and not model.startswith("/"):
        return model  # a registry name
    if model is None and spec.get("registry"):
        return str(spec["registry"])
    topic = model or spec.get("topic")
    if not topic:
        raise ValueError(f"ROS profile {prof.name!r} names no model; pass model=<URDF/MJCF path or registry name>")
    d = prof.discovery
    bus = RosBus.shared(
        Discovery(
            host=host,
            rmw=d.get("rmw"),
            domain_id=d.get("domain_id"),
            zenoh_port=int(d.get("zenoh_port", 7447)),
            iface=d.get("iface"),
        )
    )
    return fetch_urdf(
        bus,
        topic,
        type=spec.get("type", "std_msgs/msg/String"),
        qos=spec.get("qos", "latched"),
        timeout=timeout,
        name=prof.name,
    )


def connect(
    profile: str | Path = "ros2_control",
    *,
    host: str | None = None,
    model: str | Path | None = None,
    id: str | None = None,
    mesh: Any = None,
    confirm: Any = None,
    approve: Any = None,
    timeout: float = 5.0,
    **driver_options: Any,
) -> Any:
    """A hardware ``Robot`` over ROS. ``driver_options`` go to ``RosDriver`` (shadow=, released=, kp=, kd=,
    domain_id=, overrides=)."""
    from ..description import Description
    from ..robot import Robot
    from ..robot.drivers.ros import RosDriver

    prof = _profile.load(profile, overrides=driver_options.get("overrides"))
    src = model_source(prof, model, host=host, timeout=timeout)
    desc = Description.load(src)
    drv = RosDriver(desc, host=host, profile=str(profile), timeout=timeout, **driver_options)
    return Robot(
        str(src),
        driver=drv,
        id=id or prof.name,
        mesh=mesh,
        confirm=confirm,
        approve=approve,
    )
