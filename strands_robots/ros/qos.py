"""QoS by name, never by a bare depth: ``sensor``, ``reliable``, ``latched``, ``deep`` or ``auto``.

The old ROS code only took a depth, so every reader was RELIABLE and heard nothing from a BEST_EFFORT
writer (cameras, IMUs, most low-level state streams). ``auto`` copies what the live writers offer.
"""

from __future__ import annotations

from typing import Any

PROFILES = ("sensor", "reliable", "latched", "deep", "auto")


def profile(name: str, *, depth: int | None = None) -> Any:
    """The rclpy ``QoSProfile`` for a name; ``depth`` overrides the history depth of a named profile."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    if name not in PROFILES or name == "auto":
        raise ValueError(f"qos {name!r} is not one of {PROFILES[:-1]} (resolve 'auto' with resolve())")
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or not 0 < depth < 2**31):
        raise ValueError(f"qos depth must be an int in 1..2**31-1, got {depth!r}")
    rel, dur, d = {
        "sensor": (ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE, 5),
        "reliable": (ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE, 10),
        "latched": (ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL, 1),
        "deep": (ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE, 200),
    }[name]
    return QoSProfile(depth=depth or d, reliability=rel, durability=dur, history=HistoryPolicy.KEEP_LAST)


def resolve(node: Any, topic: str, name: str) -> str:
    """``auto`` -> the named profile that matches every live writer of ``topic`` (``reliable`` with none).

    A BEST_EFFORT writer anywhere means ``sensor`` (a RELIABLE reader would miss it); all writers
    TRANSIENT_LOCAL means ``latched`` (a late reader still gets the last message).
    """
    if name != "auto":
        return name
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

    infos = list(node.get_publishers_info_by_topic(topic) or ())
    if not infos:
        return "reliable"
    if any(i.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT for i in infos):
        return "sensor"
    if all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for i in infos):
        return "latched"
    return "reliable"
