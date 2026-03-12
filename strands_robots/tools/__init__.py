#!/usr/bin/env python3
"""
Strands Robotics Tools

Collection of specialized tools for robot control, camera management,
teleoperation, inference services, dataset management, serial communication,
robot mesh coordination, 3D world generation, and telemetry streaming.

Tools are lazily imported to avoid pulling in heavy dependencies (strands SDK,
torch, etc.) until actually needed. This preserves the ability to import
core strands_robots functionality (Robot, Policy, resolve_policy) without
requiring every optional dependency.
"""

_LAZY_IMPORTS: dict[str, str] = {
    "gr00t_inference": ".gr00t_inference",
    "inference": ".inference",
    "lerobot_camera": ".lerobot_camera",
    "pose_tool": ".pose_tool",
    "serial_tool": ".serial_tool",
    "teleoperator": ".teleoperator",
    "lerobot_dataset": ".lerobot_dataset",
    "robot_mesh": ".robot_mesh",
    "reachy_mini": ".reachy_mini_tool",
    "newton_sim": ".newton_sim",
    "isaac_sim": ".isaac_sim",
    "stream": ".stream",
    "use_lerobot": ".use_lerobot",
    "use_unitree": ".use_unitree",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr  # Cache for subsequent access
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
