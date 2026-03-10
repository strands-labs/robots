#!/usr/bin/env python3
"""
Strands Robotics Tools

Collection of specialized tools for robot control, camera management,
teleoperation, inference services, and serial communication.

Tools use lazy imports to avoid pulling in heavy dependencies (torch, lerobot,
etc.) until actually needed. Import any tool directly::

    from strands_robots.tools import inference
    from strands_robots.tools import use_lerobot
"""

import importlib
from typing import TYPE_CHECKING

# Lazy-import mapping: name → relative module
_LAZY_IMPORTS: dict[str, str] = {
    "gr00t_inference": ".gr00t_inference",
    "inference": ".inference",
    "isaac_sim": ".isaac_sim",
    "lerobot_calibrate": ".lerobot_calibrate",
    "lerobot_camera": ".lerobot_camera",
    "lerobot_dataset": ".lerobot_dataset",
    "lerobot_teleoperate": ".lerobot_teleoperate",
    "marble_tool": ".marble_tool",
    "newton_sim": ".newton_sim",
    "pose_tool": ".pose_tool",
    "reachy_mini_tool": ".reachy_mini_tool",
    "reachy_mini": ".reachy_mini_tool",  # alias for the @tool function name
    "robot_mesh": ".robot_mesh",
    "serial_tool": ".serial_tool",
    "stereo_depth": ".stereo_depth",
    "stream": ".stream",
    "teleoperator": ".teleoperator",
    "use_lerobot": ".use_lerobot",
    "use_unitree": ".use_unitree",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .gr00t_inference import gr00t_inference  # noqa: F401
    from .inference import inference  # noqa: F401
    from .isaac_sim import isaac_sim  # noqa: F401
    from .lerobot_calibrate import lerobot_calibrate  # noqa: F401
    from .lerobot_camera import lerobot_camera  # noqa: F401
    from .lerobot_dataset import lerobot_dataset  # noqa: F401
    from .lerobot_teleoperate import lerobot_teleoperate  # noqa: F401
    from .marble_tool import marble_tool  # noqa: F401
    from .newton_sim import newton_sim  # noqa: F401
    from .pose_tool import pose_tool  # noqa: F401
    from .reachy_mini_tool import reachy_mini_tool  # noqa: F401
    from .robot_mesh import robot_mesh  # noqa: F401
    from .serial_tool import serial_tool  # noqa: F401
    from .stereo_depth import stereo_depth  # noqa: F401
    from .stream import stream  # noqa: F401
    from .teleoperator import teleoperator  # noqa: F401
    from .use_lerobot import use_lerobot  # noqa: F401
    from .use_unitree import use_unitree  # noqa: F401
