"""MuJoCo Simulation — programmatic simulation environment for Strands Robots.

An AgentTool that gives AI agents full control over MuJoCo simulation environments.
Agents can create worlds, add robots (via URDF), add objects, run policies, observe,
and adjust — all through natural language mapped to tool actions.

Mirrors robot.py's architecture but targets simulated robots instead of real hardware.
Shares the same Policy ABC so policies trained in sim transfer to real and vice versa.
"""

from ._builder import MJCFBuilder
from ._core import MujocoBackend
from ._registry import (
    list_available_models,
    list_registered_urdfs,
    register_urdf,
    resolve_model,
    resolve_urdf,
)
from ._types import (
    SimCamera,
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)

__all__ = [
    "MujocoBackend",
    "SimStatus",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "SimWorld",
    "TrajectoryStep",
    "MJCFBuilder",
    "register_urdf",
    "resolve_model",
    "resolve_urdf",
    "list_registered_urdfs",
    "list_available_models",
]
