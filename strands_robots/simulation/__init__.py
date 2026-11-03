"""One world: a Simulation owns time and holds many robots; a Backend steps the physics.

MuJoCo is the first backend and is imported lazily (``load_backend``), so ``import strands_robots``
never pulls in ``mujoco``.
"""

from __future__ import annotations

from .backend import Backend
from .object import Object
from .physics import Contact, Physics
from .scene import Scene, SceneCamera, SceneLight, SceneObject, SceneRobot
from .simulation import Simulation

__all__ = [
    "Simulation",
    "Scene",
    "SceneRobot",
    "SceneObject",
    "SceneCamera",
    "SceneLight",
    "Object",
    "Backend",
    "Physics",
    "Contact",
]
