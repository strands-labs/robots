"""A handle to a scene object: ``.pose`` / ``.set_pose`` like a Robot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..types import Pose
from .scene import SceneObject

if TYPE_CHECKING:
    from .simulation import Simulation


class Object:
    """A handle to a scene object: ``.pose`` / ``.set_pose`` like a Robot."""

    def __init__(self, sim: Simulation, spec: SceneObject) -> None:
        self.sim = sim
        self.spec = spec
        self.name = spec.name

    @property
    def pose(self) -> Pose:
        with self.sim.lock:
            return self.sim.backend.body_pose(self.name)

    def set_pose(self, position: Any = None, orientation: Any = None) -> Pose:
        """Teleport this object; returns where it is (``Simulation.set_pose``)."""
        return self.sim.set_pose(self.name, position=position, orientation=orientation)

    def __repr__(self) -> str:
        return f"Object({self.name!r}, {self.spec.shape or self.spec.path})"
