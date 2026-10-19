"""Everything a backend needs to build the world, as plain data. No engine objects inside."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..description import Description

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SceneRobot:
    id: str
    description: Description
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = _IDENTITY_QUAT
    keyframe: str | None = None
    _twin: bool = False  # a hardware Robot's twin (robot/twin.py marks it): it touches the floor and nothing else


@dataclass
class SceneObject:
    name: str
    shape: str | None = None  # box | sphere | cylinder | capsule
    path: str | None = None  # a mesh/MJCF/URDF file instead of a primitive
    size: tuple[float, ...] = (0.05, 0.05, 0.05)
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    orientation: tuple[float, float, float, float] = _IDENTITY_QUAT
    mass: float | None = None
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    free: bool = True


@dataclass
class SceneCamera:
    name: str
    position: tuple[float, float, float] = (1.5, -1.5, 1.2)
    target: tuple[float, float, float] | None = (0.0, 0.0, 0.3)
    orientation: tuple[float, float, float, float] | None = None
    fov: float = 45.0
    size: tuple[int, int] = (640, 480)
    attach: str | None = None  # robot id whose base body carries the camera


@dataclass
class Scene:
    """Everything a backend needs to build the world. No engine objects inside."""

    robots: list[SceneRobot] = field(default_factory=list)
    objects: list[SceneObject] = field(default_factory=list)
    cameras: list[SceneCamera] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    ground: bool = True
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    light_scale: float = 1.0

    def robot(self, id: str) -> SceneRobot:
        for r in self.robots:
            if r.id == id:
                return r
        raise KeyError(f"no robot {id!r} in scene; robots are {[r.id for r in self.robots]}")
