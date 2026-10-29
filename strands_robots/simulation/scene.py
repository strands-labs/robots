"""Everything a backend needs to build the world, as plain data. No engine objects inside."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..description import Description

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_BACK = np.array([0.0, -1.0, 0.65]) / np.linalg.norm([0.0, -1.0, 0.65])  # the overview looks from -y, from above
_RIGHT, _UP = np.array([1.0, 0.0, 0.0]), np.cross(_BACK, [1.0, 0.0, 0.0])
_TAN = np.tan(np.radians(45.0 / 2))  # half the default vertical field of view


def _frame(points: np.ndarray, center: np.ndarray, aspect: float) -> tuple[float, np.ndarray]:
    """The distance along ``_BACK`` from ``center`` that keeps every point in view with a 10 % margin, and the
    points' image coordinates (right, up, in units of the half view) seen from there."""
    off = points - center
    side, high = np.abs(off @ _RIGHT) / (_TAN * aspect), np.abs(off @ _UP) / _TAN
    dist = 1.1 * max(float((off @ _BACK + np.maximum(side, high)).max()), 0.2)
    depth = dist - off @ _BACK
    return dist, np.array([off @ _RIGHT / (depth * _TAN * aspect), off @ _UP / (depth * _TAN)])


@dataclass
class SceneRobot:
    id: str
    description: Description
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = _IDENTITY_QUAT
    keyframe: str | None = None


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
    attach: str | None = None  # a robot id (its base carries the camera) or "<robot>/<link>" (a wrist camera)

    @property
    def robot(self) -> str | None:
        """The id of the robot this camera rides, base or link; None for a world camera."""
        return None if self.attach is None else self.attach.split("/", 1)[0]


@dataclass
class SceneLight:
    name: str
    position: tuple[float, float, float] = (0.0, 0.0, 2.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # diffuse RGB 0-1, scaled by randomize(lighting=)
    directional: bool = False  # True: a sun (parallel rays along direction); False: a spot at position


@dataclass
class Scene:
    """Everything a backend needs to build the world. No engine objects inside."""

    robots: list[SceneRobot] = field(default_factory=list)
    objects: list[SceneObject] = field(default_factory=list)
    cameras: list[SceneCamera] = field(default_factory=list)
    lights: list[SceneLight] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    ground: bool | str = True  # True/"checker" textured, "plain" flat grey, False none
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    light_scale: float = 1.0
    scales: dict[str, tuple[float, float]] = field(default_factory=dict)  # (friction, mass) of one robot or object
    camera_jitter: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # metres, per camera

    def overview(self, outline: np.ndarray | None = None, aspect: float = 4 / 3) -> dict[str, Any]:
        """A camera that sees every robot from the -y side. Given ``outline`` (world points around the robots'
        geometry) it frames them in the default 45 degree view (``aspect`` = width / height) with a 10 % margin;
        without one it guesses from the robots' base positions."""
        if outline is not None and len(outline):
            center = (outline.min(axis=0) + outline.max(axis=0)) / 2
            for _ in range(3):  # fit the distance, centre the projected extent, fit again
                dist, (x, y) = _frame(outline, center, aspect)
                mid_x, mid_y = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
                center = center + dist * _TAN * (_RIGHT * aspect * mid_x + _UP * mid_y)
            return {"position": tuple(center + _frame(outline, center, aspect)[0] * _BACK), "target": tuple(center)}
        pts = np.array([r.position for r in self.robots] or [(0.0, 0.0, 0.0)])
        center = pts.mean(axis=0) + np.array([0.0, 0.0, 0.4])
        spread = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 0.0
        dist = max(1.8, 1.4 * spread + 1.6)
        return {"position": tuple(center + np.array([0.0, -dist, 0.65 * dist])), "target": tuple(center)}

    def copy(self) -> Scene:
        """An independent copy of the plain data (descriptions are shared: they never change)."""
        return replace(
            self,
            robots=[replace(r) for r in self.robots],
            objects=[replace(o) for o in self.objects],
            cameras=[replace(c) for c in self.cameras],
            lights=[replace(li) for li in self.lights],
            scales=dict(self.scales),
            camera_jitter=dict(self.camera_jitter),
        )

    def robot(self, id: str) -> SceneRobot:
        for r in self.robots:
            if r.id == id:
                return r
        raise KeyError(f"no robot {id!r} in scene; robots are {[r.id for r in self.robots]}")
