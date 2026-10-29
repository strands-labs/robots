"""The contract a physics world implements, and the registry of engines."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import JointState, Pose
from .physics import Contact, Physics
from .scene import Scene, SceneCamera

_BACKENDS: dict[str, str] = {
    "mujoco": "strands_robots.simulation.backends.mujoco:MuJoCoBackend",
    "isaac": "strands_robots.simulation.backends.isaac:IsaacBackend",
    "mujoco+newton": "strands_robots.simulation.backends.newton:NewtonBackend",  # a solver of the MuJoCo world
}


class Backend(ABC):
    """The twelve abstract methods a physics world implements, plus two it may refuse with
    ``NotImplementedError``: ``set_gains`` and ``set_joints``, and three with an honest default: ``move_camera``,
    ``reach`` and ``links``. Nothing else is engine-specific."""

    deterministic = True  # the same scene and actions give a bit-identical trajectory, run to run

    def __init__(
        self,
        *,
        physics: str = "native",
        headless: bool = True,
        device: str | None = None,
        num_envs: int = 1,
        render_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.physics_name = physics
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.render_size = render_size

    @abstractmethod
    def load(self, scene: Scene) -> None: ...  # (re)build the world; keep joint state of surviving bodies

    @abstractmethod
    def step(self, n: int) -> None: ...

    @abstractmethod
    def reset(self, keyframe: str | None) -> None: ...

    @abstractmethod
    def time(self) -> float: ...

    @abstractmethod
    def joint_state(self, body: str) -> JointState: ...

    @abstractmethod
    def set_targets(self, body: str, targets: np.ndarray) -> None: ...

    @abstractmethod
    def body_pose(self, body: str) -> Pose: ...

    # teleport; a fixed body's new pose is already recorded in the scene, so the backend rebuilds it there
    @abstractmethod
    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None: ...

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        """Actuator PD gains for one robot (``action_keys`` order). Engines that cannot say so."""
        raise NotImplementedError(f"{type(self).__name__} cannot change actuator gains at runtime")

    def set_joints(self, body: str, positions: np.ndarray) -> None:
        """Joints AT ``positions``, at rest (how a hardware twin shows a measurement). Engines that cannot say so."""
        raise NotImplementedError(f"{type(self).__name__} cannot place joints directly")

    @abstractmethod
    def render(self, camera: str) -> np.ndarray: ...

    @abstractmethod
    def contacts(self) -> list[Contact]: ...

    @abstractmethod
    def physics(self) -> Physics: ...

    def move_camera(self, old: SceneCamera, cam: SceneCamera) -> bool:
        """Re-aim a built camera in place; False tells the world to rebuild instead (the default)."""
        return False

    def reach(self, body: str) -> dict[str, Any]:
        """A robot's farthest link and its motion per +1 rad of each joint (world m); empty if the engine cannot say."""
        return {}

    def links(self, body: str) -> list[str]:
        """The link (body) names of a robot, for ``camera(attach="<robot>/<link>")``; empty if the engine cannot say."""
        return []

    def outline(self) -> np.ndarray | None:
        """World points (N x 3) around every robot's visible geometry, for the overview camera to frame; None if
        the engine cannot say (the overview then guesses from base positions)."""
        return None

    @abstractmethod
    def close(self) -> None: ...


def load_backend(name: str, physics: str = "native") -> type[Backend]:
    """The engine class; a solver choice (``physics=``) may swap in its own subclass of that engine."""
    engines = sorted(k for k in _BACKENDS if "+" not in k)
    if name not in engines:
        raise ValueError(f"unknown simulation backend {name!r}; choose from {engines}")
    target = _BACKENDS.get(f"{name}+{physics}", _BACKENDS[name])
    mod_name, cls_name = target.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)
