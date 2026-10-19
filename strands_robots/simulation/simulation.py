"""One world. Owns time. Holds many robots.

``Simulation`` is backend-agnostic: it owns the ``Scene`` (plain data), the robot registry of the
world and the time policy. A ``Backend`` implements a dozen methods and nothing else; MuJoCo is the
first. Adding a robot = append to the Scene + ``backend.load(scene)`` with joint state carried over
for the bodies that still exist. The scene-editing verbs live in the private ``_SceneOps`` base.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from ._sceneops import _SceneOps
from .backend import Backend, load_backend
from .member import Member
from .physics import Physics
from .scene import Scene

if TYPE_CHECKING:
    from ..policy import PolicyLike
    from ..robot import Robot
    from ..robot.run.result import RunResult
    from ..types import Observation, Pose

_PHYSICS = ("native", "newton")


class Simulation(_SceneOps):
    """One world. Owns time. Holds many robots.

    ``Simulation()`` is MuJoCo, headless. ``Simulation.default()`` is the process-wide world
    every Robot joins when it is given none.
    """

    _default: ClassVar[Simulation | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        backend: str = "mujoco",
        *,
        physics: str = "native",
        timestep: float = 0.002,
        substeps: int = 10,
        headless: bool = True,
        device: str | None = None,
        num_envs: int = 1,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        ground: bool = True,
        render_size: tuple[int, int] = (640, 480),
    ) -> None:
        if physics not in _PHYSICS:
            raise ValueError(f"physics={physics!r}; choose from {_PHYSICS}")
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if num_envs > 1 and physics != "newton":
            raise ValueError(f"num_envs={num_envs} needs physics='newton'; {physics!r} steps one world")
        self.backend_name = backend
        self.physics_name = physics
        self.headless = headless
        self.device = device
        self.num_envs = num_envs
        self.render_size = render_size
        self.scene = Scene(gravity=gravity, timestep=timestep, ground=ground)
        self._substeps = substeps
        self._robots: dict[str, Member] = {}
        self.revision = 0  # bumped on every scene change (add/remove/add_object/camera/set_pose/randomize)
        self.lock = threading.RLock()  # one lock around the world: agents' act() and a dashboard's reads share it
        self._manual = 0
        self._closed = False
        cls = load_backend(backend, physics)
        self.backend: Backend = cls(
            physics=physics, headless=headless, device=device, num_envs=num_envs, render_size=render_size
        )
        self.backend.load(self.scene)

    @classmethod
    def default(cls) -> Simulation:
        """The world a Robot attaches to when none is given: MuJoCo, headless, created on first use."""
        with cls._lock:
            if cls._default is None or cls._default._closed:
                cls._default = cls()
            return cls._default

    @property
    def is_default(self) -> bool:
        return Simulation._default is self

    def step(self, n: int = 1) -> None:
        """Advance ``n`` physics steps (``timestep`` each)."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        with self.lock:
            self._open()  # under the lock close() takes: never a step on a model another thread just closed
            self.backend.step(int(n))

    def control_step(self) -> None:
        """Advance exactly one control step (``physics.substeps`` physics steps) unless time is manual."""
        with self.lock:
            self._open()
            if not self._manual:
                self.backend.step(self._substeps)

    @contextmanager
    def manual_time(self) -> Iterator[Simulation]:
        """Inside: ``robot.act()`` only writes targets; the caller steps the world."""
        self._manual += 1
        try:
            yield self
        finally:
            self._manual -= 1

    @property
    def time_is_manual(self) -> bool:
        return self._manual > 0

    def reset(self, *, keyframe: str | None = None) -> None:
        """Every body back to its spawn state, or to ``keyframe`` where a robot has one."""
        with self.lock:  # the keyframe check too: close() empties self._robots under this lock
            self._open()
            named = {k for r in self._robots.values() for k in r.description.keyframes}
            if keyframe is not None and keyframe not in named:
                raise ValueError(
                    f"keyframe {keyframe!r} is not on any robot in this world; keyframes are {sorted(named)}"
                )
            self.backend.reset(keyframe)

    @property
    def time(self) -> float:
        return self.backend.time()

    @property
    def physics(self) -> Physics:
        p = self.backend.physics()
        return Physics(p.name, p.timestep, self._substeps)

    @property
    def robots(self) -> Mapping[str, Member]:
        return dict(self._robots)

    @property
    def envs(self) -> int:
        return self.num_envs

    def run(
        self,
        policies: Mapping[str | Robot, PolicyLike | Callable[[Observation], Any]],
        *,
        hz: float | None = None,
        seconds: float | None = None,
        steps: int | None = None,
        instruction: str | None = None,
        units: str | None = "auto",
    ) -> dict[str, RunResult]:
        """Drive N robots in lock-step (robot id -> policy spec): each observes and acts, then the world steps once."""
        from ..robot.run import lockstep

        self._open()
        unknown = [r for r in policies if isinstance(r, str) and r not in self._robots]
        if unknown:
            raise KeyError(f"no robot {unknown} in this world; robots are {list(self._robots)}")
        bodies = {(self._robots[r] if isinstance(r, str) else r): p for r, p in policies.items()}
        return lockstep(self, bodies, hz=hz, seconds=seconds, steps=steps, instruction=instruction, units=units)  # type: ignore[arg-type]

    def pose(self, body: str) -> Pose:
        """Where a robot base or object is now: position m, orientation quaternion w,x,y,z."""
        if body not in self._robots and not any(o.name == body for o in self.scene.objects):
            raise KeyError(f"no body {body!r} in this world")
        with self.lock:
            return self.backend.body_pose(body)

    @property
    def tool(self) -> Any:
        """This world as a strands agent tool, generated from its public verbs (imports strands on first use)."""
        from ..tools.agent import sim_tool

        return sim_tool(self)

    def describe(self) -> dict[str, Any]:
        """Robots (spawn frame + live ``pose``), objects where they are now, cameras, physics, time; JSON-safe."""
        with self.lock:
            return self._describe()

    def _describe(self) -> dict[str, Any]:
        p = self.physics
        return {
            "backend": self.backend_name,
            "physics": {"name": p.name, "timestep": p.timestep, "substeps": p.substeps, "envs": self.num_envs},
            "time": self.time,
            "revision": self.revision,
            "manual_time": self.time_is_manual,
            "robots": {
                rid: r.description.describe()
                | {"position": list(self.scene.robot(rid).position), "pose": self._where(rid)}
                for rid, r in self._robots.items()
            },
            "objects": [{"name": o.name, "shape": o.shape or o.path} | self._where(o.name) for o in self.scene.objects],
            "cameras": [{"name": c.name, "size": list(c.size), "attach": c.attach} for c in self.scene.cameras],
        }

    def _where(self, body: str) -> dict[str, list[float]]:
        p = self.backend.body_pose(body)
        return {"position": p.position.tolist(), "orientation": p.orientation.tolist()}

    def close(self) -> None:
        with self.lock:  # a step, a rebuild or a twin's mirror in another thread ends first; none starts after it
            if self._closed:
                return
            self._closed = True
            for r in list(self._robots.values()):
                r.detached()
            self._robots.clear()
            self.backend.close()
        if Simulation._default is self:
            Simulation._default = None

    def __enter__(self) -> Simulation:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Simulation({self.backend_name!r}, physics={self.physics_name!r}, "
            f"robots={list(self._robots)}, t={self.time:.3f})"
        )
