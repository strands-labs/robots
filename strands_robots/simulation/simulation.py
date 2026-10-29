"""One world. Owns time. Holds many robots. ``Simulation`` is backend-agnostic: it owns the ``Scene`` (plain data),
the robot registry and the time policy; a ``Backend`` implements a dozen methods and nothing else (MuJoCo is the
first). Adding a robot = append to the Scene + ``backend.load(scene)`` with joint state carried over for the
bodies that still exist. The scene-editing verbs live in the private ``_SceneOps`` base."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from . import _describe
from ._sceneops import _SceneOps
from .backend import Backend, load_backend
from .lock import WorldLock
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
    """One world. Owns time. Holds many robots. ``Simulation()`` is MuJoCo, headless; ``Simulation.default()`` is
    the process-wide world every Robot joins when it is given none."""

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
        ground: bool | str = True,
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
        if ground not in (True, False, "checker", "plain"):
            raise ValueError(f"ground={ground!r}; choose True, False, 'checker' or 'plain'")
        self.backend_name, self.physics_name = backend, physics
        self.headless, self.device = headless, device
        self.num_envs, self.render_size = num_envs, render_size
        self.scene = Scene(gravity=gravity, timestep=timestep, ground=ground)
        self._substeps = substeps
        self._robots: dict[str, Member] = {}
        self.revision = 0  # bumped on every scene change (add/remove/add_object/camera/set_pose/randomize)
        self.lock = WorldLock()  # one lock around the world: agents' act() and a dashboard's reads share it
        self._manual, self._editing, self._pending, self._gone = 0, 0, False, []
        self._built: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}  # each robot's frame as compiled
        self._closed = False
        cls = load_backend(backend, physics)
        self.backend: Backend = cls(
            physics=physics, headless=headless, device=device, num_envs=num_envs, render_size=render_size
        )
        self._load()

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
        self._open()
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        with self.lock:
            self._flush()
            self.backend.step(int(n))

    def control_step(self) -> None:
        """Advance exactly one control step (``physics.substeps`` physics steps) unless time is manual."""
        self._open()
        if not self._manual:
            with self.lock:
                self.backend.step(self._substeps)

    @contextmanager
    def manual_time(self) -> Iterator[Simulation]:
        """Inside: ``robot.act()`` only writes targets; the caller steps the world."""
        self._manual += 1
        try:
            yield self
        finally:
            self._manual -= 1

    @contextmanager
    def editing(self) -> Iterator[Simulation]:
        """Batch scene edits into one rebuild on exit (on newton a new scene shape compiles kernels). A block that
        fails, in its body or in its rebuild, restores the scene and robots as they were; step, render and set_pose
        outside a block flush pending edits first."""
        with self.lock:
            scene, robots, pending, gone = self.scene.copy(), dict(self._robots), self._pending, list(self._gone)
            self._editing += 1
            try:
                try:
                    yield self
                finally:
                    self._editing -= 1
                self._flush()
            except BaseException:  # the body raised (a KeyboardInterrupt too) or the exit rebuild did: restore in
                for rid in set(self._robots) - set(robots):  # place, the backend holds this same Scene object
                    self._robots.pop(rid).detached()  # added in the failed block: a dead handle, not half-in
                vars(self.scene).update(vars(scene))
                self._robots.update(robots)  # removed in it: back, and still live
                self._pending, self._gone = pending, gone  # a nested block owns only the edits since its entry
                raise

    def _reload(self) -> None:
        with self.lock:
            if self._editing:  # inside ``with sim.editing():`` the rebuild waits for the block's exit
                self._pending = True
                return
            self._load()
            self.revision += 1
            self._pending = False
            while self._gone:  # a robot removed inside ``editing()`` detaches once its rebuild holds
                self._gone.pop().detached()

    def _load(self) -> None:
        """The one rebuild: the backend compiles the scene, and the frame each robot was compiled from is kept (a
        deferred ``set_pose`` composes its delta onto that frame, not onto a scene entry an earlier call moved)."""
        self.backend.load(self.scene)
        self._built = {r.id: (tuple(r.position), tuple(r.orientation)) for r in self.scene.robots}

    def _flush(self) -> None:
        if self._pending and not self._editing:
            self._reload()

    @property
    def time_is_manual(self) -> bool:
        return self._manual > 0

    def reset(self, *, keyframe: str | None = None) -> None:
        """Every body back to its spawn state, or to ``keyframe`` where a robot has one."""
        self._open()
        named = {k for r in self._robots.values() for k in r.description.keyframes}
        if keyframe is not None and keyframe not in named:
            raise ValueError(f"keyframe {keyframe!r} is not on any robot in this world; keyframes are {sorted(named)}")
        with self.lock:
            self.backend.reset(keyframe)
            for r in self._robots.values():  # every body is back at its spawn pose: no old target survives
                r._held = None

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
        """Where a robot base, an object or a robot link (``"so101/gripper"``) is now: position m, quat w,x,y,z."""
        with self.lock:
            return self.backend.body_pose(_describe.known(self, body))

    @property
    def tool(self) -> Any:
        """This world as a strands agent tool, generated from its public verbs (imports strands on first use)."""
        from ..tools.agent import sim_tool

        return sim_tool(self)

    def describe(self) -> dict[str, Any]:
        """Robots (spawn frame, live ``pose``, links), objects where they are now, cameras, physics, time; JSON-safe."""
        with self.lock:
            return _describe.describe(self)

    def close(self) -> None:
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
