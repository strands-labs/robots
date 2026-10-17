"""A Robot is a description plus a Body: observe() / act() / calibrate() / run().

The Body is the only thing that differs between a simulated robot (``SimBody``, a handle into a
``Simulation``) and a real one (a ``Driver`` on a bus). Every loop is written once against ``Robot``.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..description import Description
from ..types import Action, JointState, Observation, Pose
from . import gate as _gate
from . import twin as _twin
from .body import Body, SimBody
from .device import DeviceVerbs, Emitter, position_hold_of
from .safety import Latch, Validator, guarded_write

if TYPE_CHECKING:
    from ..policy import PolicyLike
    from ..simulation import Simulation
    from .drivers.calibration import Calibration
    from .run.result import RunResult
    from .run.task import Task


class Robot(DeviceVerbs):
    """A body in a world (``sim=``, or ``Simulation.default()``) or on a bus (``port=``/``driver=``): same verbs."""

    def __init__(
        self,
        name: str,
        *,
        sim: Simulation | None = None,
        position: Any = None,
        orientation: Any = None,
        keyframe: str | None = None,
        id: str | None = None,
        port: str | None = None,
        driver: str | Any | None = None,
        cameras: Mapping[str, int | str] | None = None,
        twin: bool = True,
        confirm: Callable[[np.ndarray], bool] | None = None,
        mesh: Any = None,
    ) -> None:
        """``mesh=`` (a ``strands_robots.mesh.Mesh``) mounts this robot as a device on it once it is built."""
        from ..simulation import Simulation

        self.description: Description = Description.load(name)
        self.name = self.description.name
        self.is_real = port is not None or driver is not None
        if confirm is not None and not self.is_real:
            raise TypeError("confirm= is the operator gate of a hardware Robot; a simulated body is never gated")
        self.confirm = confirm
        self.sim: Simulation | None = None
        self.twin: Robot | None = None  # on a hardware Robot: the simulated body that mirrors it
        self.twin_of: Robot | None = None  # on that twin: the hardware Robot it mirrors
        self._task: Task | None = None
        self._body: Body | None = None
        self._bus_lock = threading.RLock()  # a hardware body: one half-duplex bus shared by every thread
        self.estop_latch, self.events = Latch(), Emitter(self)
        self.validators: list[Validator] = []  # extra checks on every agent/dashboard write, after the interlocks
        self.interlocks: list[Validator] = []  # append a Validator to either list; it sees every target, in order

        if self.is_real:
            if sim is not None:
                raise TypeError(
                    "a hardware Robot takes port=/driver=, not sim= (its twin lives in Simulation.default())"
                )
            from .drivers import make_driver

            self.id = id or self.name
            self._body = make_driver(self.description, port=port, driver=driver, cameras=cameras)
            self.supports_position_hold = position_hold_of(self.description, self._body)
            if twin:
                self.twin = Robot(
                    name, position=position, orientation=orientation, keyframe=keyframe, id=f"{self.id}_twin"
                )
                self.twin.twin_of = self
        else:
            if cameras is not None:
                raise TypeError("cameras= is a hardware argument; in a world use sim.camera(name, attach=robot)")
            world = sim if sim is not None else Simulation.default()
            self.id = world.unique_id(id or self.name)
            self.sim = world
            world.add(self, position=position, orientation=orientation, keyframe=keyframe)
            self._body = SimBody(world, self.id, self.description.action_keys)
            self.supports_position_hold = True  # a simulated actuator keeps its last target
        if mesh is not None:
            mesh.mount(self)

    @property
    def body(self) -> Body:
        if self._body is None:
            raise RuntimeError(f"robot {self.id!r} has been removed from its world")
        return self._body

    @property
    def _lock(self) -> Any:
        """The world's lock in a world; on hardware, this body's own lock around its bus."""
        return self.sim.lock if self.sim is not None else self._bus_lock

    def detached(self) -> None:
        self._body = None
        self.sim = None

    @property
    def action_keys(self) -> tuple[str, ...]:
        return self.description.action_keys

    @property
    def pose(self) -> Pose:
        with self._lock:
            p = self.body.pose()
        if p is None:
            raise RuntimeError(f"{self.id}: this body does not report a base pose")
        return p

    def set_pose(self, position: Any = None, orientation: Any = None) -> None:
        """Teleport (sim only). A hardware robot raises; its twin can still be teleported."""
        if self.sim is None:
            raise NotImplementedError(f"{self.id} is hardware: a real robot cannot be teleported (try .twin.set_pose)")
        self.sim.set_pose(self, position=position, orientation=orientation)

    def set_gains(self, kp: Any, kd: Any) -> None:
        """PD gains of this body's actuators, ``action_keys`` order (sim only; a servo bus keeps its own)."""
        if self.sim is None:
            raise NotImplementedError(f"{self.id} is hardware: its servo gains live in the firmware")
        kp, kd = np.asarray(kp, dtype=float).reshape(-1), np.asarray(kd, dtype=float).reshape(-1)
        n = len(self.description.action_keys)
        if kp.shape != (n,) or kd.shape != (n,):
            raise ValueError(f"{self.id}: kp {kp.shape} / kd {kd.shape} for {n} actuators {list(self.action_keys)}")
        with self.sim.lock:
            self.sim.backend.set_gains(self.id, kp, kd)

    def observe(self, *, images: bool = True) -> Observation:
        body = self.body
        with self._lock:
            js = body.read()
            obs = Observation(
                t=self.sim.time if self.sim is not None else _time.time(),
                joints=js,
                pose=body.pose(),
                images=body.images() if images else {},
                extras=body.extras(),
            )
        if self.twin is not None:  # the twin shows what the real body measured
            _twin.mirror(self, self._full_targets(js.positions))
        return obs

    def gate(self, action: Action, *, confirmed: bool = False) -> np.ndarray:
        """The operator gate for commands from an agent tool or dashboard (see ``robot.gate``)."""
        return _gate.gate(self, action, confirmed=confirmed)

    def confirm_act(self, action: Action, *, steps: int | None = None, confirmed: bool = False) -> Observation:
        """``act`` through the operator gate: what an agent tool and the dashboard call."""
        return _gate.confirm_act(self, action, steps=steps, confirmed=confirmed)

    def act(self, action: Action, *, steps: int | None = None) -> Observation:
        """Write targets for THIS body and advance its world one control step (unless manual time)."""
        targets = self.coerce(action)
        with self._lock:
            guarded_write(self, targets)
            if self.sim is not None:
                if steps is None:
                    self.sim.control_step()
                else:
                    self.sim.step(steps)
        if self.twin is not None:
            _twin.mirror(self, targets)
        return self.observe(images=False)

    def calibrate(self, *, interactive: bool = True) -> Calibration:
        """Read (or, at the arm, record) the bus calibration: LeRobot's calibration JSON, unchanged."""
        from .drivers import Driver

        if not isinstance(self.body, Driver):
            raise RuntimeError("calibrate() is for hardware; a simulated body needs no calibration")
        return self.body.calibrate(interactive=interactive)

    def coerce(self, action: Action, *, clip: bool = True) -> np.ndarray:
        """Any accepted Action -> a full target vector in ``action_keys`` order, clipped to limits.

        A joint a mapping does not name holds its measured position, clipped into its limits even when
        ``clip=False``: only the caller's own values reach the safety chain unclipped.
        """
        keys = self.description.action_keys
        if isinstance(action, JointState):
            action = action.as_dict()
        if isinstance(action, Mapping):
            unknown = [k for k in action if k not in keys]
            if unknown:
                raise KeyError(f"unknown joint(s) {unknown} for {self.id}; action_keys are {list(keys)}")
            with self._lock:
                targets = self._full_targets(np.array(self.body.read().positions, dtype=float))
            for k, v in action.items():
                targets[keys.index(k)] = float(v)
        else:
            targets = np.asarray(action, dtype=float).reshape(-1)
            if targets.shape != (len(keys),):
                raise ValueError(
                    f"action has {targets.size} values; {self.id} has {len(keys)} action keys {list(keys)}"
                )
        return self._full_targets(targets) if clip else targets

    def _full_targets(self, targets: np.ndarray) -> np.ndarray:
        out = np.array(targets, dtype=float)
        for i, k in enumerate(self.description.action_keys):
            lim = self.description.limits.get(k)
            if lim is not None:
                out[i] = min(max(out[i], lim[0]), lim[1])
        return out

    def run(self, policy: PolicyLike | Callable[[Observation], Any], **kw: Any) -> RunResult:
        """observe -> policy -> act at ``hz`` until ``seconds``/``steps``/``stop_when`` (see ``robot.run``)."""
        from .run import run

        return run(self, policy, **kw)

    def start(self, policy: PolicyLike | Callable[[Observation], Any], **kw: Any) -> Task:
        """``run`` in a background thread; ``Task.stop()`` / ``.status`` / ``.result``."""
        from .run import start

        return start(self, policy, **kw)

    def stop(self) -> RunResult | None:
        return self._task.stop() if self._task is not None else None

    def close(self) -> None:
        self.events.close()
        if self._task is not None:
            self._task.halt()
        if self.sim is not None:
            self.sim.remove(self)
        elif self._body is not None:
            self._body.close()
            self._body = None
        if self.twin is not None:
            self.twin.close()
            self.twin = None

    def __repr__(self) -> str:
        where = "hardware" if self.is_real else (self.sim.backend_name if self.sim else "detached")
        return f"Robot({self.name!r}, id={self.id!r}, {where}, dof={self.description.dof})"
