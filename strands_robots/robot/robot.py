"""A Robot is a description plus a Body: observe() / act() / calibrate() / run().

The Body is the only thing that differs between a simulated robot (``SimBody``, a handle into a
``Simulation``) and a real one (a ``Driver`` on a bus). Every loop is written once against ``Robot``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..description import Description
from ..types import Action, Observation, Pose
from . import gate as _gate
from . import reach as _reach
from . import targets as _targets
from . import twin as _twin
from .body import Body, SimBody
from .clock import MONOTONIC, Clock, WorldClock
from .device import DeviceVerbs, Emitter, position_hold_of
from .run import loop as _run
from .safety import Latch, Validator, guarded_write

if TYPE_CHECKING:
    from ..simulation import Simulation
    from .drivers.calibration import Calibration
    from .run.result import RunResult
    from .run.source import RunPlan
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
        calibration: Any = None,
        mesh: Any = None,
        clock: Clock | None = None,
        approve: Callable[[RunPlan], bool] | None = None,
        **driver_options: Any,  # feetech robot_id=/baud=, unitree timeout=... reach the driver unchanged (D3)
    ) -> None:
        from ..simulation import Simulation

        self.description: Description = Description.load(name)
        self.name = self.description.name
        self.is_real = port is not None or driver is not None
        hw_only = [n for n, v in (("confirm", confirm), ("approve", approve), ("clock", clock)) if v is not None]
        if hw_only and not self.is_real:  # the operator gates and a bus clock are hardware's; a world owns its time
            raise TypeError(f"{', '.join(hw_only)}= are hardware Robot arguments; a simulated body takes none")
        self.confirm = confirm  # one command from an agent/dashboard: asked for exactly these targets
        self.approve = approve  # one RUN from an agent on a real body: asked once with its RunPlan (K6)
        self.sim: Simulation | None = None
        self.twin: Robot | None = None  # on a hardware Robot: the simulated body that mirrors it
        self.twin_of: Robot | None = None  # on that twin: the hardware Robot it mirrors
        self._task: Task | None = None
        self._body: Body | None = None
        self._bus_lock = threading.RLock()  # a hardware body: one half-duplex bus shared by every thread
        self.clock: Clock = clock or MONOTONIC  # a world's robot reads its world's time, set below
        self.estop_latch, self.events = Latch(lambda: self.clock.now()), Emitter(self)
        self.validators: list[Validator] = []  # extra checks on every agent/dashboard write, after the interlocks
        self.interlocks: list[Validator] = []  # append a Validator to either list; it sees every target, in order

        if self.is_real:
            if sim is not None:
                raise TypeError("a hardware Robot takes port=/driver=, not sim=: its twin is in Simulation.default()")
            from .drivers import make_driver

            self.id = id or self.name
            hw = {"port": port, "driver": driver, "cameras": cameras, "calibration": calibration}
            self._body = make_driver(self.description, **hw, options=driver_options, id=self.id)
            self.supports_position_hold = position_hold_of(self.description, self._body)
            if twin:
                self.twin = Robot(
                    name, position=position, orientation=orientation, keyframe=keyframe, id=f"{self.id}_twin"
                )
                self.twin.twin_of = self
        else:
            if driver_options:
                raise TypeError(f"{sorted(driver_options)} are hardware driver options; a world's robot takes none")
            if cameras is not None or calibration is not None:
                raise TypeError(
                    "cameras=/calibration= are hardware arguments; in a world: sim.camera(name, attach=robot)"
                )
            world = sim if sim is not None else Simulation.default()
            self.sim, self.id = world, world.unique_id(id or self.name)
            self.clock = WorldClock(world)
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
        """One read of the body; ``t`` is the capture time on ``self.clock``."""
        body = self.body
        with self._lock:
            t = self.clock.now()
            js = body.read()
            obs = Observation(
                t=t,
                joints=js,
                pose=body.pose(),
                images=body.images() if images else {},
                extras=body.extras(),
            )
        if self.twin is not None:  # the twin shows what the real body measured
            _twin.mirror(self, _targets.clip_to_limits(self.description, js.positions), measured=True)
        return obs

    def gate(
        self, action: Action, *, confirmed: bool = False, approved_run: bool = False, measured: np.ndarray | None = None
    ) -> np.ndarray:
        """The operator gate for commands from an agent tool or dashboard (see ``robot.gate``)."""
        return _gate.gate(self, action, confirmed=confirmed, approved_run=approved_run, measured=measured)

    def confirm_act(
        self, action: Action, *, steps: int | None = None, seconds: float | None = None, confirmed: bool = False
    ) -> Observation:
        """``act`` through the operator gate (asked once, for the target): what an agent tool and the dashboard call."""
        return _gate.confirm_act(self, action, steps=steps, seconds=seconds, confirmed=confirmed)

    def act(self, action: Action, *, steps: int | None = None, seconds: float | None = None) -> Observation:
        """Write targets for THIS body and advance its world (a real body: its twin's) one control step.

        ``seconds``: reach the pose instead, ramping there from the measured one over that long (``robot.reach``).
        Returns what the body reports after (one read); a run loop calls ``command`` instead.
        """
        if seconds is None:
            self.command(self.coerce(action), steps=steps)
        elif steps is not None:
            raise ValueError("act takes steps= or seconds=, not both")
        else:
            _reach.reach(self, self.coerce(action), seconds)
        return self.observe(images=False)

    def command(self, targets: np.ndarray, *, steps: int | None = None) -> None:
        """Write already-coerced full ``targets`` and advance the world, with no read: the run loop's write."""
        with self._lock:
            guarded_write(self, targets)
            if self.sim is not None:
                if steps is None:
                    self.sim.control_step()
                else:
                    self.sim.step(steps)
        if self.twin is not None:  # the twin's world advances one control step too
            _twin.mirror(self, targets, step=True)

    def calibrate(self, *, interactive: bool = True) -> Calibration:
        """Read (or, at the arm, record) the bus calibration: LeRobot's calibration JSON, unchanged."""
        from .drivers import Driver

        if not isinstance(self.body, Driver):
            raise RuntimeError("calibrate() is for hardware; a simulated body needs no calibration")
        return self.body.calibrate(interactive=interactive)

    def coerce(self, action: Action, *, clip: bool = True, measured: np.ndarray | None = None) -> np.ndarray:
        """Any accepted Action -> full targets in ``action_keys`` order (see ``robot.targets.coerce``)."""
        return _targets.coerce(self, action, clip=clip, measured=measured)

    run = _run.run  # the run loop's own signature, robot first: one place spells every keyword
    start = _run.start

    def stop(self, timeout: float | None = None) -> RunResult | None:
        """End the background run: its result, or its error raised once; None when nothing runs."""
        return self._task.stop(timeout) if self._task is not None else None

    @property
    def tool(self) -> Any:
        """This robot as a strands agent tool, generated from its rpcs (imports strands on first use)."""
        from ..tools.agent import agent_tool

        return agent_tool(self)

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
