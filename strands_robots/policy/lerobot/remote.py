"""Any ``lerobot-policy-server``, spoken through LeRobot's own client — and run the way it was trained.

Two executions. ``"splice"`` is LeRobot's engine unchanged: the next chunk is requested one turnaround before the
queue empties, from a moving arm, and merged (``chunk_merge``, the RTC settings). ``"sequential"`` is
hold-for-observation: the frame goes out only when the queue is EMPTY, the body holds its last target until the
chunk arrives (``act`` returns None — a hold, never a deadline miss), then one row per tick at the server's rate.
That is what a model trained on whole chunks from still frames expects. ``execution="auto"`` (the default) takes
the server's word: a descriptor ``execution`` field when it has one, else ``"splice"`` only for a deployment that
advertises an RTC contract (``rtc_horizon > 0`` or ``rtc_model_space_v1``), ``"sequential"`` for every other
chunked deployment.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ...types import Action, Observation
from ...units import Units
from ..base import PolicyFault
from ..requirements import Requirements
from .contract import Contract, lerobot
from .link import check_channel, wan_defaults, zenoh_config
from .redial import Redial
from .session import Session, open_session

if TYPE_CHECKING:
    from ...description import Description

Execution = Literal["auto", "sequential", "splice"]
EXECUTIONS = ("auto", "sequential", "splice")


class ZenohPolicy:
    """``Policy("zenoh://host:7447/<deployment>")``: LeRobot remote inference as a strands policy.

    ``attach`` discovers the deployment, reads its features against the robot, refuses a missing joint,
    camera or unknown semantics BEFORE asking for a session, then admits it and resolves the execution.
    Under ``"splice"`` ``act`` publishes the frame and runs one LeRobot motor tick; a held/starved tick returns
    the last applied action; chunk behaviour uses LeRobot's settings unchanged (``mode``, ``chunk_merge``,
    ``refill_seconds``, ``blend_*`` ...). Under ``"sequential"`` ``act`` owns the loop (module docstring).
    A terminal fault raises ``PolicyFault`` with the server's reason. After ``attach``, ``policy.execution``
    names the one running and ``policy.calls`` lists every chunk the session received, under either execution.
    """

    realtime = True  # LeRobot ages observations and paces chunks on time.monotonic()
    requires_position_hold = True  # LeRobot admits only a robot that holds between motor ticks

    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        execution: Execution = "auto",
        start_pose: Mapping[str, float] | Sequence[float] | None = None,
        start_seconds: float = 3.0,
        tasks: Sequence[str] | None = None,
        trusted: bool = False,
        zenoh_config: str | None = None,
        reconnect_s: float = 10.0,
        **settings: Any,
    ) -> None:
        """``execution``: see the module docstring. ``start_pose`` (the deployment's units; a mapping of joint -> value
        or a full vector) is where its episodes begin: ``robot.run`` ramps there over ``start_seconds`` first. Left
        out, the server's descriptor may say (``start_pose``); else the run begins where the body is. ``tasks``: the
        instructions it was trained on — any other ``run(instruction=)`` is refused naming them before a frame goes
        out; left out, the descriptor may say (``tasks``); else any instruction is sent. ``zenoh_config`` (or
        ``STRANDS_LEROBOT_ZENOH_CONFIG``): the session's zenoh config file. ``trusted``: see ``link.check_channel``.
        ``reconnect_s``: when the server goes away, hold and reopen the session for this long (0: fault at once)."""
        if execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {', '.join(map(repr, EXECUTIONS))}, got {execution!r}")
        if not reconnect_s >= 0:
            raise ValueError(f"reconnect_s must be a number of seconds >= 0, got {reconnect_s!r}")
        self.endpoint, self.deployment, self.settings, self.trusted = endpoint, deployment, settings, trusted
        self.reconnect_s = float(reconnect_s)
        self.zenoh_config = zenoh_config or settings.pop("zenoh_config_path", None)  # LeRobot's own name, same file
        self._wire = endpoint  # what the session opens: the endpoint, pinned at attach for a real robot
        self.asked: Execution = execution
        self.execution: str | None = None  # resolved at attach, from the server
        self._start_pose, self.start_seconds = start_pose, float(start_seconds)
        self.start_pose: Mapping[str, float] | Sequence[float] | None = start_pose
        self._tasks = None if tasks is None else tuple(str(t) for t in tasks)
        self.tasks: tuple[str, ...] = self._tasks or ()
        self.contract: Contract | None = None
        self.hz: float | None = None
        self.calls: list[dict[str, Any]] = []  # every chunk a session received (wall s, rows, server, instruction)
        self.reconnects: list[dict[str, Any]] = []  # every lost server the run got back (down s, attempts)
        self._s: Session | None = None
        self._redial: Redial | None = None
        self._resume: str | None = None  # the instruction a reopened session carries on with
        self._robot: Any = None
        self._last: np.ndarray | None = None
        self._task: str | None = None

    def describe(self) -> dict[str, Any]:
        """The server's descriptor (checkpoint identity, features, fps, modes, semantics), without a session."""
        with closing(self._connect()) as client:
            return dict(client.descriptor)

    def _config(self, semantics: str = "?", descriptor: dict[str, Any] | None = None, **forced: Any) -> Any:
        from lerobot.inference import RemoteInferenceConfig

        kw = wan_defaults(self.endpoint, self.settings, descriptor)
        kw["zenoh_config_path"] = zenoh_config(self.zenoh_config)
        blend = kw.get("blend_components")
        kw["blend_components"] = [f"{k}.pos" if "." not in k else k for k in blend or []]
        kw.update(forced)
        return RemoteInferenceConfig(endpoint=self._wire, deployment=self.deployment, semantics=semantics, **kw)

    def _connect(self, **forced: Any) -> Any:
        lerobot()
        from lerobot.remote_inference import RemoteClient

        return RemoteClient.connect(self._config(**forced))

    def attach(self, robot: Any) -> None:
        self.detach()
        mode = self.settings.get("zenoh_mode")  # the caller's word over wan_defaults' client
        self._wire = check_channel(repr(self), self.endpoint, robot, self.trusted, self.zenoh_config, mode)
        s = open_session(self, robot, check=lambda c: c.requirements().check(robot.observe(), who=repr(self)))
        self._adopt(s, robot)
        if self._tasks is None:  # the server's word on what it was trained to do, when it has one
            said = s.client.descriptor.get("tasks")
            self.tasks = tuple(str(t) for t in said) if isinstance(said, list | tuple) else ()
        if self._start_pose is None:  # the server's word on where its episodes begin, when it has one
            said = s.client.descriptor.get("start_pose")
            self.start_pose = said if isinstance(said, Mapping | list | tuple) and len(said) else None

    def _adopt(self, s: Session, robot: Any) -> None:
        self._s, self._robot, self.contract, self.execution = s, robot, s.contract, s.execution
        self.hz = 1.0 / s.interval

    @property
    def _seq(self) -> Any:
        return None if self._s is None else self._s.seq

    @property
    def _engine(self) -> Any:
        return None if self._s is None else self._s.engine

    def detach(self) -> None:
        if self._redial is not None:
            self._redial.stop()
        if self._s is not None:
            self._s.close()
        self._s = self._redial = None
        self._last = self._task = None  # a new session starts with the engine's empty task

    def _need(self) -> Contract:
        if self.contract is None:
            raise RuntimeError(f"{self!r} is not attached to a robot; run it with robot.run(policy, ...)")
        return self.contract

    def units_for(self, description: Description) -> Units | None:
        return self._need().units()

    def requirements(self) -> Requirements:
        return self.contract.requirements() if self.contract is not None else Requirements()

    def check_task(self, instruction: str) -> None:
        """An instruction outside ``tasks`` (when known) is refused by name: the model never saw it."""
        if self.tasks and instruction.strip() not in self.tasks:
            shown = ", ".join(repr(t) for t in self.tasks[:20]) + (", …" if len(self.tasks) > 20 else "")
            raise ValueError(
                f"{self!r} refused the instruction {instruction!r}: not one of the {len(self.tasks)} it was trained on:"
                f" {shown}"
            )

    def reset(self) -> None:
        if self._engine is not None and self._last is not None:  # a fresh session needs no reset
            self._engine.reset()
            self._engine.resume()
        if self._seq is not None:
            self._seq.retarget(None)
        self._last = None

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action | None:
        c = self._need()
        if self._redial is not None and not self._redialled():
            return None  # the server is gone: hold while the session is reopened
        try:
            return self._act(c, obs, instruction)
        except PolicyFault:
            assert self._s is not None
            if self.reconnect_s <= 0 or self._s.present:  # a fault of a live server is not a lost link
                raise
        lost = self._s
        self._resume = self._task if lost.seq is None else lost.seq.task
        lost.close()
        self._s = None  # closed: detach must not close it again
        self._redial = Redial(lambda: open_session(self, self._robot, like=lost), self.reconnect_s, repr(self))
        return None

    def _redialled(self) -> bool:
        """Adopt the reopened session once it is there; False while still trying. The window's end raises."""
        assert self._redial is not None
        s = self._redial.poll()
        if s is None:
            return False
        task, r = self._resume, self._redial
        self._adopt(s, self._robot)
        self._redial, self._last = None, None
        self.reconnects.append({"down_s": round(r.down_s, 3), "attempts": r.attempts})
        if s.seq is not None:
            s.seq.retarget(task)
        elif task:
            s.engine.set_task(task)
        self._task = task if s.seq is None else None
        return True

    def _act(self, c: Contract, obs: Observation, instruction: str | None) -> Action | None:
        if self._seq is not None:
            if instruction is not None and instruction != self._seq.task:  # a new instruction drops the old plan
                self.check_task(instruction)
                self._seq.retarget(instruction)
            out = self._seq.step(c.frame(obs))
            if out is not None:
                self._last = out
            return out
        return self._act_splice(c, obs, instruction)

    def _act_splice(self, c: Contract, obs: Observation, instruction: str | None) -> Action:
        from lerobot.inference import dispatch_tick

        engine, interp = self._engine, self._s.interp  # type: ignore[union-attr]
        if instruction is not None and instruction != self._task:
            self.check_task(instruction)
            if error := engine.text_input_error(instruction, instruction=True):
                raise ValueError(f"{self!r} refused the instruction: {error}")
            engine.set_task(instruction)
            self._task = instruction
        engine.submit_observation(c.frame(obs), capture_time=time.monotonic())
        sent: list[Any] = []
        target = dispatch_tick(engine, interp, send=sent.append, hold=lambda: None)
        if engine.failed:
            raise PolicyFault(f"{self!r}: {engine.runtime.failure or engine.failure_traceback or 'LeRobot fault'}")
        if target is not None:
            self._last = c.to_keys(target.detach().cpu().numpy())
        if self._last is None:  # startup: hold where the body is (already in the policy's units)
            self._last = np.asarray(obs.joints.positions, dtype=float).copy()
        return self._last

    def __repr__(self) -> str:
        mode = "" if self.asked == "auto" else f", execution={self.asked!r}"
        return f"Policy('zenoh://{self.endpoint.removeprefix('tcp/')}/{self.deployment}'{mode})"
