"""The one run loop: observe -> policy -> act. ``Robot.run``, ``Robot.start`` and ``Simulation.run`` call it."""

from __future__ import annotations

import contextvars
import threading
import time as _time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...recording import Episode, NotRecording, episode
from ...types import Observation
from ..clock import MONOTONIC
from ..safety import apply_on_disconnect
from .bind import Bound, bind, looks, release
from .common import announce, approve_run, emit, rate
from .pace import OnMiss, Pacer
from .result import RunResult, StoppedBy
from .source import RunPlan, is_agent, is_confirmed
from .task import Task
from .tick import stopped, tick

if TYPE_CHECKING:
    from ...policy import PolicyLike
    from ..robot import Robot

__all__ = ["run", "start", "Task", "RunResult"]
_CLAIM = threading.Lock()  # start(): the check for a running task and the claim of robot._task are one step


def run(
    robot: Robot,
    policy: PolicyLike | Callable[[Observation], Any],
    *,
    instruction: str | None = None,
    hz: float | None = None,
    seconds: float | None = None,
    steps: int | None = None,
    stop_when: Callable[[Observation], bool] | None = None,
    units: str | None = "auto",
    realtime: bool = False,
    stop_event: threading.Event | None = None,
    record: str | Path | None = None,
    push: bool = False,
    on_miss: OnMiss = "skip",
) -> RunResult:
    """observe -> policy -> act at ``hz`` (default: the policy's ``hz``, else 30) until a limit or ``stop()``.

    ``record=dir | "hf://owner/name"`` appends this run as one episode of a LeRobot v3 dataset;
    ``push=True`` (an ``hf://`` target only) then uploads it. Nothing leaves the machine otherwise.
    A paced body (hardware, or ``realtime``) spaces its writes at least one period apart; a write past the next
    tick's start is a deadline miss, counted and re-phased (``on_miss="skip"``) or the end (``"fault"``).
    A policy that returns ``None`` holds that tick: nothing is written, the body keeps its last target, the tick
    counts (``RunResult.holds``) and is never a miss — how a sequential remote policy waits for its chunk.
    """
    if seconds is None and steps is None and stop_when is None:
        raise ValueError("run needs seconds=, steps= or stop_when=")
    if hz is not None and hz <= 0:
        raise ValueError(f"hz must be > 0, got {hz}")  # before bind: a bad hz loads no checkpoint
    if is_agent() and robot.is_real and not is_confirmed():  # a real agent run asks the operator ONCE (K6),
        approve_run(robot, RunPlan(robot.id, repr(policy), instruction, hz, seconds, steps))  # before anything acquired
    bound = bind(robot, policy, units)
    try:
        if not start_pose(robot, bound):  # a policy that names where its episodes begin is taken there first
            emit(robot, "run.started", {"policy": repr(bound.policy)})
        ep = episode(
            record,
            description=robot.description,
            hz=rate(bound, hz),
            instruction=instruction,
            units=bound.units,
            push=push,
        )
        try:
            result = _loop(robot, bound, instruction, hz, seconds, steps, stop_when, realtime, stop_event, ep, on_miss)
        except BaseException:
            ep.abort()
            raise
    finally:
        release(bound)
    announce(robot, result)
    return result


def start_pose(robot: Robot, bound: Bound) -> bool:
    """Ramp to ``policy.start_pose`` (in the policy's units; a mapping or a full vector) over ``policy.start_seconds``.

    A chunked policy trained from one rest pose plans its first chunk from there; starting a run anywhere else hands
    it a jump it never saw. An agent's run passes every ramp target through the same chain as its steps.
    """
    pose = getattr(bound.policy, "start_pose", None)
    if pose is None:
        return False
    seconds = float(getattr(bound.policy, "start_seconds", 3.0))
    q = robot.observe(images=False).joints.positions
    if isinstance(pose, Mapping):
        keys = robot.description.action_keys
        unknown = [k for k in pose if k.removesuffix(".pos") not in keys]
        if unknown:
            raise KeyError(f"{bound.policy!r}.start_pose names {unknown}; action_keys are {list(keys)}")
        seen = np.array(q if bound.units is None else bound.units.from_rad(q), dtype=float)  # a writable copy
        for k, v in pose.items():
            seen[keys.index(k.removesuffix(".pos"))] = float(v)
        pose = seen
    pose = np.asarray(pose, dtype=float).reshape(-1)
    targets = robot.coerce(pose if bound.units is None else bound.units.to_rad(pose), measured=q)
    gate = (lambda t: robot.gate(t, approved_run=True, measured=q)) if is_agent() else None
    emit(robot, "run.started", {"policy": repr(bound.policy), "start_pose": targets.tolist(), "seconds": seconds})
    from ..reach import reach  # at call time: robot.reach is the layer above this loop (robot.py imports both)

    reach(robot, targets, seconds, gate=gate)
    return True


def _loop(
    robot: Robot,
    bound: Bound,
    instruction: str | None,
    hz: float | None,
    seconds: float | None,
    steps: int | None,
    stop_when: Callable[[Observation], bool] | None,
    realtime: bool,
    stop_event: threading.Event | None,
    ep: Episode | NotRecording,
    on_miss: OnMiss,
) -> RunResult:
    hz = rate(bound, hz)
    dt = 1.0 / hz
    sim_steps = max(1, round(dt / robot.sim.physics.timestep)) if robot.sim is not None else None
    max_steps = steps if steps is not None else (int(round(seconds * hz)) if seconds is not None else None)
    n, holds, stops0, fault = 0, 0, robot.estop_latch.count, None
    last: np.ndarray | None = None  # the latest written targets: what a held tick records as its action
    why: StoppedBy = "steps"
    always = robot.sim is None or stop_when is not None or ep.wants_images  # a recorded step keeps its cameras
    pacer = None
    if robot.sim is None or realtime or bound.realtime:  # hardware (or a watched world): pace on the body's clock
        clock = robot.clock if robot.sim is None else MONOTONIC
        sleep = robot.estop_latch.wait if clock is MONOTONIC else clock.sleep  # an e-stop never waits out a period
        pacer = Pacer(dt, on_miss=on_miss, now=clock.now, sleep=sleep)
    t0, wall0 = robot.clock.now(), _time.monotonic()
    gated = is_agent()  # an agent's run passes the gate on every step
    while True:
        if stopped(robot, stops0):
            why = "estop"
            break
        if stop_event is not None and stop_event.is_set():
            why = "stopped"
            break
        if max_steps is not None and n >= max_steps:
            break
        if pacer is not None:
            pacer.start()
        step = tick(
            robot,
            bound,
            instruction,
            images=always or looks(bound, instruction),
            gated=gated,
            stop_when=stop_when,
            steps=sim_steps,
            before_write=pacer.before_write if pacer is not None else None,
        )
        if step.ended is not None:
            why, fault = step.ended, step.fault
            break
        if step.held:  # the policy held: the body kept ``last``; the dataset keeps its frame rate; no miss
            ep.add(step.obs, step.obs.joints.positions if last is None else last)
            n, holds = n + 1, holds + 1
            if pacer is not None:
                pacer.hold()
            continue
        ep.add(step.obs, step.targets)  # after the write: a step the body refused is not in the dataset
        n, last = n + 1, step.targets
        if pacer is not None and not pacer.after_write():
            why, fault = "deadline", f"tick {n} ended {pacer.worst_late * 1e3:.1f} ms past its deadline"
            break
    if pacer is not None and n:  # the last write owns its period: the body's next write (any run) is >= dt later
        pacer.before_write()
    if why == "fault":  # the policy can no longer act safely: put the body in its declared on_disconnect state
        apply_on_disconnect(robot)
    final = robot.observe(images=False)
    span = final.t - t0 if robot.sim is not None else (pacer.span() if pacer is not None else 0.0)
    if robot.sim is None and pacer is not None and holds and n > holds:  # held ticks are part of the commanded span
        span = max(span, n * dt)
    recorded = ep.close(why)
    misses, late = (pacer.misses, pacer.worst_late) if pacer is not None else (0, 0.0)
    wall = _time.monotonic() - wall0
    achieved = n / span if span > 0 else 0.0
    return RunResult(n, span, hz, achieved, why, recorded, final, fault, misses, late, wall, holds, holds * dt)


def start(
    robot: Robot,
    policy: PolicyLike | Callable[[Observation], Any],
    *,
    instruction: str | None = None,
    hz: float | None = None,
    seconds: float | None = None,
    steps: int | None = None,
    stop_when: Callable[[Observation], bool] | None = None,
    units: str | None = "auto",
    realtime: bool = False,
    record: str | Path | None = None,
    push: bool = False,
    on_miss: OnMiss = "skip",
) -> Task:
    """``run`` in a background thread (every ``run`` keyword but ``stop_event``); ``Task.stop()`` / ``.result``."""
    with _CLAIM:  # two threads at once (two tool calls of one model turn): one run, one refusal
        if robot._task is not None and robot._task.running:
            raise RuntimeError(f"robot {robot.id!r} already has a running task; call stop() first")
        ctx = contextvars.copy_context()  # who commanded it (``source.from_agent``) travels to the task's thread
        robot._task = task = Task(
            lambda ev: ctx.run(
                run,
                robot,
                policy,
                instruction=instruction,
                hz=hz,
                seconds=seconds,
                steps=steps,
                stop_when=stop_when,
                units=units,
                realtime=realtime,
                stop_event=ev,
                record=record,
                push=push,
                on_miss=on_miss,
            )
        )
    return task
