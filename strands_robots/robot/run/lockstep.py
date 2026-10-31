"""Several robots in one world, in lock-step: every robot decides, then every robot writes, the world steps once."""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...types import Observation
from ..clock import MONOTONIC
from ..safety import apply_on_disconnect
from .bind import Bound, bind, install_gains, looks, release
from .common import announce, declared, emit
from .loop import start_pose
from .pace import Pacer
from .result import RunResult, StoppedBy
from .source import is_agent
from .tick import Tick, commit, decided, stopped

if TYPE_CHECKING:
    from ...policy import PolicyLike
    from ...simulation import Simulation
    from ..robot import Robot


def lockstep(
    sim: Simulation,
    policies: Mapping[Robot, PolicyLike | Callable[[Observation], Any]],
    *,
    hz: float | None = None,
    seconds: float | None = None,
    steps: int | None = None,
    instruction: str | None = None,
    units: str | None = "auto",
    stop_event: threading.Event | None = None,
) -> dict[str, RunResult]:
    """Drive N robots in lock-step: every robot observes and decides, then every robot writes, the world steps once.

    A fault, a refusal or an e-stop on any robot while they decide ends the run for all of them at that tick, written
    for none: no robot acts on a tick another refused. A press after the decisions finds robots written; one in a paced
    run's wait for the world's step (after the writes) lets that one step go ahead, the pressed robot holding.
    """
    if seconds is None and steps is None:
        raise ValueError("Simulation.run needs seconds= or steps=")
    for r in policies:
        if r.sim is not sim:
            raise ValueError(f"robot {r.id!r} is not in this world")
    bound: dict[Robot, Bound] = {}
    try:
        for r, p in policies.items():
            bound[r] = bind(r, p, units)
        for r, b in bound.items():
            if not start_pose(r, b):  # a policy that names where its episodes begin is taken there first
                emit(r, "run.started", {"policy": repr(b.policy)})
        results = _lockstep(sim, bound, hz, seconds, steps, instruction, stop_event)
    finally:
        for b in bound.values():
            release(b)
    for r in bound:
        announce(r, results[r.id])
    return results


def _lockstep(
    sim: Simulation,
    bound: Mapping[Robot, Bound],
    hz: float | None,
    seconds: float | None,
    steps: int | None,
    instruction: str | None,
    stop_event: threading.Event | None,
) -> dict[str, RunResult]:
    said = declared(bound.values(), hz)  # refuses policies that disagree, or an hz against theirs
    rate = hz if hz is not None else said
    dt = sim.physics.control_dt if rate is None else 1.0 / rate
    per = max(1, round(dt / sim.physics.timestep))
    total = steps if steps is not None else int(round(float(seconds or 0.0) / dt))
    paced = any(b.realtime for b in bound.values())
    pacer = Pacer(dt, now=MONOTONIC.now, sleep=MONOTONIC.sleep) if paced else None
    held = install_gains(bound)  # after the rate, the step count and the pacer, the loop's last checks
    limit: StoppedBy = "steps" if steps is not None else "seconds"  # the limit the caller gave
    why = limit
    faults: dict[str, str] = {}
    ended: list[Robot] = []
    n = 0
    t0, wall0, gated = sim.time, _time.monotonic(), is_agent()  # an agent's run passes the gate per step
    stops0 = {r: r.estop_latch.count for r in bound}
    with sim.manual_time():
        while n < total:
            ended = held or [r for r in bound if stopped(r, stops0[r])]
            if ended:
                why = "estop"
                break
            if stop_event is not None and stop_event.is_set():
                why = "stopped"
                break
            if pacer is not None:
                pacer.start()  # the wait for this tick: a stop or an e-stop in it ends the run before any robot reads
                if any(stopped(r, stops0[r]) for r in bound) or (stop_event is not None and stop_event.is_set()):
                    continue  # the checks above name why (an e-stop outranks stop())
            stop = _tick(bound, stops0, instruction, gated, faults)
            if stop is not None:  # a fault, a refusal or an e-stop ended this tick
                why, ended = stop
                break
            if pacer is not None:
                pacer.before_write()  # the world's step is lockstep's write: spaced by one period
            sim.step(per)
            n += 1
            if pacer is not None:
                pacer.after_write()
    for faulted in (r for r in bound if r.id in faults):  # faulted bodies go to their on_disconnect state
        apply_on_disconnect(faulted)
    el, wall = sim.time - t0, _time.monotonic() - wall0
    reason = f"e-stop engaged on {', '.join(r.id for r in ended)}" if why == "estop" else None
    misses, late = (pacer.misses, pacer.worst_late) if pacer is not None else (0, 0.0)
    final = {r: r.observe(images=False) for r in bound}
    return {
        r.id: RunResult(
            n, el, 1.0 / dt, n / el if el else 0.0, why, None, final[r], faults.get(r.id, reason), misses, late, wall
        )
        for r in bound
    }


def _tick(
    bound: Mapping[Robot, Bound],
    stops0: Mapping[Robot, int],
    instruction: str | None,
    gated: bool,
    faults: dict[str, str],
) -> tuple[StoppedBy, list[Robot]] | None:
    """One tick for every robot: all decide, then all write; None once all wrote. A fault or a refusal, or an e-stop
    on any of them (even one resumed since) while they decide, ends the run with this tick written for none."""
    ticks: list[tuple[Robot, Tick]] = []
    for r, b in bound.items():
        step = decided(r, b, instruction, images=r.sim is None or looks(b, instruction), gated=gated)
        if step.ended is not None:  # the robots after it do not decide, and none has written this tick
            return _stop(r, step, faults)
        pressed = [x for x in bound if stopped(x, stops0[x])]
        if pressed:  # pressed while they decide: the robots after do not decide, and nothing is written
            return "estop", pressed
        ticks.append((r, step))
    for r, step in ticks:
        done = commit(r, step, gated=gated, stops0=stops0[r])
        if done.ended is not None:  # an e-stop landing during the writes: the robots before it have written
            return _stop(r, done, faults)
    return None


def _stop(robot: Robot, step: Tick, faults: dict[str, str]) -> tuple[StoppedBy, list[Robot]]:
    assert step.ended is not None
    if step.fault is not None:
        faults[robot.id] = step.fault
    return step.ended, [robot]
