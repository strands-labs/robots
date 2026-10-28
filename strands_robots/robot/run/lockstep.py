"""Several robots in one world, in lock-step: every robot runs THE ``tick``, then the world steps once."""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...types import Observation
from ..clock import MONOTONIC
from ..safety import apply_on_disconnect
from .bind import Bound, bind, looks, release
from .common import announce, declared, emit
from .loop import start_pose
from .pace import Pacer
from .result import RunResult, StoppedBy
from .source import is_agent
from .tick import stopped, tick

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
    """Drive N robots in lock-step: every robot observes, every policy acts, the world steps once.

    One robot's fault or e-stop ends the run for all of them at that tick: no robot acts on a tick another refused.
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
    why: StoppedBy = "steps"
    faults: dict[str, str] = {}
    ended: list[Robot] = []
    n = 0
    t0, wall0, gated = sim.time, _time.monotonic(), is_agent()  # an agent's run passes the gate per step
    stops0 = {r: r.estop_latch.count for r in bound}
    with sim.manual_time():
        while n < total:
            ended = [r for r in bound if stopped(r, stops0[r])]
            if ended:
                why = "estop"
                break
            if stop_event is not None and stop_event.is_set():
                why = "stopped"
                break
            if pacer is not None:
                pacer.start()
            for r, b in bound.items():
                step = tick(r, b, instruction, images=r.sim is None or looks(b, instruction), gated=gated)
                if step.ended is not None:  # the robots after it do not act on this tick
                    why, ended = step.ended, [r]
                    if step.fault is not None:
                        faults[r.id] = step.fault
                    break
            if why != "steps":
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
