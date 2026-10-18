"""Several robots in one world, in lock-step: ``Simulation.run`` calls it."""

from __future__ import annotations

import time as _time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...policy.base import PolicyFault
from ...types import Observation
from ..safety import EStopped, NonFiniteTarget, apply_on_disconnect
from .bind import Bound, bind, decide, install_gains, looks, release
from .loop import _announce, _declared, _emit
from .result import RunResult
from .source import is_agent

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
) -> dict[str, RunResult]:
    """Drive N robots in lock-step: every robot observes, every policy acts, the world steps once."""
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
            _emit(r, "run.started", {"policy": repr(b.policy)})
        results = _lockstep(sim, bound, hz, seconds, steps, instruction)
    finally:
        for b in bound.values():
            release(b)
    for r in bound:
        _announce(r, results[r.id])
    return results


def _lockstep(
    sim: Simulation,
    bound: Mapping[Robot, Bound],
    hz: float | None,
    seconds: float | None,
    steps: int | None,
    instruction: str | None,
) -> dict[str, RunResult]:
    declared = _declared(bound.values(), hz)
    rate = hz if hz is not None else declared
    dt = sim.physics.control_dt if rate is None else 1.0 / rate
    per = max(1, round(dt / sim.physics.timestep))
    total = steps if steps is not None else int(round(float(seconds or 0.0) / dt))
    held = install_gains(bound)  # after the rate and the step count, the loop's last checks
    realtime = any(b.realtime for b in bound.values())
    faults: dict[str, str] = {}
    stopped: list[Robot] = []
    t0, n, next_tick = sim.time, 0, _time.perf_counter()
    stops0 = {r: r.estop_latch.count for r in bound}
    write = {r: r.confirm_act if is_agent() else r.act for r in bound}  # an agent's run passes the gate per step
    with sim.manual_time():
        for _ in range(total):
            stopped = held or [r for r in bound if r.estopped or r.estop_latch.count != stops0[r]]
            for r, b in bound.items():
                if stopped:
                    break
                try:
                    write[r](decide(r, b, r.observe(images=r.sim is None or looks(b, instruction)), instruction))
                except (PolicyFault, NonFiniteTarget) as e:
                    faults[r.id] = str(e)
                except EStopped:
                    stopped.append(r)
            if faults or stopped:
                break
            sim.step(per)
            n += 1
            if realtime:
                next_tick += dt
                _time.sleep(max(0.0, next_tick - _time.perf_counter()))
    el = sim.time - t0
    for faulted in (r for r in bound if r.id in faults):  # faulted bodies go to their on_disconnect state
        apply_on_disconnect(faulted)
    why = "fault" if faults else ("estop" if stopped else "steps")
    reason = f"e-stop engaged on {', '.join(r.id for r in stopped)}" if stopped else None
    final = {r: r.observe(images=False) for r in bound}
    return {
        r.id: RunResult(n, el, 1.0 / dt, n / el if el else 0.0, why, None, final[r], faults.get(r.id, reason))  # type: ignore[arg-type]
        for r in bound
    }
