"""One clock per body: an observation's capture time and every duration a run reports are read from it.

A simulated body's clock is its world's time (``sim.time``: deterministic, frozen between steps); a hardware
body's clock is the monotonic clock (never wall time, which jumps under NTP) unless ``Robot(clock=)`` names
another. ``Observation.t``, the e-stop latch, the watchdog and ``RunResult.seconds`` all read ``robot.clock``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..simulation import Simulation


@runtime_checkable
class Clock(Protocol):
    """Seconds on one monotone axis: ``now()`` never decreases; ``sleep(s)`` lets ``s`` seconds pass on it."""

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    """The hardware clock: ``time.monotonic``."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def __repr__(self) -> str:
        return "MonotonicClock()"


class WorldClock(MonotonicClock):
    """A world's simulated time; it passes only when the world steps, so a watched world sleeps on the host's."""

    def __init__(self, sim: Simulation) -> None:
        self._sim = sim

    def now(self) -> float:
        return float(self._sim.time)

    def __repr__(self) -> str:
        return f"WorldClock({self._sim!r})"


MONOTONIC = MonotonicClock()


def checked(clock: object) -> Clock:
    """``Robot(clock=)`` as given, refused unless it is a Clock instance; ``None`` is the monotonic clock."""
    if clock is None:
        return MONOTONIC
    if isinstance(clock, type) or not isinstance(clock, Clock):
        raise TypeError(
            "clock= must be a Clock instance, an object with now() -> seconds and sleep(seconds) "
            f"(leave clock= out for the monotonic clock), got {clock!r}"
        )
    return clock
