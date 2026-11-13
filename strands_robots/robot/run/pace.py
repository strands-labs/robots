"""Pacing on hardware (or a watched world): ticks start on a grid, WRITES are spaced by one period.

``next_tick += dt`` never re-phased, so after a stall every missed command went out back to back. Here a tick
starts at its grid point (``start``), observes and decides; its write waits until one period after the previous
write (``before_write``), whatever the inference jitter. A write held back to that floor counts as written ON it:
the sleep that held it wakes late on a real host, and carrying that overshoot into the next floor slowed every
held tick by it, with no miss counted. So consecutive writes are never closer than ``dt`` less one such overshoot,
and with no miss the writes keep the rate. A write that lands after the next tick should have started, without
having been held to its floor, is a deadline miss (``after_write``): counted, the worst lateness kept, and the
grid re-phases to that write (so a sleep's overshoot after a re-phase is not another miss); ``on_miss="fault"``
ends the run instead. A held write is on its floor, never past its deadline: only a miss, or a ``hold`` longer
than its period, re-phases the grid.

A plain sleep wakes late (on macOS up to ~2 ms at p99), and at 100 Hz that lateness alone made writes miss. With
``spin`` (the hardware loop passes ``SPIN``) a wait sleeps until ``spin`` before its target and spins the rest,
asking ``sleep(0)`` each turn (an engaged e-stop latch answers True and ends the wait) and yielding the GIL so
capture threads keep running. ``spin`` stays 0 for any other clock: a clock that only moves when slept would
never reach the target in a spin.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from ..clock import MONOTONIC

OnMiss = Literal["skip", "fault"]
SPIN = 0.002  # seconds spun before each target on the monotonic clock: covers the OS's usual late wake


def spin_for(clock: object) -> float:
    """``SPIN`` on the host's monotonic clock, else 0: a clock of the caller's own may only move when slept."""
    return SPIN if clock is MONOTONIC else 0.0


class Pacer:
    """``start()`` before observing, ``before_write()`` before the write, ``after_write()`` after it — or ``hold()``
    instead of both when the policy held (returned None): a deliberate hold is a tick, never a miss."""

    def __init__(
        self,
        dt: float,
        *,
        on_miss: OnMiss = "skip",
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
        spin: float = 0.0,
    ) -> None:
        if on_miss not in ("skip", "fault"):
            raise ValueError(f"on_miss must be 'skip' or 'fault', got {on_miss!r}")
        if not spin >= 0.0:
            raise ValueError(f"spin must be 0 or more seconds, got {spin!r}")
        self._spin = min(spin, dt / 2)  # a fast loop spins at most half its period
        self.dt, self.on_miss, self._now, self._sleep = dt, on_miss, now, sleep
        self._start: float | None = None  # this tick's grid point
        self._next: float | None = None  # the next tick's grid point (None: the first tick starts at once)
        self._first: float | None = None
        self._last: float | None = None  # when the latest write began
        self._writing: float | None = None  # when this tick's write began (its own duration is not lateness)
        self._held = False
        self.misses = 0
        self.worst_late = 0.0  # seconds the latest write landed past its deadline

    def _until(self, t: float) -> None:
        left = t - self._now()
        if left > self._spin:
            self._sleep(left - self._spin)
        while self._spin and self._now() < t:
            if self._sleep(0):  # the e-stop latch engaged: the wait ends now
                return
            time.sleep(0)  # let other threads run while spinning

    def start(self) -> None:
        if self._next is not None:
            self._until(self._next)
        self._start = self._now() if self._next is None else self._next

    def before_write(self) -> None:
        floor = None if self._last is None else self._last + self.dt
        self._held = floor is not None and floor > self._now()  # a write held back to its floor is never late
        if floor is not None:
            self._until(floor)
        self._writing = floor if self._held else self._now()  # held: on its floor, however late the sleep woke

    def after_write(self) -> bool:
        """Record the write; False when it missed its deadline and ``on_miss="fault"``."""
        w = self._last = self._now() if self._writing is None else self._writing
        self._writing = None
        self._first = w if self._first is None else self._first
        deadline = (w if self._start is None else self._start) + self.dt
        if w > deadline and not self._held:
            self.misses += 1
            self.worst_late = max(self.worst_late, w - deadline)
            self._next = w  # re-phase onto the late write: the next tick starts now, its write still waits
            return self.on_miss != "fault"
        self._next = max(deadline, w)  # a held write is on its floor, <= deadline: the grid never trails the writes
        return True

    def hold(self) -> None:
        """This tick wrote nothing on purpose: the grid moves one period on, no write, no miss.

        A hold that outlasted its period (the chunk took longer than a tick to land) re-phases the grid onto now:
        the time a run spends deliberately waiting is not lateness, so the write that ends the wait is not late.
        """
        start = self._now() if self._start is None else self._start
        self._next = max(start + self.dt, self._now())

    def span(self) -> float:
        """The time the body was commanded: first write to one period after the last (no miss: n * dt, within dt)."""
        return 0.0 if self._first is None or self._last is None else self._last - self._first + self.dt
