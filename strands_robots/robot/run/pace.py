"""Pacing on hardware (or a watched world): ticks start on a grid, WRITES are spaced by at least one period.

``next_tick += dt`` never re-phased, so after a stall every missed command went out back to back. Here a tick
starts at its grid point (``start``), observes and decides; its write waits until one period after the previous
write (``before_write``), so consecutive writes are never closer than ``dt`` whatever the inference jitter. A
write that lands after the next tick should have started, without having been held to that floor, is a deadline
miss (``after_write``): counted, the worst lateness kept, and the grid re-phases to that write (so a sleep's
overshoot after a re-phase is not another miss); ``on_miss="fault"`` ends the run instead. A held write that still
lands past its deadline re-phases the grid as well, so lateness is always measured from the writes themselves.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

OnMiss = Literal["skip", "fault"]


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
    ) -> None:
        if on_miss not in ("skip", "fault"):
            raise ValueError(f"on_miss must be 'skip' or 'fault', got {on_miss!r}")
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
        if left > 0:
            self._sleep(left)

    def start(self) -> None:
        if self._next is not None:
            self._until(self._next)
        self._start = self._now() if self._next is None else self._next

    def before_write(self) -> None:
        floor = None if self._last is None else self._last + self.dt
        self._held = floor is not None and floor > self._now()  # a write held back to its floor is never late
        if floor is not None:
            self._until(floor)
        self._writing = self._now()

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
        self._next = max(deadline, w)  # a held write past its deadline moves the grid too: lateness is measured
        return True  # from where the writes are, never from a grid they left behind one sleep overshoot at a time

    def hold(self) -> None:
        """This tick wrote nothing on purpose: the grid moves one period on, no write, no miss.

        A hold that outlasted its period (the chunk took longer than a tick to land) re-phases the grid onto now:
        the time a run spends deliberately waiting is not lateness, so the write that ends the wait is not late.
        """
        start = self._now() if self._start is None else self._start
        self._next = max(start + self.dt, self._now())

    def span(self) -> float:
        """The time the body was commanded: first write to one period after the last (n * dt with no miss)."""
        return 0.0 if self._first is None or self._last is None else self._last - self._first + self.dt
