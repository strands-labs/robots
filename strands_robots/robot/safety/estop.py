"""The e-stop: the one path that bypasses the safety chain. It latches until an operator approves ``resume``.

``engage`` never takes the world lock or the run-loop lock: it sets the latch, then asks the body for
the fastest stop it has (``Body.estop``). The run loop polls the latch every tick.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .validators import RequestRefused

if TYPE_CHECKING:
    from ..robot import Robot


class EStopped(RequestRefused):
    """A command sent while the e-stop is latched."""


class Latch:
    """Engaged / released, with who stopped it, who released it, and when (the body's clock and wall clock).

    ``engaged_at`` is on the body's clock (``robot.clock``) for elapsed math; ``engaged_wall`` is ``time.time``
    so the stop can be placed on a calendar in an audit record. ``stopped_by`` names the operator who pressed
    it (``None`` for a local, unattributed press).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._now = clock
        self._set = threading.Event()
        self._lock = threading.Lock()
        self.engaged_at: float | None = None  # the body's clock, for elapsed math
        self.engaged_wall: float | None = None  # time.time, to place the stop on a calendar
        self.reason: str | None = None
        self.stopped_by: str | None = None  # who pressed it (None = local, unattributed)
        self.resumed_by: str | None = None
        self.count = 0  # engagements so far: a run that saw a different count was stopped, even if resumed since

    @property
    def engaged(self) -> bool:
        return self._set.is_set()

    def engage(self, reason: str, by: str | None = None) -> Exception | None:
        """Latch; what the clock raised stamping ``engaged_at`` is returned, not raised (the stop holds, unstamped)."""
        with self._lock:  # two racing e-stops count once; the first keeps the latch
            if not self._set.is_set():  # latch first, stamp after: a clock that raises never un-latches a stop
                self.engaged_at, self.engaged_wall = None, time.time()
                self.reason, self.stopped_by = reason, by
                self.count += 1
                self._set.set()
                try:
                    self.engaged_at = self._now()
                except Exception as e:  # nor hides it: the caller reports the clock and still announces the stop
                    return e
        return None

    def release(self, by: str, *, stop: int | None = None) -> bool:
        """Clear the latch; with ``stop``, only while it is still engagement #``stop`` (compare-and-release).

        The pin check and the clear are one critical section under the lock ``engage`` takes, so a stop
        landing between them latches and stays latched instead of being cleared by a yes meant for the one before.
        """
        with self._lock:
            if stop is not None and stop != self.count:
                return False
            self.resumed_by = by
            self.reason = self.engaged_at = self.engaged_wall = None  # no stale derived state after a resume (K4)
            self._set.clear()
            return True

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` s, waking at once when the latch engages; True when it is engaged."""
        return self._set.wait(timeout) if timeout > 0 else self._set.is_set()

    def check(self, robot_id: str) -> None:
        if self._set.is_set():
            by = f" by {self.stopped_by}" if self.stopped_by else ""  # the person at the arm reads who pressed it
            raise EStopped(f"{robot_id}: e-stop is engaged{by} ({self.reason}); resume(approved_by=<operator>) first")


def engage(robot: Robot, reason: str = "estop()", by: str | None = None) -> Exception | None:
    """Latch first (so the next tick refuses), then the body's own stop; returns what the clock raised stamping it.

    A body whose own stop raises still raises: that stop may not have happened."""
    try:
        return robot.estop_latch.engage(reason, by)
    finally:
        robot._held, body = None, robot._body  # the stop holds the measured pose; no old target outlives it
        if body is not None:
            body.estop()


def guarded_write(robot: Robot, targets: object) -> None:
    """Write, refusing while latched (the caller holds ``robot._lock``); a stop landing mid-write is re-applied."""
    latch, body = robot.estop_latch, robot.body
    latch.check(robot.id)
    body.write(targets)  # type: ignore[arg-type]
    robot._held = np.array(targets, dtype=float)  # a joint the next Action omits keeps this
    if latch.engaged:
        robot._held = None  # the stop that landed mid-write wins here too
        body.estop()
        latch.check(robot.id)


def resume(robot: Robot, *, approved_by: str | None, stop: int | None = None) -> None:
    """Release the latch; only a named operator can, and only for the stop they saw.

    ``stop`` pins the engagement being cleared to ``Latch.count``: a yes typed for stop #1 is refused
    once a later, different stop #2 is latched, so an approver never clears a hazard they never saw.
    """
    if not approved_by or not str(approved_by).strip():
        raise RequestRefused(f"{robot.id}: resume needs approved_by=<the operator who cleared the stop>")
    latch = robot.estop_latch
    if not latch.release(str(approved_by), stop=stop):  # compare-and-release: refused only once decided under the lock
        raise RequestRefused(
            f"{robot.id}: resume is for stop #{stop}, but the current stop is #{latch.count}; approve the stop you saw"
        )
