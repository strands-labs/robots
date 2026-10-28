"""A joint speed cap on a real body's writes: a far goal becomes a ramp, trained motion is untouched.

A position servo given a goal 50° away goes there at full torque. A policy's first chunk row from a folded rest, a
mapping with one wrong number, an ``act`` typed by hand: all of them reach the bus as one write. ``Slew`` caps each
write to ``max_speed`` radians per second times the time since the previous write (a first write ramps from the last
measured position), so the body moves at most that fast toward the goal and the next write continues from where the
goal got to. The default (``MAX_SPEED``) is well above what a trained policy commands and well below the servo's own.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

MAX_SPEED = 2.0  # rad/s (~115 deg/s): a trained chunk moves slower, an STS3215 can move ~5 rad/s
DT_MIN, DT_MAX = 1 / 200, 0.5  # the step a write may take: never under one bus period, never a half second's worth


class Slew:
    """``apply(targets)`` on every write; ``seen(positions)`` on every read; ``release()`` when torque goes off."""

    def __init__(self, max_speed: float | None = MAX_SPEED, *, now: Callable[[], float] = time.monotonic) -> None:
        if max_speed is not None and not max_speed > 0:
            raise ValueError(f"max_speed must be > 0 rad/s or None (no cap), got {max_speed}")
        self.max_speed, self._now = max_speed, now
        self._goal: np.ndarray | None = None  # the last goal written: the ramp continues from it
        self._seen: np.ndarray | None = None  # the last read: a first write ramps from where the body is
        self._t: float | None = None  # when the anchor was set
        self.count = 0  # writes the cap shortened

    def seen(self, positions: np.ndarray) -> None:
        self._seen = np.asarray(positions, dtype=float).copy()
        if self._goal is None:
            self._t = self._now()

    def release(self) -> None:
        """Torque off: the body may be moved by hand, so the next write ramps from a fresh read, not an old one."""
        self._goal = self._seen = self._t = None

    def reset(self, anchor: np.ndarray) -> None:
        """A hold: the goal IS ``anchor`` (the measured pose), whatever the last goal was and however far away.

        The cap exists to stop a far goal; a hold is the opposite, Goal_Position := Present_Position, and must not
        be bent toward the old goal. The ramp continues from ``anchor`` afterwards."""
        self._goal, self._t = np.asarray(anchor, dtype=float).copy(), self._now()

    def apply(self, targets: np.ndarray, read: Callable[[], object]) -> np.ndarray:
        """The goal to write now: ``targets``, or as far toward them as ``max_speed`` allows since the last write.

        With a cap and no anchor (a fresh driver, or torque went off since the last read) ``read()`` is called
        first, so the cap never depends on the caller having read."""
        t = np.asarray(targets, dtype=float)
        if self.max_speed is not None and self._goal is None and self._seen is None:
            read()  # -> seen(): one bus round-trip, once
        now = self._now()
        anchor = self._goal if self._goal is not None else self._seen
        if self.max_speed is not None and anchor is not None:
            dt = DT_MAX if self._t is None else min(DT_MAX, max(DT_MIN, now - self._t))
            step = np.clip(t - anchor, -self.max_speed * dt, self.max_speed * dt)
            if np.any(np.abs(t - anchor) > self.max_speed * dt + 1e-12):
                self.count += 1
            t = anchor + step
        self._goal, self._t = t.copy(), now
        return t
