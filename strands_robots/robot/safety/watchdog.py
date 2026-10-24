"""The watchdog: a heartbeat on the command path. When it lapses, every control applies its ``on_disconnect``.

The clock is ``robot.clock``: the world's time for a simulated body (so ``manual_time`` tests are exact),
the body's clock on hardware. ``poll()`` checks once; ``start()`` polls on a thread until ``stop()`` or the
robot is closed. An apply that raises leaves the lapse open: the next poll retries it.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..robot import Robot


class Watchdog:
    """``beat()`` on every command; a lapse longer than ``timeout_ms`` applies ``on_disconnect`` once."""

    def __init__(self, robot: Robot, timeout_ms: float, *, clock: Callable[[], float] | None = None) -> None:
        if not 0 < timeout_ms < float("inf"):  # a NaN lapses at every poll after a beat, an inf never lapses
            raise ValueError(f"timeout_ms must be a finite number > 0, got {timeout_ms}")
        if robot.is_real and robot.description.hardware.on_disconnect is None:
            from .validators import RequestRefused

            raise RequestRefused(
                f"{robot.id}: no proven safe on_disconnect for this body (registry hardware.on_disconnect); "
                "it cannot be mounted where a watchdog is required"
            )
        self.robot = robot
        self.timeout = timeout_ms / 1000.0
        self.clock = clock or robot.clock.now
        self.last = self.clock()
        self.lapsed = False
        self.errors = 0  # polls that raised on the start() thread; each one is retried the next period
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        self.last = self.clock()
        self.lapsed = False

    def poll(self) -> bool:
        """True when this call found a fresh lapse and applied ``on_disconnect``.

        An apply that raises propagates, and the lapse stays open for the next poll to retry."""
        if self.lapsed or self.clock() - self.last <= self.timeout:
            return False
        self.lapsed = True  # claimed before the apply: a beat() that lands during it starts a new heartbeat
        try:
            apply_on_disconnect(self.robot)
        except BaseException:
            self.lapsed = False
            raise
        return True

    def start(self, period_s: float = 0.01) -> Watchdog:
        """Poll every ``period_s`` on a daemon thread until ``stop()`` or the robot is closed.

        A poll that raises is counted in ``errors`` and retried the next period. Each run of failures is one ``fault``
        event, whose subscribers run on this thread: the retry waits until they return."""
        period = float(period_s)  # Event.wait() takes a float or an int: an np.float32 or a Decimal ends the thread
        if not 0 < period <= threading.TIMEOUT_MAX:  # 0, negative or NaN spin a core; above the max, wait() overflows
            raise ValueError(f"period_s must be in (0, {threading.TIMEOUT_MAX:.0f}] seconds, got {period_s}")

        def loop() -> None:
            failing = False
            while not self._stop.wait(period) and self.robot._body is not None:
                try:
                    self.poll()
                except Exception as e:  # a corrupted reply or an I/O error must not end the watchdog
                    self.errors += 1
                    if not failing:
                        with contextlib.suppress(Exception):  # nor may an error that cannot print, or a subscriber
                            why = f"watchdog: poll failed ({type(e).__name__}: {e}); retrying every {period:g} s"
                            self.robot.events.emit("fault", {"device_id": self.robot.id, "reason": why})
                    failing = True
                else:
                    failing = False

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)


def apply_on_disconnect(robot: Robot) -> np.ndarray:
    """Write each control's declared disconnect target (hold / zero / safe / apply) straight to the body."""
    body = robot.body
    with robot._lock:
        q = np.array(body.read().positions, dtype=float)
        out = q.copy()
        for i, c in enumerate(robot.manifest.controls):
            if c.on_disconnect == "zero":
                out[i] = 0.0
            elif c.on_disconnect == "apply":
                out[i] = float(c.disconnect_value)  # type: ignore[arg-type]
            elif c.on_disconnect == "safe" and c.safe is not None:
                out[i] = min(max(q[i], c.safe[0]), c.safe[1])
        if all(c.on_disconnect == "hold" for c in robot.manifest.controls):
            body.hold()
        else:
            from .estop import guarded_write

            guarded_write(robot, out)
    return out
