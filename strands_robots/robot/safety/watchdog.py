"""The watchdog: a heartbeat on the command path. When it lapses, every control applies its ``on_disconnect``.

The clock is ``robot.clock``: the world's time for a simulated body (so ``manual_time`` tests are exact),
the body's clock on hardware. ``poll()`` checks once; ``start()`` polls on a thread until ``stop()``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..robot import Robot


class Watchdog:
    """``beat()`` on every command; a lapse longer than ``timeout_ms`` applies ``on_disconnect`` once."""

    def __init__(self, robot: Robot, timeout_ms: float, *, clock: Callable[[], float] | None = None) -> None:
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be > 0, got {timeout_ms}")
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        self.last = self.clock()
        self.lapsed = False

    def poll(self) -> bool:
        """True when this call found a fresh lapse and applied ``on_disconnect``."""
        if self.lapsed or self.clock() - self.last <= self.timeout:
            return False
        self.lapsed = True
        apply_on_disconnect(self.robot)
        return True

    def start(self, period_s: float = 0.01) -> Watchdog:
        def loop() -> None:
            while not self._stop.wait(period_s):
                self.poll()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)


def apply_on_disconnect(robot: Robot) -> np.ndarray:
    """Write each control's declared disconnect target (hold / zero / safe / apply) straight to the body.

    While the e-stop is latched it writes nothing and returns the measured positions: ``estop()`` already asked
    the body for its own stop (``Body.estop``), and only ``resume`` lifts the latch."""
    body = robot.body
    with robot._lock:
        q = np.array(body.read().positions, dtype=float)
        if robot.estopped:
            return q
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
            robot._held = None
        else:
            from .estop import guarded_write

            guarded_write(robot, out)
    return out
