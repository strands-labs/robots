"""The watchdog: a heartbeat on the command path. When it lapses, every control applies its ``on_disconnect``.

The clock is ``robot.clock``: the world's time for a simulated body (so ``manual_time`` tests are exact),
the body's clock on hardware. ``poll()`` checks once; ``start()`` polls on a thread until ``stop()`` or the
robot is closed. An apply that raises leaves the lapse open: the next poll retries it. Every apply, from any
caller, leaves one ``strands_robots.audit`` record (``apply_on_disconnect``). A lapse while the e-stop
is latched is the e-stop's.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..robot import Robot

AUDIT = logging.getLogger("strands_robots.audit")


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
        self.errors = 0  # polls that raised on the start() thread; the next period polls again
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        self.last = self.clock()
        self.lapsed = False

    def poll(self) -> bool:
        """True when this call found a fresh lapse and applied ``on_disconnect``, or left it to a latched e-stop.

        While the e-stop is latched the lapse is left to it, with nothing read or written. An apply that raises
        propagates, and the lapse stays open for the next poll to retry, unless an e-stop landed during it."""
        if self.lapsed or self.clock() - self.last <= self.timeout:
            return False
        self.lapsed = True  # claimed before the apply: a beat() that lands during it starts a new heartbeat
        if self.robot.estopped:  # estop() already asked the body to stop: a failed read would be retried after resume()
            return True
        stops = self.robot.estop_latch.count
        try:
            apply_on_disconnect(self.robot, reason="watchdog lapse")
        except BaseException:
            if self.robot.estop_latch.count == stops:  # a stop that landed during the apply takes the lapse too
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
                    if self._stop.is_set() or self.robot._body is None:
                        return  # stopped or closed during the poll: not a fault, as a state stream treats it
                    self.errors += 1
                    if not failing:
                        try:
                            what = f"{type(e).__name__}: {e}"
                        except Exception:  # an error whose message cannot be formatted is reported by its type
                            what = type(e).__name__
                        why = f"watchdog: poll failed ({what}); retrying every {period:g} s"
                        with contextlib.suppress(Exception):  # nor may a fault subscriber that raises end the thread
                            self.robot.events.emit("fault", {"device_id": self.robot.id, "reason": why})
                    failing = True
                else:
                    failing = False

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(1.0)  # a fault subscriber runs on that thread, and may call stop()


def apply_on_disconnect(
    robot: Robot,
    *,
    reason: str = "on_disconnect",
    peer: str | None = None,
    tenant: str | None = None,
    device_id: str | None = None,
) -> np.ndarray:
    """Write each control's declared disconnect target (hold / zero / safe / apply) straight to the body.

    While the e-stop is latched it writes nothing and returns the measured positions: ``estop()`` already asked
    the body for its own stop (``Body.estop``), and only ``resume`` lifts the latch. Every call leaves one record on
    the ``strands_robots.audit`` logger: why (``reason``), for whom (``peer``), what it applied and its outcome, under
    ``device_id`` (the robot's id unless it is mounted under another); one that raises is recorded with the
    exception's type, then re-raised."""
    where = (tenant, str(device_id or getattr(robot, "id", "?")))
    applied = "nothing"
    try:
        out, applied = _apply(robot)
    except BaseException as exc:
        _record(where, reason, peer, applied, type(exc).__name__)
        raise
    _record(where, reason, peer, applied, "ok")
    return out


def _apply(robot: Robot) -> tuple[np.ndarray, str]:
    body = robot.body
    with robot._lock:
        q = np.array(body.read().positions, dtype=float)
        if robot.estopped:
            return q, "nothing (e-stop latched)"
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
            return out, "hold"
        from .estop import guarded_write

        guarded_write(robot, out)
    return out, "targets"


def _record(where: tuple[str | None, str], reason: str, peer: str | None, applied: str, outcome: str) -> None:
    tenant, device = where
    rec = {"tenant": tenant, "device_id": device, "rpc": "on_disconnect", "peer": peer, "reason": reason}
    rec.update(applied=applied, outcome=outcome)
    line = "%s/%s on_disconnect reason=%s peer=%s applied=%s %s"
    AUDIT.warning(line, tenant, device, reason, peer, applied, outcome, extra=rec)
