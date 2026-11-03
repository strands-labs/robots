"""In-process device events. ``state`` is the joints dict a policy sees, plus ``t`` (monotonic) and ``seq``.

``seq`` counts a stream's ticks: a tick whose sample failed (a corrupted reply, an I/O error) leaves a gap.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..robot import Robot

EVENTS: tuple[str, ...] = ("state", "fault", "estop", "run.started", "run.stopped")


def state_payload(robot: Robot, seq: int) -> dict[str, Any]:
    js = robot.observe(images=False).joints
    out: dict[str, Any] = {"t": time.monotonic(), "seq": seq}
    for i, name in enumerate(js.names):
        out[f"{name}.pos"] = float(js.positions[i])
        out[f"{name}.vel"] = float(js.velocities[i])
        out[f"{name}.effort"] = float(js.efforts[i])
    return out


class Emitter:
    """Subscribers per event; ``state`` subscribers each get their own sampling thread at their ``hz``.

    A tick that raises is counted in ``errors`` and reported as one ``fault`` event until a sample goes through
    again; the next read waits a full period, counted from when that event's subscribers (on the stream's thread)
    return. A stream ends only when unsubscribed or when the robot is closed."""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self._subs: dict[str, list[Callable[[dict[str, Any]], Any]]] = {e: [] for e in EVENTS}
        self._lock = threading.Lock()
        self._streams: list[Callable[[], None]] = []
        self.errors = 0  # state samples that raised, over every stream; each stream carries on at its rate

    def subscribe(
        self, event: str, cb: Callable[[dict[str, Any]], Any], *, hz: float | None = None
    ) -> Callable[[], None]:
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; events are {list(EVENTS)}")
        if event == "state":
            return self._state_stream(cb, hz)
        if hz is not None:
            raise TypeError(f"hz= only applies to the 'state' event, not {event!r}")
        with self._lock:
            self._subs[event].append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._subs[event]:
                    self._subs[event].remove(cb)

        return unsubscribe

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs[event])
        for cb in subs:
            cb(payload)

    def _state_stream(self, cb: Callable[[dict[str, Any]], Any], hz: float | None) -> Callable[[], None]:
        rate = (
            float(hz) if hz is not None else next(e.hz for e in self.robot.manifest.events if e.name == "state") or 10.0
        )
        if not 0 < rate < float("inf"):  # at an inf or nan hz the stream never waits between reads of the bus
            raise ValueError(f"hz must be a finite number > 0, got {rate}")
        stop = threading.Event()

        def loop() -> None:
            seq, period, nxt, failing = 0, 1.0 / rate, time.monotonic(), False
            while not stop.is_set():
                if self.robot._body is None:
                    return
                try:
                    cb(state_payload(self.robot, seq))
                except Exception as e:  # a corrupted reply or an I/O error must not end the stream
                    if stop.is_set() or self.robot._body is None:
                        return  # unsubscribed or closed during the sample: not a fault
                    self._failed(e, period, report=not failing)
                    failing, nxt = True, time.monotonic()  # a full period before the next read: no burst to catch up
                else:
                    failing = False
                seq += 1
                nxt += period
                stop.wait(max(0.0, nxt - time.monotonic()))

        t = threading.Thread(target=loop, daemon=True, name=f"{self.robot.id}-state")
        t.start()

        def unsubscribe() -> None:
            stop.set()
            if t is not threading.current_thread():
                t.join(1.0)

        self._streams.append(unsubscribe)
        return unsubscribe

    def _failed(self, e: Exception, period: float, *, report: bool) -> None:
        """Count a state sample that raised; ``report`` it as a ``fault`` event (the first of a run of failures)."""
        with self._lock:
            self.errors += 1
        if not report:
            return
        try:
            what = f"{type(e).__name__}: {e}"
        except Exception:  # an error whose message cannot be formatted is reported by its type
            what = type(e).__name__
        why = f"state stream: a sample failed ({what}); retrying {period:g} s after each failure"
        with contextlib.suppress(Exception):  # nor may a fault subscriber that raises end the stream
            self.emit("fault", {"device_id": self.robot.id, "reason": why})

    def close(self) -> None:
        """Stop every state stream (the body is going away)."""
        for unsubscribe in self._streams:
            unsubscribe()
        self._streams.clear()
