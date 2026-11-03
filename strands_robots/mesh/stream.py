"""A mounted device's state and events, published on its keys for every subscriber on the mesh.

State is the in-process ``state`` payload (``<joint>.pos/.vel/.effort``, ``t``, ``seq``), sampled once on the
device at ``state_hz`` and published to every transport, however many peers listen. A subscriber sees gaps in
``seq`` as drops. Events (``estop``, ``fault``, ``run.started``, ``run.stopped``) are forwarded as they happen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from . import wire
from .transports.base import Transport, Unsubscribe

EVENTS: tuple[str, ...] = ("state", "fault", "estop", "run.started", "run.stopped")


def publish(transports: Sequence[Transport], tenant: str, robot: Any, device_id: str, state_hz: float) -> Unsubscribe:
    """Start publishing ``robot``'s state and events. Returns the undo."""
    seq = [0]

    def send(slot: str, kind: str, name: str, payload: dict[str, Any]) -> None:
        seq[0] += 1
        raw = wire.encode(wire.envelope(tenant, device_id, kind, name, seq=seq[0], result=payload))
        k = wire.key(slot, tenant, device_id, name if slot == "event" else "")
        for t in transports:
            try:
                t.publish(k, raw)
            except Exception:  # a closing transport must not kill the device's sampling thread
                pass

    undos = [robot.subscribe("state", lambda p: send("state", "state", "state", p), hz=state_hz)]
    for name in EVENTS[1:]:
        undos.append(robot.subscribe(name, lambda p, n=name: send("event", "event", n, p)))

    def undo() -> None:
        for u in undos:
            u()

    return undo


def listen(transport: Transport, tenant: str, device_id: str, event: str, cb: Callable[[Any], Any]) -> Unsubscribe:
    """Call ``cb(payload)`` for each ``event`` from ``device_id``. Malformed envelopes are dropped."""
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}; events are {list(EVENTS)}")
    k = wire.key("state", tenant, device_id) if event == "state" else wire.key("event", tenant, device_id, event)

    def on(_key: str, raw: bytes) -> None:
        try:
            env = wire.decode_envelope(raw)
        except ValueError:
            return
        cb(env.get("result"))

    return transport.subscribe(k, on)
