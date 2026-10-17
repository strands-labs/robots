"""One command to many devices: e-stop, resume, hold, stop.

``estop`` goes out twice, at once: on the tenant's ``@estop`` key, which every mesh in the tenant listens to and
answers by latching its matching devices on the spot (no discovery, no ack, no queue), and as an rpc to each
discovered match, whose replies are the acks. A late or undiscovered device still latches.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import wire
from .device import Device, Selector
from .transports.base import Transport, Unsubscribe

if TYPE_CHECKING:
    from .mesh import Mesh

COMMANDS = ("estop", "resume", "hold", "stop")


def broadcast(mesh: Mesh, command: str, selector: str, deadline: float, approved_by: str | None) -> list[str]:
    """Send ``command`` to every device matching ``selector``; return the ids that acked within ``deadline`` s."""
    if command not in COMMANDS:
        raise ValueError(f"cannot broadcast {command!r}; commands are {list(COMMANDS)}")
    sel = Selector.parse(selector)
    args: dict[str, Any] = {}
    if command == "estop":
        args = {"by": mesh.operator, "epoch": uuid.uuid4().hex}
        publish_estop(mesh, selector, args)
    elif command == "resume":
        args = {"approved_by": approved_by}
    if deadline <= 0:
        return []
    acked: list[str] = []
    lock = threading.Lock()

    def one(d: Device) -> None:
        try:
            mesh.invoke(d.id, command, timeout=deadline, **args)
        except Exception:
            return
        with lock:
            acked.append(d.id)

    threads = [threading.Thread(target=one, args=(d,), daemon=True) for d in mesh.discover() if sel.matches(d)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(deadline)
    with lock:
        return sorted(acked)


def publish_estop(mesh: Mesh, selector: str, args: dict[str, Any]) -> None:
    """The tenant-wide e-stop: every mesh in the tenant latches its devices matching ``selector`` on hearing it."""
    raw = wire.encode(wire.envelope(mesh.tenant, mesh.peer_id, "event", "estop", args={**args, "selector": selector}))
    for t in mesh.transports:
        try:
            t.publish(wire.key("estop", mesh.tenant), raw)
        except Exception:  # one transport down must not stop the e-stop going out on the others
            pass


def listen_estop(transport: Transport, tenant: str, on_estop: Callable[[Selector, str, str], None]) -> Unsubscribe:
    """Call ``on_estop(selector, by, epoch)`` for every tenant-wide e-stop; malformed ones latch everything."""

    def on(_key: str, raw: bytes) -> None:
        try:
            a = wire.decode_envelope(raw).get("args") or {}
            sel = Selector.parse(str(a.get("selector", "device(*)")))
        except ValueError:
            sel, a = Selector(), {}
        on_estop(sel, str(a.get("by") or "unknown"), str(a.get("epoch") or uuid.uuid4().hex))

    return transport.subscribe(wire.key("estop", tenant), on)
