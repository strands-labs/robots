"""``Mesh``: every device, anywhere. Mount a robot to make it a device; discover, invoke and subscribe to the rest.

``Mesh("zenoh://", tenant="lab-a")`` joins the strands peers on the LAN with zero configuration; ``Mesh(("zenoh://",
"iot://us-west-2"))`` spans several transports with one id space. ``Robot("so101", mesh=m)`` mounts it on ``m``."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from typing import Any

from . import authority as _authority
from . import watchdog as _watchdog
from . import wire
from .broadcast import broadcast, listen_estop, publish_estop
from .codec import observation_from_dict
from .device import Device, Selector
from .iot.options import attach as _iot_attach
from .presence import Directory, announce, refuse_shared, refuse_taken
from .record import RAISE, check_kill_switch, default_operator, default_tenant, record, state_rate
from .remote import connect
from .reply import the_reply
from .serve import ARGS, act_wait, estop_each, serve
from .stream import listen, publish
from .transports.base import Transport, Unsubscribe, open_transport

_OBSERVATION_RPCS = ("observe", "act")


class Mesh:
    """One tenant's view of the mesh over one or more transports."""

    def __init__(
        self,
        transports: str | Transport | Sequence[str | Transport] = "zenoh://",
        *,
        tenant: str | None = None,
        presence_ttl: float = 3.0,
        operator: str | None = None,
        resume_authority: _authority.Authority | None = None,
        resume_key: Any = None,
        own_transports: bool = True,
    ) -> None:
        """``tenant`` defaults to ``STRANDS_TENANT`` or the login name; ``operator`` (who stops or resumes) to
        ``user@host``. ``resume_authority`` decides who may resume devices mounted here; ``resume_key`` signs."""
        check_kill_switch()
        items = [transports] if isinstance(transports, str) or not isinstance(transports, Sequence) else transports
        self.tenant = wire.check_id(tenant or default_tenant(), "tenant")
        self.operator = operator or default_operator()
        self.resume_authority = resume_authority or _authority.Authority()
        self.resume_key = resume_key
        self.peer_id = uuid.uuid4().hex
        self.presence_ttl = float(presence_ttl)
        self.transports: list[Transport] = [
            open_transport(t, presence_ttl=self.presence_ttl) if isinstance(t, str) else t for t in items
        ]
        self._own = own_transports
        self._lock = threading.Lock()
        self._mounted: dict[str, tuple[Any, Device, list[Unsubscribe], _authority.Lockout]] = {}
        self._dirs = [Directory(t, self.tenant) for t in self.transports]
        self._closed = False
        self._seq = 0
        self.tasks: dict[str, Any] = {}  # runs started through the tool, by device id
        self._subs: list[Unsubscribe] = [listen_estop(t, self.tenant, self._on_estop) for t in self.transports]
        self._subs += [t.liveliness(wire.key("peer", self.tenant, self.peer_id)) for t in self.transports]

    def mount(
        self,
        robot: Any,
        *,
        device_id: str | None = None,
        labels: Mapping[str, str] | None = None,
        state_hz: float | None = None,
        watchdog_ms: float | None = 500.0,
        iot: Any = None,
    ) -> Any:
        """Make ``robot`` a device: announce it, answer its rpcs, publish its state and events; a mover silent past
        ``watchdog_ms`` gets ``on_disconnect``. ``iot`` adds shadows. A ``Simulation`` mounts each robot in it."""
        from ..simulation import Simulation

        self._open()
        if isinstance(robot, Simulation):
            kw = {"labels": labels, "state_hz": state_hz, "watchdog_ms": watchdog_ms, "iot": iot}
            return [self.mount(r, **kw) for r in robot.robots.values()]  # type: ignore[arg-type]
        did = wire.check_id(device_id or getattr(robot, "id", None) or robot.describe()["name"], "device_id")
        with ExitStack() as stack, self._lock:  # the stack unwinds a failed mount after the lock is released
            if did in self._mounted:
                raise ValueError(f"device {did!r} is already mounted on this mesh")
            refuse_taken(self._dirs, did, self.tenant)
            rec = record(robot, did, labels or {}, self.tenant, tuple(ARGS))
            lockout = _authority.Lockout()
            undo: list[Unsubscribe] = []
            stack.callback(lambda: [u() for u in reversed(undo)])
            undo += [] if iot is None else [_iot_attach(self.transports, iot, robot, rec, lockout)]
            dog = _watchdog.start(robot, watchdog_ms, self.transports, self.tenant)
            stack.callback(_watchdog.stopper(dog))  # a failed mount stops it first; close is idempotent
            on_rpc = getattr(dog, "on_rpc", None)
            undo += [serve(t, self.tenant, robot, did, lockout, self._authority, on_rpc) for t in self.transports]
            undo.append(_watchdog.stopper(dog))
            if hasattr(robot, "subscribe"):
                undo.append(publish(self.transports, self.tenant, robot, did, state_hz or state_rate(rec["manifest"])))
            undo += [announce(t, self.tenant, rec, self.peer_id) for t in self.transports]
            device = Device.from_dict({**rec, "last_seen": time.time()})
            self._mounted[did] = (robot, device, undo, lockout)
            stack.pop_all()
        return device

    def unmount(self, device_id: str) -> None:
        """Withdraw a mounted device; peers see it leave."""
        with self._lock:
            entry = self._mounted.pop(device_id, None)
        if entry is None:
            raise KeyError(f"device {device_id!r} is not mounted on this mesh")
        for undo in reversed(entry[2]):
            undo()

    def discover(self, selector: str = "device(*)", *, min_count: int = 0, timeout: float = 0.0) -> list[Device]:
        """Devices present now that match ``selector``; wait up to ``timeout`` s for at least ``min_count``."""
        sel = Selector.parse(selector)
        end = time.monotonic() + timeout
        while True:
            found = [d for d in self._present() if sel.matches(d)]
            left = end - time.monotonic()
            if len(found) >= min_count or left <= 0:
                return found
            self._dirs[0].wait_changed(min(left, 0.05))

    def invoke(self, device_id: str, rpc: str, *, timeout: float = 2.0, **args: Any) -> Any:
        """Call ``rpc`` on a device with its own ``args`` (it refuses others); ``act(seconds=)`` waits that much longer.
        A refusal from the device's safety chain raises ``RequestRefused`` with the device's message verbatim."""
        self._open()
        if rpc not in ARGS:
            raise LookupError(f"{device_id} has no rpc {rpc!r} over the mesh; known: {sorted(ARGS)}")
        if rpc == "estop":  # also on the tenant key: a lost or diverted rpc is never the only stop
            args.setdefault("by", self.operator)
            args.setdefault("epoch", uuid.uuid4().hex)
            publish_estop(self, f"device(id:{device_id})", dict(args))
        if rpc in ("act", "hold", "stop"):  # who moved it, for the device's audit record
            args.setdefault("by", self.operator)
        refuse_shared(self._dirs, device_id, rpc, self.tenant)
        transport = self._transport_of(device_id)
        if rpc == "resume" and self.resume_key is not None and "assertion" not in args:
            lock = self.invoke(device_id, "lockout", timeout=timeout)
            if lock["engaged"]:
                by = str(args.get("approved_by") or self.operator)
                args["assertion"] = _authority.sign(self.resume_key, lock["epoch"], [device_id], by)
        self._seq = seq = self._seq + 1
        env = wire.envelope(self.tenant, device_id, "rpc", rpc, seq=seq, args=args, peer=self.peer_id)
        timeout += act_wait(args) if rpc == "act" else 0.0
        replies = transport.query(wire.key("rpc", self.tenant, device_id, rpc), wire.encode(env), timeout)
        if not replies:
            raise TimeoutError(f"{device_id}.{rpc}: no reply within {timeout} s")
        reply = the_reply(replies, self.tenant, device_id, rpc, seq)
        if reply["kind"] == "error":
            err = reply.get("error") or {}
            raise RAISE.get(err.get("type", ""), RuntimeError)(err.get("message", "remote error"))
        for d in self._dirs:
            d.touch(device_id)
        result = reply.get("result")
        return observation_from_dict(result) if rpc in _OBSERVATION_RPCS and result is not None else result

    def broadcast(
        self, command: str, *, selector: str = "device(*)", deadline: float = 1.0, approved_by: str | None = None
    ) -> list[str]:
        """``estop``/``resume``/``hold``/``stop`` every device matching ``selector``; the ids that acked within
        ``deadline`` s. An ``estop`` also reaches devices this mesh has not discovered (they latch, unacked)."""
        self._open()
        return broadcast(self, command, selector, deadline, approved_by)

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self!r} is closed; make a new Mesh")

    def _authority(self) -> _authority.Authority:
        return self.resume_authority

    def _on_estop(self, sel: Selector, by: str, epoch: str) -> None:
        with self._lock:
            mounted = list(self._mounted.values())
        estop_each(mounted, sel, by, epoch, self.tenant)

    @property
    def tool(self) -> Any:
        """``use_mesh``: every device present as one strands agent tool, generated from their manifests."""
        from .tool import use_mesh

        return use_mesh(self)

    def connect(self, device_id: str, *, images: bool = False) -> Any:
        """A ``Robot`` whose Body is the device ``device_id`` on this mesh: observe/act/run work unchanged, every
        write goes through the device's own safety chain. ``images=True`` carries camera frames too."""
        self._open()
        return connect(self, device_id, images)

    def subscribe(self, device_id: str, event: str, cb: Callable[[Any], Any]) -> Unsubscribe:
        """Call ``cb(payload)`` for each ``event`` (``state``, ``estop``, ``fault``, ``run.started``, ``run.stopped``)
        from ``device_id``, with the payload the device's own ``subscribe`` gives. Returns ``unsubscribe``."""
        self._open()
        wire.check_id(device_id, "device_id")
        undos = [listen(t, self.tenant, device_id, event, cb) for t in self.transports]
        self._subs.extend(undos)

        def unsubscribe() -> None:
            for u in undos:
                u()

        return unsubscribe

    def _transport_of(self, device_id: str) -> Transport:
        for t, directory in zip(self.transports, self._dirs, strict=True):
            if any(d.id == device_id for d in directory.devices()):
                return t
        raise LookupError(
            f"no device {device_id!r} in tenant {self.tenant!r}; present: {[d.id for d in self._present()]}"
        )

    def _present(self) -> list[Device]:
        seen: dict[str, Device] = {}
        for directory in self._dirs:
            for d in directory.devices():
                if d.id not in seen or d.last_seen > seen[d.id].last_seen:
                    seen[d.id] = d
        return sorted(seen.values(), key=lambda d: d.id)

    def close(self) -> None:
        """Unmount everything, stop watching, close owned transports. Idempotent; every other verb then refuses."""
        if self._closed:
            return
        self._closed = True
        for did in list(self._mounted):
            self.unmount(did)
        for stop in [*self._subs, *(d.close for d in self._dirs)]:
            stop()
        if self._own:
            for t in self.transports:
                t.close()

    def __enter__(self) -> Mesh:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        urls = ", ".join(t.url for t in self.transports)
        return f"Mesh({urls!r}, tenant={self.tenant!r}, mounted={sorted(self._mounted)})"
