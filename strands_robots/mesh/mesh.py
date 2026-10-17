"""``Mesh``: every device, anywhere. Mount a robot to make it a device; discover, invoke and subscribe to the rest.

``Mesh("zenoh://", tenant="lab-a")`` joins the strands peers on the LAN with zero configuration;
``Mesh(("zenoh://", "iot://us-west-2"))`` spans several transports with one id space.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import authority as _authority
from . import watchdog as _watchdog
from . import wire
from .broadcast import broadcast, listen_estop, publish_estop
from .codec import observation_from_dict
from .device import Device, Selector
from .iot.options import attach as _iot_attach
from .presence import Directory, announce
from .record import RAISE, check_kill_switch, default_operator, default_tenant, record, state_rate
from .remote import RemoteBody
from .serve import ARGS, estop, serve
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
        """``tenant`` defaults to the user's own (``STRANDS_TENANT`` or the login name); ``operator`` (who this
        process is when it stops or resumes) to ``user@host``. ``resume_authority`` decides who may resume the
        devices mounted here; ``resume_key`` is this operator's private key for signed resumes."""
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
        """Make ``robot`` a device on every transport: announce it, answer its rpcs, publish its state (at
        ``state_hz``, default the manifest's rate) and events. Whoever moves it must keep talking within
        ``watchdog_ms`` and stay present, or the device applies ``on_disconnect`` (``None``: no watchdog). ``iot``
        (``iot.IotOptions``) adds shadows/telemetry/cameras on ``iot://``. A ``Simulation`` mounts every robot."""
        from ..simulation import Simulation

        if isinstance(robot, Simulation):
            kw = {"labels": labels, "state_hz": state_hz, "watchdog_ms": watchdog_ms, "iot": iot}
            return [self.mount(r, **kw) for r in robot.robots.values()]  # type: ignore[arg-type]
        did = wire.check_id(device_id or getattr(robot, "id", None) or robot.describe()["name"], "device_id")
        with self._lock:
            if did in self._mounted:
                raise ValueError(f"device {did!r} is already mounted on this mesh")
            rec = record(robot, did, labels or {}, self.tenant, tuple(ARGS))
            lockout = _authority.Lockout()
            undo = [] if iot is None else [_iot_attach(self.transports, iot, robot, rec, watchdog_ms, lockout)]
            dog = _watchdog.start(robot, watchdog_ms, self.transports, self.tenant)
            on_rpc = None if dog is None else dog.on_rpc
            undo += [serve(t, self.tenant, robot, did, lockout, self._authority, on_rpc) for t in self.transports]
            undo.append(_watchdog.stopper(dog))
            if hasattr(robot, "subscribe"):
                undo.append(publish(self.transports, self.tenant, robot, did, state_hz or state_rate(rec["manifest"])))
            undo += [announce(t, self.tenant, rec) for t in self.transports]
            device = Device.from_dict({**rec, "last_seen": time.time()})
            self._mounted[did] = (robot, device, undo, lockout)
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
        """Call ``rpc`` on a device; ``args`` are the rpc's own arguments (the device refuses any it does not take).

        A refusal from the device's safety chain raises ``RequestRefused`` with the device's message verbatim.
        """
        if rpc not in ARGS:
            raise LookupError(f"{device_id} has no rpc {rpc!r} over the mesh; known: {sorted(ARGS)}")
        if rpc == "estop":  # also on the tenant key: a lost or diverted rpc is never the only stop
            args.setdefault("by", self.operator)
            args.setdefault("epoch", uuid.uuid4().hex)
            publish_estop(self, f"device(id:{device_id})", dict(args))
        transport = self._transport_of(device_id)
        if rpc == "resume" and self.resume_key is not None and "assertion" not in args:
            lock = self.invoke(device_id, "lockout", timeout=timeout)
            if lock["engaged"]:
                by = str(args.get("approved_by") or self.operator)
                args["assertion"] = _authority.sign(self.resume_key, lock["epoch"], [device_id], by)
        self._seq += 1
        env = wire.envelope(self.tenant, device_id, "rpc", rpc, seq=self._seq, args=args, peer=self.peer_id)
        replies = transport.query(wire.key("rpc", self.tenant, device_id, rpc), wire.encode(env), timeout)
        if not replies:
            raise TimeoutError(f"{device_id}.{rpc}: no reply within {timeout} s")
        reply = wire.decode_envelope(replies[0])
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
        return broadcast(self, command, selector, deadline, approved_by)

    def _authority(self) -> _authority.Authority:
        return self.resume_authority

    def _on_estop(self, sel: Selector, by: str, epoch: str) -> None:
        with self._lock:
            mounted = list(self._mounted.values())
        for robot, device, _undo, lockout in mounted:
            if sel.matches(device):
                estop(robot, lockout, by, epoch)

    @property
    def tool(self) -> Any:
        """``use_mesh``: every device present as one strands agent tool, generated from their manifests."""
        from .tool import use_mesh

        return use_mesh(self)

    def connect(self, device_id: str, *, images: bool = False) -> Any:
        """A ``Robot`` whose Body is the device ``device_id`` on this mesh: observe/act/run work unchanged, every
        write goes through the device's own safety chain. ``images=True`` carries camera frames too."""
        from ..robot import Robot

        found = [d for d in self._present() if d.id == device_id]
        if not found:
            raise LookupError(f"no device {device_id!r} in tenant {self.tenant!r}")
        manifest = found[0].manifest
        name = str((manifest.get("description") or {}).get("name") or manifest.get("name"))
        names = tuple(str(k) for k in manifest.get("action_keys") or ())
        hold = bool(manifest.get("position_hold", False))
        body = RemoteBody(self, device_id, names, images=images, position_hold=hold)
        return Robot(name, driver=body, twin=False, id=device_id)

    def subscribe(self, device_id: str, event: str, cb: Callable[[Any], Any]) -> Unsubscribe:
        """Call ``cb(payload)`` for each ``event`` (``state``, ``estop``, ``fault``, ``run.started``, ``run.stopped``)
        from ``device_id``, with the payload the device's own ``subscribe`` gives. Returns ``unsubscribe``."""
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
        """Unmount everything, stop watching, close owned transports. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for did in list(self._mounted):
            self.unmount(did)
        for u in self._subs:
            u()
        for directory in self._dirs:
            directory.close()
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
