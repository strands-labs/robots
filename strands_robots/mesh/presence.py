"""Presence: a mounted device holds a liveliness token and answers for its record; a Directory watches the tokens.

A device that closes, crashes or loses the network drops its token (zenoh notices a dead peer within its lease,
set from ``presence_ttl``), and the Directory forgets it. The record (id, type, labels, manifest) is fetched once
per appearance, never polled.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from . import wire
from .device import Device
from .transports.base import Transport, Unsubscribe


def announce(transport: Transport, tenant: str, record: dict[str, Any]) -> Unsubscribe:
    """Answer for ``record`` on its manifest key, then take the presence token. Returns the undo (token first)."""
    did = record["id"]

    def answer(_key: str, _payload: bytes) -> bytes:
        return wire.encode(wire.envelope(tenant, did, "manifest", "manifest", result=record))

    stop_answering = transport.queryable(wire.key("manifest", tenant, did), answer)
    drop_token = transport.liveliness(wire.key("presence", tenant, did))

    def undo() -> None:
        drop_token()
        stop_answering()

    return undo


class Directory:
    """Every device of one tenant currently present on one transport."""

    def __init__(self, transport: Transport, tenant: str, *, fetch_timeout: float = 2.0) -> None:
        self.transport = transport
        self.tenant = tenant
        self.fetch_timeout = fetch_timeout
        self._lock = threading.Lock()
        self._devices: dict[str, Device] = {}
        self._alive: set[str] = set()
        self._closed = False
        self._changed = threading.Condition(self._lock)
        self._stop = transport.watch_liveliness(wire.key("presence", tenant, "*"), self._on_presence)

    def _on_presence(self, key: str, alive: bool) -> None:
        tenant, did = wire.parse_presence(key)
        if tenant != self.tenant:
            return
        with self._lock:
            if alive:
                self._alive.add(did)
            else:
                self._alive.discard(did)
                self._devices.pop(did, None)
                self._changed.notify_all()
        if alive:
            threading.Thread(target=self._fetch, args=(did,), daemon=True, name=f"mesh-fetch-{did}").start()

    def _fetch(self, did: str) -> None:
        end = time.monotonic() + self.fetch_timeout
        while time.monotonic() < end:
            with self._lock:
                if did not in self._alive or self._closed:
                    return
            try:
                replies = self.transport.query(wire.key("manifest", self.tenant, did), b"", 0.5)
            except Exception:  # the transport closed under us
                return
            for raw in replies:
                try:
                    env = wire.decode_envelope(raw)
                    device = Device.from_dict({**env["result"], "tenant": self.tenant, "last_seen": time.time()})
                except (ValueError, KeyError, TypeError):
                    continue
                if env["device_id"] != did or device.id != did:
                    continue  # a peer answering for another id never shadows the device that holds the token
                with self._lock:
                    if did in self._alive:
                        self._devices[did] = device
                        self._changed.notify_all()
                return
            time.sleep(0.05)

    def touch(self, did: str) -> None:
        """Note that ``did`` was just heard from (a state sample, an rpc reply)."""
        with self._lock:
            d = self._devices.get(did)
            if d is not None:
                self._devices[did] = Device(d.id, d.device_type, d.tenant, d.labels, d.manifest, time.time())

    def devices(self) -> list[Device]:
        with self._lock:
            return sorted(self._devices.values(), key=lambda d: d.id)

    def wait_changed(self, timeout: float) -> None:
        with self._changed:
            self._changed.wait(timeout)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._stop()
