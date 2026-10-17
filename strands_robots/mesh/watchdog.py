"""The device watchdog over the mesh: whoever last moved a device must stay present and keep talking.

A motion rpc (``act``) arms the watchdog for its invoker. Any rpc from that invoker is a heartbeat. If the invoker
goes quiet for ``watchdog_ms`` or its Mesh's liveliness token vanishes (closed, crashed, network gone), the device
applies its ``on_disconnect`` once and disarms until the next motion. Reads never arm it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

from ..robot.safety import Watchdog, apply_on_disconnect
from . import wire
from .transports.base import Transport, Unsubscribe

MOTION_RPCS = ("act",)


class MeshWatchdog:
    def __init__(self, robot: Any, timeout_ms: float, transports: Sequence[Transport], tenant: str) -> None:
        self.robot = robot
        self.dog = Watchdog(robot, timeout_ms, clock=time.monotonic)
        self.peer: str | None = None
        self._lock = threading.Lock()
        self._armed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mesh-watchdog")
        self._thread.start()
        self._unwatch = [t.watch_liveliness(wire.key("peer", tenant, "*"), self._on_peer) for t in transports]

    def on_rpc(self, name: str, peer: str | None) -> None:
        with self._lock:
            if name in MOTION_RPCS and peer is not None:
                self.peer = peer
                self.dog.beat()
                self._armed.set()
            elif self._armed.is_set() and peer == self.peer:
                self.dog.beat()

    def _on_peer(self, key: str, alive: bool) -> None:
        if alive or not self._armed.is_set() or key.rsplit("/", 1)[-1] != self.peer:
            return
        self._fire()

    def _fire(self) -> None:
        with self._lock:
            if not self._armed.is_set():
                return
            self._armed.clear()
            self.peer = None
        try:
            apply_on_disconnect(self.robot)
        except Exception:  # a closed robot cannot hold; nothing left to protect
            pass

    def _loop(self) -> None:
        period = min(0.01, self.dog.timeout / 10)
        while not self._stop.is_set():
            if not self._armed.wait(0.1):
                continue
            if time.monotonic() - self.dog.last > self.dog.timeout:
                self._fire()
            self._stop.wait(period)

    def close(self) -> None:
        for u in self._unwatch:
            u()
        self._stop.set()
        self._thread.join(1.0)


def start(robot: Any, timeout_ms: float | None, transports: Sequence[Transport], tenant: str) -> MeshWatchdog | None:
    """A watchdog for a device with controls (a manifest says what ``on_disconnect`` means); ``None`` otherwise."""
    if timeout_ms is None or not hasattr(robot, "manifest"):
        return None
    return MeshWatchdog(robot, timeout_ms, transports, tenant)


def stopper(dog: MeshWatchdog | None) -> Unsubscribe:
    return (lambda: None) if dog is None else dog.close
