"""The device watchdog over the mesh: whoever last moved a device must stay present and keep talking.

A motion rpc (``act``) arms the watchdog for its invoker; one the device refuses (bad arguments, the safety chain)
leaves it as it was, so a request that moved nothing can never make the device apply ``on_disconnect``. Any rpc
from that invoker is a heartbeat. If the invoker goes quiet for ``watchdog_ms`` or its Mesh's liveliness token
vanishes (closed, crashed, network gone), the device applies its ``on_disconnect`` once (an audit warning says so)
and disarms until the next motion. Reads never arm it. The quiet clock does not run
while a motion rpc is still moving the device (an ``act(seconds=)`` reach): its invoker is waiting on the reply. A
skill (``mesh/skills.py``) holds the watchdog for its whole run (``busy``): the quiet clock pauses, a vanished
invoker still ends it (``cuts``) before ``on_disconnect`` is applied.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..robot.safety import Watchdog, apply_on_disconnect
from . import wire
from .transports.base import Transport, Unsubscribe

MOTION_RPCS = ("act",)
AUDIT = logging.getLogger("strands_robots.audit")


class MeshWatchdog:
    def __init__(self, robot: Any, timeout_ms: float, transports: Sequence[Transport], tenant: str) -> None:
        self.robot = robot
        self.dog = Watchdog(robot, timeout_ms, clock=time.monotonic)
        self.peer: str | None = None
        self._lock = threading.Lock()
        self._moving = 0  # motion rpcs in progress: their invoker is waiting, not quiet
        self.cuts: list[Callable[[], None]] = []  # what must stop before on_disconnect (a running skill)
        self._armed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mesh-watchdog")
        self._thread.start()
        self._unwatch = [t.watch_liveliness(wire.key("peer", tenant, "*"), self._on_peer) for t in transports]

    def on_rpc(self, name: str, peer: str | None) -> Callable[[bool], None] | None:
        """Every rpc, before it runs; a motion rpc returns what to call once it ends (that is a beat too). A motion
        pauses the quiet clock while it runs; only one the device carried out (``ok``) arms the watchdog for its
        invoker, so a refused request, or several at once, never touches who is watched, nor its quiet clock."""
        with self._lock:
            if name in MOTION_RPCS and peer is not None:
                if peer == self.peer:  # only the watched mover's own requests keep its clock alive
                    self.dog.beat()
                self._moving += 1
                return functools.partial(self._moved, peer=peer)
            if self._armed.is_set() and peer == self.peer:
                self.dog.beat()
        return None

    def _moved(self, ok: bool = True, peer: str | None = None) -> None:
        with self._lock:
            self._moving -= 1
            if ok or peer == self.peer:  # a refusal from another peer is no sign of life from the mover
                self.dog.beat()
            if ok and peer is not None:  # the device moved for this peer: it is the one that must stay present
                self.peer = peer
                self._armed.set()

    def busy(self, peer: str) -> Callable[[], None]:
        """A skill started for ``peer``: armed for it now, quiet clock paused until the returned release is called."""
        with self._lock:
            self._moving += 1
            self.peer = peer
            self._armed.set()
            self.dog.beat()
        released = threading.Event()

        def release() -> None:
            if released.is_set():
                return
            released.set()
            with self._lock:
                self._moving -= 1
                self.dog.beat()

        return release

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
        AUDIT.warning("%s on_disconnect: its mover went quiet or vanished", getattr(self.robot, "id", "?"))
        for cut in self.cuts:  # a skill still running for the vanished mover ends first, then the robot holds
            cut()
        try:
            apply_on_disconnect(self.robot)
        except Exception:  # a closed robot cannot hold; nothing left to protect
            pass

    def _loop(self) -> None:
        period = min(0.01, self.dog.timeout / 10)
        while not self._stop.is_set():
            if not self._armed.wait(0.1):
                continue
            if not self._moving and time.monotonic() - self.dog.last > self.dog.timeout:
                self._fire()
            self._stop.wait(period)

    def close(self) -> None:
        while self._unwatch:  # idempotent: a failed mount and its undo may both close it
            self._unwatch.pop()()
        self._stop.set()
        self._thread.join(1.0)


def start(robot: Any, timeout_ms: float | None, transports: Sequence[Transport], tenant: str) -> MeshWatchdog | None:
    """A watchdog for a device with controls (a manifest says what ``on_disconnect`` means); ``None`` otherwise."""
    if timeout_ms is None or not hasattr(robot, "manifest"):
        return None
    return MeshWatchdog(robot, timeout_ms, transports, tenant)


def stopper(dog: MeshWatchdog | None) -> Unsubscribe:
    return (lambda: None) if dog is None else dog.close
