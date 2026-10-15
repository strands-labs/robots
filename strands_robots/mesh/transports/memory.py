"""An in-process transport: every ``mem://<name>`` opened in this process shares one bus.

For tests and single-process worlds; it behaves like a peer-to-peer transport with zero latency.
"""

from __future__ import annotations

import threading
from typing import Any

from .base import OnAlive, OnQuery, OnSample, RemoteError, Unsubscribe, key_matches

_BUSES: dict[str, _Bus] = {}
_LOCK = threading.Lock()


class _Bus:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.subs: dict[int, tuple[str, OnSample]] = {}
        self.queryables: dict[int, tuple[str, OnQuery]] = {}
        self.tokens: dict[int, str] = {}
        self.watchers: dict[int, tuple[str, OnAlive]] = {}
        self.next = 0

    def add(self, table: dict[int, Any], item: Any) -> int:
        with self.lock:
            self.next += 1
            table[self.next] = item
            return self.next


class MemoryTransport:
    def __init__(self, url: str) -> None:
        self.url = url
        name = url.split("://", 1)[1] or "default"
        with _LOCK:
            self.bus = _BUSES.setdefault(name, _Bus())
        self._mine: list[Unsubscribe] = []

    def _own(self, table: dict[int, Any], item: Any) -> Unsubscribe:
        n = self.bus.add(table, item)

        def undo() -> None:
            with self.bus.lock:
                table.pop(n, None)

        self._mine.append(undo)
        return undo

    def publish(self, key: str, payload: bytes) -> None:
        with self.bus.lock:
            targets = [cb for expr, cb in self.bus.subs.values() if key_matches(expr, key)]
        for cb in targets:
            cb(key, payload)

    def subscribe(self, key_expr: str, cb: OnSample) -> Unsubscribe:
        return self._own(self.bus.subs, (key_expr, cb))

    def queryable(self, key_expr: str, handler: OnQuery, *, inline: bool = False) -> Unsubscribe:
        return self._own(self.bus.queryables, (key_expr, handler))

    def query(self, key: str, payload: bytes, timeout: float) -> list[bytes]:
        with self.bus.lock:
            handlers = [h for expr, h in self.bus.queryables.values() if key_matches(expr, key)]
        out, errors = [], []
        for h in handlers:
            try:
                out.append(h(key, payload))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors and not out:
            raise RemoteError(f"{key}: {errors[0]}")
        return out

    def _alive(self, key: str, alive: bool) -> None:
        with self.bus.lock:
            targets = [cb for expr, cb in self.bus.watchers.values() if key_matches(expr, key)]
        for cb in targets:
            cb(key, alive)

    def liveliness(self, key: str) -> Unsubscribe:
        undo = self._own(self.bus.tokens, key)
        self._alive(key, True)

        live = [True]

        def drop() -> None:
            if live[0]:
                live[0] = False
                undo()
                self._alive(key, False)

        self._mine[-1] = drop
        return drop

    def watch_liveliness(self, key_expr: str, cb: OnAlive) -> Unsubscribe:
        with self.bus.lock:
            present = [k for k in self.bus.tokens.values() if key_matches(key_expr, k)]
        undo = self._own(self.bus.watchers, (key_expr, cb))
        for k in present:
            cb(k, True)
        return undo

    def close(self) -> None:
        for undo in reversed(self._mine):
            undo()
        self._mine.clear()


def open(url: str, *, presence_ttl: float = 3.0) -> MemoryTransport:
    return MemoryTransport(url)
