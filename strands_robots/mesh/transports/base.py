"""The backend protocol a Mesh runs over, and the URL → transport front door.

A transport moves bytes on keys (``wire.KEYS``): publish/subscribe for state and events, query/queryable for
rpcs, and liveliness for presence. It knows nothing about devices, manifests or safety.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

Unsubscribe = Callable[[], None]
OnSample = Callable[[str, bytes], None]
OnQuery = Callable[[str, bytes], bytes]
OnAlive = Callable[[str, bool], None]

_SCHEMES: dict[str, str] = {
    "zenoh": "strands_robots.mesh.transports.zenoh",
    "iot": "strands_robots.mesh.transports.iot",
    "mem": "strands_robots.mesh.transports.memory",
}


class RemoteError(RuntimeError):
    """A queryable's handler raised instead of answering."""


@runtime_checkable
class Transport(Protocol):
    """Bytes on keys. Every callback runs on a transport thread and must not block."""

    url: str

    def publish(self, key: str, payload: bytes) -> None: ...

    def subscribe(self, key_expr: str, cb: OnSample) -> Unsubscribe: ...

    def queryable(self, key_expr: str, handler: OnQuery, *, inline: bool = False) -> Unsubscribe:
        """Answer queries with ``handler``. Handlers run on a worker pool so a slow one never delays another;
        ``inline=True`` answers on the transport thread itself (the e-stop path: no pool, no queue)."""
        ...

    def query(self, key: str, payload: bytes, timeout: float) -> list[bytes]:
        """Every reply within ``timeout``; raises ``RemoteError`` when the only answers were handler failures."""
        ...

    def liveliness(self, key: str) -> Unsubscribe: ...

    def watch_liveliness(self, key_expr: str, cb: OnAlive) -> Unsubscribe: ...

    def close(self) -> None: ...


def open_transport(url: str, *, presence_ttl: float = 3.0) -> Transport:
    """``zenoh://…``, ``iot://<region>``, ``mem://<name>`` → an open transport. A peer that dies is noticed
    within ``presence_ttl`` seconds."""
    scheme = urlparse(url).scheme
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown mesh transport {scheme!r} in {url!r}; known: {sorted(_SCHEMES)}")
    module = importlib.import_module(_SCHEMES[scheme])
    transport: Transport = module.open(url, presence_ttl=presence_ttl)
    return transport


def key_matches(key_expr: str, key: str) -> bool:
    """Key-expression match: ``*`` is one segment, ``**`` is any number of segments; a segment starting with ``@``
    is verbatim (only a literal matches it), as in zenoh."""
    return _match(key_expr.split("/"), key.split("/"))


def _match(pat: list[str], parts: list[str]) -> bool:
    if not pat:
        return not parts
    if pat[0] == "**":
        n = len(parts)
        stop = next((i for i, p in enumerate(parts) if p.startswith("@")), n)
        return any(_match(pat[1:], parts[i:]) for i in range(stop + 1))
    if not parts:
        return False
    wild = pat[0] == "*" and not parts[0].startswith("@")
    return (wild or pat[0] == parts[0]) and _match(pat[1:], parts[1:])
