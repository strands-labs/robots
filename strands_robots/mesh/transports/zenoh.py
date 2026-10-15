"""The zenoh transport: peer mode, scouting only on the strands group, liveliness tokens for presence.

``zenoh://`` scouts on ``224.0.0.224:7447`` (zero config on a LAN). ``zenoh://?listen=tcp/0.0.0.0:7447`` and/or
``?connect=tcp/host:7447`` give explicit endpoints and switch scouting off. ``?scout=addr:port`` or
``STRANDS_ZENOH_SCOUT`` move the group.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..._extras import hint
from .base import OnAlive, OnQuery, OnSample, RemoteError, Unsubscribe

SCOUT_GROUP = "224.0.0.224:7447"
"""zenoh's default group (port 7446) is shared with every zenoh application on the LAN; foreign peers there were
measured to make ``session.close()`` hang. One port up, only strands-robots peers find each other."""

_OPEN: dict[int, Any] = {}


def _zenoh() -> Any:
    try:
        import zenoh
    except ImportError as exc:
        raise ImportError(f"the zenoh mesh transport needs eclipse-zenoh: {hint('zenoh')}") from exc
    return zenoh


def config(url: str, presence_ttl: float = 3.0) -> dict[str, Any]:
    """The zenoh config for a ``zenoh://`` URL. The link lease is two thirds of ``presence_ttl``, so a peer killed
    without a goodbye is dropped (and its liveliness tokens with it) inside the ttl."""
    u = urlparse(url)
    if u.scheme != "zenoh":
        raise ValueError(f"not a zenoh url: {url!r}")
    q = parse_qs(u.query)
    lease_ms = max(500, int(presence_ttl * 1000 * 2 / 3))
    cfg: dict[str, Any] = {"mode": "peer", "transport": {"link": {"tx": {"lease": lease_ms, "keep_alive": 4}}}}
    if "listen" in q:
        cfg["listen"] = {"endpoints": q["listen"]}
    if "connect" in q:
        cfg["connect"] = {"endpoints": q["connect"]}
    if "listen" in q or "connect" in q:
        cfg["scouting"] = {"multicast": {"enabled": False}}
    else:
        group = (q.get("scout") or [os.environ.get("STRANDS_ZENOH_SCOUT", SCOUT_GROUP)])[0]
        cfg["scouting"] = {"multicast": {"enabled": True, "address": group}}
    return cfg


def _close(session: Any, timeout: float) -> bool:
    """Close in a thread: a close waiting on a dead peer must not hang the caller."""
    _OPEN.pop(id(session), None)
    th = threading.Thread(target=session.close, daemon=True)
    th.start()
    th.join(timeout)
    return not th.is_alive()


def _close_all() -> None:
    for session in list(_OPEN.values()):
        _close(session, 1.0)


# A session left open keeps a non-daemon callback thread alive and the interpreter hangs at exit; close every
# session from the hook that runs before threads are joined.
_register = getattr(threading, "_register_atexit", None)
if _register is not None:
    _register(_close_all)
else:  # pragma: no cover
    atexit.register(_close_all)


def _undeclarer(entity: Any) -> Unsubscribe:
    done = [False]

    def undo() -> None:
        if not done[0]:
            done[0] = True
            try:
                entity.undeclare()
            except Exception:  # the session closed first and took the declaration with it
                pass

    return undo


class ZenohTransport:
    def __init__(self, url: str, presence_ttl: float = 3.0) -> None:
        zenoh = _zenoh()
        self.url = url
        self._zenoh = zenoh
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mesh-rpc")
        self.session = zenoh.open(zenoh.Config.from_json5(json.dumps(config(url, presence_ttl))))
        _OPEN[id(self.session)] = self.session

    def publish(self, key: str, payload: bytes) -> None:
        self.session.put(key, payload)

    def subscribe(self, key_expr: str, cb: OnSample) -> Unsubscribe:
        def on(sample: Any) -> None:
            cb(str(sample.key_expr), sample.payload.to_bytes())

        return _undeclarer(self.session.declare_subscriber(key_expr, on))

    def queryable(self, key_expr: str, handler: OnQuery, *, inline: bool = False) -> Unsubscribe:
        def answer(query: Any) -> None:
            try:
                payload = query.payload.to_bytes() if query.payload is not None else b""
                query.reply(query.key_expr, handler(str(query.key_expr), payload))
            except Exception as exc:
                try:
                    query.reply_err(f"{type(exc).__name__}: {exc}")
                except Exception:  # the caller is gone: no one is left to tell, and its own timeout reports it
                    pass
            finally:
                query.drop()

        def on(query: Any) -> None:
            if inline:
                answer(query)
            else:
                self._pool.submit(answer, query)

        return _undeclarer(self.session.declare_queryable(key_expr, on))

    def query(self, key: str, payload: bytes, timeout: float) -> list[bytes]:
        out, errors = [], []
        for r in self.session.get(key, payload=payload, timeout=timeout):
            if r.ok is not None:
                out.append(r.ok.payload.to_bytes())
            elif r.err is not None:
                errors.append(r.err.payload.to_string())
        if errors and not out:
            raise RemoteError(f"{key}: {errors[0]}")
        return out

    def liveliness(self, key: str) -> Unsubscribe:
        return _undeclarer(self.session.liveliness().declare_token(key))

    def watch_liveliness(self, key_expr: str, cb: OnAlive) -> Unsubscribe:
        put = self._zenoh.SampleKind.PUT

        def on(sample: Any) -> None:
            cb(str(sample.key_expr), sample.kind == put)

        sub = self.session.liveliness().declare_subscriber(key_expr, on, history=True)
        return _undeclarer(sub)

    def close(self) -> None:
        if not self.session.is_closed():
            _close(self.session, 2.0)
        self._pool.shutdown(wait=False, cancel_futures=True)


def open(url: str, *, presence_ttl: float = 3.0) -> ZenohTransport:
    return ZenohTransport(url, presence_ttl)
