"""Loopback is not a fence against the browser: any page the user has open can POST to 127.0.0.1.

Three checks, on every request and WebSocket handshake:
- ``Host`` must name this server on loopback (``127.0.0.1:<port>``, ``localhost:<port>``, ``[::1]:<port>``), which
  stops DNS rebinding;
- an ``Origin``, when present, must be this server's own origin;
- a POST must be ``Content-Type: application/json``, which a cross-site page cannot send without a CORS preflight
  (and this server answers no preflight).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.responses import JSONResponse

_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
Scope = MutableMapping[str, Any]


def _headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def refusal(scope: Scope) -> str | None:
    """Why this request is refused, or None."""
    h = _headers(scope)
    port = scope.get("server", (None, None))[1]
    allowed = {f"{name}:{port}" for name in _LOOPBACK}
    host = h.get("host", "")
    if host not in allowed:
        return f"Host {host!r} is not this dashboard (expected one of {sorted(allowed)})"
    origin = h.get("origin")
    if origin is not None and origin.removeprefix("http://").removeprefix("ws://") not in allowed:
        return f"Origin {origin!r} is not this dashboard: cross-site requests are refused"
    if scope["type"] == "http" and scope.get("method") == "POST":
        ctype = h.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return f"POST needs Content-Type: application/json (got {ctype or 'none'!r})"
    return None


class SameOrigin:
    """ASGI middleware applying :func:`refusal` to HTTP and WebSocket scopes."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            why = refusal(scope)
            if why is not None:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 4403, "reason": why[:120]})
                    return
                await JSONResponse({"detail": why}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
