"""Loopback is not a fence against the browser: any page the user has open can POST to 127.0.0.1.

Three checks, on every request and WebSocket handshake:
- ``Host`` must name this server on loopback (``127.0.0.1:<port>``, ``localhost:<port>``, ``[::1]:<port>``), which
  stops DNS rebinding;
- an ``Origin``, when present, must be this server's own origin;
- a POST must be ``Content-Type: application/json``, which a cross-site page cannot send without a CORS preflight
  (and this server answers no preflight).

Every HTML page also carries ``script-src 'self'``: script runs only from this server's own files, so text that
reaches the page as markup by mistake cannot run. Every HTTP response also says who may frame the page
(``Content-Security-Policy: frame-ancestors``): this machine's own pages (the docs served on another loopback port
embed it in their hero) and the published docs site, nobody else, so a random site cannot overlay the page to trick a
click. A framing page never presses a button itself: it is another
origin, so it cannot script the frame, and the frame's own requests carry the dashboard's origin.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Awaitable, Callable, MutableMapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse

_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
FRAME_ANCESTORS = "frame-ancestors 'self' http://127.0.0.1:* http://localhost:* https://strands-labs.github.io"
Scope = MutableMapping[str, Any]


class _InlineScripts(HTMLParser):
    """The bodies of a page's inline ``<script>`` elements (those without ``src``), as the browser hashes them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.bodies: list[str] = []
        self._open: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and not any(k == "src" for k, _ in attrs):
            self._open = []

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._open.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._open is not None:
            self.bodies.append("".join(self._open))
            self._open = None


def _inline_hashes() -> str:
    """CSP hashes of index.html's inline scripts (the colour-scheme picker that must run before first paint)."""
    parser = _InlineScripts()
    parser.feed((Path(__file__).parent / "static" / "index.html").read_text())
    parser.close()
    bodies = parser.bodies
    return " ".join(f"'sha256-{base64.b64encode(hashlib.sha256(b.encode()).digest()).decode()}'" for b in bodies)


#: On HTML pages only: script runs only from this server's own files, never from markup that reached the page. Workers
#: (MuJoCo's needs eval for its bindings) get their policy from their own response, so they are not affected.
PAGE_POLICY = f"script-src 'self' {_inline_hashes()}; object-src 'none'; base-uri 'none'"


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
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def framed(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                html = any(k.lower() == b"content-type" and v.startswith(b"text/html") for k, v in headers)
                policy = f"{FRAME_ANCESTORS}; {PAGE_POLICY}" if html else FRAME_ANCESTORS
                message["headers"] = [*headers, (b"content-security-policy", policy.encode())]
            await send(message)

        await self.app(scope, receive, framed)
