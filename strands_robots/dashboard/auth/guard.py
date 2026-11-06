"""Every ``/api/*`` route and every ``/ws/*`` socket needs a signed-in session; the page shell, its assets and the
``/auth/*`` ceremony routes do not (they are how one signs in).

Runs inside the same-origin guard (``dashboard/guard.py``), so a request reaches it only with this server's Host,
Origin and JSON content type. A refused request is a 401 (a socket closes with 4401 before it is accepted). The page
shell asked for on ``127.0.0.1`` / ``[::1]`` is redirected to ``localhost``, the one loopback name passkeys work on.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse, RedirectResponse

if TYPE_CHECKING:
    from .core import Auth

GUARDED = ("/api/", "/ws/")
SIGN_IN = "sign in: open the dashboard in a browser and use your passkey, or send Authorization: Bearer <token>"


def guarded(path: str) -> bool:
    return path.startswith(GUARDED) or path in ("/api", "/ws")


class RequireSession:
    """ASGI middleware: :data:`GUARDED` paths need ``auth.signed_in(scope)``."""

    def __init__(self, app: Callable[..., Awaitable[None]], auth: Auth) -> None:
        self.app, self.auth = app, auth

    async def __call__(self, scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket") or not self.auth.enabled:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        host = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"host"), "")
        if scope["type"] == "http" and path == "/" and host.startswith(("127.0.0.1:", "[::1]:")):
            query = scope.get("query_string", b"").decode("latin-1")
            url = f"http://localhost:{host.rsplit(':', 1)[1]}/" + (f"?{query}" if query else "")
            await RedirectResponse(url, status_code=307)(scope, receive, send)
            return
        if guarded(path):
            claims = self.auth.signed_in(scope)
            if claims is None:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 4401, "reason": "sign in first"})
                    return
                await JSONResponse({"detail": SIGN_IN}, status_code=401)(scope, receive, send)
                return
            scope.setdefault("state", {})["session"] = claims
        await self.app(scope, receive, send)
