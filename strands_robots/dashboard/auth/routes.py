"""``/auth/*``: the ceremony routes (open: they are how one signs in) and ``/api/passkeys`` (a signed-in session).

GET  /auth/status                 {enabled, signed_in, setup_required, rp_id}
POST /auth/enrol/begin            {setup?, label?}  -> {challenge_id, options}  (setup code, or a session)
POST /auth/enrol/finish           {challenge_id, credential} -> sets the session cookie
POST /auth/login/begin            {} -> {challenge_id, options}
POST /auth/login/finish           {challenge_id, credential} -> sets the session cookie
POST /auth/logout                 clears the cookie
GET  /api/passkeys · POST /api/passkeys/{id} {name} · DELETE /api/passkeys/{id} (never the last one)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import session
from .guard import RequireSession

if TYPE_CHECKING:
    from .core import Auth


def _status(exc: Exception) -> HTTPException:
    for kind, code in ((PermissionError, 403), (KeyError, 404), (ValueError, 400), (RuntimeError, 409)):
        if isinstance(exc, kind):
            return HTTPException(code, str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc))
    raise exc


async def _run(fn: Callable[[], Any]) -> Any:
    try:
        return await asyncio.to_thread(fn)
    except (PermissionError, KeyError, ValueError, RuntimeError) as exc:
        raise _status(exc) from exc


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    return body


def _signed_in(auth: Auth, token: str) -> JSONResponse:
    out = JSONResponse({"signed_in": True})
    out.headers.append("set-cookie", session.set_cookie(token, auth.ttl))
    return out


def install(app: FastAPI, auth: Auth) -> None:
    """The session guard (inside whatever middleware is added after this) and the routes below."""
    app.add_middleware(RequireSession, auth=auth)

    @app.get("/auth/status")
    async def status(request: Request) -> JSONResponse:
        signed = await asyncio.to_thread(auth.signed_in, request.scope)
        setup = await asyncio.to_thread(lambda: auth.setup_required)
        return JSONResponse(
            {"enabled": auth.enabled, "signed_in": signed is not None, "setup_required": setup, "rp_id": auth.rp_id}
        )

    @app.post("/auth/enrol/begin")
    async def enrol_begin(request: Request) -> JSONResponse:
        b, host = await _body(request), request.headers.get("host", "")
        return JSONResponse(
            await _run(lambda: auth.begin_enrol(host, setup=b.get("setup"), scope=request.scope, label=b.get("label")))
        )

    @app.post("/auth/enrol/finish")
    async def enrol_finish(request: Request) -> JSONResponse:
        b, host = await _body(request), request.headers.get("host", "")
        return _signed_in(auth, await _run(lambda: auth.finish_enrol(host, b.get("challenge_id"), b.get("credential"))))

    @app.post("/auth/login/begin")
    async def login_begin(request: Request) -> JSONResponse:
        host = request.headers.get("host", "")
        return JSONResponse(await _run(lambda: auth.begin_login(host)))

    @app.post("/auth/login/finish")
    async def login_finish(request: Request) -> JSONResponse:
        b, host = await _body(request), request.headers.get("host", "")
        return _signed_in(auth, await _run(lambda: auth.finish_login(host, b.get("challenge_id"), b.get("credential"))))

    @app.post("/auth/logout")
    async def logout() -> JSONResponse:
        out = JSONResponse({"signed_in": False})
        out.headers.append("set-cookie", session.clear_cookie())
        return out

    @app.get("/api/passkeys")
    async def passkeys() -> JSONResponse:
        return JSONResponse({"passkeys": await _run(auth.passkeys)})

    @app.post("/api/passkeys/{cid}")
    async def rename(cid: str, request: Request) -> JSONResponse:
        b = await _body(request)
        return JSONResponse(await _run(lambda: auth.rename(cid, b.get("name"))))

    @app.delete("/api/passkeys/{cid}")
    async def remove(cid: str) -> JSONResponse:
        return JSONResponse(await _run(lambda: auth.remove(cid)))
