"""The Connect routes: GET /api/connect/peers · /api/connect/robots · /api/connect/policies ·
POST /api/connect/mesh · /api/connect/robot · /api/connect/robot/{id}/{act,detach,resume} · /api/connect/policy ·
GET /api/robots/{id}/cameras/{cam}.png (a hardware or remote robot's own camera, not a sim render)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import cameras

if TYPE_CHECKING:
    from .dashboard import Dashboard


async def _call(fn: Callable[[], Any], status: int = 200) -> JSONResponse:
    from .app import http_error

    try:
        return JSONResponse(await asyncio.to_thread(fn), status_code=status)
    except Exception as exc:  # noqa: BLE001 — every refusal is an HTTP status with its reason
        raise http_error(exc) from exc


async def _body(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        from fastapi import HTTPException

        raise HTTPException(400, "body must be a JSON object")
    return body


def register(app: FastAPI, dash: Dashboard) -> None:
    """Add the Connect routes to ``app`` (before the static mount, which answers every other path)."""

    @app.get("/api/connect/peers")
    async def peers() -> JSONResponse:
        return await _call(lambda: dash.connections.peers())

    @app.get("/api/connect/robots")
    async def robots() -> JSONResponse:
        return await _call(lambda: {"robots": dash.connections.robots()})

    @app.get("/api/connect/policies")
    async def policies() -> JSONResponse:
        return await _call(lambda: {"policies": list(dash.connections.policies.values())})

    @app.post("/api/connect/mesh")
    async def join(request: Request) -> JSONResponse:
        b = await _body(request)
        return await _call(lambda: dash.connections.join(str(b.get("address") or "lan"), b.get("tenant")))

    @app.post("/api/connect/robot")
    async def attach(request: Request) -> JSONResponse:
        b = await _body(request)
        peer, address = b.get("peer") or None, b.get("address") or None
        return await _call(lambda: dash.connections.attach(peer, address), 201)

    @app.post("/api/connect/robot/{rid}/detach")
    async def detach(rid: str) -> JSONResponse:
        return await _call(lambda: dash.connections.detach(rid))

    @app.post("/api/connect/robot/{rid}/act")
    async def act(rid: str, request: Request) -> JSONResponse:
        b = await _body(request)
        confirmed = request.headers.get("x-confirm", "").lower() == "yes"
        return await _call(lambda: dash.connections.act(rid, b.get("targets"), b.get("seconds"), confirmed=confirmed))

    @app.post("/api/connect/robot/{rid}/resume")
    async def resume(rid: str, request: Request) -> JSONResponse:
        b = await _body(request)
        by, confirmed = b.get("approved_by"), request.headers.get("x-confirm", "").lower() == "yes"

        def go() -> Any:
            if not confirmed:  # before the device is asked: a resume from this page is the operator's yes
                raise PermissionError(f"{rid} is a remote robot: resume it from this page with header X-Confirm: yes")
            name = dash.connections.resume(rid, by if isinstance(by, str) and by else None)
            return dash.resume(rid, approved_by=name, confirmed=True)

        return await _call(go)

    @app.post("/api/connect/policy")
    async def policy(request: Request) -> JSONResponse:
        b = await _body(request)
        c = dash.connections
        return await _call(lambda: c.add_policy(b.get("endpoint"), b.get("kind"), b.get("deployment"), b.get("name")))

    @app.get("/api/robots/{rid}/cameras/{cam}.png")
    async def camera(rid: str, cam: str) -> Response:
        from .app import http_error

        try:
            img = await asyncio.to_thread(dash.connections.camera, rid, cam)
        except Exception as exc:  # noqa: BLE001 — an unknown robot or camera is a 404 with the names
            raise http_error(exc) from exc
        return Response(cameras.png(img), media_type="image/png", headers={"cache-control": "no-store"})
