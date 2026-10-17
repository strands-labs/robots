"""The routes: GET / · /api/scene · /api/assets/{path} · /api/policies · /api/mesh · WS /ws/state · /ws/chat ·
POST /api/robots · POST|GET /api/robots/{id}/run · POST /api/robots/{id}/{act,stop,estop,resume} ·
POST /api/mesh/{id}/estop."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import runs
from .guard import SameOrigin

if TYPE_CHECKING:
    from .dashboard import Dashboard

STATIC = Path(__file__).parent / "static"
_MEDIA = {".stl": "model/stl", ".obj": "text/plain", ".png": "image/png", ".xml": "application/xml"}


def http_error(exc: Exception) -> HTTPException:
    from ..robot.safety.estop import EStopped

    if isinstance(exc, EStopped):
        return HTTPException(409, str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(404, str(exc.args[0]) if exc.args else "not found")
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(400, str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(409, str(exc))
    return HTTPException(500, f"{type(exc).__name__}: {exc}")


def build_app(dash: Dashboard) -> FastAPI:
    app = FastAPI(title="strands-robots dashboard", docs_url=None, redoc_url=None)
    app.add_middleware(SameOrigin)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/scene")
    async def scene() -> JSONResponse:
        return JSONResponse(await asyncio.to_thread(dash.scene_json))

    @app.get("/api/assets/{path:path}")
    async def asset(path: str) -> Response:
        try:
            file = dash.asset_file(path)
        except KeyError as exc:
            raise http_error(exc) from exc
        return FileResponse(file, media_type=_MEDIA.get(Path(file).suffix.lower(), "application/octet-stream"))

    @app.post("/api/robots")
    async def add_robot(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            out = await asyncio.to_thread(
                dash.add_robot, body.get("name"), id=body.get("id"), position=body.get("position")
            )
        except Exception as exc:  # noqa: BLE001 — an unknown name is a 4xx with the reason
            raise http_error(exc) from exc
        return JSONResponse(out, status_code=201)

    @app.get("/api/policies")
    async def policies() -> JSONResponse:
        return JSONResponse(await asyncio.to_thread(runs.policies_json))

    @app.post("/api/robots/{rid}/run")
    async def run(rid: str, request: Request) -> JSONResponse:
        body = await request.json()
        confirmed = request.headers.get("x-confirm", "").lower() == "yes"

        def go() -> dict[str, Any]:
            return runs.start(
                dash._robot(rid),
                body.get("policy", "hold"),
                instruction=body.get("instruction"),
                seconds=body.get("seconds"),
                hz=body.get("hz"),
                confirmed=confirmed,
            )

        try:
            return JSONResponse(await asyncio.to_thread(go), status_code=202)
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.get("/api/robots/{rid}/run")
    async def run_status(rid: str) -> JSONResponse:
        try:
            return JSONResponse(await asyncio.to_thread(lambda: runs.status(dash._robot(rid))))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.post("/api/robots/{rid}/stop")
    async def stop(rid: str) -> JSONResponse:
        try:
            return JSONResponse(await asyncio.to_thread(lambda: runs.stop(dash._robot(rid))))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.get("/api/mesh")
    async def mesh() -> JSONResponse:
        try:
            return JSONResponse(await asyncio.to_thread(dash.mesh_json))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.post("/api/mesh/{device_id}/estop")
    async def mesh_estop(device_id: str) -> JSONResponse:
        try:
            return JSONResponse({"id": device_id, "result": await asyncio.to_thread(dash.mesh_estop, device_id)})
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.post("/api/robots/{rid}/estop")
    async def estop(rid: str) -> JSONResponse:
        try:
            return JSONResponse(await asyncio.to_thread(dash.estop, rid))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.post("/api/robots/{rid}/resume")
    async def resume(rid: str, request: Request) -> JSONResponse:
        body = await request.json()
        try:
            confirmed = request.headers.get("x-confirm", "").lower() == "yes"
            out = await asyncio.to_thread(dash.resume, rid, approved_by=body.get("approved_by"), confirmed=confirmed)
            return JSONResponse(out)
        except Exception as exc:  # noqa: BLE001 — a resume without an operator is a 403 with the reason
            raise http_error(exc) from exc

    @app.post("/api/robots/{rid}/act")
    async def act(rid: str, request: Request) -> JSONResponse:
        body = await request.json()
        confirmed = request.headers.get("x-confirm", "").lower() == "yes"
        try:
            out = await asyncio.to_thread(dash.act, rid, body.get("targets"), confirmed=confirmed)
        except Exception as exc:  # noqa: BLE001 — every refusal is an HTTP status with its reason
            raise http_error(exc) from exc
        return JSONResponse(out)

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket) -> None:
        await ws.accept()
        period = 1.0 / dash.hz
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    frame = await asyncio.to_thread(dash.frame)
                except Exception:  # noqa: BLE001 — one bad read (a body mid-reload) skips a tick, never ends the stream
                    frame = None
                if frame is not None:
                    await ws.send_json(frame)
                await asyncio.sleep(max(0.0, period - (time.perf_counter() - t0)))
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket) -> None:
        from . import chat

        await ws.accept()
        try:
            while True:
                msg = await ws.receive_json()
                prompt = str(msg.get("prompt", "")).strip() if isinstance(msg, dict) else ""
                if not prompt:
                    await ws.send_json({"type": "error", "message": "empty prompt"})
                    continue
                try:
                    agent = await asyncio.to_thread(lambda: dash.agent)
                except RuntimeError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                    await ws.send_json({"type": "done"})
                    continue
                async for ev in chat.stream_chat(agent, prompt):
                    await ws.send_json(ev)
        except WebSocketDisconnect:
            return

    app.mount("/", StaticFiles(directory=STATIC), name="static")
    return app
