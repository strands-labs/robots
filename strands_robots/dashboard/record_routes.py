"""The record routes (``dashboard/record.py``): start, save or discard an episode, stop, status, options.

The recorder only observes, so none of these needs X-Confirm; it is ``app.state.recorder`` for other modules.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .record import Recorder

if TYPE_CHECKING:
    from .dashboard import Dashboard


def add(app: FastAPI, dash: Dashboard) -> None:
    from .app import http_error

    rec = app.state.recorder = Recorder(dash)

    @app.post("/api/record/start")
    async def record_start(request: Request) -> JSONResponse:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, 'body must be {"robots": [...], "cameras": {...}, "target": ..., "fps": ...}')
        try:
            out = await asyncio.to_thread(
                rec.start,
                robots=body.get("robots"),
                cameras=body.get("cameras"),
                target=body.get("target"),
                task=body.get("task"),
                fps=body.get("fps", 30),
                units=body.get("units", "auto"),
            )
        except Exception as exc:  # noqa: BLE001 — a bad body or target is a 4xx with the reason
            raise http_error(exc) from exc
        return JSONResponse(out, status_code=201)

    @app.post("/api/record/episode/{verb}")
    async def record_episode(verb: str) -> JSONResponse:
        if verb not in ("save", "discard"):
            raise HTTPException(404, f"no episode verb {verb!r}; use save or discard")
        try:
            return JSONResponse(await asyncio.to_thread(getattr(rec, verb)))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.post("/api/record/stop")
    async def record_stop(request: Request) -> JSONResponse:
        body = await request.json()
        save = body.get("save", True) if isinstance(body, dict) else True
        try:
            return JSONResponse(await asyncio.to_thread(rec.stop, save=save))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.get("/api/record/options")
    async def record_options() -> JSONResponse:
        try:
            return JSONResponse(await asyncio.to_thread(rec.options))
        except Exception as exc:  # noqa: BLE001
            raise http_error(exc) from exc

    @app.get("/api/record/status")
    async def record_status() -> JSONResponse:
        return JSONResponse(await asyncio.to_thread(rec.status))
