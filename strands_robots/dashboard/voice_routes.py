"""The voice routes, mounted by ``app.build_app`` with one line: GET /api/voice · WS /ws/voice.

``dashboard.voice`` (and strands) is imported only when a voice route is hit, so the page loads without the
[voice] extra.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from .protocols import DashboardLike

NO_VOICE = 'voice is off: start the dashboard with --voice auto (or openai, nova), or pass Dashboard(sim, voice="auto")'


def mount(app: FastAPI, dash: DashboardLike) -> None:
    """Add GET /api/voice (is voice on, which provider) and WS /ws/voice (one conversation per socket)."""

    @app.get("/api/voice")
    async def voice_info() -> JSONResponse:
        if dash.voice is None:
            return JSONResponse({"enabled": False, "provider": None, "hint": NO_VOICE})
        from . import voice

        return JSONResponse({"enabled": True, "provider": await asyncio.to_thread(voice.provider_of, dash.voice)})

    @app.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket) -> None:
        from . import voice

        await ws.accept()
        if dash.voice is None:
            await ws.send_json({"type": "error", "message": NO_VOICE})
            await ws.close()
            return
        await voice.session(ws, dash)
