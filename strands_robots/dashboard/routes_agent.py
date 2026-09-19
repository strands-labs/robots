"""``/ws/agent``: one operator conversation with the dashboard agent.

Client -> server frames: ``{"type":"say","text":...}`` starts a turn;
``{"type":"resume","id":...,"approve":bool,"always":bool}`` answers a consent
card. Server -> client frames are the console's events (text, tool_use,
tool_result, interrupt, done, error). One turn at a time per socket: a second
``say`` while a turn streams is refused with an error frame, not queued.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

from strands_robots.dashboard import access, agent_console

logger = logging.getLogger(__name__)
router = APIRouter()


def _console_factory(app: Any) -> Any:
    """Tests install their own; production builds a Console over the app's Safety."""
    factory = getattr(app.state, "console_factory", None)
    return factory if factory is not None else (lambda: agent_console.Console(app.state.safety))


@router.get("/api/agent")
async def agent_info(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Which model the console will use and which tools ask first."""
    return {
        "model": agent_console.model_id(),
        "asks_first": sorted(agent_console.MOTION_TOOLS),
        "interrupt": agent_console.INTERRUPT_NAME,
    }


@router.websocket("/ws/agent")
async def agent_socket(ws: WebSocket) -> None:
    """A conversation. Admission is the same as every route; a stranger is closed with 4401."""
    try:
        access.caller(ws)  # type: ignore[arg-type]
    except HTTPException:
        await access.refuse_socket(ws, 4401)
        return
    await ws.accept()
    try:
        console = _console_factory(ws.app)()
    except Exception as exc:  # noqa: BLE001 - no model, no creds: the operator reads why
        await ws.send_json({"type": "error", "message": f"agent unavailable: {type(exc).__name__}: {exc}"})
        await ws.close(code=4503)
        return
    busy = asyncio.Lock()
    try:
        while True:
            frame = await ws.receive_json()
            kind = frame.get("type") if isinstance(frame, dict) else None
            if kind == "say":
                text = str(frame.get("text") or "").strip()
                if not text:
                    await ws.send_json({"type": "error", "message": "say what?"})
                    continue
                if len(text) > agent_console.MAX_PROMPT_CHARS:
                    await ws.send_json(
                        {"type": "error", "message": f"prompt longer than {agent_console.MAX_PROMPT_CHARS} chars"}
                    )
                    continue
                prompt: Any = text
            elif kind == "resume":
                prompt = agent_console.Console.resume(
                    str(frame.get("id") or ""), bool(frame.get("approve")), bool(frame.get("always"))
                )
            else:
                await ws.send_json({"type": "error", "message": "frames are {type: say|resume}"})
                continue
            if busy.locked():
                await ws.send_json({"type": "error", "message": "a turn is still streaming"})
                continue
            async with busy:
                async for event in console.run(prompt):
                    await ws.send_json(event)
    except WebSocketDisconnect:
        pass
