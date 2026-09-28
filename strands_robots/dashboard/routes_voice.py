"""``/ws/voice`` - speech-to-speech fleet control.

The browser streams PCM16 (16 kHz mono) frames in and receives the agent's audio
and transcript back. Admission is the same as every other socket. The voice agent
holds exactly one tool, :func:`voice.make_fleet_tool`, which fails closed on any
physical peer because a spoken conversation has no confirm rail to pause on.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from strands_robots.dashboard import access

router = APIRouter(tags=["voice"])

_VOICE_EXTRA = "strands-robots[voice] plus a voice provider key (OPENAI_API_KEY, or AWS credentials for nova_sonic)"


async def _voice_error(ws: WebSocket, text: str, *, code: int) -> None:
    """One error frame under the ``error`` key the page reads (voiceSession.ts), then close with *code*."""
    try:
        await ws.send_json({"type": "error", "error": text})
    except Exception:  # noqa: BLE001 - already gone
        pass
    try:
        await ws.close(code=code)
    except Exception:  # noqa: BLE001 - already closed
        pass


@router.websocket("/ws/voice")
async def voice_socket(ws: WebSocket) -> None:
    """One spoken conversation; strangers are closed with 4401, a missing provider with 4503."""
    try:
        access.caller(ws)  # type: ignore[arg-type]
    except HTTPException:
        await access.refuse_socket(ws, 4401)
        return
    await ws.accept()
    try:
        # ``strands_robots.dashboard.voice`` imports the bidi surface lazily, so importing it proves nothing;
        # probe the SDK surface here so a [dashboard]-only install is told what it lacks.
        import strands.experimental.bidi  # noqa: F401

        from strands_robots.dashboard.voice import run_voice_session
    except ImportError as exc:
        await _voice_error(ws, f"voice unavailable: {exc}. Needs {_VOICE_EXTRA}.", code=4503)
        return
    bridge: Any = getattr(ws.app.state, "bridge", None)
    try:
        await run_voice_session(ws, bridge=bridge)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except ImportError as exc:  # a provider package missing behind the SDK surface
        await _voice_error(ws, f"voice unavailable: {exc}. Needs {_VOICE_EXTRA}.", code=4503)
        return
    except Exception as exc:  # noqa: BLE001 - the operator hears why, the socket closes clean
        try:
            await ws.send_json({"type": "error", "error": f"voice session failed: {type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001 - already gone
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001 - already closed by the peer
            pass
