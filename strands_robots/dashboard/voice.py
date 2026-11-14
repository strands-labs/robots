"""Speech to speech: one ``strands.bidi.BidiAgent`` per ``/ws/voice`` socket, holding the chat agent's tools.

    Dashboard(sim, voice="openai")   # gpt-realtime-2 (OPENAI_API_KEY); "nova": Amazon Nova Sonic; "auto": either

The browser streams its microphone as PCM16 mono at the model's input rate; the agent answers with audio (PCM16 at
the output rate), transcripts and tool calls. The tools are the chat agent's (``sim.tool`` and every ``robot.tool``)
plus the page's run controls: ``dashboard_run`` starts a policy in the background (the robot tool's ``run`` waits for
the run to end), ``dashboard_stop`` ends it, ``dashboard_estop`` latches the e-stop. Every call is an agent's call, so
the operator gate stands: voice never carries the page's X-Confirm, and a real robot refuses to move from it.

Socket protocol:
    server -> page   {"type": "ready", provider, model, input_rate, output_rate} first, then
                     binary frames (assistant audio) and JSON events: transcript {role, delta | text, final} ·
                     tool_use {name, input} · tool_result {name, status, text} · interrupt · turn_done · error {message}
    page -> server   binary frames (microphone PCM16 mono little-endian at input_rate) · {"type": "text", "text"}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import chat
from .voice_tools import SYSTEM_PROMPT, build_agent, build_tools

if TYPE_CHECKING:
    from .protocols import DashboardLike

log = logging.getLogger(__name__)
__all__ = ["SYSTEM_PROMPT", "build_agent", "build_tools", "make_model", "page_events", "provider_of", "session"]

OPENAI_MODEL = "gpt-realtime-2"
NOVA_MODEL = "amazon.nova-2-5-sonic"
PROVIDERS = ("openai", "nova", "auto")
VoiceSpec = str | Callable[[], Any] | None


def _key_from_secrets() -> None:
    """``OPENAI_API_KEY`` from ``STRANDS_VOICE_SECRETS`` (a KEY=VALUE env file) when the environment lacks it."""
    path = os.environ.get("STRANDS_VOICE_SECRETS")
    if os.environ.get("OPENAI_API_KEY") or not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            k, sep, v = line.strip().removeprefix("export ").partition("=")
            if sep and k.strip() == "OPENAI_API_KEY":
                os.environ["OPENAI_API_KEY"] = v.strip().strip("'\"")


def provider_of(spec: VoiceSpec) -> str:
    """``"openai"`` | ``"nova"`` for a provider string (``"auto"``: OpenAI when its key is set); ``"custom"`` else."""
    if callable(spec):
        return "custom"
    if spec not in PROVIDERS:
        raise ValueError(f"voice must be one of {PROVIDERS} or a BidiModel factory, got {spec!r}")
    if spec == "auto":
        _key_from_secrets()
        return "openai" if os.environ.get("OPENAI_API_KEY") else "nova"
    return str(spec)


def make_model(spec: VoiceSpec) -> Any:
    """A fresh BidiModel (one per conversation: a model holds its connection)."""
    if spec is None:
        raise RuntimeError('voice needs Dashboard(sim, voice="openai" | "nova" | "auto" | <BidiModel factory>)')
    if callable(spec):
        return spec()
    provider = provider_of(spec)
    try:
        if provider == "openai":
            _key_from_secrets()
            from strands.bidi.models.openai.realtime import OpenAIRealtimeModel

            return OpenAIRealtimeModel(
                model_id=os.environ.get("STRANDS_VOICE_OPENAI_MODEL", OPENAI_MODEL),
                transcription_model_id=os.environ.get("STRANDS_VOICE_TRANSCRIBE", "gpt-4o-mini-transcribe"),
                voice=os.environ.get("STRANDS_VOICE_NAME", "alloy"),
                params={"audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 600}}}},
            )
        from strands.bidi.models.bedrock import BedrockNovaSonicModel

        return BedrockNovaSonicModel(
            model_id=os.environ.get("STRANDS_VOICE_NOVA_MODEL", NOVA_MODEL),
            region=os.environ.get("AWS_REGION", "us-east-1"),
            voice=os.environ.get("STRANDS_VOICE_NAME", "matthew"),
            audio={"input": {"sample_rate": 16000}, "output": {"sample_rate": 24000}},
        )
    except ImportError as e:
        need = '"strands-robots[voice]"' if provider == "openai" else '"strands-robots[voice]" "strands-agents[bidi]"'
        raise ImportError(f"voice ({provider}) needs: pip install {need} ({e})") from e


def rates(model: Any) -> tuple[int, int]:
    cfg = model.get_audio_config() if hasattr(model, "get_audio_config") else None
    if not cfg:
        return 16000, 24000
    return int(cfg["input"]["sample_rate"]), int(cfg["output"]["sample_rate"])


def page_events(ev: Any, names: dict[str, str]) -> list[Any]:
    """One agent event -> what the page gets: ``bytes`` (assistant audio) or JSON dicts."""
    kind = ev.get("type") if isinstance(ev, dict) else None
    if kind == "bidi_audio_delta":
        return [base64.b64decode(ev["audio"])]
    if kind == "bidi_transcript_delta":
        return [{"type": "transcript", "role": ev["role"], "delta": ev["delta"], "id": ev["content_id"]}]
    if kind == "bidi_transcript_block":
        return [
            {"type": "transcript", "role": ev["role"], "text": ev["transcript"], "final": True, "id": ev["content_id"]}
        ]
    if kind == "bidi_barge_in":
        return [{"type": "interrupt"}]
    if kind == "bidi_response_stop":
        return [{"type": "turn_done"}]
    if kind == "bidi_connection_start":
        return [{"type": "connected", "model": ev.get("model")}]
    if kind == "bidi_tool_use_blocks":
        out = []
        for tu in ev["tool_uses"]:
            names[tu["toolUseId"]] = tu["name"]
            out.append({"type": "tool_use", "name": tu["name"], "input": tu.get("input", {})})
        return out
    if kind == "tool_result":
        tr = ev["tool_result"]
        return [
            {
                "type": "tool_result",
                "name": names.get(tr["toolUseId"], "?"),
                "status": tr.get("status", "success"),
                "text": chat._text_of(tr.get("content") or [])[:4000],
            }
        ]
    return []


async def session(ws: Any, dash: DashboardLike) -> None:
    """Run one conversation over an accepted Starlette WebSocket until either side ends it."""
    try:
        model = await asyncio.to_thread(make_model, dash.voice)
        agent = await asyncio.to_thread(build_agent, dash, model)
    except Exception as exc:  # noqa: BLE001 — the page shows why voice is off
        await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await ws.close()
        return
    in_rate, out_rate = rates(model)
    provider = provider_of(dash.voice) if dash.voice is not None else "custom"
    await ws.send_json(
        {
            "type": "ready",
            "provider": provider,
            "model": getattr(model, "model_id", "?"),
            "input_rate": in_rate,
            "output_rate": out_rate,
        }
    )
    try:
        await agent.start()
    except Exception as exc:  # noqa: BLE001
        await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await ws.close()
        return

    from strands.bidi.types.media import AudioDelta

    async def upstream() -> None:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                await agent.send(AudioDelta(format="pcm", source={"bytes": msg["bytes"]}))
            elif msg.get("text"):
                try:
                    body = json.loads(msg["text"])
                except ValueError:
                    continue
                if isinstance(body, dict) and body.get("type") == "text" and str(body.get("text", "")).strip():
                    await agent.send(str(body["text"]).strip())
                elif isinstance(body, dict) and body.get("type") == "bye":
                    return

    async def downstream() -> None:
        names: dict[str, str] = {}
        try:
            async for ev in agent.receive():
                for out in page_events(ev, names):
                    if isinstance(out, bytes):
                        await ws.send_bytes(out)
                    else:
                        await ws.send_json(out)
        except Exception as exc:  # noqa: BLE001 — a model error ends the conversation with a reason
            await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    up, down = asyncio.create_task(upstream()), asyncio.create_task(downstream())
    try:
        await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (up, down):
            t.cancel()
        try:
            await agent.stop()
        except Exception:  # noqa: BLE001 — the socket is gone; nothing to tell
            log.debug("voice: agent.stop failed", exc_info=True)
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
