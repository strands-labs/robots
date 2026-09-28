"""Speech-to-speech fleet control - browser mic <-> Strands bidi agent. PCM16 audio flows over
/ws/voice (binary in, base64 JSON out).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

VOICE_PROMPT = """You are the Strands Robots fleet voice operator. You control real robots and
simulations on a mesh via the fleet tool. Keep spoken replies SHORT - one or
two sentences. Confirm before actuating real hardware (peer ids without 'sim'
in them). fleet(action='peers') shows who's online; fleet(action='task',
target=..., instruction=..., duration=...) runs a task;
fleet(action='stop_all') stops everything - use it immediately when asked to
stop.
Starting a task on a REAL robot may be refused because this dashboard does not
let an agent start physical motion on its own. That refusal is final for you:
say in one sentence that the operator has to allow it on screen (a card just
appeared) or press play themselves, and do NOT retry, reword or pick another
robot. A spoken yes cannot grant it - only their tap can. Stopping is never
refused, so always act on a stop request immediately."""

_DEFAULT_VOICES = {"openai": "marin", "nova_sonic": "tiffany"}


_refusal_listeners: list[Any] = []


def add_refusal_listener(cb: Any) -> Any:
    """Register ``cb(text)`` for every refusal the fleet tool raises; returns an unsubscribe."""
    _refusal_listeners.append(cb)

    def _off() -> None:
        with contextlib.suppress(ValueError):
            _refusal_listeners.remove(cb)

    return _off


def _notify_refusal(text: str) -> None:
    for cb in list(_refusal_listeners):
        with contextlib.suppress(Exception):
            cb(text)


def make_fleet_tool(bridge: Any) -> Any:
    """The one tool the voice agent holds: read the fleet, task a peer, stop it.

    Voice has no confirm rail (bidi cannot pause on an interrupt), so a task on a
    physical peer is decided by :func:`agent_motion.agent_motion_allowed` alone and
    fails closed; the refusal is spoken once through the listeners above.
    """
    import json as _json

    from strands import tool

    from strands_robots.dashboard.agent_motion import agent_motion_allowed
    from strands_robots.dashboard.mesh_bridge import route_task_target

    @tool
    def fleet(
        action: str,
        target: str = "",
        instruction: str = "",
        policy_provider: str = "mock",
        duration: float = 15.0,
        robot_name: str = "",
    ) -> dict[str, Any]:
        """Coordinate robots on the mesh (dashboard gateway). Actions: peers, task, stop, stop_all, status."""
        if bridge is None:
            return {"status": "error", "content": [{"text": "mesh bridge offline"}]}

        if action == "peers":
            snap = bridge.snapshot()
            lines = []
            for pid, p in sorted((snap.get("peers") or {}).items()):
                if p.get("stale"):
                    continue
                pres = p.get("presence") or {}
                st = p.get("state") or {}
                task = st.get("task") or {}
                cams = list((p.get("cameras") or {}).keys())
                lines.append(
                    f"- {pid}: type={pres.get('robot_type', '?')} hw_connected={pres.get('connected')} "
                    f"cameras={cams} joints={len(st.get('joints') or {})} task={task.get('status', 'idle')} "
                    f"instruction={task.get('instruction', '')!r}"
                )
            text = "Online peers:\n" + "\n".join(lines) if lines else "No live peers on the mesh."
            return {"status": "success", "content": [{"text": text}]}

        if action == "task":
            if not target or not instruction:
                return {"status": "error", "content": [{"text": "task requires target and instruction"}]}
            try:
                peers = bridge.snapshot().get("peers") or {}
            except Exception:  # noqa: BLE001 - an unreadable snapshot means UNKNOWN, i.e. metal
                peers = {}
            verdict = agent_motion_allowed("task", peer=peers.get(target), target=target)
            if not verdict["allowed"]:
                _notify_refusal(verdict["reason"])
                return {"status": "error", "content": [{"text": verdict["reason"]}]}
            cmd: dict[str, Any] = {
                "action": "execute",
                "instruction": instruction,
                "policy_provider": policy_provider,
                "duration": float(duration),
            }
            if robot_name:
                cmd["robot_name"] = robot_name
            target, cmd = route_task_target(target, cmd)
            res = bridge.send_cmd(target, cmd, timeout=float(duration) + 30.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:1500]}]}

        if action == "stop":
            if not target:
                return {"status": "error", "content": [{"text": "stop requires target"}]}
            res = bridge.send_cmd(target, {"action": "stop"}, timeout=10.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        if action == "stop_all":
            results = {}
            for pid, p in (bridge.snapshot().get("peers") or {}).items():
                if p.get("stale"):
                    continue
                results[pid] = bridge.send_cmd(pid, {"action": "stop"}, timeout=5.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(results)[:1500]}]}

        if action == "status":
            if not target:
                return {"status": "error", "content": [{"text": "status requires target"}]}
            res = bridge.send_cmd(target, {"action": "status"}, timeout=10.0, source="voice")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        return {
            "status": "error",
            "content": [{"text": f"unknown action {action!r}. Valid: peers, task, stop, stop_all, status"}],
        }

    return fleet


#: The provider model each backend speaks when ``VOICE_MODEL`` is unset.
_DEFAULT_MODEL_IDS = {"openai": "gpt-realtime", "nova_sonic": "amazon.nova-2-sonic-v1:0"}

#: What the browser sends: PCM16 mono at 16 kHz (``useVoice.ts`` downsamples to it).
BROWSER_INPUT_RATE = 16000


def _build_bidi_model(provider: str, voice: str | None = None) -> Any:
    """One speech-to-speech model, built against the ``strands.experimental.bidi`` 1.57 surface.

    ``model_id`` is a required field there, so each backend carries a default the
    operator overrides with ``VOICE_MODEL``.
    """
    provider = provider.lower()
    v = voice or _DEFAULT_VOICES.get(provider)
    model_id = os.getenv("VOICE_MODEL") or _DEFAULT_MODEL_IDS.get(provider)

    if provider in ("nova_sonic", "novasonic", "nova"):
        from strands.experimental.bidi.models.bedrock import BedrockNovaSonicModel

        kwargs: dict[str, Any] = {"model_id": model_id, "region": os.getenv("AWS_REGION", "us-east-1")}
        if v:
            kwargs["voice"] = v
        return BedrockNovaSonicModel(**kwargs)

    if provider in ("openai", "openai_realtime"):
        from strands.experimental.bidi.models.openai import OpenAIRealtimeModel

        kwargs = {"model_id": model_id, "transcription_model_id": None}
        if v:
            kwargs["voice"] = v
        if os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        return OpenAIRealtimeModel(**kwargs)

    # No Gemini Live: its SDK (google-genai) caps websockets below 17 and this
    # package requires websockets>=17.0, so the two cannot be installed together.
    raise ValueError(f"unknown voice provider: {provider!r} (openai | nova_sonic)")


def _resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear resample of mono PCM16; identity when the rates already agree."""
    if src_rate == dst_rate or not data:
        return data
    import numpy as np

    pcm = np.frombuffer(data[: len(data) - (len(data) % 2)], dtype="<i2").astype(np.float32)
    n_out = max(1, int(round(len(pcm) * dst_rate / src_rate)))
    x_out = np.linspace(0.0, len(pcm) - 1, n_out)
    out = np.interp(x_out, np.arange(len(pcm), dtype=np.float32), pcm)
    return bytes(np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes())


def build_voice_agent(provider: str | None = None, voice: str | None = None, *, bridge: Any = None) -> Any:
    """BidiAgent with the fleet toolset. Caller supplies browser audio IO.

    Deliberately NO robot_mesh here, and NO touching STRANDS_MESH_HITL_ACTIONS:
    bidi cannot pause a tool for a human answer (the SDK's agent/loop.py raises
    "tool interrupts are not supported in bidi"), so a gated robot_mesh action
    would blow up the tool task instead of asking. An earlier version worked
    around that by setdefault-ing STRANDS_MESH_HITL_ACTIONS="none" - a
    PROCESS-WIDE write that silently disarmed the chat agent's robot_mesh
    confirm gate the moment one voice session was opened. Voice safety instead
    rests on the fleet tool's own backstop (agent_motion_allowed fail-closes
    physical tasks - voice has no confirm rail, so no grant can ever appear).

    For the same reason this function writes NOTHING to ``os.environ``. In
    particular it never sets ``BYPASS_TOOL_CONSENT``: ``_command_gate`` reads
    that name at call time as a full consent bypass for every gated hardware
    surface in the process (hardware_robot, serial_tool, pose_tool, use_unitree,
    the ROS gate), and a robot child spawned afterwards would inherit it. The
    fleet tool here holds no strands_tools tool that prompts, so it needs no
    bypass to run.
    """
    from strands.experimental.bidi import BidiAgent
    from strands.experimental.bidi.tools import stop_conversation

    provider = provider or os.getenv("VOICE_PROVIDER", "openai")
    voice = voice or os.getenv("VOICE_NAME") or None
    model = _build_bidi_model(provider or "openai", voice)
    return BidiAgent(
        model=model,
        tools=[make_fleet_tool(bridge), stop_conversation],
        system_prompt=os.getenv("DASHBOARD_VOICE_PROMPT", VOICE_PROMPT),
    )


async def run_voice_session(ws: Any, *, bridge: Any = None) -> None:
    """Bridge one /ws/voice websocket to a fresh BidiAgent session. Browser -> binary PCM16 frames (16
    kHz mono) or {"type":"stop"} text.
    """
    import asyncio
    import json
    import queue as _queue

    # The bidi event vocabulary is experimental and is renamed between SDK
    # releases; output events are classified by the ``type`` string they all
    # carry, so a rename there is a no-op rather than an ImportError at the
    # first spoken word. Input is the 1.57 ``AudioDelta`` (format + bytes only:
    # the rate is the model's, read from ``get_audio_config()``).
    from fastapi import WebSocketDisconnect
    from strands.experimental.bidi.types.media import AudioDelta

    def _event_type(event: Any) -> str:
        try:
            return str(event.get("type", "") if hasattr(event, "get") else getattr(event, "type", ""))
        except Exception:  # noqa: BLE001 - an event that cannot say its type is not one we forward
            return ""

    in_q: asyncio.Queue[bytes] = asyncio.Queue()
    stop_evt = asyncio.Event()

    # A refusal raised inside the fleet tool is spoken once and gone: no transcript rail carries a
    # decision, and the operator cannot grant a permission by talking.
    from strands_robots.dashboard.consent import classify_refusal

    need_q: _queue.Queue[dict] = _queue.Queue()

    def _on_refusal(text: str) -> None:
        need = classify_refusal(text)
        if need is not None:
            need_q.put({"type": "needs_consent", "need": need.as_dict(), "spoken": text[:400]})

    drop_listener = add_refusal_listener(_on_refusal)

    async def _drain_needs() -> None:
        while not stop_evt.is_set():
            try:
                frame = need_q.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.2)
                continue
            try:
                await ws.send_text(json.dumps(frame))
            except Exception:  # noqa: BLE001 - the session is going away; the refusal still held
                break

    class _BrowserInput:
        _in_rate: int = BROWSER_INPUT_RATE

        async def start(self, agent: Any) -> None:
            cfg = agent.model.get_audio_config()
            self._in_rate = int(cfg["input"].get("sample_rate", BROWSER_INPUT_RATE))

        async def stop(self) -> None:
            pass

        async def __call__(self) -> Any:
            data = await in_q.get()
            return AudioDelta(
                format="pcm",
                source={"bytes": _resample_pcm16(data, BROWSER_INPUT_RATE, self._in_rate)},
            )

    class _BrowserOutput:
        async def start(self, agent: Any) -> None:
            rate = int(agent.model.get_audio_config()["output"]["sample_rate"])
            await ws.send_text(json.dumps({"type": "voice_meta", "rate": rate}))

        async def stop(self) -> None:
            pass

        async def __call__(self, event: Any) -> None:
            kind = _event_type(event)
            if kind.startswith("bidi_audio") and hasattr(event, "get") and event.get("audio"):
                await ws.send_text(json.dumps({"type": "audio", "data": event["audio"]}))
            elif kind.startswith("bidi_transcript") and hasattr(event, "get"):
                # 1.57 streams ``delta`` per transcript event; older wheels sent ``text``.
                text = event.get("delta", event.get("text"))
                if text is None:
                    return
                try:
                    await ws.send_text(json.dumps({"type": "transcript", "role": event.get("role", ""), "text": text}))
                except Exception:  # noqa: BLE001 - the socket is going away; the words were spoken
                    pass

    try:
        agent = build_voice_agent(bridge=bridge)
    except Exception as e:
        # This return happens BEFORE the finally below exists, so the listener has to be dropped here
        # too: one left behind would outlive the session, push into a queue nobody drains and pin this
        # closure for every later turn on the machine.
        drop_listener()
        await ws.send_text(json.dumps({"type": "error", "error": f"voice agent: {e}"}))
        return

    async def _reader() -> None:
        while not stop_evt.is_set():
            try:
                raw = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                break
            if raw.get("bytes") is not None:
                await in_q.put(raw["bytes"])
            elif raw.get("text"):
                try:
                    if json.loads(raw["text"]).get("type") == "stop":
                        break
                except Exception:
                    pass
        stop_evt.set()

    import asyncio as _a

    reader_task = _a.create_task(_reader())
    needs_task = _a.create_task(_drain_needs())
    runner = _a.create_task(agent.run(inputs=[_BrowserInput()], outputs=[_BrowserOutput()]))
    try:
        await stop_evt.wait()
    finally:
        # Unregister FIRST: a listener left behind would keep pushing into a queue nobody drains and
        # would hold this session's closure alive for every later turn on the machine.
        drop_listener()
        runner.cancel()
        reader_task.cancel()
        needs_task.cancel()
