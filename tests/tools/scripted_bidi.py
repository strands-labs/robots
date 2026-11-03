"""A scripted strands.bidi model (no network): the first audio it hears becomes one turn of tool calls.

    ScriptedBidi(heard="open the gripper", calls=[("robot_so101", {"rpc": "act", ...})], say="done")

On the first non-silent audio delta it emits the user's transcript and the tool-use group; when the agent sends the
tool results back it answers with an assistant transcript, a short audio chunk and a response stop.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from strands.bidi.models.configs import AudioConfig, AudioStreamConfig, ModelConfig
from strands.bidi.models.model import BidiModel
from strands.bidi.types.content import BidiMessage
from strands.bidi.types.events import (
    BidiAudioDeltaEvent,
    BidiConnectionStartEvent,
    BidiResponseStartEvent,
    BidiResponseStopEvent,
    BidiToolUseBlocksEvent,
    BidiTranscriptDeltaEvent,
    BidiTranscriptStartEvent,
    BidiTranscriptStopEvent,
)
from strands.bidi.types.media import AudioDelta

RATE = 16000


class ScriptedBidi(BidiModel):
    def __init__(self, *, heard: str, calls: list[tuple[str, dict[str, Any]]], say: str = "done") -> None:
        self._config = ModelConfig(model_id="scripted-bidi")
        self.heard, self.calls, self.say = heard, calls, say
        self.tools_seen: list[str] = []
        self.results: list[dict[str, Any]] = []
        self.audio_bytes = 0
        self._q: asyncio.Queue[Any] = asyncio.Queue()
        self._fired = False

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)  # type: ignore[typeddict-item]

    def get_config(self) -> ModelConfig:
        return self._config

    def get_audio_config(self) -> AudioConfig:
        s = AudioStreamConfig(sample_rate=RATE, channels=1, format="pcm")
        return AudioConfig(input=s, output=s)

    async def start(self, system_prompt: str | None = None, tools: Any = None, messages: Any = None, **_: Any) -> None:
        self.system_prompt = system_prompt
        self.tools_seen = [t["name"] for t in tools or []]
        await self._q.put(BidiConnectionStartEvent("scripted", "scripted-bidi"))

    async def stop(self) -> None:
        await self._q.put(None)

    async def receive(self) -> AsyncIterator[Any]:  # type: ignore[override]
        while (ev := await self._q.get()) is not None:
            yield ev

    async def send(self, content: Any) -> None:
        if isinstance(content, AudioDelta):
            data = content.source.get("bytes") or b""
            self.audio_bytes += len(data)
            if not self._fired and any(data):
                self._fired = True
                for ev in self._user_turn():
                    await self._q.put(ev)
            return
        if isinstance(content, BidiMessage):
            blocks = [b for b in content.content if getattr(b, "tool_use_id", None)]
            if blocks:
                self.results.extend({"id": b.tool_use_id, "status": b.status} for b in blocks)
                for ev in self._answer():
                    await self._q.put(ev)

    def _user_turn(self) -> list[Any]:
        uses = [{"toolUseId": f"call-{i}", "name": name, "input": inp} for i, (name, inp) in enumerate(self.calls)]
        return [
            BidiTranscriptStartEvent("user", "u1"),
            BidiTranscriptDeltaEvent(self.heard, "user", "u1"),
            BidiTranscriptStopEvent("user", "u1"),
            BidiResponseStartEvent("r1"),
            BidiToolUseBlocksEvent(uses),  # type: ignore[arg-type]
            BidiResponseStopEvent("r1"),
        ]

    def _answer(self) -> list[Any]:
        pcm = base64.b64encode(b"\x01\x00" * 160).decode()
        return [
            BidiResponseStartEvent("r2"),
            BidiTranscriptStartEvent("assistant", "a1"),
            BidiTranscriptDeltaEvent(self.say, "assistant", "a1"),
            BidiTranscriptStopEvent("assistant", "a1"),
            BidiAudioDeltaEvent(pcm, "pcm", RATE, 1, "a1"),
            BidiResponseStopEvent("r2"),
        ]


__all__ = ["ScriptedBidi"]
