"""One strands Agent over the same tools a script would use (``sim.tool``, ``robot.tool``), streamed as small events.

Events: {type:"text", delta} · {type:"tool_use", name, input} · {type:"tool_result", name, status, text}
        · {type:"error", message} · {type:"done"}
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from strands import Agent

SYSTEM_PROMPT = (
    "You operate robots that share one simulated world (and possibly real ones). Use the robot_* tools "
    "to observe and act, the sim_* tool for the world. Observe before you act. Be brief."
)


def build_tools(sim: Any, real: Iterable[Any] = ()) -> list[Any]:
    return [sim.tool, *(r.tool for r in sim.robots.values()), *(r.tool for r in real)]


def build_agent(sim: Any, model: Any, real: Iterable[Any] = ()) -> Agent:
    return Agent(model=model, tools=build_tools(sim, real), system_prompt=SYSTEM_PROMPT, callback_handler=None)


def refresh_tools(agent: Agent, sim: Any, real: Iterable[Any] = ()) -> None:
    """Make the agent's tools follow the world's robots (called after a revision bump)."""
    reg = agent.tool_registry
    want = {t.tool_name: t for t in build_tools(sim, real)}
    for name in list(reg.registry):
        if name.startswith(("robot_", "sim_")) and name not in want:
            reg.registry.pop(name, None)
    for name, tool in want.items():
        if name not in reg.registry:
            reg.register_tool(tool)


def _text_of(content: list[dict[str, Any]]) -> str:
    parts = []
    for c in content:
        if "text" in c:
            parts.append(c["text"])
        elif "image" in c:
            parts.append(f"[image/{c['image'].get('format', '?')}]")
        elif "json" in c:
            parts.append(json.dumps(c["json"]))
    return "\n".join(parts)


async def stream_chat(agent: Agent, prompt: str) -> AsyncIterator[dict[str, Any]]:
    """Translate ``agent.stream_async`` into page events. Never raises: an error becomes an event."""
    names: dict[str, str] = {}
    try:
        async for ev in agent.stream_async(prompt):
            if isinstance(ev.get("data"), str) and ev["data"]:
                yield {"type": "text", "delta": ev["data"]}
                continue
            msg = ev.get("message")
            if not isinstance(msg, dict):
                continue
            for block in msg.get("content") or []:
                if "toolUse" in block:
                    tu = block["toolUse"]
                    names[tu["toolUseId"]] = tu["name"]
                    yield {"type": "tool_use", "name": tu["name"], "input": tu.get("input", {})}
                elif "toolResult" in block:
                    tr = block["toolResult"]
                    yield {
                        "type": "tool_result",
                        "name": names.get(tr["toolUseId"], "?"),
                        "status": tr.get("status", "success"),
                        "text": _text_of(tr.get("content") or [])[:4000],
                    }
    except Exception as exc:  # noqa: BLE001 — the page shows the failure instead of a dead socket
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
    yield {"type": "done"}
