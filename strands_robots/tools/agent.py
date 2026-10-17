"""The strands AgentTool for a device (strands is imported here, only when ``.tool`` is used)."""

from __future__ import annotations

from typing import Any

from strands.tools.tools import PythonAgentTool
from strands.types.tools import ToolResult, ToolUse

from ..robot.run.source import from_agent
from .call import dispatch
from .spec import tool_spec
from .wire import content


def agent_tool(device: Any, *, prefix: str = "robot") -> PythonAgentTool:
    """A strands tool whose spec is generated from ``device``'s rpcs; errors go back to the model as text."""
    spec = tool_spec(device, prefix=prefix)

    def run(tool_use: ToolUse, **_invocation_state: Any) -> ToolResult:
        tid = tool_use["toolUseId"]
        try:
            with from_agent():
                result = dispatch(device, dict(tool_use.get("input") or {}))
        except Exception as exc:  # noqa: BLE001 - the model reads the refusal and corrects its next call
            return {"toolUseId": tid, "status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
        return {"toolUseId": tid, "status": "success", "content": content(result)}  # type: ignore[typeddict-item]

    return PythonAgentTool(spec["name"], spec, run)  # type: ignore[arg-type]


def sim_tool(sim: Any) -> PythonAgentTool:
    """The world's verbs (add robot/object/camera, step, reset, render, ...) as one strands tool."""
    from .world import WorldDevice

    return agent_tool(WorldDevice(sim), prefix="sim")
