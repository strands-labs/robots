"""The strands AgentTool for a device (strands is imported here, only when ``.tool`` is used).

``robot.tool`` is the tool; ``robot.tool(policies=[ttt])`` is the same tool with developer-registered policy handles
the model may run by name (``run(policy="ttt", instruction=..., seconds=...)``; with exactly one handle, ``policy``
may be left out). A handle is a policy object the developer built — a remote one included — so the model never types
a network address: ``policy="zenoh://..."`` from a model stays refused.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from strands.tools.tools import PythonAgentTool
from strands.types.tools import ToolResult, ToolUse

from ..robot.run.source import from_agent
from .call import dispatch
from .spec import tool_spec
from .wire import content

_TURNS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Lock]] = weakref.WeakKeyDictionary()


class _DeviceTool(PythonAgentTool):
    """A device's tool. It is never hot-reloaded, so strands refuses a second tool of the same name in one agent
    instead of keeping the last: a simulated and a hardware so101 are both ``robot_so101`` until one gets an ``id=``.
    Calling it (``robot.tool(policies=[...])``) returns the same tool with policy handles registered.

    Calls on one world (its tool and its robots' tools) run one at a time in the order the model sent them, whatever
    the agent's tool executor: strands' default runs one message's calls at once, so ``camera c`` then ``render c``
    would otherwise race (render first, no camera yet)."""

    device: Any
    prefix: str

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> AsyncGenerator[Any]:
        world = getattr(self.device, "sim", None) or self.device
        turns = _TURNS.setdefault(asyncio.get_running_loop(), {})  # an asyncio.Lock belongs to one loop
        async with turns.setdefault(id(world), asyncio.Lock()):  # FIFO: the calls queue in the order they were sent
            async for event in super().stream(tool_use, invocation_state, **kwargs):
                yield event

    @property
    def supports_hot_reload(self) -> bool:
        return False

    def __call__(self, *, policies: Iterable[Any] | Mapping[str, Any] | None = None) -> _DeviceTool:
        return agent_tool(self.device, prefix=self.prefix, policies=policies)


def handles(policies: Iterable[Any] | Mapping[str, Any] | None) -> dict[str, Any]:
    """``{name: policy}``: a mapping as given; a list named by ``.name``, then a remote policy's ``.deployment``.

    Two list entries that derive the same name are refused, never collapsed: the tool spec, the refusal hint and
    ``run(policy=...)`` all read this dict, so a silent overwrite would move the arm with whichever came last."""
    if policies is None:
        return {}
    if isinstance(policies, Mapping):
        out = dict(policies)
    else:
        out = {}
        for i, p in enumerate(policies):
            name = str(getattr(p, "name", None) or getattr(p, "deployment", None) or f"policy{i + 1}")
            if name in out:
                raise ValueError(
                    f"policies: two handles are both named {name!r} ({out[name]!r} and {p!r}); "
                    "pass a {name: policy} mapping to name them yourself"
                )
            out[name] = p
    for name, p in out.items():
        if isinstance(p, str | bytes) or not (callable(p) or hasattr(p, "act")):
            raise TypeError(f"policies[{name!r}] must be a policy object (Policy(...), a callable), got {p!r}")
    return out


def agent_tool(
    device: Any, *, prefix: str = "robot", policies: Iterable[Any] | Mapping[str, Any] | None = None
) -> _DeviceTool:
    """A strands tool whose spec is generated from ``device``'s rpcs; errors go back to the model as text."""
    registered = handles(policies)
    spec = tool_spec(device, prefix=prefix, policies=tuple(registered))

    def run(tool_use: ToolUse, **_invocation_state: Any) -> ToolResult:
        tid = tool_use["toolUseId"]
        try:
            with from_agent():
                result = dispatch(device, dict(tool_use.get("input") or {}), policies=registered)
        except Exception as exc:  # noqa: BLE001 - the model reads the refusal and corrects its next call
            return {"toolUseId": tid, "status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
        return {"toolUseId": tid, "status": "success", "content": content(result)}  # type: ignore[typeddict-item]

    tool = _DeviceTool(spec["name"], spec, run)  # type: ignore[arg-type]
    tool.device, tool.prefix = device, prefix
    return tool


def sim_tool(sim: Any) -> PythonAgentTool:
    """The world's verbs (add robot/object/camera, step, reset, render, ...) as one strands tool."""
    from .world import WorldDevice

    return agent_tool(WorldDevice(sim), prefix="sim")
