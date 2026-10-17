"""Dispatch one tool input ``{"rpc": name, name: {args}}`` to the device, as an agent's command."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .spec import OPERATOR_ONLY, handler, rpc_names
from .wire import decode

# An agent's ``act`` is never the developer's own call: it goes through the robot's operator gate.
_AS_AGENT = {"act": "confirm_act"}


def target(device: Any, rpc: str) -> Callable[..., Any]:
    """The method a tool call runs: the declared rpc, or its gated form when an agent sends it."""
    if rpc in OPERATOR_ONLY:
        raise PermissionError(f"{rpc!r} is the operator's to call on {device.id}, not an agent's")
    names = rpc_names(device)
    if rpc not in names:
        raise ValueError(f"unknown rpc {rpc!r}; {device.id} offers {list(names)}")
    gated = _AS_AGENT.get(rpc)
    return getattr(device, gated) if gated and hasattr(device, gated) else handler(device, rpc)


def dispatch(device: Any, tool_input: Mapping[str, Any]) -> Any:
    rpc = tool_input.get("rpc")
    if not isinstance(rpc, str):
        raise ValueError(f"rpc is required; one of {list(rpc_names(device))}")
    extra = sorted(k for k in tool_input if k not in ("rpc", rpc))
    if extra:
        raise TypeError(f"arguments of {rpc!r} go under {rpc!r}; unexpected {extra}")
    args = tool_input.get(rpc) or {}
    if not isinstance(args, Mapping):
        raise TypeError(f"{rpc!r} must be an object of arguments, got {type(args).__name__}")
    fn = target(device, rpc)
    decode(handler(device, rpc), args)  # refuse what the declared rpc does not take, before anything moves
    return fn(**decode(fn, args))
