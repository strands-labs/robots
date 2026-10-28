"""Dispatch one tool input ``{"rpc": name, name: {args}}`` to the device, as an agent's command."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

from ..robot.run.source import REMOTE_SPEC
from .spec import DEVELOPER_ONLY_ARGS, OPERATOR_ONLY, handler, rpc_names
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
    if rpc == "hold" and hasattr(device, "gate"):  # torque on at the measured pose: the operator says go first
        return lambda: _gated_hold(device)
    if rpc == "run" and hasattr(device, "start"):  # the robot's one task: a second run is refused, stop() ends it
        return _run_as_the_task(device)
    gated = _AS_AGENT.get(rpc)
    return getattr(device, gated) if gated and hasattr(device, gated) else handler(device, rpc)


def _gated_hold(device: Any) -> None:
    device.gate(device.observe(images=False).joints)
    device.hold()


def _run_as_the_task(device: Any) -> Callable[..., Any]:
    """``run`` as ``device.start(...)``, waited on until the run ends (``decode`` reads ``start``'s signature)."""

    @functools.wraps(device.start)
    def run(**args: Any) -> Any:
        return device.start(**args)._wait()

    return run


def resolve_policy(args: Mapping[str, Any], policies: Mapping[str, Any]) -> dict[str, Any]:
    """``policy="ttt"`` -> the registered handle; left out with exactly one handle -> that one; a string that is not
    a handle stays a spec for ``run`` (a preset, a Hub repo; a network address was refused before this)."""
    out = dict(args)
    spec = out.get("policy")
    if isinstance(spec, str) and spec in policies:
        out["policy"] = policies[spec]
    elif spec is None and "policy" not in out and len(policies) == 1:
        out["policy"] = next(iter(policies.values()))
    elif spec is None and "policy" not in out and policies:
        raise ValueError(f"policy is required: one of the registered {sorted(policies)}")
    return out


def dispatch(device: Any, tool_input: Mapping[str, Any], *, policies: Mapping[str, Any] | None = None) -> Any:
    rpc = tool_input.get("rpc")
    if not isinstance(rpc, str):
        raise ValueError(f"rpc is required; one of {list(rpc_names(device))}")
    extra = sorted(k for k in tool_input if k not in ("rpc", rpc))
    if extra:
        raise TypeError(f"arguments of {rpc!r} go under {rpc!r}; unexpected {extra}")
    args = tool_input.get(rpc) or {}
    if not isinstance(args, Mapping):
        raise TypeError(f"{rpc!r} must be an object of arguments, got {type(args).__name__}")
    kept = sorted(set(args) & DEVELOPER_ONLY_ARGS)
    if kept:
        raise PermissionError(f"{kept} of {rpc!r} are the developer's to pass, not an agent's")
    policies = policies or {}
    many = args.get("policies")  # a world's run takes {robot id: spec}; a robot's run takes policy=
    for spec in [args.get("policy"), *(many.values() if isinstance(many, Mapping) else ())]:
        if isinstance(spec, str) and REMOTE_SPEC in spec and spec not in policies:  # the cameras would stream there,
            hint = f"; registered: {sorted(policies)}" if policies else ""  # before the gate sees an action
            raise PermissionError(
                f"a policy at a network address is the developer's to pass, not an agent's: {spec}{hint}"
            )
    if rpc == "run":
        args = resolve_policy(args, policies)
    fn = target(device, rpc)
    decode(handler(device, rpc), args)  # refuse what the declared rpc does not take, before anything moves
    return fn(**decode(fn, args))
