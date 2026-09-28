"""Bus-guarded direct-serial tools for the dashboard agent.

The SDK's pose_tool / serial_tool open a serial port DIRECTLY -- but this
dashboard's robots are child processes that HOLD those buses. A second owner on a
half-duplex servo bus corrupts both conversations, so the guarded wrappers
here refuse a held port BEFORE the SDK tool opens it, naming the holder, and
refuse a missing port with the machine's actual choices instead of letting
the SDK default (/dev/ttyACM0, absent on macOS) produce a confusing error.

Neither SDK tool raises tool_context.interrupt, so the dashboard's
agent_hitl.MOTION_ACTIONS layer is their ONLY human gate -- registration adds
those rows (see :mod:`strands_robots.dashboard.agent_hitl`); this module only owns bus safety.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any, cast

__all__ = [
    "POSE_PORT_FREE",
    "SERIAL_PORT_FREE",
    "port_refusal",
    "build_direct_serial_tools",
]

#: actions that never touch a serial port -- the guard waves them through.
POSE_PORT_FREE: frozenset[str] = frozenset({"list_poses", "show_pose", "delete_pose"})
SERIAL_PORT_FREE: frozenset[str] = frozenset({"list_ports"})


def _scan_hint(scan: Callable[[], list[dict[str, Any]]]) -> str:
    try:
        devices = [str(p.get("device")) for p in scan() if p.get("device")]
    except Exception:  # noqa: BLE001 - the hint must not break the refusal
        devices = []
    if not devices:
        return "no candidate servo ports are attached right now (scan_serial_ports found none)."
    return "ports on this machine: " + ", ".join(sorted(devices)) + "."


def port_refusal(
    port: Any,
    action: str,
    *,
    port_free: frozenset[str],
    holders: Callable[[str], list[int]],
    tracked: Callable[[], Mapping[int, str]],
    scan: Callable[[], list[dict[str, Any]]],
    exists: Callable[[str], bool] = os.path.exists,
) -> str | None:
    """The refusal text, or None when the SDK tool may open the port.

    Pure judgement: every I/O dependency is injected, so tests hand in fake
    holders and the live wiring hands in bus_claim / DeviceManager readers.
    """
    act = str(action or "").strip()
    if act in port_free:
        return None
    p = str(port or "").strip()
    if not p or not exists(p):
        what = f"port {p!r} does not exist on this machine" if p else "no serial port was given"
        return f"{what}, so the {act or 'requested'} action cannot proceed. Pass port= explicitly -- {_scan_hint(scan)}"
    try:
        pids = holders(p)
    except Exception:  # noqa: BLE001 - an unreadable holder list means UNKNOWN, refuse loudly
        return (
            f"{p}: could not determine whether another process holds this bus, "
            f"so direct serial access is refused rather than risked."
        )
    if not pids:
        return None
    known = dict(tracked() or {})
    ours = {pid: known[pid] for pid in pids if pid in known}
    strangers = [pid for pid in pids if pid not in known]
    parts: list[str] = []
    if ours:
        names = ", ".join(f"{peer} (pid {pid})" for pid, peer in sorted(ours.items()))
        parts.append(
            f"held by this dashboard's own spawned robot: {names} -- despawn that peer first "
            f"(or drive it through its proxy tool / robot_mesh instead of raw serial)"
        )
    if strangers:
        parts.append(
            "held by process(es) this dashboard does not manage: pid " + ", ".join(str(pid) for pid in strangers)
        )
    return (
        f"{p} is {'; also '.join(parts)}. A second owner on a half-duplex serial bus corrupts "
        f"both conversations, so the call is refused. Other {_scan_hint(scan)}"
    )


def available_names() -> list[str]:
    """Tool names build_direct_serial_tools WOULD produce -- same import rules.

    The not-yet-built agent badge reports this instead of a hardcoded guess
    (the ['fleet'] badge lied once already; agent_bridge.agent_status).
    """
    out: list[str] = []
    try:
        from strands_robots.tools.pose_tool import pose_tool  # noqa: F401

        out.append("pose_tool")
    except Exception:  # noqa: BLE001 - absent SDK tool means absent badge entry
        pass
    try:
        from strands_robots.tools.serial_tool import serial_tool  # noqa: F401

        out.append("serial_tool")
    except Exception:  # noqa: BLE001
        pass
    return out


def _default_holders(port: str) -> list[int]:
    from strands_robots.dashboard import bus_claim

    return bus_claim.bus_holders(port)


def _default_scan() -> list[dict[str, Any]]:
    from strands_robots.dashboard.device_manager import scan_serial_ports

    return scan_serial_ports()


def _guard(sdk_tool: Any, port_free: frozenset[str], deps: dict[str, Any]) -> Any:
    """Re-decorate the SDK tool's original function behind the bus guard.

    functools.wraps carries __name__/__doc__ and sets __wrapped__, which
    inspect.signature follows -- so the strands @tool decorator reads the
    ORIGINAL signature and the agent sees an honest, identical surface.
    """
    from strands import tool

    fn = getattr(sdk_tool, "_tool_func", None) or inspect.unwrap(sdk_tool)
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        refusal = port_refusal(
            bound.arguments.get("port"),
            str(bound.arguments.get("action") or ""),
            port_free=port_free,
            **deps,
        )
        if refusal:
            return {"status": "error", "content": [{"text": refusal}]}
        return cast("dict[str, Any]", fn(*args, **kwargs))

    return tool(guarded)


def build_direct_serial_tools(
    tracked: Callable[[], Mapping[int, str]],
    *,
    holders: Callable[[str], list[int]] | None = None,
    scan: Callable[[], list[dict[str, Any]]] | None = None,
    exists: Callable[[str], bool] = os.path.exists,
) -> list[Any]:
    """Guarded pose_tool + serial_tool, or [] when the SDK tools cannot import.

    ``tracked`` maps pid -> peer_id for the dashboard's own spawned children
    (DeviceManager.robots), so a refusal can NAME the holding peer.
    """
    deps = {
        "holders": holders or _default_holders,
        "tracked": tracked,
        "scan": scan or _default_scan,
        "exists": exists,
    }
    out: list[Any] = []
    try:
        from strands_robots.tools.pose_tool import pose_tool as _pose

        out.append(_guard(_pose, POSE_PORT_FREE, deps))
    except Exception:  # noqa: BLE001 - the agent must build even if a tool cannot import
        pass
    try:
        from strands_robots.tools.serial_tool import serial_tool as _serial

        out.append(_guard(_serial, SERIAL_PORT_FREE, deps))
    except Exception:  # noqa: BLE001
        pass
    return out
