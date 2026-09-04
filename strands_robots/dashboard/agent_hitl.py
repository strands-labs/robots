"""Human-in-the-loop gate for agent tool calls that would move real hardware.

The hook pauses the agent (SDK interrupt) instead of refusing, so a human
yes resumes the SAME turn and the tool executes. Stopping is never gated.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any, cast

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

from strands_robots.dashboard.agent_motion import MOTION_ENV, peer_is_physical

logger = logging.getLogger(__name__)

INTERRUPT_NAME = "physical_motion"

#: tool name -> actions that can put a real robot in motion. Absent tool = never gated.
#: stop / emergency_stop / status / peers are deliberately NOT here: stopping is never gated.
MOTION_ACTIONS: dict[str, frozenset[str]] = {
    "fleet": frozenset({"task"}),
    # robot_mesh is deliberately ABSENT: it raises its own SDK-native interrupt
    # (tool_context.interrupt in strands_robots/tools/robot_mesh.py) on every
    # physical action, so listing it here would ask the operator twice for one
    # command. This dict gates only the dashboard's bespoke tools.
    # The bus-guarded direct-serial tools (dashboard/direct_serial.py) raise NO
    # interrupt of their own (grep tool_context.interrupt in the SDK tools = 0),
    # so this layer is their ONLY human gate. Reads, emergency_stop and
    # delete_pose stay out: stopping is never gated. serial "monitor" only ever
    # calls ser.read (serial_tool.py) so it is a read too.
    "pose_tool": frozenset({"load_pose", "move_motor", "move_multiple", "incremental_move", "reset_to_home"}),
    "serial_tool": frozenset({"send", "send_read", "feetech_position", "feetech_velocity"}),
}

#: tools whose gated input names the motion in FIELDS, not an instruction string.
DIRECT_SERIAL_TOOLS: frozenset[str] = frozenset({"pose_tool", "serial_tool"})

#: the motion-bearing fields, in the order an operator reads them.
_DETAIL_FIELDS = ("pose_name", "motor_name", "positions", "position", "delta", "steps", "data")


def _direct_serial_detail(action: str, tool_input: dict) -> str:
    """The gated call's own motion fields as one readable line -- never invented."""
    parts = [action]
    for key in _DETAIL_FIELDS:
        value = tool_input.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts) if len(parts) > 1 else ""


def _resolve_target(
    tool_name: str,
    tool_input: Mapping[str, Any],
    bound_targets: Mapping[str, str] | None,
) -> str:
    """The peer or port a yes would move, read from the most trusted source.

    Precedence is by TRUST, never by presence. The model authors ``tool_input``
    and the ``peers`` action that lists a sim's name is deliberately ungated, so
    a sim peer's name is always within its reach; resolving the target from a
    field it writes lets it choose which robot the gate believes it is asking
    about. Each tool therefore has exactly one trusted source, and a field the
    model wrote is read only where the tool itself reads the same field:

    * a proxy tool IS its peer, so the per-build binding names the target and no
      input can move it. This is the guarantee the binding exists to make.
    * a direct-serial tool addresses a ``port``, one of its own declared
      parameters. Neither ``pose_tool`` nor ``serial_tool`` declares ``target``,
      and the SDK drops undeclared keys before the call, so a ``target`` on such
      an input is unconsumed by construction: reading it would let the model
      name a robot that is not the one the port moves.
    * every other gated tool (``fleet``) declares ``target`` itself, so the gate
      and the tool resolve the same peer from the same field.

    An unresolvable target is returned empty, which is never a key on the peers
    snapshot, so :func:`~strands_robots.dashboard.agent_motion.peer_is_physical`
    treats it as metal and the call is gated.
    """
    if bound_targets is not None and tool_name in bound_targets:
        return str(bound_targets.get(tool_name) or "").strip()
    if tool_name in DIRECT_SERIAL_TOOLS:
        return str(tool_input.get("port") or "").strip()
    return str(tool_input.get("target") or "").strip()


_TRUE = ("1", "true", "yes", "on")


def _granted(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(MOTION_ENV, "")).strip().lower() in _TRUE


def motion_intent(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    peers: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
    *,
    extra_actions: Mapping[str, frozenset[str]] | None = None,
    bound_targets: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Would this tool call start physical motion needing a human yes?

    Returns the structured interrupt reason, or None when the call may proceed
    (not a motion action, target is a sim, or the always-allow grant is set).
    """
    actions = MOTION_ACTIONS.get(tool_name)
    if actions is None and extra_actions is not None:
        # per-peer proxy tools: their gate rows are DERIVED per agent build
        # (peer_tools.motion_actions_for), never hand-kept here.
        actions = extra_actions.get(tool_name)
    if actions is None:
        return None
    tool_input = tool_input or {}
    action = str(tool_input.get("action") or "").strip()
    if action not in actions:
        return None
    if _granted(env):
        return None  # the always-allow fast lane: no interrupt is raised

    target = _resolve_target(tool_name, tool_input, bound_targets)
    peer = (peers or {}).get(target)
    physical, why = peer_is_physical(peer)
    if not physical:
        return None

    instruction = str(tool_input.get("instruction") or tool_input.get("message") or "")
    if not instruction and tool_name in DIRECT_SERIAL_TOOLS:
        # pose/serial inputs carry the motion in named fields, not an
        # instruction string; show the operator WHAT a yes moves, verbatim.
        instruction = _direct_serial_detail(action, cast("dict[str, Any]", tool_input))
    reason: dict[str, Any] = {
        "tool": tool_name,
        "action": action,
        "target": target or "(unnamed peer)",
        "instruction": instruction,
        "why_physical": why,
    }
    if tool_input.get("duration") is not None:
        try:
            reason["duration"] = float(tool_input["duration"])
        except (TypeError, ValueError):
            # A duration the model wrote as prose ("30s") or as a structure is
            # dropped rather than raised on. This field is one extra line the
            # operator reads, not part of deciding WHETHER to ask: the gate has
            # already resolved the action as motion and the target as metal by
            # this point, so the interrupt fires with or without it and the tool
            # still cannot run without a yes. Raising instead would take an
            # unparseable optional field and abort the operator's turn with an
            # exception out of a BeforeToolCallEvent hook, so the human is never
            # asked the question this gate exists to ask them. The tuple is the
            # exact pair float() raises: TypeError for a non-numeric type,
            # ValueError for a string that does not parse.
            pass
    return reason


def response_approves(response: Any) -> bool:
    """Interpret a human interrupt response. Anything but an explicit yes is a no."""
    if isinstance(response, bool):
        return response
    if isinstance(response, Mapping):
        return response_approves(response.get("approve"))
    if isinstance(response, str):
        return response.strip().lower() in _TRUE + ("y", "approve", "approved")
    return False


# --- one-shot approval grants ------------------------------------------------
# The fleet tool's own agent_motion_allowed() gate stays as a backstop; a human
# yes deposits a grant here that the tool consumes for exactly one call.

_grants_lock = threading.Lock()
_grants: set[str] = set()


def _grant_key(tool_name: str, tool_input: Mapping[str, Any] | None) -> str:
    tool_input = tool_input or {}
    return "|".join(
        (
            tool_name,
            str(tool_input.get("action") or ""),
            str(tool_input.get("target") or ""),
            str(tool_input.get("instruction") or tool_input.get("message") or ""),
        )
    )


def deposit_grant(tool_name: str, tool_input: Mapping[str, Any] | None) -> None:
    with _grants_lock:
        _grants.add(_grant_key(tool_name, tool_input))


def consume_grant(tool_name: str, tool_input: Mapping[str, Any] | None) -> bool:
    """True exactly once per deposited grant for this call's shape."""
    key = _grant_key(tool_name, tool_input)
    with _grants_lock:
        if key in _grants:
            _grants.discard(key)
            return True
    return False


def cancel_sentence(reason: Mapping[str, Any]) -> str:
    """What the model relays when the human says no."""
    return (
        f"the human declined: {reason.get('target')} was NOT sent "
        f"{reason.get('instruction') or 'that instruction'!r}. Nothing moved. "
        f"Ask them what they would like instead."
    )


class MotionInterruptHook(HookProvider):
    """Pause before any tool call that would start physical motion.

    ``peers_snapshot`` is a callable returning the current fleet peers dict, so
    the physicality verdict is read at call time, never cached.
    """

    def __init__(
        self,
        peers_snapshot: Callable[[], Mapping[str, Any]],
        proxy_motion: Mapping[str, frozenset[str]] | None = None,
        proxy_targets: Mapping[str, str] | None = None,
    ) -> None:
        self._peers_snapshot = peers_snapshot
        self._proxy_motion = dict(proxy_motion or {})
        self._proxy_targets = dict(proxy_targets or {})

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._gate)

    def _gate(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = str(tool_use.get("name") or "")
        tool_input = tool_use.get("input") or {}
        try:
            peers = self._peers_snapshot() or {}
        except Exception:  # noqa: BLE001 - unreadable snapshot means UNKNOWN, i.e. metal
            peers = {}
        reason = motion_intent(
            name,
            tool_input,
            peers,
            extra_actions=self._proxy_motion,
            bound_targets=self._proxy_targets,
        )
        if reason is None:
            return
        # Raises InterruptException on first pass; returns the human response on resume.
        response = event.interrupt(INTERRUPT_NAME, reason=reason)
        approved = response_approves(response)
        # Record the operator's reply in the local audit log for both outcomes.
        # The reply itself never reaches the model (cancel_sentence returns a
        # flat sentinel); the audit row is the only place it survives.
        from strands_robots.tools._hitl_audit import log_operator_response

        log_operator_response(
            "dashboard_agent_hitl",
            str(reason.get("action", "")),
            str(reason.get("target", "")),
            approved=approved,
            response=response,
        )
        if approved:
            deposit_grant(name, tool_input)
            return
        event.cancel_tool = cancel_sentence(reason)
