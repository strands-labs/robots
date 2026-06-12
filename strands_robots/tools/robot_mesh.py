"""Agent-facing tool for coordinating robots on the Zenoh mesh.

Every :class:`~strands_robots.robot.Robot` and
:class:`~strands_robots.simulation.Simulation` constructed in this process is
already a mesh peer (see :mod:`strands_robots.mesh`); this tool exposes that
mesh to a Strands agent via a single ``robot_mesh`` action dispatcher.

The action vocabulary mirrors the underlying :class:`~strands_robots.mesh.Mesh`
API plus a few discovery helpers:

==================  ===================================================
``peers``           List local + remote peers
``status``          One-line summary of mesh state
``tell``            ``mesh.tell(target, instruction, ...)``
``send``            ``mesh.send(target, json.loads(command), ...)``
``broadcast``       ``mesh.broadcast(json.loads(command), ...)``
``stop``            Send ``{"action": "stop"}`` to a single peer
``emergency_stop``  Broadcast stop to every peer (audited)
``subscribe``       ``mesh.subscribe(target, name=...)`` (buffer mode)
``watch``           ``mesh.on_stream(target)``
``inbox``           Read buffered messages from a subscription
``unsubscribe``     Unsubscribe from a topic by name
==================  ===================================================

The tool always returns a Strands-compatible dict::

    {"status": "success" | "error", "content": [{"text": "..."}]}

It never raises out of the dispatcher: every error path renders a
human-readable text payload so the calling agent can recover.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_robots.mesh import security as _security

logger = logging.getLogger(__name__)


# Per-action sliding-window rate limiter for LLM-facing actions.
_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "tell": (30, 60.0),
    "send": (30, 60.0),
    "broadcast": (10, 60.0),
    "stop": (20, 60.0),
    "rpc": (30, 60.0),
    "emergency_stop": (3, 60.0),
}
_RATE_HISTORY: dict[str, collections.deque[float]] = {}
_RATE_LOCK = threading.Lock()

# Actions whose physical effect is fleet-wide. Each one routes through a
# Strands SDK interrupt so the calling host can request explicit human
# approval before the mesh issues the command. Unlike a boolean tool
# parameter the interrupt response is delivered by the framework
# out-of-band of the LLM's tool-argument flow, so an injected prompt
# cannot smuggle approval.
_INTERRUPT_REQUIRED: frozenset[str] = frozenset({"emergency_stop", "broadcast", "rpc"})

# Affirmative responses accepted from the interrupt prompt. Anything else
# (empty string, "n", "no", "cancel", whitespace) is treated as decline.
_AFFIRMATIVE_RESPONSES: frozenset[str] = frozenset({"y", "yes", "approve", "approved"})


def _interrupt_approves(response: object) -> bool:
    """True iff *response* is an explicit affirmative.

    The interrupt mechanism returns whatever the operator submitted, which
    is normally a string but the contract is "JSON-serialisable any". We
    accept the canonical short forms only — defence in depth against
    accidental approval from a typo.
    """
    if not isinstance(response, str):
        return False
    return response.strip().lower() in _AFFIRMATIVE_RESPONSES


def _rate_limit_check(action: str) -> str | None:
    """Return None if a slot is available, else the rejection message.

    Inspects the sliding-window history but does NOT consume a slot.
    Use :func:`_rate_limit_record` after a fleet-wide action's HITL
    approval is positively granted (or unconditionally for actions
    that do not require approval).

    Splitting check from record means a *declined* HITL approval no
    longer consumes a slot — without the split, three nuisance LLM
    prompts that an operator declined within a minute would lock the
    agent out of issuing a real ``emergency_stop``. That's the
    opposite of the intended safety property: the rate limit exists
    to bound LLM-driven nuisance, not to inhibit a genuine emergency.
    """
    cfg = _RATE_LIMITS.get(action)
    if cfg is None:
        return None
    max_calls, window = cfg
    now = time.monotonic()
    with _RATE_LOCK:
        bucket = _RATE_HISTORY.setdefault(action, collections.deque())
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= max_calls:
            wait = window - (now - bucket[0])
            return (
                f"rate limit exceeded for action '{action}': "
                f"max {max_calls} calls per {window:.0f}s window. "
                f"Try again in {wait:.1f}s."
            )
    return None


def _rate_limit_record(action: str) -> None:
    """Append a slot to *action*'s sliding-window history.

    Call this only after a HITL-required action's approval is granted,
    or unconditionally for actions that do not require an interrupt.
    Pairs with :func:`_rate_limit_check`.
    """
    cfg = _RATE_LIMITS.get(action)
    if cfg is None:
        return
    _, window = cfg
    now = time.monotonic()
    with _RATE_LOCK:
        bucket = _RATE_HISTORY.setdefault(action, collections.deque())
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        bucket.append(now)


def _reset_rate_limits() -> None:
    """Test helper: clear sliding-window history."""
    with _RATE_LOCK:
        _RATE_HISTORY.clear()


def _rate_limit_check_and_record(action: str) -> str | None:
    """Atomic check+record under a single _RATE_LOCK acquisition.

    Used on the post-HITL-approval path to close the TOCTOU between
    :func:`_rate_limit_check` (called BEFORE the operator interrupt) and
    :func:`_rate_limit_record` (called AFTER). Without this, two
    concurrent emergency_stop or broadcast invocations could each pass
    the pre-interrupt check, both get operator-approved on different
    threads, and both record -- briefly exceeding the configured limit.

    Returns None if the slot was atomically reserved, else the
    rejection message (caller should treat as 'rate limit raced past us
    while we were waiting on the operator interrupt; reject this one').
    """
    cfg = _RATE_LIMITS.get(action)
    if cfg is None:
        return None
    max_calls, window = cfg
    now = time.monotonic()
    with _RATE_LOCK:
        bucket = _RATE_HISTORY.setdefault(action, collections.deque())
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= max_calls:
            wait = window - (now - bucket[0])
            return (
                f"rate limit exceeded for action '{action}' between check "
                f"and record (concurrent approval raced past): max {max_calls} "
                f"calls per {window:.0f}s window. Try again in {wait:.1f}s."
            )
        bucket.append(now)
        return None


def _audit_tool_action(action: str, target: str, success: bool, detail: str) -> None:
    """Best-effort audit log of every safety-significant tool call.

    R7-5: a swallowed exception with no log line means a broken audit
    path silently disappears. Match the ``core.py:_on_cmd`` pattern —
    log at DEBUG so operators investigating "why don't I see my LLM
    tool actions in the audit log?" get a breadcrumb without flooding
    production. Audit failures must NEVER propagate up into the safety
    code path; the catch is intentionally wide for that reason and
    documented here so AGENTS.md > "Exception Clauses Must Be Narrow"
    is not violated implicitly.
    """
    try:
        from strands_robots.mesh.audit import log_safety_event

        log_safety_event(
            "llm_tool_action",
            "robot_mesh_tool",
            {
                "action": action,
                "target": target,
                "success": success,
                "detail": detail[:500],
            },
        )
    except Exception as audit_exc:  # noqa: BLE001 — see docstring
        logger.debug("[robot_mesh] audit log unavailable: %s", audit_exc)


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _resolve_mesh(target: str) -> Any | None:
    """Return a local Mesh in this process to use as the gateway for RPC.

    The agent does not need to know its own peer_id: any local mesh in
    ``_LOCAL_ROBOTS`` is functionally equivalent for outbound calls because
    they all share the same Zenoh session.

    Important: when *target* matches a local peer_id, we deliberately pick a
    *different* local mesh as the gateway. Using the target as its own
    gateway triggers ``_on_cmd``'s self-loop drop (``sender_id == peer_id``)
    and the call silently times out. When the target IS the only local mesh,
    we still return it — the caller will get a timeout, which is the
    expected behaviour for "send to yourself".
    """
    from strands_robots.mesh import get_local_robots

    locals_ = get_local_robots()
    if not locals_:
        return None
    if target:
        # Prefer a local mesh whose peer_id is NOT the target so we don't
        # send-to-self via the target's own session.
        for pid, m in locals_.items():
            if pid != target:
                return m
    # Either no target was specified or every local mesh IS the target —
    # fall back to "any one" (matching the original behaviour for the
    # single-mesh case).
    return next(iter(locals_.values()))


# ── Device Connect dispatch helpers ────────────────────────────────────────
# Device Connect is the primary discovery + RPC layer; the Zenoh mesh above is
# the fallback. These helpers are invoked by robot_mesh() AFTER its safety
# gates, so DC dispatch inherits the rate limit, HITL approval, validation, and
# audit. When DC is unavailable or has discovered no devices the helpers return
# None and robot_mesh() falls through to the built-in mesh path.

_dc_state = {"connected": False}


class _DCResult(dict):
    """Strands tool-response dict whose ``str()`` renders the text block cleanly."""

    def __str__(self) -> str:
        content = self.get("content", [])
        if content and isinstance(content[0], dict):
            return content[0].get("text", super().__str__())
        return super().__str__()


def _dc_ensure_connected() -> None:
    """Establish the Device Connect agent-side connection (idempotent)."""
    if _dc_state["connected"]:
        return
    os.environ.setdefault("MESSAGING_BACKEND", "zenoh")
    # Security hardening: do NOT force insecure transport here. Previously this
    # set DEVICE_CONNECT_ALLOW_INSECURE=true process-wide, silently downgrading
    # every connection in the process. Insecure mode is now strictly opt-in by
    # the operator. If they have opted in, surface a warning so it is visible.
    if os.environ.get("DEVICE_CONNECT_ALLOW_INSECURE", "").lower() in ("true", "1", "yes"):
        logger.warning(
            "DEVICE_CONNECT_ALLOW_INSECURE is enabled — agent-side Device "
            "Connect traffic is unencrypted and unauthenticated. Use only on "
            "a trusted, isolated network."
        )
    from device_connect_agent_tools.connection import connect, get_connection

    try:
        get_connection()
    except Exception:
        connect()
    _dc_state["connected"] = True


def _try_device_connect(
    action: str,
    target: str,
    instruction: str,
    command: str,
    policy_provider: str,
    policy_port: int,
    duration: float,
    timeout: float,
    function: str = "",
    validated_command: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Dispatch *action* through Device Connect, or return None to fall back.

    Returns None — signalling robot_mesh() to use the built-in mesh — when
    Device Connect is unavailable, has discovered no devices, or the action is
    one DC does not handle (subscribe / watch / inbox / unsubscribe).
    """
    if action in ("subscribe", "watch", "inbox", "unsubscribe"):
        return None  # mesh-only actions — let the built-in mesh handle them
    if os.environ.get("STRANDS_ROBOT_MESH_DC", "on").strip().lower() in ("off", "0", "false", "no"):
        return None  # Device Connect dispatch disabled (e.g. hermetic unit tests)
    try:
        _dc_ensure_connected()
        from device_connect_agent_tools.connection import get_connection

        conn = get_connection()
        devices = conn.list_devices()
    except Exception as exc:  # noqa: BLE001 — DC is optional; fall back to mesh
        logger.debug("Device Connect unavailable, using mesh fallback: %s", exc)
        return None
    # A well-formed connection returns a list of device dicts. Anything else
    # (e.g. a malformed/stubbed connection) means DC is not usable here, so fall
    # back to the built-in mesh rather than misdispatch.
    if not isinstance(devices, (list, tuple)) or not devices:
        return None
    return _device_connect_dispatch(
        action,
        target,
        instruction,
        command,
        policy_provider,
        policy_port,
        duration,
        timeout,
        function,
        validated_command,
    )


def _device_connect_dispatch(
    action: str,
    target: str,
    instruction: str,
    command: str,
    policy_provider: str,
    policy_port: int,
    duration: float,
    timeout: float,
    function: str = "",
    validated_command: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Render a robot_mesh action through Device Connect (dev-compatible API).

    Fetches the agent-side connection via ``get_connection()`` (patchable in
    unit tests) and returns a Strands tool-response dict (``_DCResult``).
    """
    try:
        from device_connect_agent_tools.connection import get_connection

        conn = get_connection()
        if action in ("peers", "status"):
            devices = conn.list_devices()
            text = (
                f"Discovered {len(devices)} device(s):\n"
                if action == "peers"
                else f"Network: {len(devices)} device(s)\n"
            )
            for d in devices:
                dtype = d.get("device_type", "?")
                icon = {"strands_robot": "robot", "strands_sim": "sim", "reachy_mini": "reachy"}.get(dtype, dtype)
                status = d.get("status", {})
                avail = status.get("availability", "?") if isinstance(status, dict) else "?"
                text += f"  [{icon}] {d['device_id']} — {avail}\n"
                if action == "peers":
                    funcs = d.get("functions", [])
                    if funcs:
                        names = [f["name"] if isinstance(f, dict) else f for f in funcs]
                        text += f"    Functions: {', '.join(names)}\n"
            return _DCResult(_ok(text))

        if action == "tell":
            if not target or not instruction:
                return _DCResult(_err("tell requires both target and instruction"))
            kwargs: dict[str, Any] = {"policy_provider": policy_provider, "duration": duration}
            if policy_port:
                kwargs["policy_port"] = policy_port
            # Inherit the mesh path's per-action command validation.
            try:
                _security.validate_command({"action": "execute", "instruction": instruction, **kwargs})
            except _security.ValidationError as exc:
                _audit_tool_action(action, target, False, f"validation: {exc}")
                return _DCResult(_err(f"tell rejected: {exc}"))
            result = conn.invoke(target, "execute", {"instruction": instruction, **kwargs}, timeout=timeout)
            r = result.get("result", result)
            _audit_tool_action(action, target, True, f"instruction={instruction[:200]}")
            return _DCResult(_ok(f"-> {target}: {instruction}\n  {json.dumps(r, default=str)}"))

        if action == "send":
            if not target:
                return _DCResult(_err("send requires target"))
            if not command:
                return _DCResult(_err("send requires command (JSON string)"))
            try:
                cmd = json.loads(command)
            except json.JSONDecodeError as exc:
                return _DCResult(_err(f"command is not valid JSON: {exc}"))
            if not isinstance(cmd, dict):
                return _DCResult(_err("command must decode to a JSON object (dict)"))
            try:
                cmd = _security.validate_command(cmd)
            except _security.ValidationError as exc:
                _audit_tool_action(action, target, False, f"validation: {exc}")
                return _DCResult(_err(f"send rejected: {exc}"))
            func = cmd.pop("action", cmd.pop("function", "getStatus"))
            result = conn.invoke(target, func, cmd, timeout=timeout)
            r = result.get("result", result)
            _audit_tool_action(action, target, True, f"action={func}")
            return _DCResult(_ok(f"{target}:\n{json.dumps(r, indent=2, default=str)[:2000]}"))

        if action == "rpc":
            # Device-native RPC (e.g. Reachy nod/look/playMove). Validated via
            # security.validate_device_rpc (charset + bounded params) WITHOUT
            # the policy-action allowlist, then invoked directly on the device.
            if not target:
                return _DCResult(_err("rpc requires target"))
            if not function:
                return _DCResult(_err("rpc requires function (the device-native function name)"))
            rpc_params: dict[str, Any] = {}
            if command:
                try:
                    parsed = json.loads(command)
                except json.JSONDecodeError as exc:
                    return _DCResult(_err(f"rpc params (command) is not valid JSON: {exc}"))
                if not isinstance(parsed, dict):
                    return _DCResult(_err("rpc params (command) must decode to a JSON object (dict)"))
                rpc_params = parsed
            try:
                func_name, rpc_params = _security.validate_device_rpc(function, rpc_params)
            except _security.ValidationError as exc:
                _audit_tool_action(action, target, False, f"validation: {exc}")
                return _DCResult(_err(f"rpc rejected: {exc}"))
            result = conn.invoke(target, func_name, rpc_params, timeout=timeout)
            r = result.get("result", result) if isinstance(result, dict) else result
            _audit_tool_action(action, target, True, f"function={func_name}")
            return _DCResult(
                _ok(f"{target}.{func_name}({rpc_params}) ->\n{json.dumps(r, indent=2, default=str)[:2000]}")
            )

        if action == "stop":
            if not target:
                return _DCResult(_err("stop requires target"))
            result = conn.invoke(target, "stop", timeout=min(timeout, 5.0))
            r = result.get("result", result)
            _audit_tool_action(action, target, True, "")
            return _DCResult(_ok(f"Stop {target}: {json.dumps(r, default=str)}"))

        if action == "emergency_stop":
            devices = conn.list_devices()
            stopped = 0
            for d in devices:
                try:
                    conn.invoke(d["device_id"], "stop", timeout=3.0)
                    stopped += 1
                except Exception:  # noqa: BLE001 — best-effort fan-out
                    pass
            _audit_tool_action(action, "*", True, f"stopped={stopped}/{len(devices)}")
            return _DCResult(_ok(f"E-STOP: {stopped}/{len(devices)} devices stopped"))

        if action == "broadcast":
            # Security hardening: dispatch the *validated* command that the
            # operator approved at the HITL gate — never re-parse the raw
            # caller-supplied string here (that would allow a payload whose
            # validated form differs from what actually executes).
            if validated_command is None:
                return _DCResult(_err("broadcast reached Device Connect dispatch without a validated command"))
            cmd = dict(validated_command)
            func = cmd.pop("action", cmd.pop("function", "getStatus"))
            params = cmd
            results = conn.broadcast(func, params, timeout=timeout)
            _audit_tool_action(action, "*", True, f"action={func} responses={len(results)}")
            text = f"[broadcast] {len(results)} responses\n"
            for r in results[:10]:
                sstr = "ok" if "result" in r else f"error: {r.get('error', '?')}"
                text += f"  {r.get('device_id', '?')}: {sstr}\n"
            return _DCResult(_ok(text.rstrip()))

        # subscribe / watch / inbox / unsubscribe → handled by the mesh path
        return None
    except Exception as exc:  # noqa: BLE001 — never raise out of the dispatcher
        logger.debug("Device Connect dispatch error for %s: %s", action, exc)
        return _DCResult(_err(f"[{action}] Device Connect error: {exc}"))


@tool(context=True)
def robot_mesh(
    action: str,
    tool_context: ToolContext | None = None,
    target: str = "",
    instruction: str = "",
    command: str = "",
    policy_provider: str = "mock",
    policy_port: int = 0,
    duration: float = 30.0,
    timeout: float = 30.0,
    name: str = "",
    limit: int = 50,
    function: str = "",
) -> dict[str, Any]:
    """Coordinate every robot, sim, and agent on the local Zenoh mesh.

    Args:
        action: One of ``peers`` / ``status`` / ``tell`` / ``send`` /
            ``rpc`` / ``broadcast`` / ``stop`` / ``emergency_stop`` /
            ``subscribe`` / ``unsubscribe`` / ``watch`` / ``inbox``.
            ``rpc`` calls a device's NATIVE Device Connect function (e.g.
            the Reachy's ``nod`` / ``look`` / ``playMove``) directly,
            bypassing the policy-action allowlist that ``tell`` / ``send``
            enforce. Pass the function name in ``function`` and any kwargs
            as a JSON object in ``command``.
        target: Peer id (for ``tell`` / ``send`` / ``stop`` / ``watch``) or
            Zenoh topic pattern (for ``subscribe``).
        instruction: Natural-language instruction for ``tell``.
        command: JSON-encoded command body for ``send`` / ``broadcast``.
        policy_provider: Policy provider tag forwarded with ``tell``.
        policy_port: Optional policy port forwarded with ``tell``.
        duration: Task duration (seconds) forwarded with ``tell``.
        timeout: Response timeout for RPC actions (seconds).
        name: Optional subscription name for ``subscribe`` / ``inbox``.
        limit: Max messages returned by ``inbox`` (default: 50).
        function: Device-native function name for ``rpc`` (e.g. ``nod``).

    Returns:
        A Strands tool response dict with status and a single text block.

    Examples::

        robot_mesh(action="peers")
        robot_mesh(action="tell", target="so100_sim-a1b2",
                   instruction="pick up the cube")
        robot_mesh(action="send", target="peer-b",
                   command='{"action": "status"}')
        robot_mesh(action="emergency_stop")    # raises a HITL interrupt;
                                               # runs only on operator approval

    Safety controls:
        * **Human-in-the-loop interrupts** for ``emergency_stop`` and
          ``broadcast``. The tool calls
          ``tool_context.interrupt("robot_mesh-<action>-approval", reason=...)``
          and only proceeds if the operator's response is an affirmative
          ("y" / "yes" / "approve"). The Strands SDK delivers the response
          out-of-band of the LLM's tool arguments, so prompt-injection that
          flips a boolean cannot bypass this gate.
        * Per-action sliding-window rate limit (e.g. emergency_stop is capped
          at 3 calls/min). Reject reason includes wait-time estimate.
        * ``send`` / ``broadcast`` payloads are validated through
          :func:`strands_robots.mesh.security.validate_command` before
          leaving the agent. The same validator runs on the receiver side,
          so a malformed or out-of-policy payload is rejected client-side
          before it hits the wire.
        * Every ``tell`` / ``send`` / ``broadcast`` / ``stop`` /
          ``emergency_stop`` is audited.
    """
    # Check the per-action rate limit before doing any work — but
    # do NOT consume a slot until we know the action is going to run.
    # See _rate_limit_check / _rate_limit_record for rationale.
    rl_err = _rate_limit_check(action)
    if rl_err is not None:
        _audit_tool_action(action, target, False, f"rate_limit: {rl_err}")
        return _err(rl_err)

    # R8-7: for broadcast, parse + validate the command BEFORE the HITL
    # interrupt so the operator does not approve an action that the
    # validator then rejects (which would burn an audit "operator
    # approved" record and a rate-limit slot for an action that never
    # ran). emergency_stop has no command body so the existing order
    # is fine for that path.
    validated_broadcast_cmd: dict[str, Any] | None = None
    if action == "broadcast":
        if not command:
            _audit_tool_action(action, "*", False, "missing command")
            return _err("broadcast requires command (JSON string)")
        try:
            parsed = json.loads(command)
        except json.JSONDecodeError as exc:
            _audit_tool_action(action, "*", False, f"bad json: {exc}")
            return _err(f"command is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            _audit_tool_action(action, "*", False, "command not a dict")
            return _err("command must decode to a JSON object (dict)")
        try:
            validated_broadcast_cmd = _security.validate_command(parsed)
        except _security.ValidationError as exc:
            _audit_tool_action(action, "*", False, f"validation: {exc}")
            return _err(f"broadcast rejected: {exc}")

    # Human-in-the-loop approval gate for fleet-wide actions. The Strands
    # runtime pauses the agent loop on tool_context.interrupt(...) and
    # returns control to the host process; the operator's response (e.g.
    # "y" / "n") is delivered back outside the LLM's tool-argument flow,
    # so an injected prompt cannot smuggle approval.
    if action in _INTERRUPT_REQUIRED:
        if tool_context is None:
            _audit_tool_action(action, target, False, "interrupt unavailable: no tool_context")
            return _err(
                f"action '{action}' requires a human-in-the-loop interrupt, "
                "but no tool_context is available in this calling context."
            )
        try:
            response = tool_context.interrupt(
                f"robot_mesh-{action}-approval",
                reason={
                    "action": action,
                    "target": target if target else "*ALL_PEERS*",
                    # Surface the device-native function name for rpc so the
                    # operator approves the specific function being invoked.
                    "function": function if action == "rpc" else "",
                    # R8-7: surface the validated command so the operator
                    # approves the post-validation form, not the raw LLM
                    # string. emergency_stop has no command body so we
                    # fall back to the raw value.
                    "command": (validated_broadcast_cmd if validated_broadcast_cmd is not None else command),
                    "instruction": instruction,
                    "warning": ("Fleet-wide physical effect. Reply 'y' to approve, anything else to deny."),
                },
            )
        except RuntimeError as exc:
            # ToolContext.interrupt raises RuntimeError when no agent
            # instance is attached — i.e. the tool is being invoked
            # outside a Strands agent loop (a direct
            # ``agent.tool.robot_mesh(...)`` call, a unit test that did
            # not wire up the SDK, etc.). In those contexts there is no
            # operator to ask, so fail-closed.
            #
            # NB: the SDK's ``InterruptException`` MUST propagate up to
            # pause the agent loop, so we deliberately do NOT catch
            # ``Exception`` here — that would swallow the normal
            # interrupt-pause flow and turn every approval into an
            # immediate "interrupt unavailable" error.
            _audit_tool_action(action, target, False, f"interrupt unavailable: {exc}")
            return _err(
                f"action '{action}' requires a human-in-the-loop interrupt. Interrupts are not available here: {exc}"
            )

        if not _interrupt_approves(response):
            # Declined approval does NOT consume a rate-limit slot —
            # see _rate_limit_check docstring for the safety rationale.
            _audit_tool_action(action, target, False, f"operator declined: {response!r}")
            return _err(f"action '{action}' was declined by the operator interrupt (response={response!r}).")
        # Approval granted. Re-check under the lock and consume the
        # slot atomically -- a concurrent invocation that ALSO passed
        # the pre-interrupt check (different operator thread, etc.)
        # will be rejected here so the configured limit cannot be
        # exceeded by HITL races.
        rl_race_err = _rate_limit_check_and_record(action)
        if rl_race_err is not None:
            _audit_tool_action(action, target, False, f"rate_limit_race: {rl_race_err}")
            return _err(rl_race_err)
        _audit_tool_action(action, target, True, f"operator approved: {response!r}")
    else:
        # No interrupt required for this action — consume the slot
        # unconditionally (matches the pre-split behaviour for
        # non-fleet-wide actions like ``tell``, ``send``, ``stop``).
        _rate_limit_record(action)

    # ── Device Connect dispatch (primary networking layer) ─────────────────
    # Every safety gate above (rate limit, HITL approval, broadcast
    # pre-validation, audit) has already run, so Device Connect inherits them.
    # _try_device_connect returns None when DC is unavailable or has discovered
    # no devices, in which case we fall through to the built-in mesh below.
    _dc_result = _try_device_connect(
        action,
        target,
        instruction,
        command,
        policy_provider,
        policy_port,
        duration,
        timeout,
        function,
        validated_broadcast_cmd,
    )
    if _dc_result is not None:
        return _dc_result

    try:
        from strands_robots.mesh import get_local_robots
        from strands_robots.mesh.session import get_peers
    except ImportError as exc:
        return _err(f"mesh module unavailable: {exc}")

    locals_ = get_local_robots()
    peers = get_peers()

    # ── action: peers ─────────────────────────────────────────────────────
    if action == "peers":
        lines = [f"[mesh] {len(locals_)} local, {len(peers)} remote"]
        if locals_:
            lines.append("")
            lines.append("Local (this process):")
            for pid, m in locals_.items():
                lines.append(f"  - {pid} ({m.peer_type})")
        if peers:
            lines.append("")
            lines.append("Discovered peers:")
            for p in peers:
                age = p.get("age", 0)
                ptype = p.get("type", "?")
                host = p.get("hostname", "?")
                lines.append(f"  - {p['peer_id']} ({ptype}) host={host} age={age}s")
                ts = p.get("task_status")
                if ts:
                    lines.append(f"      task: {ts} - {p.get('instruction', '')}")
        elif not locals_:
            lines.append("")
            lines.append("No peers. Create a Robot() or Simulation() to auto-join the mesh.")
        return _ok("\n".join(lines))

    # ── action: status ────────────────────────────────────────────────────
    if action == "status":
        return _ok(f"[mesh] local={len(locals_)} remote={len(peers)} peers={[p['peer_id'] for p in peers]}")

    # All remaining actions need an outbound mesh.
    mesh = _resolve_mesh(target)
    if mesh is None:
        return _err("no local mesh found. Construct a Robot()/Simulation() first to join the mesh, then retry.")

    # ── action: tell ──────────────────────────────────────────────────────
    if action == "tell":
        if not target or not instruction:
            _audit_tool_action(action, target, False, "missing target/instruction")
            return _err("tell requires both target and instruction")
        kwargs: dict[str, Any] = {
            "policy_provider": policy_provider,
            "duration": duration,
        }
        if policy_port:
            kwargs["policy_port"] = policy_port
        # Validate the synthesised command through mesh.security.
        synthesised = {"action": "execute", "instruction": instruction, **kwargs}
        try:
            _security.validate_command(synthesised)
        except _security.ValidationError as exc:
            _audit_tool_action(action, target, False, f"validation: {exc}")
            return _err(f"tell rejected: {exc}")
        try:
            result = mesh.tell(target, instruction, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # Audit dispatch failures (mesh.tell may raise on transport
            # error, lockout, etc.). Previously only ``success=True`` was
            # emitted, leaving a forensic gap on failure paths.
            _audit_tool_action(action, target, False, f"dispatch error: {type(exc).__name__}: {exc}")
            return _err(f"[tell -> {target}] dispatch error: {type(exc).__name__}: {exc}")
        _audit_tool_action(action, target, True, f"instruction={instruction[:200]}")
        return _ok(f"[tell -> {target}] {json.dumps(result, default=str)[:600]}")

    # ── action: send ──────────────────────────────────────────────────────
    if action == "send":
        if not target:
            _audit_tool_action(action, target, False, "missing target")
            return _err("send requires target")
        if not command:
            _audit_tool_action(action, target, False, "missing command")
            return _err("send requires command (JSON string)")
        try:
            cmd = json.loads(command)
        except json.JSONDecodeError as exc:
            _audit_tool_action(action, target, False, f"bad json: {exc}")
            return _err(f"command is not valid JSON: {exc}")
        if not isinstance(cmd, dict):
            _audit_tool_action(action, target, False, "command not a dict")
            return _err("command must decode to a JSON object (dict)")
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            _audit_tool_action(action, target, False, f"validation: {exc}")
            return _err(f"send rejected: {exc}")
        try:
            result = mesh.send(target, cmd, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            _audit_tool_action(action, target, False, f"dispatch error: {type(exc).__name__}: {exc}")
            return _err(f"[send -> {target}] dispatch error: {type(exc).__name__}: {exc}")
        _audit_tool_action(action, target, True, f"action={cmd.get('action')}")
        return _ok(f"[send -> {target}] {json.dumps(result, default=str)[:600]}")

    # ── action: broadcast ─────────────────────────────────────────────────
    if action == "broadcast":
        # R8-7: pre-validated above before the HITL interrupt fired, so
        # the cmd here is already a clean validated dict.
        # Use explicit raise (not assert) -- assert is stripped under
        # ``python -O`` / ``PYTHONOPTIMIZE=1`` which would silently send
        # an unvalidated cmd to mesh.broadcast.
        if validated_broadcast_cmd is None:
            raise RuntimeError("broadcast reached its handler without pre-validation -- R8-7 contract broken")
        cmd = validated_broadcast_cmd
        try:
            results = mesh.broadcast(cmd, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            _audit_tool_action(action, "*", False, f"dispatch error: {type(exc).__name__}: {exc}")
            return _err(f"[broadcast] dispatch error: {type(exc).__name__}: {exc}")
        _audit_tool_action(action, "*", True, f"action={cmd.get('action')} responses={len(results)}")
        text = f"[broadcast] {len(results)} responses\n"
        for r in results[:10]:
            text += f"  - {json.dumps(r, default=str)[:200]}\n"
        if len(results) > 10:
            text += f"  ... and {len(results) - 10} more"
        return _ok(text.rstrip())

    # ── action: stop ──────────────────────────────────────────────────────
    if action == "stop":
        if not target:
            _audit_tool_action(action, target, False, "missing target")
            return _err("stop requires target")
        try:
            result = mesh.send(target, {"action": "stop"}, timeout=min(timeout, 5.0))
        except Exception as exc:  # noqa: BLE001
            _audit_tool_action(action, target, False, f"dispatch error: {type(exc).__name__}: {exc}")
            return _err(f"[stop -> {target}] dispatch error: {type(exc).__name__}: {exc}")
        _audit_tool_action(action, target, True, "")
        return _ok(f"[stop -> {target}] {json.dumps(result, default=str)[:600]}")

    # ── action: emergency_stop ────────────────────────────────────────────
    if action == "emergency_stop":
        # Operator approval was already obtained above through the
        # interrupt gate; this branch only runs on an affirmative response.
        try:
            results = mesh.emergency_stop()
        except Exception as exc:  # noqa: BLE001
            _audit_tool_action(action, "*", False, f"dispatch error: {type(exc).__name__}: {exc}")
            return _err(f"[emergency_stop] dispatch error: {type(exc).__name__}: {exc}")
        _audit_tool_action(action, "*", True, f"responses={len(results)}")
        return _ok(f"[E-STOP] broadcast complete - {len(results)} responses (audit log written)")

    # ── action: subscribe ─────────────────────────────────────────────────
    if action == "subscribe":
        if not target:
            _audit_tool_action(action, target, False, "missing target")
            return _err("subscribe requires target (Zenoh topic pattern)")
        sub_name = name or target
        out = mesh.subscribe(target, name=sub_name)
        if out is None:
            _audit_tool_action(action, target, False, "subscribe returned None")
            return _err("subscribe failed (mesh not running?)")
        _audit_tool_action(action, target, True, f"name={sub_name}")
        return _ok(
            f"[sub] subscribed to '{target}' as '{sub_name}'. "
            f"Use action='inbox' name='{sub_name}' to read buffered messages."
        )

    # ── action: watch ─────────────────────────────────────────────────────
    if action == "watch":
        if not target:
            _audit_tool_action(action, target, False, "missing target")
            return _err("watch requires target (peer id)")
        out = mesh.on_stream(target)
        if out is None:
            _audit_tool_action(action, target, False, "watch returned None")
            return _err("watch failed (mesh not running?)")
        _audit_tool_action(action, target, True, f"stream_name={out}")
        return _ok(f"[watch] watching peer '{target}'. Use action='inbox' name='{out}' to read buffered steps.")

    # ── action: inbox ─────────────────────────────────────────────────────
    if action == "inbox":
        sub_name = name or target
        if not sub_name:
            return _err("inbox requires name (or target)")
        msgs = mesh.inbox.get(sub_name, [])
        if not msgs:
            return _ok(f"[inbox '{sub_name}'] no messages")
        head = msgs[-limit:] if limit > 0 else msgs
        text = f"[inbox '{sub_name}'] {len(msgs)} total, showing last {len(head)}\n"
        for topic, data in head:
            text += f"  - {topic}: {json.dumps(data, default=str)[:200]}\n"
        return _ok(text.rstrip())

    # ── action: unsubscribe ────────────────────────────────────────────────
    if action == "unsubscribe":
        sub_name = name or target
        if not sub_name:
            return _err("unsubscribe requires name (or target)")
        mesh.unsubscribe(sub_name)
        return _ok(f"[unsub] unsubscribed from '{sub_name}'")

    if action == "rpc":
        return _err(
            "rpc (device-native function call) requires Device Connect, which is "
            "unavailable or has discovered no devices in this context. The built-in "
            "Zenoh mesh has no equivalent. Ensure the agent connected via "
            "device_connect_agent_tools.connect() and the target is online."
        )

    return _err(
        f"unknown action: {action!r}. Valid: peers, status, tell, send, rpc, "
        "broadcast, stop, emergency_stop, subscribe, unsubscribe, watch, inbox."
    )


__all__ = ["robot_mesh"]
