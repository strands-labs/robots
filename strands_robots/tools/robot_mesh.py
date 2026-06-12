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
import functools
import json
import logging
import os
import re
import threading
import time
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_robots.mesh import security as _security

# Literal peer-id pattern for watch(target=...). Peer ids are an enumerable
# surface (per AGENTS.md > Review Learnings (PR #92) > "Allowlist enumerable
# values"); rejecting Zenoh wildcards (`*`, `**`) and path separators here
# prevents the agent from defeating per-peer scoping by interpolating a
# wildcard segment into ``strands/<target>/stream`` even when an operator has
# extended ``STRANDS_MESH_SUBSCRIBE_ALLOW`` with a wildcard pattern such as
# ``strands/*/stream``. Allowed: alphanumerics, ``.``, ``_``, ``-``; first
# char must be alphanumeric; max 64 chars (Zenoh peer-id practical limit).
_PEER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

logger = logging.getLogger(__name__)


# Per-action sliding-window rate limiter for LLM-facing actions.
_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "tell": (30, 60.0),
    "send": (30, 60.0),
    "broadcast": (10, 60.0),
    "stop": (20, 60.0),
    "emergency_stop": (3, 60.0),
}
_RATE_HISTORY: dict[str, collections.deque[float]] = {}
_RATE_LOCK = threading.Lock()

# Actions that CAN be placed behind a human-in-the-loop interrupt. Each
# routes through a Strands SDK interrupt so the calling host can request
# explicit human approval before the mesh issues the command. Unlike a
# boolean tool parameter the interrupt response is delivered by the
# framework out-of-band of the LLM's tool-argument flow, so an injected
# prompt cannot smuggle approval.
#
# ``subscribe`` / ``watch`` are gateable too (they expose mesh telemetry
# to the agent context) but are NOT in the default set -- they have no
# physical actuation effect, and gating them by default would interrupt
# every read-only observation. Operators who treat telemetry as sensitive
# opt them in via ``STRANDS_MESH_HITL_ACTIONS``.
_GATEABLE_ACTIONS: frozenset[str] = frozenset(
    {"emergency_stop", "broadcast", "tell", "send", "stop", "subscribe", "watch"}
)

# Default interrupt set: every action with a direct physical-actuation
# effect. ``emergency_stop`` and ``broadcast`` are fleet-wide;
# ``tell`` / ``send`` / ``stop`` actuate a single targeted peer. Gating
# all five by default means a prompt-injected agent cannot drive ANY
# physical command without an out-of-band operator approval. Consumers
# who want a narrower or wider gate override via the env var below.
_DEFAULT_INTERRUPT_ACTIONS: frozenset[str] = frozenset({"emergency_stop", "broadcast", "tell", "send", "stop"})


# Sentinel raised by the resolver when the env var holds an unknown token.
# Surfaced as a structured tool error so a typo fails loud rather than
# silently degrading the gate (AGENTS.md: warn on unrecognized env values).
class _InterruptConfigError(ValueError):
    """STRANDS_MESH_HITL_ACTIONS contained an unrecognized action token."""


@functools.lru_cache(maxsize=4)
def _parse_interrupt_actions(raw: str) -> frozenset[str]:
    """Parse ``STRANDS_MESH_HITL_ACTIONS`` into the resolved interrupt set.

    Cached on the raw env string (a ``monkeypatch.setenv`` change yields a
    different cache key and re-parses; in-place mutation tests call
    :func:`_reset_interrupt_actions_cache`). Semantics:

    * empty / unset -> :data:`_DEFAULT_INTERRUPT_ACTIONS`
    * ``"all"``     -> every action in :data:`_GATEABLE_ACTIONS`
    * ``"none"``    -> empty set (no gate; explicit opt-out)
    * comma list    -> validated subset of :data:`_GATEABLE_ACTIONS`;
                       an unknown token raises :class:`_InterruptConfigError`

    The ``"none"`` opt-out re-opens the physical-actuation surface, so the
    caller logs a one-time warning when it is in effect.
    """
    cleaned = raw.strip().lower()
    if not cleaned:
        return _DEFAULT_INTERRUPT_ACTIONS
    if cleaned == "all":
        return _GATEABLE_ACTIONS
    if cleaned == "none":
        return frozenset()
    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
    unknown = [t for t in tokens if t not in _GATEABLE_ACTIONS]
    if unknown:
        raise _InterruptConfigError(
            f"STRANDS_MESH_HITL_ACTIONS contains unknown action(s): {unknown}. "
            f"Valid tokens: 'all', 'none', or a comma-separated subset of "
            f"{sorted(_GATEABLE_ACTIONS)}."
        )
    return frozenset(tokens)


def _reset_interrupt_actions_cache() -> None:
    """Test helper: clear the cached env parse and the once-warned flag."""
    _parse_interrupt_actions.cache_clear()
    _warn_none_opt_out_once.cache_clear()


def _resolve_interrupt_actions() -> frozenset[str]:
    """Return the configured interrupt set, honoring the env override.

    Raises :class:`_InterruptConfigError` on a malformed env var so the
    dispatcher can return a structured error instead of silently running
    with the default (which would mask the operator's misconfiguration).
    """
    return _parse_interrupt_actions(os.getenv("STRANDS_MESH_HITL_ACTIONS", ""))


@functools.lru_cache(maxsize=1)
def _warn_none_opt_out_once() -> None:
    """Emit the HITL-disabled warning at most once per process.

    Implemented as an ``lru_cache`` nullary function: the first call runs
    the body (emitting the warning) and memoizes ``None``; subsequent calls
    return the cached result without re-logging. This avoids a module-level
    mutable flag entirely -- the prior ``global`` bool tripped CodeQL
    ``py/unused-global-variable`` (the rebind was not recognised as a use),
    and the list-hack before it had the same problem. ``cache_clear()`` (via
    :func:`_reset_interrupt_actions_cache`) restores the warn-once state for
    test isolation. Warn-once under concurrency is best-effort, which is fine
    for a one-shot operator notice.
    """
    logger.warning(
        "[robot_mesh] STRANDS_MESH_HITL_ACTIONS=none -- human-in-the-loop "
        "approval is DISABLED for all mesh actions. Physical-actuation "
        "commands (tell/send/stop/broadcast/emergency_stop) will dispatch "
        "without operator confirmation."
    )


# --- subscribe topic allowlist (telemetry-leak defence in depth) --------
#
# ``subscribe`` declares a Zenoh subscriber on a caller-supplied key
# expression and buffers matching traffic into ``inbox`` for the agent to
# read. Without a topic allowlist a prompt-injected agent can subscribe to
# another peer's cmd / state / camera / input streams and exfiltrate them
# into the LLM context. The transport ACL (examples/mesh_acl_example.json5)
# is the primary control; this tool-layer allowlist is defence in depth so
# the leak is blocked even on a mesh running the permissive default ACL.
#
# The default set is intentionally narrow: only fleet-shared, low-impact
# topic CLASSES with no actuation or sensor-stream content. Operators extend
# it via STRANDS_MESH_SUBSCRIBE_ALLOW (comma-separated key-expr patterns).
_DEFAULT_SUBSCRIBE_ALLOW: tuple[str, ...] = (
    "**/presence",
    "**/health",
    "**/safety/**",
)


@functools.lru_cache(maxsize=4)
def _subscribe_allowlist_cached(raw: str) -> tuple[str, ...]:
    """Cached parse of STRANDS_MESH_SUBSCRIBE_ALLOW (defaults + extras)."""
    extras = tuple(t.strip() for t in raw.split(",") if t.strip())
    return _DEFAULT_SUBSCRIBE_ALLOW + extras


def _reset_subscribe_allowlist_cache() -> None:
    """Test helper: clear the cached env parse."""
    _subscribe_allowlist_cached.cache_clear()


def _subscribe_allowlist() -> tuple[str, ...]:
    return _subscribe_allowlist_cached(os.getenv("STRANDS_MESH_SUBSCRIBE_ALLOW", ""))


def _ke_matches(pattern: str, target: str) -> bool:
    """Conservative Zenoh-style key-expr match for the subscribe allowlist.

    We do NOT import a general glob engine: Zenoh's ``**`` (any number of
    segments, including zero) and ``*`` (one segment) semantics differ from
    fnmatch, and a mismatch here would either over- or under-block. We
    implement only the shapes the allowlist actually uses:

    * exact equality (``"**/presence" == "**/presence"``),
    * a leading ``**/`` wildcard on the pattern, which matches zero or more
      leading segments (``"**/presence"`` matches ``"presence"``,
      ``"robot1/presence"``, and ``"a/b/c/presence"``),
    * a trailing ``/**`` wildcard on the pattern, which matches the prefix
      plus any deeper segments (``"a/safety/**"`` matches ``"a/safety"``,
      ``"a/safety/event"``, ``"a/safety/estop"``),
    * BOTH leading and trailing ``**`` (``"**/safety/**"`` matches
      ``"a/safety"``, ``"a/b/safety/event"``, ``"safety/estop"``),
    * segment-level ``*`` wildcard (one segment): ``"strands/*/stream"``
      matches ``"strands/peer-b/stream"`` but not ``"strands/a/b/stream"``.

    Anything more exotic must be enumerated literally in the allowlist. A
    target the matcher cannot positively confirm is treated as NOT allowed
    (fail-closed).
    """
    if not isinstance(target, str) or not target:
        return False
    if pattern == target:
        return True

    leading_dstar = pattern.startswith("**/")
    trailing_dstar = pattern.endswith("/**")

    # Both ends double-star: pattern is "**/MIDDLE/**" or just "**/X/**"
    if leading_dstar and trailing_dstar:
        middle = pattern[3:-3]  # strip "**/" and "/**"
        if not middle:
            # pattern was "**/**" — match anything non-empty
            return bool(target)
        # match if target == middle, target ends with /middle, target starts
        # with middle/, or target contains /middle/
        return (
            target == middle
            or target.startswith(middle + "/")
            or target.endswith("/" + middle)
            or ("/" + middle + "/") in target
        )

    if leading_dstar:
        suffix = pattern[3:]  # strip leading "**/"
        # match suffix exactly (zero leading segments) OR any path ending
        # in /suffix (one or more leading segments). Suffix may itself
        # contain segment-level wildcards — keep it simple: literal compare.
        return target == suffix or target.endswith("/" + suffix)

    if trailing_dstar:
        prefix = pattern[:-3]  # strip trailing "/**"
        # Allow the prefix itself or anything one-or-more segments deeper.
        return target == prefix or target.startswith(prefix + "/")

    # Segment-level single-star: split both by "/" and match segment-wise.
    # A ``*`` segment matches exactly one non-empty segment in the target.
    if "*" in pattern:
        p_parts = pattern.split("/")
        t_parts = target.split("/")
        if len(p_parts) != len(t_parts):
            return False
        return all(pp == tp or (pp == "*" and tp != "") for pp, tp in zip(p_parts, t_parts))
    return False


def _is_allowed_subscribe_target(target: str) -> bool:
    """True iff *target* matches any entry in the subscribe allowlist."""
    return any(_ke_matches(p, target) for p in _subscribe_allowlist())


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
) -> dict[str, Any]:
    """Coordinate every robot, sim, and agent on the local Zenoh mesh.

    Args:
        action: One of ``peers`` / ``status`` / ``tell`` / ``send`` /
            ``broadcast`` / ``stop`` / ``emergency_stop`` / ``subscribe`` /
            ``unsubscribe`` / ``watch`` / ``inbox``.
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
    # Resolve which actions require a human-in-the-loop interrupt for THIS
    # call. Consumers configure the set via STRANDS_MESH_HITL_ACTIONS; the
    # default gates every physical-actuation action (emergency_stop,
    # broadcast, tell, send, stop). A malformed env var fails loud here
    # rather than silently degrading the gate.
    try:
        interrupt_actions = _resolve_interrupt_actions()
    except _InterruptConfigError as exc:
        _audit_tool_action(action, target, False, f"hitl config error: {exc}")
        return _err(str(exc))

    # One-time warning when the operator has explicitly disabled the gate.
    if not interrupt_actions and os.getenv("STRANDS_MESH_HITL_ACTIONS", "").strip().lower() == "none":
        _warn_none_opt_out_once()

    # Check the per-action rate limit before doing any work — but
    # do NOT consume a slot until we know the action is going to run.
    # See _rate_limit_check / _rate_limit_record for rationale.
    rl_err = _rate_limit_check(action)
    if rl_err is not None:
        _audit_tool_action(action, target, False, f"rate_limit: {rl_err}")
        return _err(rl_err)

    # R8-7: parse + validate any command body BEFORE the HITL interrupt so
    # the operator never approves an action the validator then rejects
    # (which would burn an audit "operator approved" record and a
    # rate-limit slot for an action that never ran). This applies to
    # ``broadcast`` and ``send`` (JSON command bodies) and ``tell`` (a
    # synthesised execute command). ``stop`` / ``emergency_stop`` have no
    # validated body so they skip this pre-pass.
    validated_broadcast_cmd: dict[str, Any] | None = None
    validated_send_cmd: dict[str, Any] | None = None
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
    elif action == "send":
        if not target:
            _audit_tool_action(action, target, False, "missing target")
            return _err("send requires target")
        if not command:
            _audit_tool_action(action, target, False, "missing command")
            return _err("send requires command (JSON string)")
        try:
            parsed = json.loads(command)
        except json.JSONDecodeError as exc:
            _audit_tool_action(action, target, False, f"bad json: {exc}")
            return _err(f"command is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            _audit_tool_action(action, target, False, "command not a dict")
            return _err("command must decode to a JSON object (dict)")
        try:
            validated_send_cmd = _security.validate_command(parsed)
        except _security.ValidationError as exc:
            _audit_tool_action(action, target, False, f"validation: {exc}")
            return _err(f"send rejected: {exc}")
    elif action == "tell":
        if not target or not instruction:
            _audit_tool_action(action, target, False, "missing target/instruction")
            return _err("tell requires both target and instruction")
        _tell_kwargs: dict[str, Any] = {"policy_provider": policy_provider, "duration": duration}
        if policy_port:
            _tell_kwargs["policy_port"] = policy_port
        try:
            _security.validate_command({"action": "execute", "instruction": instruction, **_tell_kwargs})
        except _security.ValidationError as exc:
            _audit_tool_action(action, target, False, f"validation: {exc}")
            return _err(f"tell rejected: {exc}")

    # Human-in-the-loop approval gate. The Strands runtime pauses the agent
    # loop on tool_context.interrupt(...) and returns control to the host
    # process; the operator's response (e.g. "y" / "n") is delivered back
    # outside the LLM's tool-argument flow, so an injected prompt cannot
    # smuggle approval. Which actions are gated is operator-configurable
    # (see _resolve_interrupt_actions).
    if action in interrupt_actions:
        if tool_context is None:
            _audit_tool_action(action, target, False, "interrupt unavailable: no tool_context")
            return _err(
                f"action '{action}' requires a human-in-the-loop interrupt, "
                "but no tool_context is available in this calling context."
            )
        # Fleet-wide actions reach every peer; single-target actions hit
        # one peer. Surface the right scope so the operator's confirmation
        # reflects the real blast radius.
        _fleet_wide = action in ("emergency_stop", "broadcast")
        _approval_target = "*ALL_PEERS*" if _fleet_wide else (target or "*ALL_PEERS*")
        _scope_warning = (
            "Fleet-wide physical effect. Reply 'y' to approve, anything else to deny."
            if _fleet_wide
            else f"Physical effect on peer '{target}'. Reply 'y' to approve, anything else to deny."
        )
        # R8-7: surface the validated command (post-validation form) so the
        # operator approves what will actually dispatch, not the raw LLM
        # string. tell/stop/emergency_stop have no JSON command body.
        _approval_command = (
            validated_broadcast_cmd
            if validated_broadcast_cmd is not None
            else (validated_send_cmd if validated_send_cmd is not None else command)
        )
        try:
            response = tool_context.interrupt(
                f"robot_mesh-{action}-approval",
                reason={
                    "action": action,
                    "target": _approval_target,
                    "command": _approval_command,
                    "instruction": instruction,
                    "warning": _scope_warning,
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
        # target/instruction presence and synthesised-command validation
        # already ran in the pre-interrupt pass above; rebuild the dispatch
        # kwargs here (the pre-pass validated an equivalent payload).
        kwargs: dict[str, Any] = {
            "policy_provider": policy_provider,
            "duration": duration,
        }
        if policy_port:
            kwargs["policy_port"] = policy_port
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
        # Parse + validation already happened in the pre-interrupt pass
        # above (so the operator approves the validated form). Reuse that
        # result rather than re-parsing the LLM string a second time.
        if validated_send_cmd is None:
            raise RuntimeError("send reached its handler without pre-validation -- R8-7 contract broken")
        cmd = validated_send_cmd
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
        # Telemetry-leak defence in depth: only allow subscribing to the
        # low-impact, fleet-shared topic classes in the allowlist. A target
        # outside the allowlist would let the agent observe another peer's
        # cmd / state / camera / input streams. If subscribe was placed in
        # the HITL set (STRANDS_MESH_HITL_ACTIONS=all) the operator already
        # approved this specific target above, so we honour that; otherwise
        # we reject and steer the agent to RPC (tell/send) for peer status.
        if not _is_allowed_subscribe_target(target) and action not in interrupt_actions:
            _audit_tool_action(action, target, False, "target not in subscribe allowlist")
            return _err(
                f"subscribe target '{target}' is not in the allowed topic set "
                f"{list(_subscribe_allowlist())}. Subscribing to another peer's "
                "control or sensor streams is not permitted; use action='tell' or "
                "action='send' to request status from a peer instead. Operators can "
                "extend the allowlist via STRANDS_MESH_SUBSCRIBE_ALLOW."
            )
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
        # Wildcard-bypass defence: ``target`` is interpolated into
        # ``strands/{target}/stream`` and matched against the subscribe
        # allowlist. If an operator extended the allowlist with a wildcard
        # pattern (e.g. ``strands/*/stream`` per the README example), then
        # ``target="*"`` / ``target="**"`` would pass the allowlist match by
        # equality / trailing-`/**` and reach ``mesh.on_stream("*")`` —
        # subscribing to every peer's stream (the cross-peer telemetry-leak
        # this surface exists to close). Require a literal peer id BEFORE
        # interpolating, mirroring the ``_REPO_TAG_RE`` shape-validation
        # pattern in ``gr00t_inference.py`` for the same class of attack.
        if not _PEER_ID_RE.match(target):
            _audit_tool_action(action, target, False, "watch target not a literal peer id")
            return _err(
                f"watch target '{target}' is not a valid peer id "
                f"(allowed: letters, digits, '._-'; max 64 chars; no wildcards "
                "or path separators). Watch requires a literal peer id; "
                "wildcards would subscribe to every peer's stream and defeat "
                "per-peer scoping."
            )
        # Telemetry-leak defence in depth: watch(target="peer-b") subscribes
        # to strands/<peer-b>/stream which carries observations + policy
        # actions -- the same cross-peer telemetry surface the subscribe
        # allowlist was added to close. Apply the same allowlist check on
        # the equivalent Zenoh key expression so watch cannot bypass the
        # subscribe gate. If watch is in the HITL set and the operator
        # already approved this call above, we honour that approval.
        watch_key = f"strands/{target}/stream"
        if not _is_allowed_subscribe_target(watch_key) and action not in interrupt_actions:
            _audit_tool_action(action, target, False, "watch target not in subscribe allowlist")
            return _err(
                f"watch target '{target}' (Zenoh key 'strands/{target}/stream') is not "
                f"in the allowed topic set {list(_subscribe_allowlist())}. Watching "
                "another peer's stream exposes its observations and policy actions; "
                "use action='tell' to request status from a peer instead. Operators "
                "can extend the allowlist via STRANDS_MESH_SUBSCRIBE_ALLOW."
            )
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
            # Audit even the empty read so the forensic trail records every
            # inbox access (agent read attempt), not just non-empty ones.
            _audit_tool_action(action, sub_name, True, "read=0")
            return _ok(f"[inbox '{sub_name}'] no messages")
        head = msgs[-limit:] if limit > 0 else msgs
        # Audit the read: which subscription, how many frames the agent
        # pulled into its context. Gives operators the "agent read N frames
        # from sub X at time T" trail that raw telemetry access otherwise
        # lacks.
        _audit_tool_action(action, sub_name, True, f"read={len(head)} total={len(msgs)}")
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

    return _err(
        f"unknown action: {action!r}. Valid: peers, status, tell, send, "
        "broadcast, stop, emergency_stop, subscribe, unsubscribe, watch, inbox."
    )


__all__ = ["robot_mesh"]
