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

import atexit
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
from strands_robots.mesh.core import mesh_disabled_by_env
from strands_robots.tools._hitl_audit import log_operator_response
from strands_robots.utils import finite_number_error, positive_count_error, positive_finite_number_error

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
    "rpc": (30, 60.0),
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
    {"emergency_stop", "broadcast", "tell", "send", "stop", "rpc", "subscribe", "watch"}
)

# Default interrupt set: every action with a direct physical-actuation
# effect. ``emergency_stop`` and ``broadcast`` are fleet-wide;
# ``tell`` / ``send`` / ``stop`` actuate a single targeted peer, and
# ``rpc`` invokes a device-native function on a targeted peer (Device
# Connect dispatch). Gating all six by default means a prompt-injected
# agent cannot drive ANY physical command without an out-of-band operator
# approval. Consumers who want a narrower or wider gate override via the
# env var below.
_DEFAULT_INTERRUPT_ACTIONS: frozenset[str] = frozenset({"emergency_stop", "broadcast", "tell", "send", "stop", "rpc"})

#: The gated actions that reach EVERY peer rather than one target. Named here so
#: the approval prompt's scope branch and its blast-radius wording read the same
#: set, and a third fleet-wide action cannot be added to one and not the other.
_FLEET_WIDE_ACTIONS: frozenset[str] = frozenset({"emergency_stop", "broadcast"})


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
        "commands (tell/send/stop/rpc/broadcast/emergency_stop) will "
        "dispatch "
        "without operator confirmation."
    )


# --- subscribe topic allowlist (telemetry-leak defence in depth) --------
#
# ``subscribe`` declares a Zenoh subscriber on a caller-supplied key
# expression and buffers matching traffic into ``inbox`` for the agent to
# read. Without a topic allowlist a prompt-injected agent can subscribe to
# another peer's cmd / state / camera / input streams and exfiltrate them
# into the LLM context. The transport ACL (examples/mesh/mesh_acl_example.json5)
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
            # pattern was "**/**" - match anything non-empty
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
        # contain segment-level wildcards - keep it simple: literal compare.
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


def _peer_snapshot(peer_id: str) -> dict[str, Any] | None:
    """The :func:`~strands_robots.mesh.session.get_peers` entry for *peer_id*.

    ``None`` when the peer is not on the snapshot, which the classifier reads as
    metal. Reads the in-process peer registry only - it never starts the gateway
    mesh, because this runs inside the approval gate and a gate must not acquire
    a transport to decide what to tell the operator.
    """
    from strands_robots.mesh.session import get_peers

    for entry in get_peers():
        if entry.get("peer_id") == peer_id:
            return entry
    return None


def _approval_warning(action: str, target: str) -> tuple[str, bool, str]:
    """The operator prompt's warning line, and the classification behind it.

    Returns ``(warning, physical, verified)``. The warning states what the gate
    established rather than asserting a physical effect it never checked: this
    tool gates by ACTION, so before this it told the operator "Physical effect on
    peer '<target>'" for every gated single-target call, including one aimed at a
    peer that reports itself as a sim.

    Fail-closed, in both scopes:

    * single target - :func:`~strands_robots.mesh.session.peer_is_physical`
      decides, so an absent or unclassifiable peer is still announced as
      physical;
    * fleet-wide (``emergency_stop`` / ``broadcast``) - physical unless EVERY
      peer on the snapshot is a classified sim, and physical when the snapshot is
      empty, because a peer this process has not discovered yet cannot be shown
      to be a sim.

    The verdict never changes WHICH actions are gated: an operator is asked for
    exactly the same set of actions as before, and is told what the peer says
    about itself.
    """
    from strands_robots.mesh.session import get_peers, peer_is_physical

    reply = "Reply 'y' to approve, anything else to deny."
    if action in _FLEET_WIDE_ACTIONS:
        peers = get_peers()
        physical_peers = [p for p in peers if peer_is_physical(p)[0]]
        if not peers:
            verified = "no peer is on the fleet snapshot, so none can be shown to be a sim"
            return f"Fleet-wide physical effect: {verified}. {reply}", True, verified
        if physical_peers:
            verified = f"{len(physical_peers)} of {len(peers)} peers on the snapshot are not known to be sims"
            return f"Fleet-wide physical effect: {verified}. {reply}", True, verified
        verified = f"all {len(peers)} peers on the snapshot report themselves as sims"
        return f"Fleet-wide effect, but {verified}. {reply}", False, verified

    physical, verified = peer_is_physical(_peer_snapshot(target))
    if physical:
        return f"Physical effect on peer '{target}': {verified}. {reply}", True, verified
    return f"Peer '{target}' is not known to be physical: {verified}. {reply}", False, verified


def _interrupt_approves(response: object) -> bool:
    """True iff *response* is an explicit affirmative.

    The interrupt mechanism returns whatever the operator submitted, which
    is normally a string but the contract is "JSON-serialisable any". We
    accept the canonical short forms only - defence in depth against
    accidental approval from a typo.
    """
    if not isinstance(response, str):
        return False
    return response.strip().lower() in _AFFIRMATIVE_RESPONSES


def _rate_limit_check(action: str) -> str | None:
    """Return None if a slot is available, else the rejection message.

    Inspects the sliding-window history but does NOT consume a slot.
    Use :func:`_rate_limit_check_and_record` once the action is known to
    run - after a HITL approval is positively granted, or directly for an
    action the operator has taken out of the gated set. Reserving through
    that helper is what keeps this check from being a TOCTOU.

    Splitting check from record means a *declined* HITL approval no
    longer consumes a slot - without the split, three nuisance LLM
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


def _reset_rate_limits() -> None:
    """Test helper: clear sliding-window history."""
    with _RATE_LOCK:
        _RATE_HISTORY.clear()


def _rate_limit_check_and_record(action: str) -> str | None:
    """Atomic check+record under a single _RATE_LOCK acquisition.

    This is the only way a slot is consumed. It closes the TOCTOU left by
    :func:`_rate_limit_check`, which reports whether a slot is free without
    taking it: without an atomic reservation two concurrent invocations
    could each pass that check and each record, briefly exceeding the
    configured limit. Both gate paths reserve through here - the approved
    path after the operator interrupt returns, and the ungated path for an
    action outside ``STRANDS_MESH_HITL_ACTIONS`` - so neither an operator
    wait nor the validation work in between can be raced past.

    Returns None if the slot was atomically reserved, else the rejection
    message (caller should treat as 'another call took the last slot after
    our check; reject this one').
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
                f"and record (a concurrent call raced past): max {max_calls} "
                f"calls per {window:.0f}s window. Try again in {wait:.1f}s."
            )
        bucket.append(now)
        return None


#: Event source recorded on this tool's audit rows, including the
#: operator-response rows :func:`log_operator_response` writes.
_AUDIT_SOURCE = "robot_mesh_tool"


def _audit_tool_action(action: str, target: str, success: bool, detail: str) -> None:
    """Best-effort audit log of every safety-significant tool call.

    A swallowed exception with no log line means a broken audit
    path silently disappears. Match the :meth:`~strands_robots.mesh.core.Mesh._on_cmd` pattern -
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
            _AUDIT_SOURCE,
            {
                "action": action,
                "target": target,
                "success": success,
                "detail": detail[:500],
            },
        )
    except Exception as audit_exc:  # noqa: BLE001 - see docstring
        logger.debug("[robot_mesh] audit log unavailable: %s", audit_exc)


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


# ── Numeric-option domain ──────────────────────────────────────────────────
#
# Which of the two numeric options each action actually consumes. Scoped per
# action rather than validated unconditionally because a caller must never be
# refused for a value the requested action never looks at: ``peers`` lists the
# graph without a wait budget, and ``emergency_stop`` fans out on a fixed
# internal budget rather than the caller's. An action absent from this table
# reads neither option and is never refused here.
#
# ``duration`` and ``policy_port`` are deliberately absent: they travel inside
# the command body that :func:`~strands_robots.mesh.security.validate_command`
# inspects, which already bounds them (``duration`` to ``[0,
# MAX_DURATION_S]``, ``policy_port`` to ``[1, 65535]``). ``timeout`` and
# ``limit`` never enter a command body, so nothing on that path can see them.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "tell": ("timeout",),
    "send": ("timeout",),
    "rpc": ("timeout",),
    "broadcast": ("timeout",),
    "stop": ("timeout",),
    "inbox": ("limit",),
}


def _numeric_option_error(action: str, *, timeout: Any, limit: Any) -> str | None:
    """Error text for the first numeric option *action* consumes but cannot honor.

    ``timeout`` is a wait budget: every action that reads it hands it to a
    :class:`threading.Event` wait (:meth:`strands_robots.mesh.core.Mesh.send`)
    or to a Device Connect ``invoke``, and returns ``{"status": "timeout"}`` when
    nothing arrived in time. Only a positive finite number can be honored, so it
    is checked against
    :func:`~strands_robots.utils.positive_finite_number_error` - the same domain
    :mod:`~strands_robots.tools._numeric_options` applies to the ROS transports'
    ``timeout``, which is the same quantity consumed the same way. A fractional
    budget is perfectly usable, which is why the domain is the continuous one.

    ``limit`` is the number of buffered messages ``inbox`` returns, consumed
    directly as a slice index, so it is checked against
    :func:`~strands_robots.utils.positive_count_error`: an integral float raises
    ``TypeError`` from the slice rather than being coerced.

    This tool keeps its own table and calls the two shared domains directly
    rather than reusing
    :func:`~strands_robots.tools._numeric_options.numeric_option_error`, whose
    documented scope is the three tools that drive a ROS graph and whose options
    are ``timeout`` / ``count`` / ``rate``. The domains are shared; only the
    per-action scoping, a property of this tool's transports, is local.

    Args:
        action: The requested action; decides which options are effective.
        timeout: Seconds to wait for a response, as supplied.
        limit: Max messages ``inbox`` returns, as supplied.

    Returns:
        An error message naming the tool, the action and the option, or ``None``
        when every option this action reads is usable.
    """
    consumed = _ACTION_NUMERIC_OPTIONS.get(action, ())
    context = f"robot_mesh {action}"
    if "timeout" in consumed:
        error = positive_finite_number_error(timeout, "timeout", context)
        if error:
            return error
    if "limit" in consumed:
        error = positive_count_error(limit, "limit", context)
        if error:
            return error
    return None


# ── #10: robot-less gateway mesh ───────────────────────────────────────────
# A dashboard / coordinator / logger process has no Robot()/Simulation() in
# _LOCAL_ROBOTS, so historically every robot_mesh action failed with "no local
# mesh found" even with live peers on the wire. The gateway is a Mesh with
# robot=None: it subscribes presence (populating session peer tracking, so
# ``peers`` works), and its send()/broadcast() path is fully functional. It
# is never a task target itself - incoming execute/... simply report
# "unknown action" like any robot-less peer.
_GATEWAY_LOCK = threading.Lock()

#: Single-slot cache for the process-wide gateway, keyed ``"mesh"``. A mutable
#: container rather than a rebound module global: the cache is written from
#: ``_gateway_mesh`` and read there on every later call, and a ``global``
#: rebinding makes that write look dead to any single-function analysis -- which
#: is what code scanning reported it as. The dict is also the shape the rest of
#: the tree already uses for lock-guarded module state.
_GATEWAY: dict[str, Any] = {}

#: Environment override for how long gateway bring-up waits for presence.
_GATEWAY_WAIT_ENV = "STRANDS_MESH_GATEWAY_DISCOVERY_WAIT_S"

#: Default gateway presence wait (seconds) -- one heartbeat period.
_GATEWAY_DISCOVERY_WAIT_S = 3.0


def _gateway_discovery_wait_s() -> float:
    """Resolve how long gateway bring-up waits for presence to populate.

    Read through the shared numeric domain rather than by calling ``float()`` on
    the raw value inside the :func:`time.sleep` argument. That form gave one
    operator knob two failures that look nothing like a misconfigured wait. A
    non-numeric value raised :class:`ValueError`, which the best-effort handler
    around gateway bring-up absorbed, so a typo was reported as "gateway mesh
    unavailable" and every action fell back to ``no local mesh found`` -- naming
    neither the variable nor the typo. ``inf`` was worse than raising:
    :func:`time.sleep` accepts it and blocks forever while holding
    ``_GATEWAY_LOCK``, so the call never returns and no later call can take the
    lock either.

    Same shape as :func:`~strands_robots.mesh.session.stream_min_period_from_env`
    for the step-telemetry rate; this is the remaining mesh knob that was still
    read inline.

    Returns:
        The override when it names a span :func:`time.sleep` can honor,
        including ``0`` -- an operator asking not to wait is obeyed rather than
        overridden -- and :data:`_GATEWAY_DISCOVERY_WAIT_S` when it is unset or
        holds a value no sleep can honor. A wait that is merely wrong costs
        first-call peer completeness; refusing to start the gateway over it
        would cost the whole feature, so the default is the safe direction.
    """
    raw = os.environ.get(_GATEWAY_WAIT_ENV)
    if raw is None or raw.strip() == "":
        return _GATEWAY_DISCOVERY_WAIT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a number; waiting the %.1fs default for gateway presence",
            _GATEWAY_WAIT_ENV,
            raw,
            _GATEWAY_DISCOVERY_WAIT_S,
        )
        return _GATEWAY_DISCOVERY_WAIT_S
    error = finite_number_error(value, _GATEWAY_WAIT_ENV, "gateway mesh bring-up")
    if error is None and value < 0:
        error = f"{_GATEWAY_WAIT_ENV} must be >= 0 seconds, got {value}"
    if error is not None:
        logger.warning("%s; waiting the %.1fs default for gateway presence", error, _GATEWAY_DISCOVERY_WAIT_S)
        return _GATEWAY_DISCOVERY_WAIT_S
    return value


def _stop_gateway_mesh() -> None:
    """Stop the process-wide gateway mesh, if one was ever started.

    The gateway is created lazily and cached for the process lifetime, and it is
    the one :class:`~strands_robots.mesh.core.Mesh` in the tree with no owner to
    stop it: every other one is closed by the ``Robot`` or ``Simulation`` that
    built it. So it held an open session plus its heartbeat and state threads,
    and stayed advertised to the fleet as a live peer, until the interpreter
    died. Registered with :mod:`atexit`, matching the session singleton's own
    teardown.
    """
    with _GATEWAY_LOCK:
        gateway = _GATEWAY.pop("mesh", None)
    if gateway is None:
        return
    try:
        gateway.stop()
    except (AttributeError, OSError, RuntimeError) as exc:
        # Interpreter shutdown: this path makes no success claim to contradict,
        # and Mesh.stop already absorbs its own transport errors. Narrow so a
        # programmer error still surfaces rather than being swallowed at exit.
        logger.debug("robot_mesh: gateway mesh stop failed at exit: %s", exc)


atexit.register(_stop_gateway_mesh)


def _gateway_mesh() -> Any | None:
    """Lazily create the robot-less gateway Mesh.

    Returns None when zenoh is unavailable, and -- before trying -- when
    ``STRANDS_MESH`` trips the hard kill switch.

    This is the one ``Mesh`` in the tree built without going through
    :func:`~strands_robots.mesh.core.init_mesh`, so it did not inherit that
    function's kill-switch check. README documents ``STRANDS_MESH=false`` as
    overriding even an explicit ``mesh=True``, but an operator who set it and
    then used the ``robot_mesh`` tool from a robot-less process still got a real
    Zenoh session, this peer advertised to the fleet as ``gateway-*``, and the
    heartbeat, state and seven sensor threads :meth:`Mesh.start` spawns -- for
    the life of the process, since the result is cached here until
    :func:`_stop_gateway_mesh` runs at exit. A kill switch that leaves nine
    threads publishing is not a kill switch.

    The check is deliberately outside ``_GATEWAY_LOCK``: it reads one env var and
    touches no shared state, and a disabled mesh should not queue behind a
    bring-up that is holding the lock through its discovery sleep.
    """
    if mesh_disabled_by_env():
        logger.debug(
            "robot_mesh: gateway mesh not started: STRANDS_MESH=%r disables the mesh",
            os.getenv("STRANDS_MESH", ""),
        )
        return None
    with _GATEWAY_LOCK:
        cached = _GATEWAY.get("mesh")
        if cached is not None and getattr(cached, "alive", False):
            return cached
        try:
            import socket as _socket
            import uuid as _uuid

            from strands_robots.mesh.core import Mesh

            gw = Mesh(
                None,
                peer_id=f"gateway-{_socket.gethostname().split('.')[0]}-{_uuid.uuid4().hex[:4]}",
                peer_type="gateway",
            )
            gw.start()
            if not gw.alive:
                return None
            _GATEWAY["mesh"] = gw
            logger.info("robot_mesh: started robot-less gateway mesh %s", gw.peer_id)
            # First bring-up: wait one heartbeat period so presence
            # subscription can populate session peer tracking before the
            # caller reads peers. Once, here, rather than per call - a
            # per-call wait stretches a burst of calls past the rate-limit
            # window and silently raises the effective cap.
            time.sleep(_gateway_discovery_wait_s())
            return gw
        except Exception as exc:  # noqa: BLE001 - gateway is best-effort
            logger.debug("robot_mesh: gateway mesh unavailable: %s", exc)
            return None


def _resolve_mesh(target: str) -> Any | None:
    """Return a local Mesh in this process to use as the gateway for RPC.

    The agent does not need to know its own peer_id: any local mesh in
    ``_LOCAL_ROBOTS`` is functionally equivalent for outbound calls because
    they all share the same Zenoh session.

    Important: when *target* matches a local peer_id, we deliberately pick a
    *different* local mesh as the gateway. Using the target as its own
    gateway triggers ``_on_cmd``'s self-loop drop (``sender_id == peer_id``)
    and the call silently times out. When the target IS the only local mesh,
    we still return it - the caller will get a timeout, which is the
    expected behaviour for "send to yourself".
    """
    from strands_robots.mesh import get_local_robots

    locals_ = get_local_robots()
    if not locals_:
        # #10: no in-process robot - fall back to the robot-less gateway so
        # coordinator processes (dashboards, schedulers) can still reach the
        # fleet. Returns None only when zenoh itself is unavailable.
        return _gateway_mesh()
    if target:
        # Prefer a local mesh whose peer_id is NOT the target so we don't
        # send-to-self via the target's own session.
        for pid, m in locals_.items():
            if pid != target:
                return m
    # Either no target was specified or every local mesh IS the target -
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


def _agent_identity() -> str:
    """Return this agent's caller identity for Device Connect RPCs.

    Sourced from ``STRANDS_ROBOT_MESH_AGENT_ID`` (falling back to the generic
    ``DEVICE_CONNECT_CLIENT_ID``). Empty string when unset - in which case the
    agent is an anonymous caller and a device with ``DEVICE_CONNECT_RPC_ALLOW``
    set will (correctly) reject it.

    SECURITY: this identity is *self-asserted*. It lets an operator who has
    locked a device's RPC allowlist authorize this agent by id, but it is only
    a trustworthy control when the transport authenticates the sender (mTLS).
    On an insecure/trusted-LAN D2D link it is advisory - any peer can claim any
    id, so do not rely on it as the sole authorization boundary there.
    """
    return os.environ.get("STRANDS_ROBOT_MESH_AGENT_ID") or os.environ.get("DEVICE_CONNECT_CLIENT_ID") or ""


def _with_identity(params: dict[str, Any]) -> dict[str, Any]:
    """Stamp the agent's caller identity into the DC command envelope.

    The device side reads ``params["_dc_meta"]["source_device"]`` and exposes it
    via ``get_rpc_source_device()`` for the driver's caller-authorization check.
    Without this the device sees an anonymous caller (``None``). No-op when no
    identity is configured, preserving the anonymous-caller behaviour.
    """
    identity = _agent_identity()
    if not identity:
        return params
    meta = dict(params.get("_dc_meta", {}))
    meta.setdefault("source_device", identity)
    return {**params, "_dc_meta": meta}


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
            "DEVICE_CONNECT_ALLOW_INSECURE is enabled - agent-side Device "
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

    Returns None - signalling robot_mesh() to use the built-in mesh - when
    Device Connect is unavailable, has discovered no devices, or the action is
    one DC does not handle (subscribe / watch / inbox / unsubscribe).
    """
    if action in ("subscribe", "watch", "inbox", "unsubscribe"):
        return None  # mesh-only actions - let the built-in mesh handle them
    if os.environ.get("STRANDS_ROBOT_MESH_DC", "on").strip().lower() in ("off", "0", "false", "no"):
        return None  # Device Connect dispatch disabled (e.g. hermetic unit tests)
    try:
        _dc_ensure_connected()
        from device_connect_agent_tools.connection import get_connection

        conn = get_connection()
        devices = conn.list_devices()
    except Exception as exc:  # noqa: BLE001 - DC is optional; fall back to mesh
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
                text += f"  [{icon}] {d['device_id']} - {avail}\n"
                if action == "peers":
                    funcs = d.get("functions", [])
                    if funcs:
                        names = [f["name"] if isinstance(f, dict) else f for f in funcs]
                        text += f"    Functions: {', '.join(names)}\n"
            # Audit the read-only observation actions on this backend too. The
            # mesh rendering of ``peers`` / ``status`` records the fleet it read
            # (see the ``peers`` branch in the mesh dispatch below); this backend
            # is the one tried FIRST whenever Device Connect has devices, so
            # leaving it out made the audited implementations the fallback and
            # the unaudited ones the default. ``peers`` in particular returns
            # every device id and every function name the fleet exposes, so an
            # enumeration of the callable surface would otherwise leave no trail.
            _audit_tool_action(action, target, True, f"devices={len(devices)}")
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
            result = conn.invoke(
                target, "execute", _with_identity({"instruction": instruction, **kwargs}), timeout=timeout
            )
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
            result = conn.invoke(target, func, _with_identity(cmd), timeout=timeout)
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
            result = conn.invoke(target, func_name, _with_identity(rpc_params), timeout=timeout)
            r = result.get("result", result) if isinstance(result, dict) else result
            _audit_tool_action(action, target, True, f"function={func_name}")
            return _DCResult(
                _ok(f"{target}.{func_name}({rpc_params}) ->\n{json.dumps(r, indent=2, default=str)[:2000]}")
            )

        if action == "stop":
            if not target:
                return _DCResult(_err("stop requires target"))
            # A stop is capped at 5s so it cannot hang. This is a cap over an
            # already-validated positive finite budget, not a guard: min() would
            # pass nan straight through (min(nan, 5.0) is nan).
            result = conn.invoke(target, "stop", _with_identity({}), timeout=min(timeout, 5.0))
            r = result.get("result", result)
            _audit_tool_action(action, target, True, "")
            return _DCResult(_ok(f"Stop {target}: {json.dumps(r, default=str)}"))

        if action == "emergency_stop":
            devices = conn.list_devices()
            stopped = 0
            for d in devices:
                try:
                    conn.invoke(d["device_id"], "stop", _with_identity({}), timeout=3.0)
                    stopped += 1
                except Exception:  # noqa: BLE001 - best-effort fan-out
                    pass
            _audit_tool_action(action, "*", True, f"stopped={stopped}/{len(devices)}")
            return _DCResult(_ok(f"E-STOP: {stopped}/{len(devices)} devices stopped"))

        if action == "broadcast":
            # Security hardening: dispatch the *validated* command that the
            # operator approved at the HITL gate - never re-parse the raw
            # caller-supplied string here (that would allow a payload whose
            # validated form differs from what actually executes).
            if validated_command is None:
                return _DCResult(_err("broadcast reached Device Connect dispatch without a validated command"))
            cmd = dict(validated_command)
            func = cmd.pop("action", cmd.pop("function", "getStatus"))
            params = _with_identity(cmd)
            results = conn.broadcast(func, params, timeout=timeout)
            _audit_tool_action(action, "*", True, f"action={func} responses={len(results)}")
            text = f"[broadcast] {len(results)} responses\n"
            for r in results[:10]:
                sstr = "ok" if "result" in r else f"error: {r.get('error', '?')}"
                text += f"  {r.get('device_id', '?')}: {sstr}\n"
            return _DCResult(_ok(text.rstrip()))

        # subscribe / watch / inbox / unsubscribe → handled by the mesh path
        return None
    except Exception as exc:  # noqa: BLE001 - never raise out of the dispatcher
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
        timeout: Response timeout for RPC actions (seconds). A positive finite
            number; read by ``tell`` / ``send`` / ``rpc`` / ``broadcast`` /
            ``stop`` and ignored by the rest. ``stop`` additionally caps it at
            5s. Zero or negative would report ``{"status": "timeout"}`` without
            waiting at all, so an unusable value is refused rather than reported
            as a peer that did not answer.
        name: Optional subscription name for ``subscribe`` / ``inbox``.
        limit: Max messages returned by ``inbox`` (default: 50). A positive
            integer; read by ``inbox`` only.
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
          ``emergency_stop`` / ``rpc`` is audited.
    """
    # Resolve which actions require a human-in-the-loop interrupt for THIS
    # call. Consumers configure the set via STRANDS_MESH_HITL_ACTIONS; the
    # default gates every physical-actuation action (emergency_stop,
    # broadcast, tell, send, stop, rpc). A malformed env var fails loud
    # here rather than silently degrading the gate.
    try:
        interrupt_actions = _resolve_interrupt_actions()
    except _InterruptConfigError as exc:
        _audit_tool_action(action, target, False, f"hitl config error: {exc}")
        return _err(str(exc))

    # One-time warning when the operator has explicitly disabled the gate.
    if not interrupt_actions and os.getenv("STRANDS_MESH_HITL_ACTIONS", "").strip().lower() == "none":
        _warn_none_opt_out_once()

    # Check the per-action rate limit before doing any work - but
    # do NOT consume a slot until we know the action is going to run.
    # See _rate_limit_check / _rate_limit_check_and_record for rationale.
    rl_err = _rate_limit_check(action)
    if rl_err is not None:
        _audit_tool_action(action, target, False, f"rate_limit: {rl_err}")
        return _err(rl_err)

    # Reject a numeric option this action cannot honor before anything else -
    # for the same reason the command-body pre-pass below runs early, and one
    # step sooner because this check needs no parsing. A wait budget of ``nan``
    # or a message ``limit`` of ``0`` must not burn an operator approval at the
    # HITL gate, consume a rate-limit slot, or reach a transport.
    num_err = _numeric_option_error(action, timeout=timeout, limit=limit)
    if num_err is not None:
        _audit_tool_action(action, target, False, f"validation: {num_err}")
        return _err(num_err)

    # Parse + validate any command body BEFORE the HITL interrupt so
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
        _fleet_wide = action in _FLEET_WIDE_ACTIONS
        _approval_target = "*ALL_PEERS*" if _fleet_wide else (target or "*ALL_PEERS*")
        _scope_warning, _target_physical, _target_verified = _approval_warning(action, target)
        # Surface the validated command (post-validation form) so the
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
                    # Surface the device-native function name for rpc so the
                    # operator approves the specific function being invoked.
                    "function": function if action == "rpc" else "",
                    "command": _approval_command,
                    "instruction": instruction,
                    "warning": _scope_warning,
                    # What the gate established about the target, so a host UI
                    # can show the verdict without re-deriving it - and so the
                    # prompt and the structured reason cannot disagree.
                    "physical": _target_physical,
                    "verified": _target_verified,
                },
            )
        except RuntimeError as exc:
            # ToolContext.interrupt raises RuntimeError when no agent
            # instance is attached - i.e. the tool is being invoked
            # outside a Strands agent loop (a direct
            # ``agent.tool.robot_mesh(...)`` call, a unit test that did
            # not wire up the SDK, etc.). In those contexts there is no
            # operator to ask, so fail-closed.
            #
            # NB: the SDK's ``InterruptException`` MUST propagate up to
            # pause the agent loop, so we deliberately do NOT catch
            # ``Exception`` here - that would swallow the normal
            # interrupt-pause flow and turn every approval into an
            # immediate "interrupt unavailable" error.
            _audit_tool_action(action, target, False, f"interrupt unavailable: {exc}")
            return _err(
                f"action '{action}' requires a human-in-the-loop interrupt. Interrupts are not available here: {exc}"
            )

        approved = _interrupt_approves(response)
        # Record the human's verdict as soon as it is known, before any later
        # refusal can return. The rate-limit re-check below can reject an
        # APPROVED action (a concurrent invocation took the last slot while the
        # operator was deciding), and recording after it left that path with no
        # operator row at all: the audit log carried only ``rate_limit_race``,
        # which does not say a human authorised a physical actuation. One
        # unconditional site, matching use_ros and lerobot_train.
        #
        # #322: the operator's literal interrupt response is recorded in
        # the LOCAL audit row (full fidelity for forensics) but MUST NOT be
        # echoed back to the LLM. Echoing it turns the human operator into a
        # content side-channel: a prompt-injected agent could phrase the
        # approval reason so the operator's typed reply leaks data back into
        # the model context. Return a flat, fixed sentinel instead.
        log_operator_response(_AUDIT_SOURCE, action, target, approved=approved, response=response)
        if not approved:
            # Declined approval does NOT consume a rate-limit slot -
            # see _rate_limit_check docstring for the safety rationale.
            return _err(f"action '{action}' was declined by the operator interrupt.")
        # Approval granted. Re-check under the lock and consume the
        # slot atomically -- a concurrent invocation that ALSO passed
        # the pre-interrupt check (different operator thread, etc.)
        # will be rejected here so the configured limit cannot be
        # exceeded by HITL races.
        rl_race_err = _rate_limit_check_and_record(action)
        if rl_race_err is not None:
            _audit_tool_action(action, target, False, f"rate_limit_race: {rl_race_err}")
            return _err(rl_race_err)
    else:
        # No interrupt required for this action - reserve the slot with the
        # same atomic check+record the approved path uses above. The pre-gate
        # check does not consume a slot, so a concurrent invocation can take
        # the last one in between; recording unconditionally here would let
        # both callers past a full bucket. Once ``STRANDS_MESH_HITL_ACTIONS``
        # narrows the gated set this limit is the only bound left on
        # LLM-driven actuation, so it has to hold on this path too.
        rl_race_err = _rate_limit_check_and_record(action)
        if rl_race_err is not None:
            _audit_tool_action(action, target, False, f"rate_limit_race: {rl_race_err}")
            return _err(rl_race_err)

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
    if not locals_:
        # #10: robot-less process - bring up the gateway BEFORE reading peers
        # so presence subscription populates session peer tracking. The
        # gateway itself waits one heartbeat period on first bring-up.
        _gateway_mesh()
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
        # #322: audit read-only observation actions too, so the audit log is a
        # complete record of agent mesh access (not just actuation). Closes the
        # forensic gap where peers/status/inbox/unsubscribe left no trail.
        _audit_tool_action(action, target, True, f"local={len(locals_)} remote={len(peers)}")
        return _ok("\n".join(lines))

    # ── action: status ────────────────────────────────────────────────────
    if action == "status":
        # #322: read-only status is audited too (see peers branch above).
        _audit_tool_action(action, target, True, f"local={len(locals_)} remote={len(peers)}")
        return _ok(f"[mesh] local={len(locals_)} remote={len(peers)} peers={[p['peer_id'] for p in peers]}")

    # All remaining actions need an outbound mesh.
    mesh = _resolve_mesh(target)
    if mesh is None:
        if mesh_disabled_by_env():
            # Naming the variable matters more here than anywhere else in this
            # function: the generic remedy below is to construct a Robot(), and
            # the kill switch would refuse that mesh too, so an operator
            # following it learns nothing and tries twice.
            return _err(
                f"mesh disabled: STRANDS_MESH={os.getenv('STRANDS_MESH', '')!r} is a hard kill switch. "
                f"Unset it (or set it to true) to use action={action!r}."
            )
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
        # Explicit raise, not assert -- see the broadcast handler below for
        # why the sentinel check must survive ``python -O``.
        if validated_send_cmd is None:
            raise RuntimeError(
                "send reached its handler without pre-validation -- validate-before-HITL contract broken"
            )
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
        # Pre-validated above before the HITL interrupt fired, so
        # the cmd here is already a clean validated dict.
        # Use explicit raise (not assert): assert is stripped under
        # ``python -O`` / ``PYTHONOPTIMIZE=1``, and with the check gone the
        # handler dispatches the unset sentinel -- mesh.broadcast(None) is
        # issued fleet-wide, and the cmd.get(...) on the audit line below
        # then raises, so the dispatch that did happen is never recorded.
        if validated_broadcast_cmd is None:
            raise RuntimeError(
                "broadcast reached its handler without pre-validation -- validate-before-HITL contract broken"
            )
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
            # Capped at 5s so a stop cannot hang - a cap over an already-validated
            # positive finite budget, not a guard (min(nan, 5.0) is nan).
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
        # equality / trailing-`/**` and reach ``mesh.on_stream("*")`` -
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
        # ``limit`` is a validated positive int by here, so the tail slice is
        # always the cap the caller asked for. The former ``if limit > 0 else
        # msgs`` fallback returned the WHOLE buffer for a non-positive limit -
        # the opposite of a cap - and is unreachable now.
        head = msgs[-limit:]
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
        # #322: audit the unsubscribe so the read-only/observation action set
        # (peers, status, inbox, unsubscribe) leaves a complete forensic trail.
        _audit_tool_action(action, sub_name, True, "")
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
