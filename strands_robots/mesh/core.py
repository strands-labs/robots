"""Core Mesh class — lifecycle, presence, state, cameras, RPC, and subscriptions.

This is the primary component that a Robot or Simulation composes with.
Extended sensor loops (pose, IMU, health, etc.) are provided by
:class:`~strands_robots.mesh.sensors.SensorLoopsMixin`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from strands_robots.mesh import security as _security
from strands_robots.mesh.audit import log_safety_event
from strands_robots.mesh.sensors import SensorLoopsMixin
from strands_robots.mesh.session import (
    CAMERA_HZ,
    HEARTBEAT_HZ,
    STATE_HZ,
    current_session,
    get_session,
    prune_peers,
    put,
    release_session,
    update_peer,
)
from strands_robots.mesh.session import (
    get_peers as _session_get_peers,
)

logger = logging.getLogger(__name__)


# Module-level registry of local meshes
_LOCAL_ROBOTS: dict[str, Mesh] = {}
_LOCAL_ROBOTS_LOCK = threading.Lock()


def get_local_robots() -> dict[str, Mesh]:
    """Return a snapshot of in-process mesh-enabled robots."""
    with _LOCAL_ROBOTS_LOCK:
        return dict(_LOCAL_ROBOTS)


#: Sentinel stored in :attr:`Mesh._expected_responders` for
#: broadcast turn_ids. Distinct from any real peer_id (no peer_id
#: contains a NUL byte).
BROADCAST_RESPONDER: str = "<broadcast>\x00"


def _parse_positive_float_env(name: str, default: str, *, minimum: float = 0.0) -> float:
    """Parse a positive-float env var, falling back to default on bad input.

    Catches the case where an operator sets ``STRANDS_MESH_RESUME_FRESHNESS_S=abc``
    or a negative value. The module would otherwise fail to import with an opaque
    ``ValueError`` (found by running the module under bad env locally; see
    ``test_resume_env_validation.py``).
    """
    raw = os.getenv(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r (not a float); falling back to default %r.",
            name,
            raw,
            default,
        )
        return float(default)
    if value < minimum:
        logger.warning(
            "Invalid %s=%r (must be >= %s); falling back to default %r.",
            name,
            value,
            minimum,
            default,
        )
        return float(default)
    return value


def _parse_positive_int_env(name: str, default: str, *, minimum: int = 1) -> int:
    """Parse a positive-int env var, falling back to default on bad input.

    Companion to :func:`_parse_positive_float_env` for cache-size knobs.
    Rejects zero / negative because ``deque(maxlen=0)`` would silently disable
    the replay cache and ``maxlen=-1`` would raise at runtime.
    """
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r (not an int); falling back to default %r.",
            name,
            raw,
            default,
        )
        return int(default)
    if value < minimum:
        logger.warning(
            "Invalid %s=%r (must be >= %s); falling back to default %r.",
            name,
            value,
            minimum,
            default,
        )
        return int(default)
    return value


#: Default resume-envelope freshness window. Envelopes whose t field
#: is older than this are rejected as potential replays. Operators
#: on drifty NTP can extend via STRANDS_MESH_RESUME_FRESHNESS_S
#: (sane bound: keep < 600). Bad input falls back to 60.
#:
#: the module-level value below is captured
#: once at import time for backward compatibility with code/tests
#: that read these as constants. The runtime hot paths
#: (:meth:`Mesh._on_safety_estop`, :meth:`Mesh._on_safety_resume`)
#: now call :func:`_resume_freshness_window_s` etc. on every call so
#: an operator setting STRANDS_MESH_RESUME_* AFTER importing the
#: module sees the new value without a process restart. The README
#: contract ("operator-tunable env vars") now actually holds.
RESUME_FRESHNESS_WINDOW_S: float = _parse_positive_float_env("STRANDS_MESH_RESUME_FRESHNESS_S", "60")

#: Forward-skew tolerance on the envelope t field. See
#: :data:`RESUME_FRESHNESS_WINDOW_S` for the lazy-resolution note.
RESUME_FORWARD_SKEW_S: float = _parse_positive_float_env("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "5")

#: Maximum entries in the per-receiver resume replay cache.
#: See :data:`RESUME_FRESHNESS_WINDOW_S` for lazy-resolution note.
RESUME_REPLAY_CACHE_MAX: int = _parse_positive_int_env("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4096")


def _resume_freshness_window_s() -> float:
    """Lazy resolver for ``STRANDS_MESH_RESUME_FRESHNESS_S``.

    re-reads the env var on every call so operator-set values
    take effect without a process restart. Cheap (one ``os.getenv``
    + a regex parse via ``_parse_positive_float_env``) and called
    only on safety-handler entry, which is bounded by the transport
    rate cap.
    """
    return _parse_positive_float_env("STRANDS_MESH_RESUME_FRESHNESS_S", "60")


def _resume_forward_skew_s() -> float:
    """Lazy resolver for ``STRANDS_MESH_RESUME_FORWARD_SKEW_S``."""
    return _parse_positive_float_env("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "5")


def _resume_replay_cache_max() -> int:
    """Lazy resolver for ``STRANDS_MESH_RESUME_REPLAY_CACHE_MAX``."""
    return _parse_positive_int_env("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4096")


def _resume_max_fails() -> int:
    """Consecutive failed-resume attempts before the throttle engages.
    Lazy (restart-free). Defaults to 5; bad input falls back to 5."""
    return _parse_positive_int_env("STRANDS_MESH_RESUME_MAX_FAILS", "5")


def _resume_backoff_s() -> float:
    """Cooldown (seconds) the resume path is refused after the
    fail threshold is hit. Lazy. Defaults to 30s; bad input -> 30."""
    return _parse_positive_float_env("STRANDS_MESH_RESUME_BACKOFF_S", "30")


def _evict_replay_cache[K](
    cache: dict[K, float],
    *,
    max_size: int,
    ttl_s: float,
    now_mono: float,
) -> None:
    """Bound *cache* to ``max_size`` entries in-place.

    Eviction strategy (single-pass, no-op when the cache is below the cap):

    1. Drop entries whose stored monotonic timestamp is older than
       ``now_mono - ttl_s`` (TTL purge).
    2. If the cache is still over budget after the TTL purge (cap is full
       of in-window entries -- active flood), drop the oldest 20 percent
       by stored timestamp.

    Caller is responsible for passing the right ``ttl_s``. the safety-replay caches now pass
    ``RESUME_FRESHNESS_WINDOW_S + RESUME_FORWARD_SKEW_S`` so an
    accepted forward-skewed envelope (``t = now + skew``) stays in the
    cache for the full ``freshness + skew`` interval -- preventing a
    captured forward-skewed envelope from being replayed seconds after
    its original acceptance window closed.

    Single source of truth for both ``_resume_replay_cache`` (a
    direct ``dict[K, float]``) and ``_estop_replay_cache`` (a
    ``dict[float, tuple[issuer_id, mono_ts, wire_zid]]``, which the
    estop call site reduces to a ``ts_view`` ``dict[K, float]`` before
    calling this helper, then applies the surviving keyset back to its
    real cache via set-difference). The view-shim strips the
    issuer/zid attribution before this helper sees it; any future
    eviction policy that needs issuer- or zid-aware behaviour (e.g.
    "evict over-cap issuers preferentially" or "weight TTL by source
    session") cannot live in this helper -- it would have to be
    re-implemented inline at the estop call site, or this helper
    widened to ``dict[K, V]`` + ``value_to_ts: Callable[[V], float]``.
    Callers own insertion and the per-cache lock; this helper is pure
    bookkeeping.
    """
    # ALWAYS run the TTL purge -- on low-traffic meshes the cache may
    # never reach max_size but stale entries still accumulate
    # indefinitely (issue #274). The TTL purge is O(n) and bounded by
    # the cache size, so it's cheap to run unconditionally.
    cutoff = now_mono - ttl_s
    stale = [k for k, ts in cache.items() if ts < cutoff]
    for k in stale:
        cache.pop(k, None)
    if len(cache) >= max_size:
        ordered = sorted(cache.items(), key=lambda kv: kv[1])
        drop = max(1, len(ordered) // 5)
        for k, _ in ordered[:drop]:
            cache.pop(k, None)


#: Lowercase hex digest at most 32 chars. ``ZenohId`` stringifies as
#: a hex digest of the 16-byte session identifier (leading-zero
#: trimmed, so 1..32 chars). Used by ``_extract_sample_source_zid``
#: to reject obviously-bogus stand-ins (test ``MagicMock``,
#: third-party transport shims) without paying the cost of importing
#: zenoh just to ``isinstance`` check.
_ZENOH_ZID_PATTERN = re.compile(r"^[0-9a-f]{1,32}$")


def _extract_sample_source_zid(sample: Any) -> str | None:
    """Return the TLS-bound publisher ZID from a Zenoh ``sample``, or ``None``.

    Zenoh attaches ``sample.source_info.source_id.zid`` (the publishing
    session's ``ZenohId``) at the wire level. The ``ZenohId`` is established
    during the session bootstrap that follows the mTLS handshake, and the
    ``zenoh-python`` API does not expose a public constructor for either
    ``ZenohId`` or ``EntityGlobalId`` -- they can only be obtained from
    ``Session.info.zid()`` or ``Publisher.id`` on a session that has already
    completed the handshake against the trust roots in ``connect.tls``.

    Combined with mTLS this means a peer holding a valid cert for one
    session cannot mint an envelope that *also* claims the wire-level
    ``source_zid`` of a different session: the cross-session forgery is
    bounded by what their own session's ``ZenohId`` actually is.

    The body's ``peer_id`` field remains application-level metadata (chosen
    by the operator at ``init_mesh`` time and routable across reconnects);
    this helper returns the wire-level identity that the safety handlers
    pin HMAC inputs and replay caches to. The two are complementary: body
    ``peer_id`` survives a session restart, wire ``source_zid`` survives an
    attacker mutating the body.

    Returns ``None`` when:

    * the sample carries no ``source_info`` (publisher did not attach one --
      e.g. the bridge/IoT transport path or a legacy publisher),
    * the ``source_id`` is missing,
    * the stringified value does not match the strict 1..32 hex digest
      shape (a malformed sample, third-party transport shim, or unit-test
      ``MagicMock`` whose ``source_id.zid`` defaults to a Mock repr),
    * extraction raises (defence in depth -- treat as "no zid available"
      rather than crashing the safety handler).
    """
    try:
        si = getattr(sample, "source_info", None)
        if si is None:
            return None
        sid = getattr(si, "source_id", None)
        if sid is None:
            return None
        zid = getattr(sid, "zid", None)
        if zid is None:
            return None
        zid_str = str(zid)
        if not _ZENOH_ZID_PATTERN.match(zid_str):
            return None
        return zid_str
    except (AttributeError, TypeError):
        return None


class Mesh(SensorLoopsMixin):
    """Peer-to-peer mesh component embedded in a single Robot or Simulation.

    Lifecycle: construct via :func:`init_mesh`, call :meth:`stop` during cleanup.

    Thread safety:
        :meth:`start` and :meth:`stop` are protected by ``_lifecycle_lock``.
    """

    def __init__(self, robot: Any, peer_id: str, peer_type: str = "robot") -> None:
        self.robot = robot
        self.peer_id = peer_id
        self.peer_type = peer_type

        self._running: bool = False
        self._has_session_ref: bool = False
        self._subs: list[Any] = []
        self._threads: list[threading.Thread] = []
        self._lifecycle_lock = threading.Lock()
        self._subs_lock = threading.Lock()
        self._inbox_lock = threading.Lock()
        self._stop_event = threading.Event()

        # RPC correlation state.
        #
        # _expected_responders maps turn_id -> the peer_id we expect to
        # answer (set by send() at point-to-point), or the sentinel
        # ``BROADCAST_RESPONDER`` if the turn_id was created by
        # broadcast() and we accept responses from any peer. Phase-4 /
        # D1: this is what _on_response uses to reject a forged
        # response from a peer that wasn't the original target.
        self._rpc_lock = threading.Lock()
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, list[dict[str, Any]]] = {}
        self._expected_responders: dict[str, str] = {}

        # User subscribe state
        self.inbox: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._user_subs: dict[str, Any] = {}

        # Emergency-stop lockout flag. While this Event is set, every
        # action other than ``status`` and ``resume`` is refused (see
        # :meth:`_dispatch`). The flag is cleared by :meth:`_resume_lockout`,
        # which requires the operator-supplied override code.
        self._estop_lockout = threading.Event()
        self._last_estop_ts: float = 0.0
        self._last_estop_mono: float = 0.0
        # _on_safety_resume must defend
        # against replay of a previously-observed override-proof envelope.
        # The receiver caches (proof_nonce, issuer_peer_id) tuples it has
        # already accepted and refuses duplicates within a bounded window.
        # Combined with the freshness check on the envelope ``t`` field
        # this closes the recorded-and-replayed-resume surface even when
        # an attacker has live ACL access on safety/**.
        # Key shape: ((domain_tag, issuer), proof_nonce) where domain_tag is
        # "wire" for TLS-bound Zenoh wire_zid or "body" for app-level issuer_id.
        # Tuple-of-tuples prevents cross-transport namespace collision (R12).
        self._resume_replay_cache: dict[tuple[tuple[str, str], str], float] = {}
        self._resume_replay_lock = threading.Lock()
        # estop replay defense -- mirror of resume cache, keyed on
        # (issuer_peer_id, envelope_t). Closes the captured-estop-replay DoS
        # surface that previously let any peer with live ACL access to
        # safety/** replay a captured envelope indefinitely. Reuses
        # _resume_freshness_window_s() / _resume_forward_skew_s() /
        # _resume_replay_cache_max() (the safety-replay defenses are
        # symmetric in shape; sharing the bounds keeps env-var surface
        # minimal).
        # Cache shape: ``dict[float_t, (issuer_id, mono_ts, wire_zid)]``.
        # The float key preserves the peer_id-permutation defence
        # (attacker cannot mint novel keys by varying untrusted JSON
        # peer_id); the 3-tuple value carries issuer attribution,
        # the monotonic eviction timestamp, AND the TLS wire_zid at
        # capture (for corroboration gating). Per-issuer counts are
        # derived from cache contents on demand -- no separate dict
        # that can drift after eviction.
        self._estop_replay_cache: dict[float, tuple[str, float, str | None]] = {}
        self._estop_replay_lock = threading.Lock()
        # Command-replay dedup. The CMD path (_exec_cmd ->
        # _dispatch) previously had NO turn_id dedup: an attacker who captured
        # one valid command envelope on the wire could replay it indefinitely
        # and each copy would dispatch + actuate (confirmed: sim_time
        # 1->2->3->4->5 on the same turn_id). We cache (sender_id, turn_id)
        # keys we have already executed and reject duplicates within a bounded
        # TTL window. Mirrors the _resume_replay_cache / _estop_replay_cache
        # shape and reuses _evict_replay_cache for bounding. Key shape:
        # ((sender_id, turn_id)) -> monotonic insert ts.
        self._cmd_replay_cache: dict[tuple[str, str], float] = {}
        self._cmd_replay_lock = threading.Lock()
        # M-1: resume override-code brute-force throttle. The crypto
        # oracles (timing / content / length) are all closed, but the resume
        # action had NO rate limit -- a 4-digit numeric override code is
        # crackable in seconds at network speed (measured 295K attempts/s).
        # We track consecutive FAILED bad-code attempts and, after a
        # threshold, refuse further resume attempts for a cooldown window.
        # A successful resume resets the counter. The throttle is keyed on
        # attempt COUNT (not code content) so it adds no new content/timing
        # oracle beyond "you are being rate-limited", which an attacker can
        # already observe. Operator-tunable via STRANDS_MESH_RESUME_MAX_FAILS
        # and STRANDS_MESH_RESUME_BACKOFF_S.
        self._resume_fail_count: int = 0
        self._resume_locked_until_mono: float = 0.0
        self._resume_bruteforce_lock = threading.Lock()

        # Safety topic publishers. Held lazily so the receiver path
        # ``_publish_safety_envelope`` can attach a Zenoh
        # ``SourceInfo(EntityGlobalId, monotonic_sn)`` -- the only API
        # surface for getting an attacker-unforgeable wire-level zid
        # onto an outbound sample. Reusing one publisher per topic
        # also gives a stable ``EntityGlobalId.eid`` so the receiver
        # can spot a same-zid attacker mutating eid mid-flight (which
        # zenoh treats as a different publisher entity).
        self._safety_publishers: dict[str, Any] = {}
        self._safety_publishers_lock = threading.Lock()
        # Monotonically increasing per-topic sequence number. Bound
        # into ``SourceInfo.source_sn`` so the receiver can reject an
        # off-the-wire replay even from the same session: two
        # envelopes with the same ``(source_zid, source_sn)`` cannot
        # both be legitimate.
        self._safety_sn: dict[str, int] = {}
        self._safety_sn_lock = threading.Lock()

    def __repr__(self) -> str:
        state = "alive" if self._running else "stopped"
        return f"Mesh(peer_id={self.peer_id!r}, type={self.peer_type!r}, {state})"

    def _refuse_under_permissive_default_acl(self) -> bool:
        """Refuse-to-start gate per issue #218.

        Returns True (refuse) when:
        - STRANDS_MESH_AUTH_MODE == "mtls" (ACL is the third line of defence)
        - AND the resolved ACL is permissive-by-shape (built-in default
          or operator file with default_permission=allow + no rules)
        - AND STRANDS_MESH_ACCEPT_PERMISSIVE_ACL is NOT set (operator
          has not explicitly acknowledged the dev/lab posture).

        Logs an ERROR breadcrumb on refusal so the operator sees the
        actionable remediation paths (set ACL file / accept opt-in /
        disable mesh).

        On opt-in (STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1) the same shape
        is logged at INFO instead of ERROR -- the operator has
        acknowledged the posture and the WARNING contradicting their
        opt-in would be noise.

        Implementation note: takes a single ACL snapshot via
        :func:`_acl_config.snapshot_acl` and stashes the result on
        ``self._acl_snapshot`` so :func:`session._build_config`
        downstream can reuse the SAME dict (closes the
        ``Mesh.start`` -> ``_build_config`` TOCTOU window flagged in
        review at session.py:296).
        """
        from strands_robots.mesh import _acl_config, _zenoh_config

        try:
            auth_mode = _zenoh_config.resolve_auth_mode()
            namespace = _zenoh_config.resolve_namespace()
            is_permissive, resolved = _acl_config.snapshot_acl(namespace)
        except ValueError as warn_exc:
            # Narrow tuple per AGENTS.md > Review Learnings (#86):
            # ValueError surfaces bad STRANDS_MESH_AUTH_MODE / unloadable
            # ACL. Fail-CLOSED (treat as permissive) so the gate refuses
            # to bring up the wire. Wider exception types (OSError, etc.)
            # propagate so genuine bugs aren't masked at WARNING level.
            logger.warning(
                "[mesh] %s: ACL gate evaluation failed (%s) -- treating as permissive default; refusing to start",
                self.peer_id,
                warn_exc,
            )
            auth_mode = "mtls"
            is_permissive = True
            resolved = None
        # Stash the snapshot AND auth_mode on a thread-local used by
        # ``session._build_config`` so the wire-config builder picks up
        # the SAME dict the gate inspected AND the SAME auth_mode value.
        # Issue #218 + review threads session.py:296 / core.py:139.
        self._acl_snapshot = resolved
        _acl_config._set_thread_snapshot(resolved, auth_mode=auth_mode)
        if auth_mode != "mtls":
            return False
        if not is_permissive:
            return False

        accept_permissive = os.getenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if accept_permissive:
            logger.info(
                "[mesh] %s: permissive default ACL active under mtls "
                "(STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 acknowledged) -- "
                "starting in dev/lab posture",
                self.peer_id,
            )
            return False

        logger.error(
            "[mesh] %s: PERMISSIVE DEFAULT ACL ACTIVE UNDER MTLS -- "
            "Mesh NOT STARTED. Any CA-signed peer can publish/subscribe "
            "on any key. Remediate one of:\n"
            "  1. Supply STRANDS_MESH_ACL_FILE pointing at a role-separated "
            "ACL (see examples/mesh_acl_example.json5).\n"
            "  2. Set STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 to acknowledge "
            "the dev/lab posture.\n"
            "  3. Disable the mesh (do not call Mesh.start()).",
            self.peer_id,
        )
        return True

    # Lifecycle
    def start(self) -> None:
        """Acquire a Zenoh session and start all publishing loops."""
        with self._lifecycle_lock:
            if self._running:
                return

            # H-1: permanent-fleet-lockout footgun warning.
            # STRANDS_MESH_OVERRIDE_CODE is optional and defaults to empty.
            # When it is unset, _resume_lockout can NEVER succeed (no code
            # matches the empty/sentinel digest), so a SINGLE e-stop broadcast
            # permanently locks every robot until a physical restart of each
            # one -- a fleet-wide DoS from one message. We cannot safely
            # auto-generate a code (every peer must agree on it), so we emit a
            # loud startup WARNING describing the consequence + the fix. This
            # turns a silent operational landmine into an explicit, logged
            # decision. Operators who genuinely want no remote-resume posture
            # (e.g. physical-only recovery) see the warning and accept it.
            if not os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip():
                logger.warning(
                    "[safety] %s: STRANDS_MESH_OVERRIDE_CODE is not set -- the "
                    "emergency-stop lockout will be UNRECOVERABLE over the mesh. "
                    "A single estop broadcast will lock this robot until a "
                    "physical restart (fleet-wide DoS risk). Set "
                    "STRANDS_MESH_OVERRIDE_CODE to the same value on every peer "
                    "to enable remote resume.",
                    self.peer_id,
                )

            # Issue #218 / R3: refuse-to-start gate when mtls is configured
            # but the ACL is permissive-by-shape (built-in default OR
            # operator file with default_permission=allow). The gate
            # closes the "fleet thinks mTLS protects them, but ACL is
            # wide open" silent-misconfiguration footgun. Operators who
            # explicitly accept the dev/lab posture set
            # STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1.
            #
            # Review thread core.py:189 -- the gate stashes a thread-local
            # snapshot via ``_set_thread_snapshot`` (called inside
            # ``_refuse_under_permissive_default_acl``) BEFORE deciding
            # whether to refuse. Wrap both the gate and ``get_session()``
            # in the same try/finally so the snapshot is cleared on the
            # refused-start branch too, otherwise a subsequent direct
            # ``get_session()`` on the same thread (integration test, or
            # a caller bypassing Mesh) would observe a stale snapshot.
            from strands_robots.mesh import _acl_config

            try:
                if self._refuse_under_permissive_default_acl():
                    # Logged at ERROR; mesh stays not-started (mesh.alive == False).
                    # Caller's Robot() construction succeeds; only the wire is gated.
                    return
                session = get_session()
            finally:
                # Snapshot has been consumed by ``session._build_config``
                # via the thread-local single-flight (issue #218 +
                # review session.py:296), or we refused to start before
                # ``get_session`` was reached -- either way, clear it so
                # the next ``Mesh.start`` (different instance, same
                # thread) or direct ``get_session()`` call sees fresh
                # state.
                _acl_config._clear_thread_snapshot()
            if session is None:
                logger.debug("[mesh] %s: zenoh unavailable, mesh off", self.peer_id)
                return

            self._has_session_ref = True

            declared: list[Any] = []
            try:
                declared.append(session.declare_subscriber("strands/*/presence", self._on_presence))
                declared.append(session.declare_subscriber(f"strands/{self.peer_id}/cmd", self._on_cmd))
                declared.append(session.declare_subscriber("strands/broadcast", self._on_cmd))
                declared.append(session.declare_subscriber(f"strands/{self.peer_id}/response/**", self._on_response))
                # Fleet-wide e-stop: any peer broadcasting on safety/estop or
                # safety/resume engages / clears the lockout on every other
                # peer too. Without these subscribers the lockout would only
                # protect the issuing process, leaving receivers willing to
                # accept the next command after they've stopped the current
                # task.
                declared.append(session.declare_subscriber("strands/safety/estop", self._on_safety_estop))
                declared.append(session.declare_subscriber("strands/safety/resume", self._on_safety_resume))
            except (RuntimeError, OSError) as exc:
                # narrow the lifecycle cleanup catch from bare ``Exception``
                # to the tuple Zenoh's ``declare_subscriber`` actually raises
                # (``ZError`` is a subclass of ``RuntimeError``; transport-layer
                # failures surface as ``OSError``). This mirrors the wire-handler
                # tuple established earlier for ``_on_cmd`` / ``_on_safety_estop``
                # so programmer errors (``TypeError``, ``AttributeError``,
                # ``MemoryError``) on the partial-failure cleanup path surface
                # in tests rather than being silently swallowed.
                for sub in declared:
                    try:
                        sub.undeclare()
                    except (RuntimeError, OSError):
                        # Best-effort cleanup; an undeclare failure here
                        # cannot recover the parent failure that put us in
                        # this branch and surfacing it would mask the
                        # original exc. DEBUG (not WARNING) because the
                        # operator already gets the WARNING below and a
                        # second per-sub line per failure becomes log noise.
                        logger.debug(
                            "[mesh] %s: undeclare failed during cleanup",
                            self.peer_id,
                        )
                logger.warning("[mesh] %s: failed to declare subscribers: %s", self.peer_id, exc)
                release_session()
                self._has_session_ref = False
                return

            with self._subs_lock:
                self._subs.extend(declared)

            # ACL gate moved to ``_refuse_under_permissive_default_acl``,
            # called at the TOP of start() before session acquisition.
            self._running = True

            with _LOCAL_ROBOTS_LOCK:
                _LOCAL_ROBOTS[self.peer_id] = self

            # Core loops
            heartbeat = threading.Thread(
                target=self._heartbeat_loop, name=f"mesh-heartbeat-{self.peer_id}", daemon=True
            )
            state_thread = threading.Thread(target=self._state_loop, name=f"mesh-state-{self.peer_id}", daemon=True)
            self._threads = [heartbeat, state_thread]
            heartbeat.start()
            state_thread.start()

            # Optional camera loop
            camera_hz = self._resolve_camera_hz()
            if camera_hz > 0:
                cam_thread = threading.Thread(
                    target=self._camera_loop,
                    args=(camera_hz,),
                    name=f"mesh-camera-{self.peer_id}",
                    daemon=True,
                )
                self._threads.append(cam_thread)
                cam_thread.start()
                logger.info("[mesh] %s camera stream enabled @ %.1f Hz", self.peer_id, camera_hz)

            # Extended sensor loops (from SensorLoopsMixin)
            extended_loops = [
                ("pose", self._pose_loop),
                ("health", self._health_loop),
                ("imu", self._imu_loop),
                ("odom", self._odom_loop),
                ("lidar", self._lidar_loop),
                ("hand", self._hand_loop),
                ("map-info", self._map_info_loop),
            ]
            for loop_name, loop_fn in extended_loops:
                t = threading.Thread(target=loop_fn, name=f"mesh-{loop_name}-{self.peer_id}", daemon=True)
                self._threads.append(t)
                t.start()

            logger.info("[mesh] %s on mesh (%s)", self.peer_id, self.peer_type)

    def stop(self) -> None:
        """Stop all loops and release the session reference."""
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        with _LOCAL_ROBOTS_LOCK:
            _LOCAL_ROBOTS.pop(self.peer_id, None)

        with self._subs_lock:
            subs_to_drop = list(self._subs)
            self._subs.clear()
            self._user_subs.clear()
        with self._inbox_lock:
            self.inbox.clear()

        for sub in subs_to_drop:
            try:
                sub.undeclare()
            except Exception:
                pass

        # Undeclare any safety publishers we lazily declared so the
        # underlying Zenoh entity is released cleanly when the
        # process drops the last session reference. ``undeclare()``
        # is best-effort -- any failure here cannot recover state
        # (we are already in stop()) and would only mask a more
        # informative WARNING from the session teardown path.
        with self._safety_publishers_lock:
            pubs_to_drop = list(self._safety_publishers.values())
            self._safety_publishers.clear()
        for pub in pubs_to_drop:
            try:
                pub.undeclare()
            except (RuntimeError, OSError):
                logger.debug(
                    "[mesh] %s: safety publisher undeclare failed during stop()",
                    self.peer_id,
                )

        with self._rpc_lock:
            for ev in self._pending.values():
                ev.set()
            self._pending.clear()
            self._responses.clear()

        if self._has_session_ref:
            release_session()
            self._has_session_ref = False

        logger.info("[mesh] %s off mesh", self.peer_id)

    @property
    def alive(self) -> bool:
        return self._running

    @property
    def peers(self) -> list[dict[str, Any]]:
        return [p for p in _session_get_peers() if p.get("peer_id") != self.peer_id]

    @property
    def peers_by_id(self) -> dict[str, dict[str, Any]]:
        """Peers keyed by ``peer_id`` for dict-style lookup.

        Complements :attr:`peers` (a ``list[dict]``). README pseudo-code used
        ``mesh.peers[peer_id]`` expecting dict access; on the list that raises
        ``TypeError`` (GH #373 friction #8). Use this for O(1) lookup::

            info = robot.mesh.peers_by_id[other.peer_id]

        or the :meth:`get_peer` helper for a ``None``-safe single lookup.
        """
        return {p["peer_id"]: p for p in self.peers if "peer_id" in p}

    def get_peer(self, peer_id: str) -> dict[str, Any] | None:
        """Return a single peer's info dict by ``peer_id``, or ``None``.

        ``None``-safe counterpart to ``peers_by_id[peer_id]`` -- prefer this
        when the peer may not be present yet (discovery is asynchronous).
        """
        return self.peers_by_id.get(peer_id)

    # Presence — outgoing
    def _build_presence(self) -> dict[str, Any]:
        r = self.robot
        payload: dict[str, Any] = {
            "robot_id": self.peer_id,
            "robot_type": self.peer_type,
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        }

        try:
            if hasattr(r, "tool_name_str"):
                payload["tool_name"] = r.tool_name_str
        except Exception:
            pass

        try:
            ts = getattr(r, "_task_state", None)
            if ts is not None:
                status = getattr(ts, "status", None)
                payload["task_status"] = getattr(status, "value", status)
                payload["instruction"] = getattr(ts, "instruction", "")
        except Exception:
            pass

        try:
            inner = getattr(r, "robot", None)
            if inner is not None:
                if hasattr(inner, "is_connected"):
                    payload["connected"] = bool(inner.is_connected)
                if hasattr(inner, "name"):
                    payload["hw"] = inner.name
                cam_cfg = getattr(getattr(inner, "config", None), "cameras", None)
                if isinstance(cam_cfg, dict) and cam_cfg:
                    payload["cameras"] = list(cam_cfg.keys())
                input_pubs = getattr(r, "_input_publishers", None)
                if isinstance(input_pubs, dict) and input_pubs:
                    payload["inputs"] = [
                        {"device": name, "method": pub.method, "hz": pub.hz}
                        for name, pub in input_pubs.items()
                        if pub._running
                    ]
        except Exception:
            pass

        try:
            action_features = getattr(r, "_action_features", None)
            if isinstance(action_features, dict):
                payload["action_keys"] = list(action_features.keys())
        except Exception:
            pass

        try:
            world = getattr(r, "_world", None)
            if world is not None:
                payload["world"] = True
                world_robots = getattr(world, "robots", None)
                if isinstance(world_robots, dict):
                    payload["sim_robots"] = list(world_robots.keys())
        except Exception:
            pass

        # Advertise available extended topics
        available_topics: list[str] = []
        try:
            if (
                getattr(r, "_pose", None) is not None
                or getattr(r, "_slam_pose", None) is not None
                or getattr(r, "_odom_pose", None) is not None
            ):
                available_topics.append("pose")
            if getattr(r, "_imu", None) is not None:
                available_topics.append("imu")
            if getattr(r, "_odom", None) is not None:
                available_topics.append("odom")
            if getattr(r, "_lidar_summary", None) is not None or getattr(r, "_lidar_state", None) is not None:
                available_topics.append("lidar")
            if getattr(r, "_battery", None) is not None:
                available_topics.append("health")
            if getattr(r, "_hands", None) is not None:
                available_topics.append("hand")
            if getattr(r, "_map_info", None) is not None:
                available_topics.append("map")
        except Exception:
            pass
        if "health" not in available_topics:
            available_topics.append("health")
        if available_topics:
            payload["topics"] = available_topics

        return payload

    def _heartbeat_loop(self) -> None:
        period = 1.0 / HEARTBEAT_HZ
        while self._running:
            try:
                self.publish(f"strands/{self.peer_id}/presence", self._build_presence())
                prune_peers()
            except Exception as exc:
                logger.debug("[mesh] %s: heartbeat tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _on_presence(self, sample: Any) -> None:
        """Handle a peer's presence broadcast.

        Identity, fleet membership, and replay protection are enforced
        at the Zenoh transport: a sample reaching this callback has
        already cleared mTLS handshake + ACL, so its peer-id is
        cryptographically bound to the cert CN. We only parse the
        payload, update our peer registry, and log a debug line for
        first-sighting.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            # narrow per AGENTS.md > "Exception
            # Clauses Must Be Narrow". Same tuple as the four other
            # wire-input handlers (_on_cmd, _on_response,
            # _on_safety_estop, _on_safety_resume).
            # Pin: tests/mesh/test_wire_handler_narrow_except.py
            return
        if not isinstance(data, dict):
            return
        peer_id = data.get("robot_id")
        if not isinstance(peer_id, str) or peer_id == self.peer_id:
            return

        # M-3: presence-freshness check. Presence heartbeats carry a
        # ``timestamp`` (wall clock at publish). Previously _on_presence parsed
        # and stored the peer with NO freshness validation, so a captured
        # heartbeat with a 60s / 300s-old timestamp was accepted into the peer
        # registry on replay -- letting an attacker resurrect a dead peer or
        # inject phantoms with stale envelopes. Reject heartbeats whose
        # timestamp is older than the freshness window or implausibly
        # future-skewed. Heartbeats without a numeric timestamp are also
        # rejected (the publisher always sets one; a missing/garbage value is
        # either a malformed or hand-crafted replay envelope). Reuses the
        # safety-replay freshness/skew env knobs so operators tune one set of
        # clock-drift bounds for the whole mesh.
        _ts = data.get("timestamp")
        if not isinstance(_ts, (int, float)) or isinstance(_ts, bool):
            logger.debug("[mesh] %s: presence from %s missing/invalid timestamp -- dropped", self.peer_id, peer_id)
            return
        _now = time.time()
        _age = _now - float(_ts)
        _fresh = _resume_freshness_window_s()
        _skew = _resume_forward_skew_s()
        if _age > _fresh or _age < -_skew:
            logger.debug(
                "[mesh] %s: stale/future presence from %s (age=%.1fs, window=%.0fs) -- dropped",
                self.peer_id,
                peer_id,
                _age,
                _fresh,
            )
            return

        is_new = update_peer(
            peer_id=peer_id,
            peer_type=str(data.get("robot_type", "robot")),
            hostname=str(data.get("hostname", "")),
            caps=data,
        )
        if is_new:
            logger.info("[mesh] new peer: %s (%s)", peer_id, data.get("robot_type", "?"))

    # State — outgoing
    def _state_loop(self) -> None:
        period = 1.0 / STATE_HZ
        while self._running:
            try:
                state = self._read_state()
                if state:
                    self.publish(f"strands/{self.peer_id}/state", state)
            except Exception as exc:
                logger.debug("[mesh] %s: state tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _read_state(self) -> dict[str, Any] | None:
        r = self.robot
        snapshot: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}

        try:
            inner = getattr(r, "robot", None)
            if inner is not None and hasattr(inner, "get_observation") and getattr(inner, "is_connected", False):
                obs = inner.get_observation()
                cam_keys = set(getattr(getattr(inner, "config", None), "cameras", {}).keys())
                joints: dict[str, Any] = {}
                for key, value in obs.items():
                    if key in cam_keys:
                        continue
                    shape = getattr(value, "shape", None)
                    if shape is not None and len(shape) > 1:
                        continue
                    if hasattr(value, "tolist"):
                        joints[key] = value.tolist()
                    else:
                        joints[key] = value
                if joints:
                    snapshot["joints"] = joints
        except Exception:
            pass

        try:
            ts = getattr(r, "_task_state", None)
            if ts is not None:
                status = getattr(ts, "status", None)
                snapshot["task"] = {
                    "status": getattr(status, "value", status),
                    "instruction": getattr(ts, "instruction", ""),
                    "steps": getattr(ts, "step_count", 0),
                    "duration": getattr(ts, "duration", 0.0),
                }
        except Exception:
            pass

        try:
            world = getattr(r, "_world", None)
            if world is not None:
                world_data = getattr(world, "_data", None)
                if world_data is not None and hasattr(world_data, "time"):
                    snapshot["sim_time"] = float(world_data.time)
                world_robots = getattr(world, "robots", None)
                if isinstance(world_robots, dict):
                    snapshot["robots"] = {name: {"active": True} for name in world_robots}
        except Exception:
            pass

        return snapshot if len(snapshot) > 2 else None

    # Cameras — outgoing (opt-in)
    def _resolve_camera_hz(self) -> float:
        env = os.getenv("STRANDS_MESH_CAMERA_HZ")
        if env is None or env.strip() == "":
            hz = CAMERA_HZ
        else:
            try:
                hz = float(env)
            except ValueError:
                logger.warning("STRANDS_MESH_CAMERA_HZ=%r invalid; camera loop disabled", env)
                return 0.0
        return hz if hz > 0 else 0.0

    def _camera_loop(self, hz: float) -> None:
        period = 1.0 / hz
        while self._running:
            try:
                self._publish_cameras_once()
            except Exception as exc:
                logger.debug("[mesh] %s: camera tick error: %s", self.peer_id, exc)
            if self._stop_event.wait(period):
                break

    def _publish_cameras_once(self) -> None:
        # Privacy kill switch. Operators on sensitive deployments set
        # STRANDS_MESH_CAMERA_DISABLED=true to short-circuit the camera
        # loop entirely -- no frames built, no envelopes signed, nothing
        # published.
        # Lenient bool parsing matches the rest of the env-var surface
        # (STRANDS_MESH_MULTICAST, STRANDS_MESH_I_KNOW_THIS_IS_INSECURE).
        # Operators using ``=1`` / ``=yes`` / ``=on`` get the same
        # behaviour as ``=true``; bad values fail-loud rather than
        # silently re-enabling camera publishing on a privacy flag.
        from strands_robots.mesh._zenoh_config import _bool_env as _zc_bool_env  # type: ignore[import-untyped]

        if _zc_bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False):
            return
        r = self.robot
        inner = getattr(r, "robot", None)
        if inner is None or not getattr(inner, "is_connected", False):
            return
        cam_cfg = getattr(getattr(inner, "config", None), "cameras", None)
        if not isinstance(cam_cfg, dict) or not cam_cfg:
            return

        obs = None
        try:
            obs = inner.get_observation()
        except Exception:
            pass

        if obs is None:
            cameras_dict = getattr(inner, "cameras", None)
            if not isinstance(cameras_dict, dict) or not cameras_dict:
                return
            obs = {}
            for cam_name, cam_obj in cameras_dict.items():
                try:
                    if hasattr(cam_obj, "async_read"):
                        obs[cam_name] = cam_obj.async_read()
                    elif hasattr(cam_obj, "read"):
                        obs[cam_name] = cam_obj.read()
                except Exception:
                    pass
            if not obs:
                return

        try:
            import cv2

            have_cv2 = True
        except Exception:
            have_cv2 = False

        for cam_name in cam_cfg:
            try:
                frame = obs.get(cam_name)
                if frame is None:
                    continue
                shape = getattr(frame, "shape", None)
                if shape is None or len(shape) < 2:
                    continue
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                if hasattr(frame, "astype"):
                    import numpy as np

                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)

                if have_cv2:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
                    encoding = "jpeg"
                else:
                    encoded = base64.b64encode(bytes(frame)).decode("ascii")
                    encoding = "raw"

                self.publish(
                    f"strands/{self.peer_id}/camera/{cam_name}",
                    {
                        "peer_id": self.peer_id,
                        "cam": cam_name,
                        "t": time.time(),
                        "shape": list(shape),
                        "dtype": "uint8",
                        "encoding": encoding,
                        "data": encoded,
                    },
                )
            except Exception as exc:
                logger.debug("[mesh] %s: camera %s publish failed: %s", self.peer_id, cam_name, exc)

    # RPC — incoming
    def _on_cmd(self, sample: Any) -> None:
        """Handle an inbound command sample.

        The Zenoh transport has already enforced:

        * mTLS peer identity (the sender's cert CN is bound to the link).
        * ACL -- **when the operator supplies ``STRANDS_MESH_ACL_FILE``
          with role separation, only peers in the ``operator_peer``
          subject can publish on ``cmd`` / ``broadcast`` topics**. The
          default ``default_acl()`` is permissive (any CA-signed peer
          may publish/subscribe on any key) -- see CHANGELOG.md Section 8.
        * Per-key-expression frequency cap (``downsampling`` block) --
          floods are dropped pre-deserialise.
        * Per-message size cap (``low_pass_filter`` block) -- jumbo
          frames are dropped pre-deserialise.

        We only have to parse the payload and dispatch.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        sender_id = data.get("sender_id", "")
        if sender_id == self.peer_id:
            return
        threading.Thread(
            target=self._exec_cmd,
            args=(data,),
            name=f"mesh-exec-{self.peer_id}",
            daemon=True,
        ).start()

    def _exec_cmd(self, data: dict[str, Any]) -> None:
        sender = data.get("sender_id", "")
        # full 128-bit fallback. Pre-fix, an inbound command without
        # turn_id triggered a 32-bit hex which was birthday-colliding under
        # heavy concurrent load and cheap to predict for an attacker who
        # could observe the response topic. D1 closed the outbound side;
        # this closes the symmetric receive-side surface.
        turn = data.get("turn_id") or uuid.uuid4().hex
        # require an explicit ``command`` key.
        # Earlier the fallback ``data.get("command", data)`` allowed a
        # peer to publish a flat-shape envelope (sender_id, turn_id,
        # action, instruction, policy_provider all at top level) and
        # have ``data`` itself treated as the command. Earlier revisions rejected
        # bare-string non-dict commands; this closes the symmetric
        # flat-dict-envelope shape -- the wire contract REQUIRES a
        # ``command`` field whose value is a dict.
        cmd = data.get("command")
        # Reject non-dict commands at the wire boundary. A bare-string
        # coercion here would bypass validate_command's dict-shape contract --
        # any peer that survives mTLS+ACL could drive the robot at the mock
        # policy with arbitrary text simply by publishing "hello" instead of
        # {"action":"execute",...}. Outgoing send/broadcast/tell still accept
        # the ergonomic dict-or-string forms because tell() wraps internally.
        #
        # F-15 / B-09: include the responder's own peer_id as a topic
        # segment so the IoT robot policy can scope publish to
        # ``strands/+/response/${ThingName}/*`` -- a robot can only
        # publish responses tagged with its OWN ThingName, closing the
        # cross-robot response-spoof surface. The requester subscribes
        # with ``response/**`` so the extra segment matches. Operator
        # prefix (``{sender}``) is unchanged so routing is preserved.
        rkey = f"strands/{sender}/response/{self.peer_id}/{turn}" if sender else None
        if cmd is None or not isinstance(cmd, dict):
            # route non-dict envelope rejection
            # through the same audit + wire-response path as
            # ValidationError. Earlier this branch was silent on the
            # wire and in the audit log -- asymmetric with how every
            # other validation rejection produces a structured error
            # response and a forensic record.
            logger.warning(
                "[mesh] %s: rejected non-dict cmd from %s (type=%s)",
                self.peer_id,
                sender,
                type(cmd).__name__,
            )
            if rkey is not None:
                self.publish(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": "validation: command must be a dict with explicit `command` key",
                        "timestamp": time.time(),
                    },
                )
            try:
                log_safety_event(
                    "command_rejected",
                    self.peer_id,
                    {
                        "sender": sender,
                        "reason": "non-dict envelope or missing command key",
                        "type": type(cmd).__name__,
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return

        # Validate the command shape against the action allowlist + per-action
        # schema (instruction length, duration bounds, policy_host allowlist,...).
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: rejected invalid cmd from %s: %s", self.peer_id, sender, exc)
            if rkey is not None:
                self.publish(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": f"validation: {exc}",
                        "timestamp": time.time(),
                    },
                )
            try:
                log_safety_event(
                    "command_rejected",
                    self.peer_id,
                    {
                        "sender": sender,
                        "reason": str(exc),
                        "action": cmd.get("action") if isinstance(cmd, dict) else None,
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                # narrow per AGENTS.md > Review
                # Learnings (#86). ``log_safety_event`` raises TypeError
                # / ValueError on payload shape, OSError on disk failure;
                # the audit best-effort contract means we drop those, but
                # an unexpected RuntimeError from a future audit-module
                # refactor should NOT be silently swallowed.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return

        # H-3: reject replayed commands. Read-only actions (status / state /
        # features) are idempotent and safe to repeat (operator polling), so
        # we only dedup actuating actions. A duplicate (sender, turn_id) within
        # the TTL window is dropped with a structured error + audit record,
        # mirroring the LockoutError rejection path. turn_id defaults to a
        # fresh uuid when absent, so a sender that omits it cannot benefit
        # from dedup-bypass: each omitted-turn_id command gets a unique key
        # and is treated as new (the wire contract expects callers to send a
        # turn_id; the fallback exists only so a malformed envelope doesn't
        # crash dispatch).
        _action = cmd.get("action", "status") if isinstance(cmd, dict) else "status"
        _READONLY = {"status", "state", "features"}
        if _action not in _READONLY:
            _now_mono = time.monotonic()
            _key = (sender, turn)
            _is_replay = False
            with self._cmd_replay_lock:
                _ttl = _resume_freshness_window_s() + _resume_forward_skew_s()
                _evict_replay_cache(
                    self._cmd_replay_cache,
                    max_size=_resume_replay_cache_max(),
                    ttl_s=_ttl,
                    now_mono=_now_mono,
                )
                if _key in self._cmd_replay_cache:
                    _is_replay = True
                else:
                    self._cmd_replay_cache[_key] = _now_mono
            if _is_replay:
                logger.warning(
                    "[mesh] %s: rejected replayed cmd from %s (turn_id=%s, action=%s)",
                    self.peer_id,
                    sender,
                    turn,
                    _action,
                )
                if rkey is not None:
                    self.publish(
                        rkey,
                        {
                            "type": "error",
                            "responder_id": self.peer_id,
                            "turn_id": turn,
                            "error": "duplicate command rejected (replay)",
                            "timestamp": time.time(),
                        },
                    )
                try:
                    log_safety_event(
                        "command_rejected_replay",
                        self.peer_id,
                        {"sender": sender, "turn_id": turn, "action": _action},
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
                return

        try:
            result = self._dispatch(cmd)
            if rkey is not None:
                self.publish(
                    rkey,
                    {
                        "type": "response",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "result": result,
                        "timestamp": time.time(),
                    },
                )
            # M-5 ("Negative-Only Audit Logging"): record
            # SUCCESSFUL command execution, not just rejections. Pre-fix the
            # audit log only ever held *_rejected / *_denied events, so the 6
            # exploit chains produced 0 combined audit records -- post-incident
            # forensics and real-time detection were impossible (a successful
            # actuation left no trace). We log only non-readonly actions to
            # avoid flooding the audit log with operator ``status`` polls; the
            # readonly set matches the H-3 dedup exemption. Best-effort: an
            # audit failure must never break the dispatch path (same narrow
            # except tuple as every other audit call site).
            if _action not in _READONLY:
                try:
                    log_safety_event(
                        "command_executed",
                        self.peer_id,
                        {"sender": sender, "turn_id": turn, "action": _action},
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
        except _security.LockoutError as exc:
            # Lockout is the most operationally interesting rejection -- emit
            # a structured error on the response topic and audit it.
            logger.warning("[mesh] %s: rejected during lockout from %s", self.peer_id, sender)
            if rkey is not None:
                self.publish(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
            try:
                log_safety_event(
                    "command_rejected_lockout",
                    self.peer_id,
                    {"sender": sender, "action": cmd.get("action") if isinstance(cmd, dict) else None},
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                # narrow per AGENTS.md > "Exception
                # Clauses Must Be Narrow". Same tuple as the symmetric
                # narrowing at the ValidationError audit path above.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
            return
        except (
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
            OSError,
            TypeError,
        ) as exc:
            # narrowed from bare ``except Exception``
            # per AGENTS.md > Review Learnings (#86). The original goal
            # ("any unhandled exception in a robot adapter would crash
            # the dispatch thread and silently kill the mesh") is
            # achievable with a narrow tuple: this catches every
            # realistic adapter failure (LeRobot raising RuntimeError,
            # GR00T raising ValueError on bad inputs, type mismatches,
            # missing keys, OSError from device I/O) but lets
            # ``MemoryError``, ``SystemExit``, ``KeyboardInterrupt``,
            # and any future programmer-error type that doesn't fit
            # this list propagate to the test harness / supervisor.
            #
            # The "static dispatch error string on the wire" rationale
            # below stays the same -- regardless of catch width, we
            # never leak internal exception detail to a remote caller.
            # The structured ValidationError / LockoutError paths above
            # remain the preferred channel for client-distinguishable
            # rejections.
            logger.warning(
                "[mesh] %s: dispatch error from %s: %s",
                self.peer_id,
                sender,
                exc,
                exc_info=True,
            )
            if rkey is not None:
                self.publish(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": "dispatch error",
                        "timestamp": time.time(),
                    },
                )
            # Audit the dispatch-error path so a remote prober cannot
            # silently fish for adapter exceptions without leaving a
            # forensic trail (issue #257). Reuses ``command_rejected``
            # event_type with reason="dispatch error" to keep the
            # operator audit-walker grep simple.
            try:
                log_safety_event(
                    "command_rejected",
                    self.peer_id,
                    {
                        "sender": sender,
                        "reason": "dispatch error",
                        "action": cmd.get("action") if isinstance(cmd, dict) else None,
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                # Wrap audit emission in narrow except so an audit-sink
                # failure on the dispatch-error path doesn't crash back
                # through the mesh wire handler we just narrowed in R24-A.
                logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)

    def _dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        action = cmd.get("action", "status")
        r = self.robot

        # While the emergency-stop lockout is engaged, only ``status`` and
        # ``resume`` are permitted. Raise so _exec_cmd handles the rejection
        # symmetrically with ValidationError -- emitting type="error" on the
        # response topic and recording an audit entry. The wire response is
        # intentionally generic so a remote caller cannot use it to map the
        # lockout window.
        if self._estop_lockout.is_set() and action not in ("status", "resume"):
            raise _security.LockoutError("command rejected")

        if action == "resume":
            return self._resume_lockout(cmd.get("override_code", ""))

        if action == "status":
            if hasattr(r, "get_task_status"):
                return dict(r.get_task_status())
            ts = getattr(r, "_task_state", None)
            return {"status": getattr(getattr(ts, "status", None), "value", "unknown")}
        if action == "stop":
            if hasattr(r, "stop_task"):
                return dict(r.stop_task())
            return {"ok": True}
        if action == "features":
            return dict(r.get_features()) if hasattr(r, "get_features") else {}
        if action == "state":
            return self._read_state() or {}
        if action in ("execute", "start"):
            instruction = cmd.get("instruction", "")
            if not instruction:
                return {"error": "instruction required"}
            policy_provider = cmd.get("policy_provider", "mock")
            policy_port = cmd.get("policy_port")
            policy_host = cmd.get("policy_host", "localhost")
            # Defence in depth: the wire path reaches _dispatch via
            # _exec_cmd -> validate_command, which already allowlist-checks
            # policy_host. This re-check guards any current or future caller
            # that reaches _dispatch without that upstream validation (a
            # direct internal call, a refactor that reorders the pipeline,
            # etc.). is_safe_policy_host is idempotent and cheap, so the
            # double-check costs nothing and the surface stays closed even
            # if the single upstream gate is ever bypassed.
            if not _security.is_safe_policy_host(str(policy_host)):
                return {
                    "error": (
                        f"policy_host={policy_host!r} not in allowlist. Set STRANDS_MESH_POLICY_HOST_ALLOW to extend."
                    )
                }
            duration = cmd.get("duration", 30.0)
            extra = {
                k: cmd[k]
                for k in ("model_path", "server_address", "policy_type", "pretrained_name_or_path")
                if k in cmd
            }
            # Sim peer? Route to Simulation.start_policy / run_policy.
            # Detected by the presence of the ``run_policy`` callable + a
            # ``_world`` attribute (the SimEngine ABC contract). HardwareRobot
            # has neither, so this is unambiguous.
            #
            # Forwards the well-known per-call kwargs from #300
            # (``target_pose`` / ``target_joints`` / ``world_update``) plus
            # the existing ``extra`` set (model_path / server_address / ...)
            # via ``policy_config``. ``create_policy(provider, **policy_config)``
            # passes them to the Policy constructor; per the #300 contract
            # planner-style providers consume them and VLA providers ignore
            # unknown kwargs without raising.
            if (
                action in ("execute", "start")
                and hasattr(r, "run_policy")
                and hasattr(r, "_world")
                and hasattr(r, "list_robots")
            ):
                return self._dispatch_sim_policy(
                    action=action,
                    cmd=cmd,
                    instruction=instruction,
                    policy_provider=policy_provider,
                    duration=duration,
                    extra=extra,
                )
            if action == "execute" and hasattr(r, "_execute_task_sync"):
                return dict(
                    r._execute_task_sync(instruction, policy_provider, policy_port, policy_host, duration, **extra)
                )
            if action == "start" and hasattr(r, "start_task"):
                return dict(r.start_task(instruction, policy_provider, policy_port, policy_host, duration, **extra))
        if action == "step" and hasattr(r, "step"):
            return dict(r.step(cmd.get("steps", 1)))
        if action == "reset" and hasattr(r, "reset"):
            return dict(r.reset())
        if action == "teleop_status":
            if hasattr(r, "get_teleop_status"):
                return dict(r.get_teleop_status())
            return {"inputs": [], "publishers": {}, "receivers": {}}
        if action == "teleop_receive":
            source = cmd.get("source_peer_id", "")
            dev = cmd.get("device_name", "leader")
            if not source:
                return {"error": "source_peer_id required"}
            if hasattr(r, "start_teleop_receive"):
                return dict(r.start_teleop_receive(source, dev))
            return {"error": "robot does not support teleop_receive"}
        if action == "teleop_stop":
            dev = cmd.get("device_name")
            if hasattr(r, "stop_teleop"):
                return dict(r.stop_teleop(dev))
            return {"error": "robot does not support stop_teleop"}
        return {"error": f"unknown action: {action}"}

    # Well-known per-call policy kwargs from issue #300 — keys that planner-
    # style providers (cuRobo, MoveIt2, MPC) consume to encode goals beyond
    # natural-language ``instruction``. Forwarded from ``tell()`` payload
    # into ``policy_config`` so a ``policy_provider="curobo"`` peer sees the
    # ``target_pose`` it needs without the dispatch layer dropping it
    # silently.
    #
    # See AGENTS.md > Public API Hygiene: "Forward all advertised kwargs
    # end-to-end. Silent drops are bugs masquerading as features."
    _SIM_WELL_KNOWN_POLICY_KWARGS: tuple[str, ...] = (
        "target_pose",
        "target_joints",
        "world_update",
    )

    def _dispatch_sim_policy(
        self,
        *,
        action: str,
        cmd: dict[str, Any],
        instruction: str,
        policy_provider: str,
        duration: float,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch ``execute`` / ``start`` to a sim peer's policy runner.

        Sim peers expose :meth:`SimEngine.run_policy` (blocking) and
        :meth:`SimEngine.start_policy` (async) instead of ``HardwareRobot``'s
        ``_execute_task_sync`` / ``start_task``. This helper bridges the
        ``tell()`` wire payload to those methods.

        ``robot_name`` resolution:
            * Use ``cmd["robot_name"]`` when present.
            * Otherwise default to the only robot in the world if there is
              exactly one.
            * Otherwise return an error — ambiguous targets must be
              explicit so the agent can't accidentally drive the wrong arm.

        Forwards both the existing ``extra`` constructor kwargs
        (``model_path``, ``server_address``, ``policy_type``,
        ``pretrained_name_or_path``) and the issue #300 well-known per-call
        kwargs (``target_pose``, ``target_joints``, ``world_update``) via
        ``policy_config``. Per #300 the receiving Policy ignores unknown
        kwargs rather than raising, so VLA providers stay compatible.
        """
        sim = self.robot

        if sim._world is None:
            return {"error": "sim peer has no world; call create_world first"}

        try:
            available = list(sim.list_robots())
        except Exception as exc:  # noqa: BLE001 — surface as wire-level error
            return {"error": f"sim list_robots failed: {exc}"}

        robot_name = cmd.get("robot_name")
        if not robot_name:
            if len(available) == 1:
                robot_name = available[0]
            elif len(available) == 0:
                return {"error": "sim peer has no robots; add_robot first"}
            else:
                return {
                    "error": (
                        f"sim peer has {len(available)} robots {available}; "
                        "tell() must include robot_name to disambiguate"
                    )
                }
        if robot_name not in available:
            return {"error": f"robot_name={robot_name!r} not in sim (available: {available})"}

        # Build policy_config: existing constructor kwargs + well-known
        # per-call kwargs from #300. ``policy_config`` is the documented
        # passthrough on SimEngine.run_policy/start_policy.
        policy_config: dict[str, Any] = dict(extra)
        for key in self._SIM_WELL_KNOWN_POLICY_KWARGS:
            if key in cmd:
                policy_config[key] = cmd[key]

        # Optional sim-side controls. We expose only the fields that already
        # have validator coverage in the wire schema — control_frequency,
        # action_horizon, fast_mode, video — and that are safe to forward to
        # an LLM-issued ``tell()``. Anything else stays on its server-side
        # default to avoid surfacing internal knobs to untrusted agents.
        run_kwargs: dict[str, Any] = {
            "policy_provider": policy_provider,
            "policy_config": policy_config,
            "instruction": instruction,
            "duration": duration,
        }
        for opt_key in ("control_frequency", "action_horizon", "fast_mode", "n_steps"):
            if opt_key in cmd:
                run_kwargs[opt_key] = cmd[opt_key]

        # ``execute`` blocks until the rollout finishes (matches HardwareRobot
        # ``_execute_task_sync`` semantics). ``start`` returns immediately
        # with a future-tracking ack (matches ``start_task``).
        if action == "execute":
            return dict(sim.run_policy(robot_name, **run_kwargs))
        # action == "start"
        return dict(sim.start_policy(robot_name, **run_kwargs))

    def _on_response(self, sample: Any) -> None:
        """Inbound response handler.

        Identity, fleet membership, and topic ACL have already been
        enforced at the Zenoh transport. We additionally apply a
        point-to-point scope check: a response is accepted only if its
        ``responder_id`` matches the expected target recorded in
        :attr:`_expected_responders` by :meth:`send`. Broadcast turns
        use the ``BROADCAST_RESPONDER`` sentinel and accept any
        responder_id -- that is the broadcast contract.

        Without the responder-id check, an ACL-authorised peer that
        observes a turn_id (a fellow operator) could publish a response
        on someone else's pending turn and have the sender accept its
        ``result`` instead of the legitimate target's. The transport
        ACL prevents an attacker from joining at all; this check
        prevents lateral mischief between authorised peers.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        turn = data.get("turn_id")
        if not isinstance(turn, str):
            return
        responder = data.get("responder_id")
        with self._rpc_lock:
            event = self._pending.get(turn)
            if event is None:
                return
            expected = self._expected_responders.get(turn)
            # Strict scoping for point-to-point sends. Broadcast accepts any.
            if expected is not None and expected != BROADCAST_RESPONDER and responder != expected:
                # structured forensic event via the audit log
                # (``response_hijack_rejected``) plus a WARNING line is
                # the operator-and-forensic channel; an earlier draft
                # also raised-and-caught a typed exception around the
                # same code, but with no real consumer it was YAGNI
                # scaffolding and got removed.
                logger.warning(
                    "[mesh] %s: dropped response on turn %s -- "
                    "responder_id=%r does not match expected target %r "
                    "(possible response hijack)",
                    self.peer_id,
                    turn[:12],
                    responder,
                    expected,
                )
                try:
                    log_safety_event(
                        "response_hijack_rejected",
                        self.peer_id,
                        {
                            "turn_prefix": turn[:12],
                            "responder_id": responder,
                            "expected": expected,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # narrow per AGENTS.md > Review
                    # Learnings (#86). Same tuple as the other audit-publish
                    # wrappers in this file. Audit best-effort still holds;
                    # MemoryError / RuntimeError / future programmer errors
                    # propagate to the test harness instead of being
                    # silently swallowed at DEBUG.
                    logger.debug("[mesh] %s: audit log unavailable: %s", self.peer_id, audit_exc)
                return
            self._responses.setdefault(turn, []).append(data)
        event.set()

    # Safety -- inbound estop / resume
    def _on_safety_estop(self, sample: Any) -> None:
        """Engage the local emergency-stop lockout in response to a fleet-
        wide ``strands/safety/estop`` broadcast.

        Wire authentication (mTLS + ACL) admits this handler. **When the
        operator supplies an ``STRANDS_MESH_ACL_FILE`` with role
        separation (template at ``examples/mesh_acl_example.json5``),
        only peers in the ``operator_peer`` subject can publish on
        ``safety/**``.** The default ACL shipped by ``default_acl()`` is
        permissive (CHANGELOG.md Section 8 -- "any CA-signed peer may
        publish/subscribe on any key"), so any cert-holding peer can
        originate an estop on out-of-the-box deployments.

        Defense-in-depth -- captured-envelope replay protection. Even with an unrestricted ACL, a
        replay of a captured ``safety/estop`` envelope cannot keep the
        fleet locked indefinitely.  Mirrors :meth:`_on_safety_resume`:

        1. Freshness window (``_resume_freshness_window_s()``) -- envelopes
           older than the window are rejected.
        2. Forward-skew bound (``_resume_forward_skew_s()``) -- envelopes
           timestamped beyond the tolerance in the future are rejected
           (defeats clock-rollback attacks against the freshness check).
        3. Per-receiver replay cache keyed on ``float(envelope_t)`` -- bounded LRU at ``_resume_replay_cache_max()`` entries.
           Keyed on ``t`` alone (NOT ``(issuer_id, t)``) so an attacker
           who captures one envelope cannot replay it by varying the
           payload ``peer_id`` field, which is untrusted (comes from the
           JSON body, not the TLS cert CN).

        E-stop without an envelope ``t`` OR without a valid string
        ``peer_id`` is rejected as malformed (the canonical
        :meth:`emergency_stop` issuer always sets both).
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        # Cache operator-tunable freshness/skew knobs once at handler entry
        # (issue #265). Reading them per-use parsed os.getenv plus a regex
        # validation on every reference (5-6 times per envelope) and could
        # observe a mid-handler env mutation, creating an internal
        # inconsistency window. The 0.2s estop corroboration window is
        # timing-sensitive, so we also keep these reads out of the
        # _estop_replay_lock critical section. The next envelope picks up a
        # changed env value, preserving the operator-tunable contract.
        forward_skew_s = _resume_forward_skew_s()
        freshness_window_s = _resume_freshness_window_s()

        # Wire-level publisher attribution (cross-session forgery defence).
        # When the sample carries a ``source_info.source_id.zid`` set by
        # Zenoh during the mTLS-bootstrapped session handshake AND the
        # body advertises a ``source_zid`` field, the two MUST agree.
        # An attacker on a different mTLS session cannot make
        # ``sample.source_info.source_id.zid`` point at a peer's session
        # because zenoh-python exposes no public ``ZenohId`` constructor;
        # the value is forced to whatever the publishing session bootstrapped
        # to under ``connect.tls``. Bridge/IoT transports do not propagate
        # ``source_info`` -- in that case both fields are absent and we
        # fall back to the body-level HMAC-bind defences below.
        wire_zid = _extract_sample_source_zid(sample)
        body_zid = data.get("source_zid")
        if wire_zid is not None and body_zid is not None:
            if not isinstance(body_zid, str) or wire_zid != body_zid:
                logger.warning(
                    "[safety] %s: refusing remote estop -- body source_zid does not "
                    "match TLS-bound wire source_zid (cross-session forgery rejected)",
                    self.peer_id,
                )
                return
        elif wire_zid is None and body_zid is not None:
            # Body advertises a zid but the wire does not -- a mTLS
            # peer that forgot to attach SourceInfo, or a transport
            # that dropped it. Treat as malformed and reject; the
            # canonical issuer always pairs the two.
            logger.warning(
                "[safety] %s: refusing remote estop -- body source_zid present but wire "
                "source_zid absent (publisher misconfigured or attacker stripped SourceInfo)",
                self.peer_id,
            )
            return
        elif wire_zid is not None and body_zid is None:
            # Wire carries a zid but the body does not -- a publisher
            # from a pre-binding mesh version. Reject so we never
            # downgrade silently to the body-only HMAC binding when
            # the wire-level binding is available. Operators upgrade
            # all peers together.
            logger.warning(
                "[safety] %s: refusing remote estop -- wire source_zid present but body "
                "source_zid absent (publisher predates source_zid binding; upgrade required)",
                self.peer_id,
            )
            return

        # Freshness + replay defences. An estop envelope without ``t`` is
        # not from a canonical issuer -- reject (also closes the trivial
        # replay surface where an attacker strips ``t`` to bypass the
        # freshness check).
        envelope_t = data.get("t")
        now = time.time()
        if not isinstance(envelope_t, (int, float)):
            logger.warning(
                "[safety] %s: refusing remote estop -- envelope missing/invalid ``t``",
                self.peer_id,
            )
            return
        if envelope_t > now + forward_skew_s:
            logger.warning(
                "[safety] %s: refusing remote estop -- ``t``=%s in future (forward_skew_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                forward_skew_s,
                now,
            )
            return
        if (now - envelope_t) > freshness_window_s:
            logger.warning(
                "[safety] %s: refusing remote estop -- ``t``=%s too old (freshness_window_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                freshness_window_s,
                now,
            )
            return

        # reject envelopes with missing/empty ``peer_id`` outright
        # rather than coalescing to ``<unknown>``. The canonical
        # :meth:`emergency_stop` issuer always sets ``peer_id``; a
        # malformed envelope is either a programming bug or an attacker
        # probing the cache. Coalescing to a shared bucket let one
        # attacker poison the slot for legitimate operators.
        issuer_id = data.get("peer_id")
        if not isinstance(issuer_id, str) or not issuer_id:
            logger.warning(
                "[safety] %s: refusing remote estop -- envelope missing/invalid ``peer_id``",
                self.peer_id,
            )
            return

        # cache key is keyed on ``float(envelope_t)`` ALONE -- not
        # ``(issuer_id, t)``. The previous (issuer, t) key let an
        # attacker who captured one valid envelope replay it
        # indefinitely by varying the payload ``peer_id`` (which is
        # untrusted -- it comes from the JSON body, not the TLS cert
        # CN). Keying on the wall-clock ``t`` alone closes that
        # peer_id-permutation surface; the only way to mint a new key
        # is to advance the timestamp, which is bounded by the
        # freshness window above. A per-issuer slot cap
        # below to bound the denial-of-estop surface where one
        # attacker pre-publishes ``t = now + skew - eps`` to occupy
        # cache slots that legitimate same-float-tick estops would

        # (post-replay-cache: see the per-issuer denial-of-estop discussion).
        cache_key = float(envelope_t)
        # Per-issuer fairness bound: one issuer may occupy at
        # most ``per_issuer_cap`` slots so a single attacker cannot
        # fill the global cache. Default cap is _resume_replay_cache_max()
        # / 4 -- four legitimate operators always have working slots.
        per_issuer_cap = max(1, _resume_replay_cache_max() // 4)
        # cache TTL bookkeeping uses time.monotonic() so an NTP step
        # backward cannot leave entries un-evictable and a step forward
        # cannot age fresh entries out early. Envelope freshness still
        # uses time.time() above (it must compare against the issuer's
        # wall-clock).
        now_mono = time.monotonic()
        with self._estop_replay_lock:
            if cache_key in self._estop_replay_cache:
                # Corroboration vs replay disambiguation gated on the
                # TLS-bound wire ``source_zid``. The previous heuristic
                # ("lockout active + within 0.2s -> corroboration") was
                # forgeable: a same-session attacker who captured a
                # legitimate envelope could republish it within 200 ms
                # with a mutated body ``peer_id`` and earn an
                # ``estop_corroborated`` audit (severity ``info``,
                # operator-dashboard-invisible) instead of
                # ``estop_replay_rejected`` (severity ``warning``).
                #
                # The cache value tuple now carries the wire_zid in
                # effect when the slot was first populated. A second
                # envelope is treated as legitimate cross-session
                # corroboration ONLY IF:
                #   * the cached wire_zid is non-None (slot was
                #     established by a TLS-bound publisher),
                #   * the new wire_zid is non-None,
                #   * the two zids differ (two distinct mTLS
                #     sessions -> two distinct operators).
                # Same-zid replay -- including the
                # mutated-peer_id case -- audits as
                # ``estop_replay_rejected`` per the original threat
                # model. Bridge / IoT transports that legitimately have
                # no SourceInfo (wire_zid is ``None`` on either side)
                # also fall into the rejection branch: corroboration
                # over an attribution-less transport cannot be proven.
                cached_entry = self._estop_replay_cache[cache_key]
                # Cache values are always ``(issuer_id, mono_ts, wire_zid)``
                # 3-tuples. The type annotation at __init__ enforces this
                # shape and the only writer (line ~1601) emits it. No
                # defensive isinstance -- half-defensive code disagrees
                # with ts_view (line ~1545) and per-issuer iteration
                # (line ~1570) which both assume the 3-tuple shape.
                cached_wire_zid = cached_entry[2]
                wire_zids_distinct = (
                    cached_wire_zid is not None and wire_zid is not None and cached_wire_zid != wire_zid
                )
                if (
                    wire_zids_distinct
                    and self._estop_lockout.is_set()
                    and (time.monotonic() - self._last_estop_mono) < 0.2
                ):
                    try:
                        self.publish_safety_event(
                            event_type="estop_corroborated",
                            severity="info",
                            payload={
                                "issuer": issuer_id,
                                "issuer_t": envelope_t,
                                "wire_zid": wire_zid,
                                "corroborates_wire_zid": cached_wire_zid,
                            },
                        )
                    except (TypeError, ValueError, OSError) as audit_exc:
                        # Narrow per AGENTS.md > "Exception Clauses Must Be Narrow".
                        # publish_safety_event raises only TypeError/ValueError
                        # on payload shape and OSError on disk failure; everything
                        # else is a programmer bug worth seeing.
                        logger.debug(
                            "[mesh] %s: estop_corroborated audit publish failed: %s",
                            self.peer_id,
                            audit_exc,
                        )
                    return
                # Original replay rejection (now also covers same-wire-zid
                # mutated-peer_id replays and attribution-less transports).
                logger.warning(
                    "[safety] %s: REJECTED remote estop -- replay of (issuer=%s, t=%s) already accepted",
                    self.peer_id,
                    issuer_id,
                    envelope_t,
                )
                try:
                    self.publish_safety_event(
                        event_type="estop_replay_rejected",
                        severity="warning",
                        payload={"issuer": issuer_id, "issuer_t": envelope_t},
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # Audit publish is best-effort and must never block the
                    # safety path; narrow the catch tuple so a future
                    # programmer error surfaces in tests rather than
                    # being swallowed at DEBUG.
                    logger.debug(
                        "[mesh] %s: estop_replay_rejected audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
                return
            # evict using the tuple-valued cache. We extract a
            # mono_ts view, run the standard eviction, then re-key
            # the surviving entries from the original cache.
            ts_view: dict[float, float] = {k: v[1] for k, v in self._estop_replay_cache.items()}
            _evict_replay_cache(
                ts_view,
                max_size=_resume_replay_cache_max(),
                # include forward_skew so a forward-skewed envelope
                # at t=now+skew stays cached for the full freshness window
                # rather than the lesser ``freshness`` only.
                ttl_s=freshness_window_s + forward_skew_s,
                now_mono=now_mono,
            )
            # Apply the eviction back to the real cache.
            for evicted in set(self._estop_replay_cache.keys()) - set(ts_view.keys()):
                self._estop_replay_cache.pop(evicted, None)

            # Per-issuer fairness check derived from cache contents.
            # No separate dict that drifts -- count entries owned by
            # ``issuer_id`` directly. After eviction this is naturally
            # correct: an attacker who flooded their cap and waited for
            # eviction now has fewer entries (eviction dropped them) and
            # can reclaim slots, which is the intended dynamic-attacker
            # rate-limit. A sustained attacker who paces floods to land
            # just after each eviction is bounded by ``per_issuer_cap``
            # at every instant -- they never hold more than that fraction
            # of the global cache, so legitimate operators always have
            # ``_resume_replay_cache_max() - per_issuer_cap`` slots available.
            issuer_slots = sum(1 for issuer, _mono, _zid in self._estop_replay_cache.values() if issuer == issuer_id)
            if issuer_slots >= per_issuer_cap:
                logger.warning(
                    "[safety] %s: REFUSED estop cache slot -- issuer %r already at cap %d "
                    "(per-issuer fairness bound; flood suspected)",
                    self.peer_id,
                    issuer_id,
                    per_issuer_cap,
                )
                # Audit the over-cap rejection so an operator dashboard
                # can alert on this. The replay-cache slot is NOT added,
                # but the lockout below still engages -- a legitimate
                # safety event is preserved even if the cache itself
                # cannot hold it.
                try:
                    self.publish_safety_event(
                        event_type="estop_per_issuer_cap_exceeded",
                        severity="warning",
                        payload={
                            "issuer": issuer_id,
                            "issuer_t": envelope_t,
                            "cap": per_issuer_cap,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    logger.debug(
                        "[mesh] %s: estop_per_issuer_cap_exceeded audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
            else:
                self._estop_replay_cache[cache_key] = (issuer_id, now_mono, wire_zid)

            # Lockout state mutation must be inside _estop_replay_lock
            # to close the concurrent-estops race (issue #273): two
            # invocations from distinct issuers could both pass the
            # is_set() check before either calls set() and both would
            # publish remote_estop_engaged instead of one + one
            # remote_estop_redundant. Mutating + reading
            # _last_estop_ts/_last_estop_mono inside the lock also
            # prevents the inconsistent timestamp pair the
            # corroboration window check at line ~1492 depends on.
            lockout_was_engaged = self._estop_lockout.is_set()
            if not lockout_was_engaged:
                self._estop_lockout.set()
                self._last_estop_ts = time.time()
                self._last_estop_mono = time.monotonic()
            lockout_engaged_since = self._last_estop_ts

        sender = data.get("peer_id", "<remote>")
        if not lockout_was_engaged:
            logger.critical(
                "[safety] %s: lockout engaged via remote estop from %s",
                self.peer_id,
                sender,
            )
            self.publish_safety_event(
                event_type="remote_estop_engaged",
                severity="critical",
                payload={
                    "trigger": "remote",
                    "issuer": sender,
                    "issuer_t": data.get("t"),
                },
            )
        else:
            # a second legitimate estop (different issuer, fresh ``t``)
            # arriving while the lockout is already engaged would otherwise
            # be silently dropped from the audit trail -- forensics lose
            # the signal that another operator also tried to engage.
            # Mirror the corroboration audit shape so every issuer of an
            # estop is preserved on the forensic record.
            try:
                self.publish_safety_event(
                    event_type="remote_estop_redundant",
                    severity="info",
                    payload={
                        "issuer": data.get("peer_id"),
                        "issuer_t": envelope_t,
                        "lockout_engaged_since": lockout_engaged_since,
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                logger.debug(
                    "[mesh] %s: remote_estop_redundant audit publish failed: %s",
                    self.peer_id,
                    audit_exc,
                )

    def _on_safety_resume(self, sample: Any) -> None:
        """Clear the local lockout in response to ``strands/safety/resume``.

        Wire authentication (mTLS + ACL) admits this handler. **When the
        operator supplies an ``STRANDS_MESH_ACL_FILE`` with role
        separation only ``operator_peer`` peers can publish here**; the
        default permissive ACL admits any cert-holding peer. Resume is
        further gated by the operator override code: the issuer signed
        ``HMAC(STRANDS_MESH_OVERRIDE_CODE, proof_nonce)`` and we
        recompute it locally; a mismatch means the issuer's override
        code differs from ours and we refuse. This is what stops one
        operator from clearing another operator's e-stop without
        explicit shared authorisation.

        Receivers without ``STRANDS_MESH_OVERRIDE_CODE`` configured
        FAIL CLOSED -- operators must distribute the code to every peer
        for fleet-wide remote resume to work.
        """
        try:
            raw = sample.payload.to_bytes().decode()
            data = json.loads(raw)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            # Narrow exception tuple matches ``_on_safety_estop``:
            # AttributeError -> sample.payload is None or not bytes-like
            # UnicodeDecodeError -> payload not valid UTF-8
            # json.JSONDecodeError -> payload is not valid JSON
            # Wider exceptions (e.g. RuntimeError) bubble up and surface in logs
            # rather than silently leaving the fleet in a half-state.
            return
        if not isinstance(data, dict):
            return
        # Cache operator-tunable freshness/skew knobs once at handler entry
        # (issue #265). Reading them per-use parsed os.getenv plus a regex
        # validation on every reference (5-6 times per envelope) and could
        # observe a mid-handler env mutation, creating an internal
        # inconsistency window. The 0.2s estop corroboration window is
        # timing-sensitive, so we also keep these reads out of the
        # _estop_replay_lock critical section. The next envelope picks up a
        # changed env value, preserving the operator-tunable contract.
        forward_skew_s = _resume_forward_skew_s()
        freshness_window_s = _resume_freshness_window_s()

        # Wire-level publisher attribution (cross-session forgery defence).
        # Mirrors the parallel block in ``_on_safety_estop``: extract the
        # TLS-bound zid from ``sample.source_info.source_id.zid`` and the
        # body's advertised ``source_zid`` field. When both are present
        # they MUST agree; when one is present but the other is not we
        # reject (operator must upgrade all peers together so the binding
        # is never silently downgraded). The wire-level zid is bound into
        # the HMAC input below, so even a body mutation that flips
        # ``peer_id`` and recomputes the MAC under the attacker's own
        # session key cannot satisfy the compare unless the attacker can
        # also forge ``sample.source_info.source_id.zid`` -- which they
        # cannot, because ``ZenohId`` has no public Python constructor
        # and the value is bootstrapped from the mTLS-authenticated
        # session's identity.
        wire_zid = _extract_sample_source_zid(sample)
        body_zid = data.get("source_zid")
        if wire_zid is not None and body_zid is not None:
            if not isinstance(body_zid, str) or wire_zid != body_zid:
                logger.warning(
                    "[safety] %s: refusing remote resume -- body source_zid does not "
                    "match TLS-bound wire source_zid (cross-session forgery rejected)",
                    self.peer_id,
                )
                return
        elif wire_zid is None and body_zid is not None:
            logger.warning(
                "[safety] %s: refusing remote resume -- body source_zid present but wire "
                "source_zid absent (publisher misconfigured or attacker stripped SourceInfo)",
                self.peer_id,
            )
            return
        elif wire_zid is not None and body_zid is None:
            logger.warning(
                "[safety] %s: refusing remote resume -- wire source_zid present but body "
                "source_zid absent (publisher predates source_zid binding; upgrade required)",
                self.peer_id,
            )
            return

        local_code = os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip()
        if not local_code:
            logger.warning(
                "[safety] %s: refusing remote resume -- STRANDS_MESH_OVERRIDE_CODE "
                "not configured locally (operator code missing)",
                self.peer_id,
            )
            return

        proof_nonce = data.get("proof_nonce")
        provided_proof = data.get("override_proof")
        if not isinstance(proof_nonce, str) or not isinstance(provided_proof, str):
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing override_proof / proof_nonce",
                self.peer_id,
            )
            return

        # the HMAC compare moved BELOW the
        # envelope_t + issuer_id + lockout_elapsed_s shape validation
        # because the MAC input now binds those fields. The
        # ``override_proof`` is only meaningful once we've confirmed
        # the wire envelope shape that was signed.

        # freshness + replay cache.
        # The HMAC by itself authenticates the override code but says
        # nothing about when the envelope was minted -- a replay of a
        # captured envelope would still verify. Two cheap defences:
        #
        # 1. Freshness: reject envelopes whose ``t`` field is older
        #  than freshness_window_s or more than the forward
        #  skew in the future. This matches the operator NTP
        #  requirement documented in CHANGELOG.
        # 2. Per-receiver replay cache: refuse a (issuer, proof_nonce)
        #  tuple we have already accepted within the freshness
        #  window. Bounded at _resume_replay_cache_max() entries.
        envelope_t = data.get("t")
        now = time.time()
        if not isinstance(envelope_t, (int, float)):
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing/invalid ``t``",
                self.peer_id,
            )
            return
        if envelope_t > now + forward_skew_s:
            logger.warning(
                "[safety] %s: refusing remote resume -- ``t``=%s in future (forward_skew_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                forward_skew_s,
                now,
            )
            return
        if (now - envelope_t) > freshness_window_s:
            logger.warning(
                "[safety] %s: refusing remote resume -- ``t``=%s too old (freshness_window_s=%s, now=%s)",
                self.peer_id,
                envelope_t,
                freshness_window_s,
                now,
            )
            return

        # Mirror the prior estop strict-reject: an envelope without a
        # valid issuer peer_id would coalesce every "<unknown>"-issued
        # resume into a shared cache slot, polluting the bounded
        # ``_resume_replay_cache_max()`` allowance and giving any peer
        # who omits ``peer_id`` a free way to evict legitimate entries.
        # The canonical ``Mesh.emergency_stop`` issuer always sets
        # ``peer_id``; a malformed envelope is either a programming
        # bug or an attacker probing the cache.
        issuer_id = data.get("peer_id")
        if not isinstance(issuer_id, str) or not issuer_id:
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing/invalid ``peer_id``",
                self.peer_id,
            )
            return

        # the envelope ``lockout_elapsed_s``
        # must be an int/float to participate in the bound MAC input.
        # A missing/invalid value indicates a malformed envelope and
        # is rejected outright -- the canonical issuer at line 1986
        # always sets a real elapsed seconds value.
        envelope_elapsed = data.get("lockout_elapsed_s")
        if not isinstance(envelope_elapsed, (int, float)):
            logger.warning(
                "[safety] %s: refusing remote resume -- envelope missing/invalid ``lockout_elapsed_s``",
                self.peer_id,
            )
            return

        # The HMAC binds every body-routing field (peer_id, t,
        # lockout_elapsed_s, proof_nonce) so a captured envelope mutated
        # by the attacker on ANY of those fields fails the compare. When
        # the wire carries a TLS-bound ``source_zid`` we additionally
        # bind it into the MAC input so an attacker on a different mTLS
        # session who happens to also hold the override code cannot
        # mint a fresh resume claiming to be the legitimate session:
        # the receiver re-derives the MAC using the wire-level
        # ``sample.source_info.source_id.zid`` (bounded by mTLS trust
        # roots; ``ZenohId`` has no public Python ctor) so a mutation
        # of the body ``source_zid`` is provably caught and a same-body
        # resume from a different session is provably caught.
        mac_fields: dict[str, Any] = {
            "peer_id": issuer_id,
            "t": envelope_t,
            "lockout_elapsed_s": envelope_elapsed,
            "proof_nonce": proof_nonce,
        }
        if wire_zid is not None:
            mac_fields["source_zid"] = wire_zid
        mac_input = json.dumps(
            mac_fields,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        expected_proof = hmac.new(
            local_code.encode(),
            mac_input,
            "sha256",
        ).hexdigest()
        if not hmac.compare_digest(expected_proof, provided_proof):
            logger.warning(
                "[safety] %s: refusing remote resume -- override_proof mismatch "
                "(MAC binds peer_id+t+lockout_elapsed_s+proof_nonce%s; "
                "captured-and-mutated replay rejected, constant-time compared)",
                self.peer_id,
                "+source_zid" if wire_zid is not None else "",
            )
            return
        # Replay-cache key incorporates the TLS-bound wire zid when
        # available so two sessions that legitimately share the same
        # ``proof_nonce`` (e.g. two operators racing the same resume)
        # do not collide -- and so an attacker on a different session
        # cannot reuse a captured ``(issuer_peer_id, proof_nonce)`` to
        # evict legitimate cache slots.
        # Domain-tagged key prevents namespace collision between Zenoh
        # wire_zid (hex, TLS-bound) and body issuer_id (app metadata).
        # A bridge peer with peer_id="ab12cd" and a Zenoh peer with
        # wire_zid="ab12cd" no longer conflate into the same slot.
        issuer_key = ("wire", wire_zid) if wire_zid is not None else ("body", issuer_id)
        cache_key = (issuer_key, proof_nonce)
        with self._resume_replay_lock:
            if cache_key in self._resume_replay_cache:
                logger.warning(
                    "[safety] %s: REJECTED remote resume -- replay of (issuer=%s, proof_nonce=%s) already accepted",
                    self.peer_id,
                    issuer_id,
                    proof_nonce[:16] + "...",
                )
                # Audit the replay attempt -- this is exactly the
                # forensic signal an operator wants on a compromised
                # peer trying captured-and-replayed envelopes.
                try:
                    self.publish_safety_event(
                        event_type="resume_replay_rejected",
                        severity="warning",
                        payload={
                            "issuer": issuer_id,
                            "proof_nonce_prefix": proof_nonce[:16],
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # Audit publish is best-effort and must never block the
                    # safety path; narrow the catch tuple per AGENTS.md.
                    logger.debug(
                        "[mesh] %s: resume_replay_rejected audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
                return
            # TTL math uses time.monotonic() (see this PR B5) --
            # envelope freshness above stays on time.time() because it
            # compares against the issuer wall clock; cache eviction is
            # local-only bookkeeping.
            now_mono = time.monotonic()
            _evict_replay_cache(
                self._resume_replay_cache,
                max_size=_resume_replay_cache_max(),
                # see _evict_replay_cache docstring.
                ttl_s=freshness_window_s + forward_skew_s,
                now_mono=now_mono,
            )
            # Per-issuer fairness bound -- mirror of the estop path
            # (_on_safety_estop, search "per_issuer_cap"). Without this
            # a single wire_zid (or body issuer_id on an attribution-less
            # transport) holding the override code can fill all
            # _resume_replay_cache_max() slots and then churn legitimate
            # other-issuer entries out via the eviction 20%-oldest-drop
            # branch, suppressing real replay-rejection signals. The cap
            # is computed from the SAME expression as the estop site so
            # the two replay-cache defenses stay symmetric.
            per_issuer_cap = max(1, _resume_replay_cache_max() // 4)
            issuer_slots = sum(1 for k in self._resume_replay_cache if k[0] == issuer_key)
            if issuer_slots >= per_issuer_cap:
                logger.warning(
                    "[safety] %s: REFUSED resume cache slot -- issuer %r already at cap %d "
                    "(per-issuer fairness bound; flood suspected)",
                    self.peer_id,
                    issuer_key,
                    per_issuer_cap,
                )
                # Audit the over-cap rejection so an operator dashboard
                # can alert. The cache slot is NOT added; the resume
                # itself is refused (unlike estop, which still engages
                # lockout, a refused resume must NOT clear lockout --
                # returning here is the safe direction).
                try:
                    self.publish_safety_event(
                        event_type="resume_per_issuer_cap_exceeded",
                        severity="warning",
                        payload={
                            "issuer": issuer_id,
                            "proof_nonce_prefix": proof_nonce[:16],
                            "cap": per_issuer_cap,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    # Narrow per AGENTS.md > "Exception Clauses Must Be Narrow".
                    logger.debug(
                        "[mesh] %s: resume_per_issuer_cap_exceeded audit publish failed: %s",
                        self.peer_id,
                        audit_exc,
                    )
                return
            self._resume_replay_cache[cache_key] = now_mono

        sender = data.get("peer_id", "<remote>")
        if self._estop_lockout.is_set():
            self._estop_lockout.clear()
            logger.warning("[safety] %s: lockout cleared via remote resume from %s", self.peer_id, sender)
            # audit the receiver-side resume transition. Mirrors
            # _on_safety_estop above so verify_audit_integrity walkers
            # see the close of the lockout window for every peer that
            # entered one.
            self.publish_safety_event(
                event_type="remote_resume_applied",
                severity="info",
                payload={
                    "trigger": "remote",
                    "issuer": sender,
                    "issuer_t": data.get("t"),
                },
            )
        else:
            # Mirror the estop _redundant pattern: a successfully-validated
            # resume that arrives on an already-cleared lockout still
            # consumed a replay-cache slot, so forensics need the signal
            # too (issue #271). Without this audit, a fleet audit-walker
            # reconciling estop_engaged/resume_applied pairs has gaps for
            # the case where multiple operators legitimately hit resume
            # in close succession.
            try:
                self.publish_safety_event(
                    event_type="remote_resume_redundant",
                    severity="info",
                    payload={
                        "trigger": "remote",
                        "issuer": sender,
                        "issuer_t": data.get("t"),
                    },
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                logger.debug(
                    "[mesh] %s: remote_resume_redundant audit publish failed: %s",
                    self.peer_id,
                    audit_exc,
                )

    # RPC -- outgoing
    def send(self, target: str, cmd: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """Send a command to a single peer and return the first response.

        Phase-4 / D1 hardening: turn_id is a full 128-bit uuid4 (no
        truncation), and the expected responder is recorded so
        :meth:`_on_response` rejects forged responses from any peer
        other than *target*.

        explicit guard against passing the
        :data:`BROADCAST_RESPONDER` sentinel (or any string containing a
        NUL byte) as ``target``. ``init_mesh``'s peer_id regex already
        rejects NUL on the receive side, so a real peer can't collide,
        but a future refactor that loosens that rule must not reopen
        the response-hijack surface that this method's contract closes.
        """
        if not self._running:
            return {"status": "error", "error": "mesh not running"}
        if not isinstance(target, str) or not target:
            return {"status": "error", "error": "send: target must be a non-empty string"}
        if "\x00" in target or target == BROADCAST_RESPONDER:
            return {
                "status": "error",
                "error": "send: target may not contain NUL or equal the BROADCAST_RESPONDER sentinel",
            }
        # client-side validate before publishing. Prior to this fix,
        # programmatic callers (tests, third-party integrations, anything
        # that imports Mesh directly) skipped validate_command -- only the
        # robot_mesh tool path validated client-side. Receiver-side
        # _exec_cmd still validates, so this is defence-in-depth, but the
        # PR description and README claimed client-side AND server-side
        # validation; this closes the gap.
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: send to %s rejected client-side: %s", self.peer_id, target, exc)
            return {"status": "error", "error": f"validation: {exc}"}
        # 128-bit turn id -- at 32 bits the birthday-collision window
        # under heavy concurrent RPC load was practical (~65k turns
        # before 50% collision); 128 bits removes that surface entirely.
        turn = uuid.uuid4().hex
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []
            # Defensively: belt-and-suspenders. The public guard above
            # already rejects target == BROADCAST_RESPONDER and target
            # containing NUL, but a future refactor that adds another
            # path into this method (e.g. an internal helper that bypasses
            # the public guard) must not reopen the response-hijack
            # surface. Re-checking here makes the invariant explicit at
            # the assignment site.
            if target == BROADCAST_RESPONDER or "\x00" in target:
                self._pending.pop(turn, None)
                self._responses.pop(turn, None)
                raise ValueError("send: target may not equal BROADCAST_RESPONDER or contain NUL")
            self._expected_responders[turn] = target
        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}
        try:
            self.publish(f"strands/{target}/cmd", msg)
            event.wait(timeout=timeout)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
                self._expected_responders.pop(turn, None)
        return resps[0] if resps else {"status": "timeout"}

    def broadcast(self, cmd: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
        """Broadcast a command to every peer and return all responses.

        Phase-4 / D1: turn_id is a full 128-bit uuid4 (no truncation).
        Broadcast turns accept responses from any responder by design,
        so the responder_id check is bypassed (sentinel
        ``BROADCAST_RESPONDER``).
        """
        if not self._running:
            return []
        # client-side validate before publishing. broadcast()'s
        # return type is list[dict] (responses), so a validation failure
        # has no structured slot -- log the rejection and return [] so
        # callers see "no responses" rather than a partial broadcast.
        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            logger.warning("[mesh] %s: broadcast rejected client-side: %s", self.peer_id, exc)
            return []
        turn = uuid.uuid4().hex
        event = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = event
            self._responses[turn] = []
            # Sentinel -- broadcast accepts responses from any peer.
            self._expected_responders[turn] = BROADCAST_RESPONDER
        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}
        try:
            self.publish("strands/broadcast", msg)
            event.wait(timeout=timeout)
            time.sleep(0.3)
        finally:
            with self._rpc_lock:
                resps = self._responses.pop(turn, [])
                self._pending.pop(turn, None)
                self._expected_responders.pop(turn, None)
        return resps

    def tell(self, target: str, instruction: str, **kw: Any) -> dict[str, Any]:
        """Shorthand: ask a peer to execute a natural-language instruction."""
        return self.send(target, {"action": "execute", "instruction": instruction, **kw})

    # Subscribe / publish_step / on_stream
    def subscribe(
        self, topic: str, callback: Callable[[str, dict[str, Any]], None] | None = None, name: str | None = None
    ) -> str | None:
        """Subscribe to any Zenoh topic and receive parsed JSON dicts."""
        if not self._running:
            return None
        session = current_session()
        if session is None:
            return None
        sub_name = name or topic
        with self._inbox_lock:
            self.inbox.setdefault(sub_name, [])

        def handler(sample: Any) -> None:
            try:
                key = str(sample.key_expr)
                raw = sample.payload.to_bytes().decode()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}
                if callback is not None:
                    callback(key, data)
                else:
                    with self._inbox_lock:
                        buf = self.inbox.setdefault(sub_name, [])
                        buf.append((key, data))
                        if len(buf) > 1000:
                            del buf[: len(buf) - 500]
            except Exception as exc:
                logger.debug("[mesh] %s: subscribe handler error on %s: %s", self.peer_id, topic, exc)

        try:
            sub = session.declare_subscriber(topic, handler)
        except Exception as exc:
            logger.warning("[mesh] %s: declare_subscriber(%s) failed: %s", self.peer_id, topic, exc)
            return None

        with self._subs_lock:
            self._subs.append(sub)
            self._user_subs[sub_name] = sub
        logger.info("[sub] %s subscribed to: %s", self.peer_id, topic)
        return sub_name

    def unsubscribe(self, name: str) -> None:
        """Unsubscribe from a topic by name."""
        with self._subs_lock:
            sub = self._user_subs.pop(name, None)
            if sub is not None:
                try:
                    self._subs.remove(sub)
                except ValueError:
                    pass
        if sub is None:
            return
        try:
            sub.undeclare()
        except Exception:
            pass
        with self._inbox_lock:
            self.inbox.pop(name, None)

    def publish_step(
        self, step: int, observation: dict[str, Any], action: dict[str, Any], instruction: str = "", policy: str = ""
    ) -> None:
        """Publish one VLA execution step to the mesh."""
        if not self._running:
            return
        obs_numeric: dict[str, Any] = {}
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 1:
                continue
            if hasattr(value, "tolist"):
                obs_numeric[key] = value.tolist()
            elif isinstance(value, (int, float, bool, str)):
                obs_numeric[key] = value
            elif isinstance(value, (list, tuple)) and len(value) < 100:
                obs_numeric[key] = list(value)

        act_numeric: dict[str, Any] = {}
        for key, value in action.items():
            if hasattr(value, "tolist"):
                act_numeric[key] = value.tolist()
            elif isinstance(value, (int, float, bool, str, list, tuple)):
                act_numeric[key] = value if not isinstance(value, tuple) else list(value)

        self.publish(
            f"strands/{self.peer_id}/stream",
            {
                "peer_id": self.peer_id,
                "step": step,
                "t": time.time(),
                "instruction": instruction,
                "policy": policy,
                "observation": obs_numeric,
                "action": act_numeric,
            },
        )

    def on_stream(self, peer_id: str, callback: Callable[[str, dict[str, Any]], None] | None = None) -> str | None:
        """Subscribe to another peer's VLA execution stream."""
        return self.subscribe(f"strands/{peer_id}/stream", callback, name=f"stream:{peer_id}")

    # Safety — emergency stop
    def emergency_stop(self) -> list[dict[str, Any]]:
        """Broadcast a stop command to every peer and engage the local lockout.

        After this call the local mesh refuses every non-status, non-resume
        action until :meth:`_resume_lockout` is invoked with the operator
        override code (``STRANDS_MESH_OVERRIDE_CODE``). The event is also
        published on ``strands/safety/estop`` and recorded in the audit log
        (see :func:`strands_robots.mesh.audit.log_safety_event`).

        Returns the list of responses received from peers within the broadcast
        timeout -- useful for telemetry (which peers acknowledged before the
        stop fanned out).
        """
        self._estop_lockout.set()
        self._last_estop_ts = time.time()
        self._last_estop_mono = time.monotonic()
        responses = self.broadcast({"action": "stop"}, timeout=3.0)
        # Wire-level publisher attribution: bind the local TLS-bound zid
        # into both the body (so receivers can verify the body matches
        # ``sample.source_info.source_id.zid``) and the publish path (via
        # ``_publish_safety_envelope`` which attaches a ``SourceInfo``).
        # When ``local_zid`` is ``None`` we are running on the bridge/IoT
        # transport; the body-level HMAC binding still holds, only the
        # cross-session forgery defence is unavailable.
        local_zid = self._local_session_zid()
        envelope: dict[str, Any] = {
            "peer_id": self.peer_id,
            "t": self._last_estop_ts,
            "responses_received": len(responses),
            "lockout_engaged": True,
        }
        if local_zid is not None:
            envelope["source_zid"] = local_zid
        self._publish_safety_envelope("strands/safety/estop", envelope)
        self.publish_safety_event(
            event_type="emergency_stop",
            severity="critical",
            payload={
                "sender_id": self.peer_id,
                "responses_received": len(responses),
                "lockout_engaged": True,
            },
        )
        logger.critical("[safety] %s: EMERGENCY STOP engaged -- lockout active", self.peer_id)
        return responses

    def _resume_lockout(self, override_code: str) -> dict[str, Any]:
        """Clear the emergency-stop lockout if *override_code* matches.

        Compared in constant time against ``STRANDS_MESH_OVERRIDE_CODE``.

        the wire response is a single generic shape (``{"status":
        "ok"}`` on success, ``{"status": "error", "error": "resume
        rejected"}`` on every failure including "lockout not engaged" and
        "override code unconfigured") so a remote prober cannot use
        differential responses as oracles for:

        * whether the lockout is engaged at all (``noop`` vs ``error``),
        * whether ``STRANDS_MESH_OVERRIDE_CODE`` is configured (``not
          configured`` vs ``invalid code``),
        * how long the lockout was held (``lockout_elapsed_s``).

        Structured detail is preserved in the local
        ``publish_safety_event`` audit record where forensics can use it.
        Local callers (e.g. operator tooling that wants to show "already
        unlocked" UI) can still distinguish via the local audit log.
            {"status": "error", "error": "<reason>"}  # rejected

        Every attempt -- successful or not -- is recorded in the audit log
        through :meth:`publish_safety_event`.
        """
        # every non-success path returns the same generic dict so
        # a remote caller cannot use the response shape as an oracle.
        # Structured rejection reasons are preserved in the local audit
        # log via publish_safety_event.
        _generic_error = {"status": "error", "error": "resume rejected"}

        # close the timing oracle by ALWAYS
        # running ``hmac.compare_digest`` on every code path, regardless
        # of whether the lockout is engaged or the override code is
        # configured. Without this, a remote prober can distinguish
        # "compare ran" from "compare didn't run" by response time and
        # learn:
        #  - whether the lockout is engaged (lockout-not-engaged path
        #  skipped the compare)
        #  - whether STRANDS_MESH_OVERRIDE_CODE is configured (unset
        #  path skipped the compare)
        # The earlier response-shape parity (single generic error
        # dict) closed the message-shape oracle but not the timing
        # oracle.
        #
        # Strategy: capture lockout state and configured-code presence
        # FIRST, then unconditionally run the compare. Regardless of
        # outcome, the rejection branch fires in O(constant) time
        # relative to whether each pre-condition was met.
        expected = os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip()
        provided = (override_code or "").strip()
        lockout_engaged = self._estop_lockout.is_set()
        # Always perform the compare against fixed-length sha256 digests
        # so the compare runs to completion regardless of:
        #   * whether ``expected`` is configured,
        #   * the byte length of either input.
        # The previous formulation -- ``compare_digest(expected.encode()
        # or b"\x00" * len(provided), provided.encode())`` -- closed the
        # configured-vs-unconfigured oracle but left a residual
        # ``len(expected) == len(provided)`` length oracle:
        # ``hmac.compare_digest`` is documented constant-time only when
        # both operands have equal length, and CPython returns a fast
        # ``False`` on length mismatch. By pre-hashing both inputs to a
        # fixed 32-byte digest before the compare, both length oracles
        # collapse: the compare always runs over 32 bytes regardless of
        # configuration or attacker probe length.
        #
        # The pre-hash uses sha256 (collision-resistant for the 32-byte
        # output domain) so a digest collision is the only way for the
        # compare to accept a wrong code; correctness is unchanged from
        # the prior byte-equality check. When ``expected`` is empty the
        # placeholder digest is the sha256 of a fixed sentinel value, so
        # the compare always mismatches without paying a different-length
        # cost.
        _PROVIDED_HASH = hashlib.sha256(provided.encode()).digest()
        if expected:
            _EXPECTED_HASH = hashlib.sha256(expected.encode()).digest()
        else:
            # Sentinel digest: sha256(b"\x00" * 32) is a constant a
            # remote prober cannot synthesise an override-code preimage
            # for (it would require breaking sha256). Same byte length
            # (32) as any real digest, so the compare-call cost is
            # identical to the configured-code path.
            _EXPECTED_HASH = hashlib.sha256(b"\x00" * 32).digest()
        compare_ok = hmac.compare_digest(_EXPECTED_HASH, _PROVIDED_HASH)

        # M-1: brute-force throttle gate. If we are inside the cooldown window
        # (armed by a prior run of failed attempts), refuse the resume
        # regardless of whether the code is correct -- this bounds the
        # attempt rate to (max_fails / backoff_s). A legitimate operator who
        # fat-fingers the code N times waits out the (short) cooldown; an
        # attacker is reduced from 295K/s to a handful per cooldown window.
        # Lazily ensure brute-force state exists (defends against callers /
        # test stubs that construct Mesh via __new__ and bypass __init__).
        if not hasattr(self, "_resume_bruteforce_lock"):
            self._resume_bruteforce_lock = threading.Lock()
            self._resume_fail_count = 0
            self._resume_locked_until_mono = 0.0
        _now_mono_bf = time.monotonic()
        with self._resume_bruteforce_lock:
            _throttled = _now_mono_bf < self._resume_locked_until_mono

        # Issue #272: the structured ``reason`` field used to be published
        # via ``publish_safety_event`` which fans out to
        # ``strands/{peer_id}/safety/event`` -- any peer subscribed to
        # ``strands/+/safety/event`` could read the rejection reason and
        # use it as a content-channel oracle (lockout-not-engaged vs
        # not-configured vs bad-code). Now we publish ONLY an opaque
        # ``reason_code`` over the wire (uniform "denied" string) and
        # write the structured reason to the LOCAL audit log via
        # ``log_safety_event`` (file-backed; not broadcast).
        # Issue #256: every rejection branch performs the same I/O
        # work shape (one local audit + one wire publish) so the
        # latency oracle collapses too.
        def _emit_resume_denied(reason_text: str, severity: str) -> None:
            try:
                log_safety_event(
                    "resume_denied",
                    self.peer_id,
                    {"sender_id": self.peer_id, "reason": reason_text, "severity": severity},
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                logger.debug(
                    "[mesh] %s: resume_denied local audit failed: %s",
                    self.peer_id,
                    audit_exc,
                )
            try:
                # Wire-broadcast carries ONLY the opaque code, no reason.
                self.publish_safety_event(
                    event_type="resume_denied",
                    severity=severity,
                    payload={"sender_id": self.peer_id, "reason_code": "denied"},
                )
            except (TypeError, ValueError, OSError) as audit_exc:
                logger.debug(
                    "[mesh] %s: resume_denied wire publish failed: %s",
                    self.peer_id,
                    audit_exc,
                )

        if _throttled:
            _emit_resume_denied("resume rate-limited (brute-force throttle)", "warning")
            return _generic_error

        if not lockout_engaged:
            _emit_resume_denied("lockout not engaged", "info")
            return _generic_error

        if not expected:
            _emit_resume_denied("STRANDS_MESH_OVERRIDE_CODE not configured", "warning")
            return _generic_error

        if not compare_ok:
            # M-1: count this consecutive failure and arm the cooldown once we
            # cross the threshold. Done under the bruteforce lock so concurrent
            # probe threads can't race past the limit.
            with self._resume_bruteforce_lock:
                self._resume_fail_count += 1
                if self._resume_fail_count >= _resume_max_fails():
                    self._resume_locked_until_mono = time.monotonic() + _resume_backoff_s()
                    self._resume_fail_count = 0
                    logger.warning(
                        "[safety] %s: resume brute-force threshold hit -- throttling resume for %.0fs",
                        self.peer_id,
                        _resume_backoff_s(),
                    )
            _emit_resume_denied("bad override code", "warning")
            return _generic_error

        elapsed = time.time() - self._last_estop_ts
        self._estop_lockout.clear()
        # M-1: a correct code clears the brute-force counter + any cooldown.
        with self._resume_bruteforce_lock:
            self._resume_fail_count = 0
            self._resume_locked_until_mono = 0.0
        self.publish_safety_event(
            event_type="resume_ok",
            severity="info",
            payload={"sender_id": self.peer_id, "lockout_elapsed_s": elapsed},
        )

        # bind a proof-of-override-code into the resume envelope so
        # receivers can re-verify on _on_safety_resume. Without this,
        # any operator-class peer could fan-out a resume just by virtue
        # of being on the ACL; the override code adds a second factor
        # that every receiver re-verifies by recomputing
        # HMAC(local_code, proof_nonce).
        #
        # The proof_nonce is per-resume (uuid4.hex). We deliberately do
        # NOT include the override code itself in the published payload
        # or the audit log -- only the HMAC of (code, nonce).
        proof_nonce = uuid.uuid4().hex
        envelope_t = time.time()
        # The HMAC input binds every envelope-routing field plus the
        # local Zenoh session ZID. Binding ``source_zid`` closes the
        # cross-session forgery surface where an attacker holding the
        # override code on a different mTLS-authenticated session
        # would otherwise be able to mint a resume claiming to come
        # from the legitimate operator's session: the receiver
        # re-derives the MAC using the wire-level
        # ``sample.source_info.source_id.zid``, and ``ZenohId`` cannot
        # be chosen by the publisher (zenoh-python exposes no public
        # constructor; the value is established by the Zenoh bootstrap
        # that follows the mTLS handshake and bounded by the trust
        # roots in ``connect.tls``).
        #
        # Body-level fields (``peer_id``, ``t``, ``lockout_elapsed_s``,
        # ``proof_nonce``) remain bound so the cache-key + freshness
        # defences continue to hold for non-Zenoh transports where
        # ``source_zid`` is absent (bridge / IoT). Bound as a
        # deterministic JSON blob (sort_keys, no whitespace) so issuer
        # and receiver compute the same digest byte-for-byte.
        local_zid = self._local_session_zid()
        mac_fields: dict[str, Any] = {
            "peer_id": self.peer_id,
            "t": envelope_t,
            "lockout_elapsed_s": elapsed,
            "proof_nonce": proof_nonce,
        }
        if local_zid is not None:
            mac_fields["source_zid"] = local_zid
        mac_input = json.dumps(
            mac_fields,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        override_proof = hmac.new(
            expected.encode(),
            mac_input,
            "sha256",
        ).hexdigest()
        envelope: dict[str, Any] = {
            "peer_id": self.peer_id,
            "t": envelope_t,
            "lockout_elapsed_s": elapsed,
            "proof_nonce": proof_nonce,
            "override_proof": override_proof,
        }
        if local_zid is not None:
            envelope["source_zid"] = local_zid
        self._publish_safety_envelope("strands/safety/resume", envelope)
        logger.warning("[safety] %s: resume after %.1fs lockout", self.peer_id, elapsed)
        # success is also generic on the wire; the local audit
        # record (resume_ok above) carries the elapsed time for forensics.
        return {"status": "ok"}

    def _local_session_zid(self) -> str | None:
        """Return our own Zenoh session's ZID (32-char hex), or ``None``.

        The ZID is established during the Zenoh session bootstrap that
        follows the mTLS handshake; it is unique per session and cannot
        be chosen by the caller. Used in two places:

        1. As the ``source_zid`` field embedded in safety envelope
           bodies. Combined with the receiver-side check
           ``body.source_zid == sample.source_info.source_id.zid`` this
           closes the cross-session forgery window where an attacker
           in a different session would have to convince Zenoh to
           attach a wire-level ``source_id.zid`` that does not match
           the body field. Zenoh-python exposes no public constructor
           for ``ZenohId``/``EntityGlobalId``, so this binding is
           bounded by the trust roots in ``connect.tls``.

        2. As an HMAC input on resume envelopes so a captured
           override-proof bound to one session's wire identity cannot
           be replayed from a different session even if the attacker
           also holds the override code (see
           :meth:`_resume_lockout`).

        Returns ``None`` for non-Zenoh transports (bridge / IoT) and
        when no session is currently open. In that case the safety
        path falls back to the body-level HMAC binding alone -- the
        cross-session-forgery defence is Zenoh-specific because only
        Zenoh exposes a TLS-bound publisher identity.
        """
        try:
            from strands_robots.mesh.session import _current_zenoh_session_directly

            session = _current_zenoh_session_directly()
        except ImportError:
            return None
        if session is None:
            return None
        try:
            zid = session.info.zid()
        except (AttributeError, RuntimeError):
            return None
        if zid is None:
            return None
        zid_str = str(zid)
        return zid_str or None

    def _safety_publisher_for(self, key: str) -> Any | None:
        """Lazily declare and cache a Zenoh ``Publisher`` for *key*.

        The publisher is held for the lifetime of this Mesh and reused
        across :meth:`_publish_safety_envelope` calls so:

        * its ``EntityGlobalId.eid`` is stable -- a receiver-side
          downstream defence can refuse same-zid envelopes whose eid
          has shifted (which Zenoh treats as a fresh publisher entity).
        * the per-publisher monotonic counter inside Zenoh increments
          predictably; a replay across the same session is bounded by
          our own ``_safety_sn`` counter (bound into ``source_sn``).

        Returns ``None`` for non-Zenoh transports or when no session
        is currently open. The caller falls back to the legacy
        ``put()`` path in that case.
        """
        try:
            from strands_robots.mesh.session import _current_zenoh_session_directly

            session = _current_zenoh_session_directly()
        except ImportError:
            return None
        if session is None:
            return None
        with self._safety_publishers_lock:
            pub = self._safety_publishers.get(key)
            if pub is not None:
                return pub
            try:
                pub = session.declare_publisher(key)
            except (RuntimeError, OSError) as exc:
                logger.warning(
                    "[mesh] %s: declare_publisher(%s) failed: %s",
                    self.peer_id,
                    key,
                    exc,
                )
                return None
            self._safety_publishers[key] = pub
            return pub

    def _next_safety_sn(self, key: str) -> int:
        """Return a monotonically-increasing sequence number scoped to *key*.

        Bound into the ``SourceInfo.source_sn`` field of every safety
        envelope we publish. Pairs with ``source_zid`` so that the
        receiver can refuse replays of (zid, sn) it has already
        accepted, without coalescing with envelopes from other
        sessions.
        """
        with self._safety_sn_lock:
            sn = self._safety_sn.get(key, 0) + 1
            self._safety_sn[key] = sn
            return sn

    def _strip_wire_zid(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *payload* without the wire-level ``source_zid``.

        Used on the legacy ``put()`` fallback path: when no Zenoh
        ``SourceInfo`` can be attached, a body that still advertises
        ``source_zid`` is hard-rejected by the receiver (wire-absent +
        body-present == "publisher stripped SourceInfo"). Removing the
        body field makes the envelope match the receiver's
        transport-agnostic accept-without-zid contract (availability
        fix). The body-level HMAC binding still holds for non-Zenoh
        transports because those never bound ``source_zid`` in the
        first place (``_local_session_zid()`` returned ``None``).
        """
        if "source_zid" not in payload:
            return payload
        clean = dict(payload)
        clean.pop("source_zid", None)
        return clean

    def _publish_safety_envelope(self, key: str, payload: dict[str, Any]) -> None:
        """Publish a safety envelope with TLS-bound source attribution.

        Attempts the Zenoh-native path first: declare (or reuse) a
        publisher on *key*, allocate a fresh per-topic sequence number,
        and put the JSON-encoded *payload* with
        ``SourceInfo(publisher.id, sn)`` attached. The receiver
        extracts ``sample.source_info.source_id.zid`` and checks it
        matches the body's ``source_zid`` field plus the HMAC binding
        -- this closes the cross-session forgery window where an
        attacker on a different session would have had to convince
        Zenoh to attach our wire-level zid (which is bounded by mTLS
        trust roots and the absence of a public ``ZenohId`` ctor in
        zenoh-python).

        Falls back to the legacy transport-agnostic ``put()`` path
        when:

        * the transport is not Zenoh (bridge / IoT),
        * no Zenoh session is currently open,
        * ``declare_publisher`` failed for any reason (logged at
          WARNING by ``_safety_publisher_for``),
        * the ``zenoh.SourceInfo`` constructor is unavailable on the
          installed zenoh-python version.

        In the fallback case the body-level HMAC binding still holds;
        only the additional cross-session-forge defence is omitted.
        """
        pub = self._safety_publisher_for(key)
        if pub is None:
            put(key, self._strip_wire_zid(payload))
            return
        try:
            import zenoh
        except ImportError:
            put(key, self._strip_wire_zid(payload))
            return
        sn = self._next_safety_sn(key)
        try:
            source_info = zenoh.SourceInfo(pub.id, sn)
        except (TypeError, AttributeError):
            # zenoh-python without the SourceInfo ctor (very old build);
            # fall back to body-level binding only.
            put(key, self._strip_wire_zid(payload))
            return
        try:
            pub.put(json.dumps(payload).encode(), source_info=source_info)
        except (RuntimeError, OSError, TypeError) as exc:
            logger.warning(
                "[mesh] %s: safety publisher.put(%s) failed: %s",
                self.peer_id,
                key,
                exc,
            )
            put(key, self._strip_wire_zid(payload))

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        """Publish *payload* on *key* via the mesh transport.

        Wire authentication is owned by the Zenoh transport: outbound
        bytes ride a TLS link whose cert binds the peer identity, and
        the ACL gates which key-expressions this peer can publish on.
        This method simply forwards to ``put()`` -- it stays as a
        single chokepoint so a future hook (audit, telemetry,
        compression) can land in one place.

        Renamed from ``_put_signed`` after the application-layer signing
        envelope was dropped (commit 7113742). The old name was a
        historical artefact: nothing in the body ever signed anything
        once Zenoh's mTLS + ACL took over identity and authorization.
        """
        put(key, payload)


# init_mesh -- the only public constructor
def init_mesh(
    robot: Any,
    peer_id: str | None = None,
    peer_type: str = "robot",
    mesh: bool = True,
) -> Mesh | None:
    """Construct and start a Mesh for the given robot.

    Returns None when mesh is disabled (STRANDS_MESH=false or mesh=False).
    """
    env = os.getenv("STRANDS_MESH", "true").strip().lower()
    if env == "false":
        mesh = False
    if not mesh:
        return None

    if peer_id is None:
        base = getattr(robot, "tool_name_str", None) or "robot"
        peer_id = f"{base}-{uuid.uuid4().hex[:8]}"

    # Validate peer_id — reject reserved names and MQTT-unsafe characters.
    _RESERVED_PEER_IDS = {"broadcast", "safety"}
    _PEER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{0,127}$")
    if peer_id in _RESERVED_PEER_IDS:
        raise ValueError(
            f"peer_id={peer_id!r} is reserved for system use. Reserved names: {sorted(_RESERVED_PEER_IDS)}"
        )
    if not _PEER_ID_PATTERN.match(peer_id):
        raise ValueError(
            f"peer_id={peer_id!r} contains invalid characters. "
            "Must match [a-zA-Z0-9][a-zA-Z0-9._-]{{0,127}} "
            "(no /, +, # — these break MQTT topic structure and AWS Thing-name rules)."
        )

    instance = Mesh(robot, peer_id=peer_id, peer_type=peer_type)
    instance.start()

    # Auto-wire IoT enrichments when the active transport supports them.
    # Both calls are no-ops when STRANDS_MESH_BACKEND=zenoh (the default),
    # so this is purely additive — Zenoh-LAN behaviour is unchanged.
    if instance.alive:
        try:
            from strands_robots.mesh.iot import (
                enable_camera_offload_for_mesh,
                enable_shadow_for_mesh,
            )

            enable_shadow_for_mesh(instance)
            enable_camera_offload_for_mesh(instance)
        except Exception as exc:  # noqa: BLE001 — IoT enrichment is best-effort
            logger.debug("[mesh] IoT enrichment failed (continuing): %s", exc)

    return instance
