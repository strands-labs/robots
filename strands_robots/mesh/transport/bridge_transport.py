"""Bridge transport — Zenoh LAN + AWS IoT WAN behind one MeshTransport.

This is the **production** transport for fleets that have both LAN peers
(robot + sim + leader arm in the same room) AND cloud connectivity
(operator dashboards, audit, fleet ops in AWS).

How it works
------------
:class:`BridgeTransport` owns one :class:`ZenohTransport` and one
:class:`IotMqttTransport`. Every :meth:`put` fans out to both, but the
**MQTT side is filtered**: high-volume / latency-sensitive topics
(``state``, ``pose``, ``imu``, ``odom``, ``camera``, ``input``, ``hand``)
default to LAN-only. Cloud-relevant topics (``presence``, ``health``,
``cmd``, ``response``, ``broadcast``, ``safety/event``, ``safety/estop``)
default to **both transports**.

The filter is configurable via ``STRANDS_MESH_BRIDGE_TOPICS`` (a
comma-separated list of suffixes that bridge to MQTT). The default value
is conservative and matches the cost analysis in the design doc.

Subscriptions are **fanned out**: a single ``declare_subscriber("X", h)``
subscribes ``h`` on both Zenoh and MQTT. Inbound deduplication happens at
the :class:`Mesh` layer (existing :meth:`_on_presence` / :meth:`_on_cmd` /
:meth:`_on_response` already handle duplicate / self-loop dropouts via
``sender_id`` and ``turn_id`` correlation), so we don't try to be clever
here — the fact that a presence might arrive twice is harmless and the
existing peer registry deduplicates by ``peer_id``.

Failure isolation
-----------------
If either side's :meth:`connect` fails, the bridge degrades gracefully:

- Zenoh failed, IoT OK → behaves as a pure :class:`IotMqttTransport`.
- IoT failed, Zenoh OK → behaves as a pure :class:`ZenohTransport`.
- Both failed → :meth:`is_alive` returns False, all puts are no-ops.

This matches the "mesh is enrichment, never crash the robot" contract
that :class:`Mesh` already follows for failed Zenoh sessions.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots.mesh.transport.iot_transport import IotMqttTransport
    from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

logger = logging.getLogger(__name__)


def _get_zenoh_transport_class() -> type[ZenohTransport]:
    """Lazily import ZenohTransport to avoid circular dependency."""
    from strands_robots.mesh.transport.zenoh_transport import ZenohTransport as _ZT

    return _ZT


def _get_iot_transport_class() -> type[IotMqttTransport]:
    """Lazily import IotMqttTransport to avoid circular dependency."""
    from strands_robots.mesh.transport.iot_transport import IotMqttTransport as _IT

    return _IT


# Default bridge filter — derived from cost / latency analysis.
#
# Topics in this set bridge from Zenoh to MQTT. Everything else stays LAN.
# Suffixes are matched against the part of the topic AFTER ``strands/``.
#
# Why this default:
# - presence  — rare, retained on cloud, late operators need it
# - health  — rare, retained, threshold alerts via Rules
# - safety/event  — must hit cloud audit
# - safety/estop  — defence-in-depth E-stop
# - safety/resume  — paired with safety/estop; cloud audit needs the
#   resume edge to close the safety incident timeline
# - cmd  — operator-to-robot RPC (cloud → robot direction)
# - response  — robot-to-operator RPC reply
# - broadcast  — fan-out RPC
#
# Explicitly NOT bridged by default (opt in via STRANDS_MESH_BRIDGE_TOPICS):
# - state, pose, imu, odom, lidar — high volume, route via Basic Ingest if
#  cloud needs them. See AWS_IOT_MESH_INTEGRATION.md §7.2 for the cost math.
# - camera, input, hand — LAN-only by definition (size / latency).
DEFAULT_BRIDGE_SUFFIXES: frozenset[str] = frozenset(
    {
        "presence",
        "health",
        "safety/event",
        "safety/estop",
        "safety/resume",
        "cmd",
        "response",
        "broadcast",
    }
)

# Of the bridge filter entries, only ``response`` legitimately carries a
# trailing ``/<turn-id>`` segment that the bridge must accept. Every other
# entry is matched exactly. This is the post-Phase-4 hardening:
#
#  Pre-fix: a sloppy prefix-walk in _should_bridge meant that
#  ``strands/<x>/cmd/anything-attacker-tacks-on`` matched the ``cmd``
#  filter entry and was bridged to MQTT. An attacker could pollute the
#  cloud audit table / spam CloudWatch / inflate broker billing by
#  appending arbitrary suffixes to allowed prefixes
#  (``strands/x/safety/event/<10kb-blob>`` is the worst case -- it ends
#  up in the DDB audit table).
#
# Operators who need a bare-prefix match for a custom suffix can opt in
# explicitly via ``STRANDS_MESH_BRIDGE_TOPICS_PREFIX``.
_DEFAULT_BRIDGE_PREFIX_SUFFIXES: frozenset[str] = frozenset({"response"})


def _resolve_bridge_filter() -> frozenset[str]:
    """Read ``STRANDS_MESH_BRIDGE_TOPICS`` or fall back to the default.

    Returns the EXACT-match suffix set. Prefix-match suffixes (i.e.
    those whose tail is part of the topic, like ``response/<turn>``)
    are returned by :func:`_resolve_bridge_prefix_filter`.
    """
    env = os.getenv("STRANDS_MESH_BRIDGE_TOPICS")
    if not env:
        return DEFAULT_BRIDGE_SUFFIXES
    parts = [p.strip() for p in env.split(",") if p.strip()]
    if not parts:
        return DEFAULT_BRIDGE_SUFFIXES
    return frozenset(parts)


def _resolve_bridge_prefix_filter() -> frozenset[str]:
    """Read ``STRANDS_MESH_BRIDGE_TOPICS_PREFIX`` or fall back to default.

    Entries here are matched as a path prefix (``response`` matches
    ``response/abc-123``). The default is just ``response`` because that
    is the only RPC-shape topic with a per-turn tail. Operators who add
    a new RPC-shape topic must extend this list explicitly -- extending
    only ``STRANDS_MESH_BRIDGE_TOPICS`` will NOT bridge tails.
    """
    env = os.getenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX")
    if not env:
        return _DEFAULT_BRIDGE_PREFIX_SUFFIXES
    parts = [p.strip() for p in env.split(",") if p.strip()]
    if not parts:
        return _DEFAULT_BRIDGE_PREFIX_SUFFIXES
    return frozenset(parts)


def _topic_suffix(topic: str) -> str:
    """Return the suffix following ``strands/`` from a Mesh topic.

    Handles three layouts:

    - ``strands/broadcast``  -> ``broadcast``
    - ``strands/safety/estop``  -> ``safety/estop``
    - ``strands/{peer}/{kind}/...``  -> ``{kind}/...``
    """
    if not topic.startswith("strands/"):
        return ""
    rest = topic[len("strands/") :]
    parts = rest.split("/", 1)
    if len(parts) == 1:
        return rest  # e.g. "broadcast"
    head, tail = parts
    # If head is a recognised top-level kind ("safety", "broadcast"), keep it.
    if head in ("safety", "broadcast"):
        return rest
    # Otherwise head is a peer_id and the kind starts at tail.
    return tail


def _should_bridge(
    topic: str,
    allowed_suffixes: frozenset[str],
    allowed_prefixes: frozenset[str] | None = None,
) -> bool:
    """True if *topic* should be republished to MQTT.

    Match policy (Phase-4 tightening):

    * **Exact match**: ``allowed_suffixes`` entries match the topic
      suffix character-for-character. ``cmd`` matches
      ``strands/<peer>/cmd`` only -- NOT
      ``strands/<peer>/cmd/<attacker-supplied-tail>``.
    * **Prefix match**: only entries listed in ``allowed_prefixes``
      (default: ``{"response"}`` -- the only RPC-shape topic with a
      per-turn tail) accept a trailing path component. ``response``
      matches ``response/<turn>``.

    The exact / prefix split closes a cloud-pollution attack: without
    it, an attacker could append arbitrary tails to any allowed prefix
    and have the bridge republish the message to MQTT (e.g. a 10 KiB
    blob on ``strands/<x>/safety/event/<blob>`` ending up in the DDB
    audit table). Only ``response`` legitimately carries a per-turn
    tail, so it is the sole prefix-walk default.
    """
    if allowed_prefixes is None:
        allowed_prefixes = _resolve_bridge_prefix_filter()

    suffix = _topic_suffix(topic)
    if not suffix:
        return False

    # Exact match -- fast path.
    if suffix in allowed_suffixes:
        return True

    # Prefix match -- only legitimate for entries explicitly opted-in to
    # tail-acceptance.
    head = suffix.split("/", 1)[0]
    # Reject path-traversal in the head segment too -- the previous
    # check only rejected ``..`` in the tail. ``../foo`` would have
    # head=".." and skip the rest-segment scan entirely. Belt-and-
    # braces against operator misconfigurations of allowed_prefixes
    # that include ``..`` literally.
    if head == "..":
        return False
    if head in allowed_prefixes:
        # Defence-in-depth: reject any tail containing path-traversal
        # segments. Zenoh keys never legitimately contain ``..``.
        rest = suffix[len(head) + 1 :] if "/" in suffix else ""
        if rest and any(seg == ".." for seg in rest.split("/")):
            return False
        return True

    return False


# Cross-transport command deduplication.
#
# In bridge mode the same command can be delivered twice -- once via Zenoh
# and once via MQTT -- because subscriptions fan out on both sides. Without
# dedup the receiver would dispatch the action twice (move twice, broadcast
# twice, etc.).
#
# The deduplicator below caches a SHA-256 fingerprint of
# (sender_id, turn_id, command) per topic and refuses to deliver a sample
# whose identity it has seen recently. Tunable via
# ``STRANDS_MESH_DEDUP_TTL`` (seconds; default 120).
_DEFAULT_DEDUP_TTL_S = 120.0
_MAX_DEDUP_ENTRIES = 10_000
# Issue #231: hysteresis band on the sort-and-slice GC trigger. The cheap
# stale-eviction sweep still runs at the soft boundary (_MAX), but the
# heap-select-and-evict pass only runs once the cache exceeds this hard
# boundary, so its O(n log k) cost amortises across many calls instead of
# firing on every call while the cache hovers at the cap.
_DEDUP_GC_HARD_RATIO = 1.1
_MAX_DEDUP_ENTRIES_HARD = int(_MAX_DEDUP_ENTRIES * _DEDUP_GC_HARD_RATIO)


def _resolve_dedup_ttl() -> float:
    """Read ``STRANDS_MESH_DEDUP_TTL`` env var (default: 120s).

    NOTE: read once at ``BridgeTransport`` construction; mid-process
    env-var changes require a bridge restart to take effect.
    """
    raw = os.getenv("STRANDS_MESH_DEDUP_TTL")
    if raw is None:
        return _DEFAULT_DEDUP_TTL_S
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_DEDUP_TTL_S
    except ValueError:
        logger.warning("[bridge] STRANDS_MESH_DEDUP_TTL=%r invalid -- using default", raw)
        return _DEFAULT_DEDUP_TTL_S


def _resolve_dedup_strict() -> bool:
    """Read ``STRANDS_MESH_BRIDGE_DEDUP_STRICT`` env var (default: off).

    Strict mode makes the deduplicator hash the full payload when no
    canonical (sender_id, turn_id, command) tuple is present. Without it,
    non-canonical payloads bypass dedup entirely (safe for heartbeats that
    legitimately recur with the same content).

    Bridge cross-transport needs strict mode to dedup heartbeat-style
    payloads that arrive on both Zenoh and MQTT.

    NOTE: read once at ``BridgeTransport`` construction; mid-process
    env-var changes require a bridge restart to take effect.
    """
    raw = os.getenv("STRANDS_MESH_BRIDGE_DEDUP_STRICT", "").strip().lower()
    if raw in ("", "0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    logger.warning(
        "[bridge] STRANDS_MESH_BRIDGE_DEDUP_STRICT=%r invalid -- using default (off)",
        raw,
    )
    return False


class _CommandDeduplicator:
    """TTL-bounded cache of (key, dedup-id) tuples seen in the recent past.

    Thread-safe. Identity is a SHA-256 fingerprint over the canonical
    RPC triple ``(sender_id, turn_id, command)`` -- callers must not
    reuse that triple for distinct deliveries (the contract assumes
    ``turn_id`` is monotonic per-sender). Payloads with an incomplete
    canonical triple pass through in default mode or fall back to a
    full-payload hash in strict mode. The cache key is *(topic_key,
    dedup_id)* so two distinct topics with coincidentally matching
    dedup_ids don't collide.
    """

    __slots__ = ("_seen", "_lock", "_ttl", "_strict")

    def __init__(self, ttl_s: float | None = None, *, strict: bool = False) -> None:
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_s if ttl_s is not None else _resolve_dedup_ttl()
        # Strict mode: when True, _dedup_id falls back to full-payload hash
        # for payloads with no canonical (sender_id, turn_id, command) tuple.
        # Used by bridge cross-transport path to dedup heartbeats etc.
        self._strict = strict

    @property
    def ttl(self) -> float:
        return self._ttl

    def _dedup_id(self, payload: dict[str, Any]) -> str | None:
        """Return a content fingerprint identifying this message.

        Identity is the canonical RPC triple ``(sender_id, turn_id,
        command)``: callers must not reuse that triple for distinct
        deliveries (the contract assumes ``turn_id`` is monotonic
        per-sender). Two messages that share the triple but differ in
        other top-level fields (timestamps, audit metadata, future-added
        envelope fields) are treated as the same delivery -- this is the
        intentional dedup contract for cross-transport bridge mode.

        Returns ``None`` when the canonical triple is incomplete (any of
        the three fields missing). In strict mode, an incomplete triple
        falls through to a full-payload hash; in default mode it passes
        through (the existing peer registry deduplicates by
        ``peer_id``/``turn_id`` upstream).

        The previous behaviour -- canonical path on *any* non-None field
        -- aliased partial payloads (e.g. ``{"sender_id": "a"}`` would
        dedup against ``{"sender_id": "a", "extra": 1}``). Pinned by
        ``test_partial_canonical_does_not_alias``.

        JSON-encodability contract: ``command`` payloads on the canonical
        identity path MUST be pure-JSON-encodable (str/int/float/bool/None,
        list, dict). The ``default=str`` argument to ``json.dumps`` is a
        defensive coercion that prevents ``TypeError`` from non-JSON types
        (datetime, bytes, custom objects), but the resulting fingerprint is
        non-deterministic for objects whose ``str()`` includes their memory
        address (e.g. instances without a ``__str__`` override). Producers
        relying on dedup correctness for non-JSON ``command`` shapes are in
        contract violation. Tracked for resolution (drop ``default=str`` and
        let TypeError surface, vs. enforce JSON contract at producer side)
        in #233.

        The strict-mode full-payload hash (partial-canonical fallback at
        lines below) shares the same ``default=str`` non-determinism risk;
        #233 covers both paths.
        """
        if not isinstance(payload, dict):
            return None

        sender = payload.get("sender_id")
        turn = payload.get("turn_id")
        cmd = payload.get("command")

        # Canonical path requires all three fields present AND non-blank;
        # partial/empty canonical payloads fall through to the strict/pass-
        # through branch so they do not alias against each other.
        # Empty-string rejection aligns with R20 bridge-side counterpart.
        def _is_blank(v: object) -> bool:
            return v is None or (isinstance(v, str) and v.strip() == "")

        if _is_blank(sender) or _is_blank(turn) or cmd is None:
            if not self._strict:
                # Default: pass through (preserves heartbeat semantics).
                return None
            # Strict mode: full-payload hash fallback.
            try:
                # Issue #233: dropped ``default=str`` from canonical-path
                # encoders. Custom objects without ``__str__`` overrides
                # produced address-suffixed strings (``<Foo object at 0x...>``)
                # which made the fingerprint non-deterministic. Now we let
                # TypeError fall through to pass-through (same semantics as
                # missing-canonical-fields: dedup is bypassed, peer-registry
                # upstream still bounds replays).
                full = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError):
                return None
            return "p:" + hashlib.sha256(full).hexdigest()

        try:
            # Issue #233: dropped ``default=str``. Non-JSON ``command``
            # payloads now bypass dedup (return None) rather than producing
            # a non-deterministic address-suffixed fingerprint that appears
            # to dedup in tests but doesn't in production.
            canonical = json.dumps(
                {"sender": sender, "turn": turn, "cmd": cmd},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        # Full 256-bit (64 hex chars) -- no birthday-attack truncation.
        return "f:" + hashlib.sha256(canonical).hexdigest()

    def is_duplicate(self, key: str, payload: dict[str, Any]) -> bool:
        """Return True if this (key, payload) was seen within the TTL.

        Records the entry when not a duplicate so the next call returns True.
        """
        ident = self._dedup_id(payload)
        if ident is None:
            return False  # nothing to dedup against -- pass through
        cache_key = (key, ident)
        now = time.monotonic()  # NTP-safe, snapshot-resume-safe

        # Issue #231: bound GC cost on two axes -- algorithmic and lock-hold.
        #
        # Axis 1 (algorithm): a hysteresis band defers the expensive
        # heap-select pass until the cache exceeds the hard boundary
        # (_MAX_DEDUP_ENTRIES_HARD), so it amortises across calls instead of
        # firing every call while the cache hovers at the soft cap. The cheap
        # stale-eviction sweep still runs at the soft boundary (_MAX).
        #
        # Axis 2 (lock-hold): the heap walk runs OUTSIDE self._lock via a
        # snapshot-then-apply pattern. We snapshot self._seen.items() under
        # the lock, release it, compute the eviction set with heapq.nsmallest
        # (O(n log k), the only expensive step), then re-acquire to apply.
        # The lock is held only for the snapshot copy and the eviction apply,
        # never for the heap walk -- so concurrent is_duplicate() callers are
        # not serialised behind the GC compute.
        #
        # Concurrency note: another thread may insert a new entry between the
        # snapshot and the apply. We tolerate this by re-reading each entry's
        # timestamp under the apply lock and only evicting it if it still
        # matches the snapshot timestamp; a key re-inserted with a newer
        # timestamp is kept (its newer identity is the live one). Dedup
        # correctness is unaffected: identity semantics, TTL eviction, and the
        # (topic_key, dedup_id) cache key shape are all unchanged.
        with self._lock:
            size = len(self._seen)
            run_stale = size > _MAX_DEDUP_ENTRIES
            if run_stale:
                cutoff = now - self._ttl
                stale = [k for k, ts in self._seen.items() if ts < cutoff]
                for k in stale:
                    self._seen.pop(k, None)
            run_select = len(self._seen) > _MAX_DEDUP_ENTRIES_HARD
            snapshot = list(self._seen.items()) if run_select else None

        if run_stale or run_select:
            logger.debug(
                "[bridge] dedup GC: stale-eviction=%s sort-and-slice=%s size=%d",
                run_stale,
                run_select,
                size,
            )

        if run_select and snapshot is not None:
            # Heap walk OUTSIDE the lock (axis 2). O(n log k), k = n // 5.
            drop = max(1, len(snapshot) // 5)
            oldest = heapq.nsmallest(drop, snapshot, key=lambda kv: kv[1])
            with self._lock:
                for k, ts in oldest:
                    # Only evict if the live timestamp still matches the
                    # snapshot -- a key re-inserted between snapshot and apply
                    # carries a newer identity we must not drop.
                    if self._seen.get(k) == ts:
                        self._seen.pop(k, None)

        with self._lock:
            seen_ts = self._seen.get(cache_key)
            if seen_ts is not None and (now - seen_ts) <= self._ttl:
                return True
            self._seen[cache_key] = now
            return False

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


class _BridgeSubHandle:
    """Subscription handle that calls ``undeclare`` on whichever side(s)
    actually subscribed.

    Both sides may be present (typical) or only one (the other's connect
    failed). Tearing down only what's there.
    """

    __slots__ = ("_zenoh_sub", "_iot_sub", "_undeclared")

    def __init__(self, zenoh_sub: Any | None, iot_sub: Any | None) -> None:
        self._zenoh_sub = zenoh_sub
        self._iot_sub = iot_sub
        self._undeclared = False

    def undeclare(self) -> None:
        if self._undeclared:
            return
        self._undeclared = True
        for sub in (self._zenoh_sub, self._iot_sub):
            if sub is None:
                continue
            try:
                sub.undeclare()
            except (RuntimeError, AttributeError, OSError) as exc:
                # Narrow per AGENTS.md > Review Learnings: idempotent
                # teardown should swallow the documented transport-failure
                # surface (RuntimeError = already-closed handle;
                # AttributeError = mock or partial-init handle;
                # OSError = socket teardown race) and let unexpected
                # exceptions propagate.
                logger.debug("[bridge] sub.undeclare() failed: %s", exc)


class BridgeTransport:
    """:class:`MeshTransport` that fans out to both Zenoh (LAN) and AWS IoT (WAN).

    Construct with no arguments — the underlying :class:`ZenohTransport` and
    :class:`IotMqttTransport` read their config from env vars exactly as
    they would on their own.

    Lifecycle:
        :meth:`connect` succeeds if **either** side connects. The other
        side becomes a silent no-op for that path.
    """

    def __init__(
        self,
        zenoh: ZenohTransport | None = None,
        iot: IotMqttTransport | None = None,
        bridge_suffixes: frozenset[str] | None = None,
    ) -> None:
        ZenohTransportCls = _get_zenoh_transport_class()
        IotMqttTransportCls = _get_iot_transport_class()
        self._zenoh = zenoh or ZenohTransportCls()
        self._iot = iot or IotMqttTransportCls()
        self._bridge_suffixes = bridge_suffixes if bridge_suffixes is not None else _resolve_bridge_filter()
        self._bridge_prefixes = _resolve_bridge_prefix_filter()
        self._zenoh_alive = False
        self._iot_alive = False
        self._lock = threading.Lock()

        # Cross-transport command deduplicator. One instance per
        # BridgeTransport, shared between the Zenoh and IoT subscriber
        # wrappers -- whichever transport delivers a sample first wins,
        # and the other side silently drops the duplicate.
        self._dedup = _CommandDeduplicator(strict=_resolve_dedup_strict())

    # Lifecycle

    def connect(self) -> bool:
        """Connect both backends. Succeeds if at least one works."""
        with self._lock:
            self._zenoh_alive = self._zenoh.connect()
            self._iot_alive = self._iot.connect()

            if not self._zenoh_alive and not self._iot_alive:
                logger.warning("[bridge] both Zenoh and IoT failed to connect")
                return False

            if self._zenoh_alive and self._iot_alive:
                logger.info(
                    "[bridge] both transports up — bridging %d topic suffix(es): %s",
                    len(self._bridge_suffixes),
                    sorted(self._bridge_suffixes),
                )
            elif self._zenoh_alive:
                logger.info("[bridge] only Zenoh up — running LAN-only")
            else:
                logger.info("[bridge] only IoT up — running cloud-only")
            return True

    def close(self) -> None:
        """Close both backends. Idempotent.

        Narrow exception surface per AGENTS.md > Review Learnings:
        idempotent teardown swallows the documented transport-failure
        surface (RuntimeError = already-closed session,
        ConnectionError = connection drop racing with close,
        OSError = socket teardown race) and lets unexpected exceptions
        propagate.
        """
        with self._lock:
            try:
                self._zenoh.close()
            except (RuntimeError, ConnectionError, OSError) as exc:
                logger.debug("[bridge] zenoh.close() failed: %s", exc)
            try:
                self._iot.close()
            except (RuntimeError, ConnectionError, OSError) as exc:
                logger.debug("[bridge] iot.close() failed: %s", exc)
            self._zenoh_alive = False
            self._iot_alive = False

    # Inspection

    def is_alive(self) -> bool:
        return self._zenoh.is_alive() or self._iot.is_alive()

    @property
    def zenoh(self) -> ZenohTransport:
        return self._zenoh

    @property
    def iot(self) -> IotMqttTransport:
        return self._iot

    @property
    def bridge_suffixes(self) -> frozenset[str]:
        return self._bridge_suffixes

    @property
    def raw_session(self) -> Any | None:
        """The underlying ``zenoh.Session`` for backwards compatibility.

        Mesh.subscribe() historically called ``current_session().declare_subscriber(...)``
        directly. Bridge mode delegates that to the Zenoh side because user
        subscriptions to arbitrary Zenoh keys aren't necessarily MQTT-safe.
        """
        return self._zenoh.raw_session

    # Pub/Sub

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish to Zenoh always; publish to IoT only if the topic bridges.

        Failure of one side does not affect the other. Narrow exception
        surface per AGENTS.md > Review Learnings: transport-level failures
        (RuntimeError from closed session, ConnectionError from broker
        drop, OSError from socket-level write) are absorbed; everything
        else propagates.
        """
        # Always Zenoh (LAN is cheap; preserves existing behaviour).
        if self._zenoh.is_alive():
            try:
                self._zenoh.put(key, data)
            except (RuntimeError, ConnectionError, OSError) as exc:
                logger.debug("[bridge] zenoh.put error on %s: %s", key, exc)

        # Filtered IoT.
        if self._iot.is_alive() and _should_bridge(key, self._bridge_suffixes, self._bridge_prefixes):
            try:
                self._iot.put(key, data)
            except (RuntimeError, ConnectionError, OSError) as exc:
                logger.debug("[bridge] iot.put error on %s: %s", key, exc)

    def declare_subscriber(self, key_expr: str, handler: Callable[[Any], None]) -> _BridgeSubHandle:
        """Subscribe on both transports with cross-transport deduplication.

        The bridge fans subscriptions out to both Zenoh and IoT, but each
        delivered sample is funnelled through the shared
        :class:`_CommandDeduplicator`. *handler* is therefore called at most
        once per logical message even when the same payload arrives on both
        sides.

        Identity is a SHA-256 fingerprint over the canonical
        ``(sender_id, turn_id, command)`` tuple. Samples without any
        canonical fields bypass dedup and are delivered as-is (default),
        or fall back to a full-payload hash when
        ``STRANDS_MESH_BRIDGE_DEDUP_STRICT=1`` (intended for heartbeats
        that legitimately recur with identical content).
        """
        zenoh_sub: Any | None = None
        iot_sub: Any | None = None

        # One-shot warning gate shared across both transport sides (zenoh + iot).
        # A missing key_expr is a per-subscription contract drift, not per-side.
        _warned_missing_key_expr = [False]

        def make_dedup_handler(transport_label: str) -> Callable[[Any], None]:

            def _filtered(sample: Any) -> None:
                # Extract payload for dedup. We do NOT json-decode if the
                # sample doesn't expose a payload -- fall back to raw handler.
                payload: dict[str, Any] | None = None
                try:
                    raw = sample.payload.to_bytes().decode()
                    decoded = json.loads(raw)
                    if isinstance(decoded, dict):
                        payload = decoded
                except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                    # narrow per AGENTS.md > Review
                    # Learnings (#86) > "Exception Clauses Must Be Narrow".
                    # Same tuple as the four wire handlers in core.py
                    # (_on_cmd, _on_response, _on_safety_estop,
                    # _on_safety_resume). Pinned by
                    # ``test_wire_handler_narrow_except.py``.
                    payload = None

                # Use the actual delivered topic (sample.key_expr), not the
                # subscription pattern (key_expr), for dedup-cache keying.
                # A wildcard subscription like "strands/+/cmd" must not alias
                # messages delivered on distinct topics (e.g. robot-a/cmd vs
                # robot-b/cmd).  Per the _MqttSample / zenoh.Sample contracts
                # key_expr is always present; a missing attribute is a bug
                # (mock shape drift, transport refactor) so we fall back to the
                # subscription pattern AND emit a warning so the regression is
                # observable in operator logs (per AGENTS.md > Review Learnings
                # (#85) > "No silent defaults on error"). Pinned by
                # test_missing_key_expr_warns_and_falls_back.
                _sentinel = object()
                _delivered = getattr(sample, "key_expr", _sentinel)
                if _delivered is _sentinel:
                    if not _warned_missing_key_expr[0]:
                        logger.warning(
                            "[bridge] sample on subscription %r is missing"
                            " key_expr; falling back to subscription pattern"
                            " for dedup cache key (R5 contract drift -- this"
                            " reintroduces wildcard-aliasing if it persists)",
                            key_expr,
                        )
                        _warned_missing_key_expr[0] = True
                    _delivered = key_expr
                delivered_topic = str(_delivered)
                if payload is not None and self._dedup.is_duplicate(delivered_topic, payload):
                    logger.debug(
                        "[bridge] dropped duplicate from %s on %s",
                        transport_label,
                        delivered_topic,
                    )
                    return
                handler(sample)

            return _filtered

        if self._zenoh.is_alive():
            try:
                zenoh_sub = self._zenoh.declare_subscriber(key_expr, make_dedup_handler("zenoh"))
            except (RuntimeError, ConnectionError, OSError) as exc:
                # Narrow per AGENTS.md > Review Learnings: subscribe-side
                # transport failures (closed session, broker drop, socket
                # error) degrade to the surviving side; unexpected errors
                # propagate so genuine bugs aren't masked.
                logger.debug("[bridge] zenoh.declare_subscriber(%s) failed: %s", key_expr, exc)

        if self._iot.is_alive():
            try:
                iot_sub = self._iot.declare_subscriber(key_expr, make_dedup_handler("iot"))
            except (RuntimeError, ConnectionError, OSError) as exc:
                logger.debug("[bridge] iot.declare_subscriber(%s) failed: %s", key_expr, exc)

        if zenoh_sub is None and iot_sub is None:
            raise RuntimeError(f"BridgeTransport.declare_subscriber({key_expr!r}) failed on both sides")

        return _BridgeSubHandle(zenoh_sub, iot_sub)
