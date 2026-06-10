"""
Pin test for estop corroboration attribution.

When two distinct operators broadcast safety/estop at colliding t values,
the audit log should record this as estop_corroborated (positive forensic)
not estop_replay_rejected (false negative).
"""

import json
import threading
import time
from types import SimpleNamespace

from strands_robots.mesh import audit as audit_mod
from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub for safety handler testing."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._last_estop_mono = 0.0
    return m


def _envelope(t: float, peer_id: str = "issuer", **extra):
    body = {"peer_id": peer_id, "t": t, **extra}
    raw = json.dumps(body).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


_OPERATOR_A_ZID = "0123456789abcdef0123456789abcdef"
_OPERATOR_B_ZID = "fedcba9876543210fedcba9876543210"


def _envelope_with_wire_zid(t, peer_id, wire_zid, **extra):
    """Build a Zenoh-sample fake whose ``source_info.source_id.zid``
    stringifies to *wire_zid* and whose body declares ``source_zid`` to
    match (the production code requires body/wire agreement before the
    cache check is reached).
    """
    body = {"peer_id": peer_id, "t": t, "source_zid": wire_zid, **extra}
    raw = json.dumps(body).encode()

    class _Zid:
        def __init__(self, s):
            self._s = s

        def __str__(self):
            return self._s

    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda r=raw: r),
        source_info=SimpleNamespace(
            source_id=SimpleNamespace(zid=_Zid(wire_zid)),
            source_sn=1,
        ),
    )


def test_distinct_issuers_distinct_wire_zids_same_t_audited_as_corroborated():
    """
    Two distinct operators on distinct mTLS sessions with colliding t
    should audit as corroborated.

    The wire_zid distinctness is the trust anchor: peer_id is
    body-supplied (untrusted) and was the original mis-classification
    surface (R7 regression). wire_zid is bound by Zenoh's mTLS handshake
    and cannot be forged by a same-session attacker.
    """
    mesh = _stub_mesh()
    audit_calls = []

    def capture_audit(**kwargs):
        audit_calls.append(kwargs)

    mesh.publish_safety_event = capture_audit  # type: ignore[method-assign]

    # First estop from operator A on session zid A.
    envelope_t = time.time()
    mesh._estop_lockout.set()
    mesh._last_estop_ts = envelope_t
    mesh._last_estop_mono = time.monotonic()  # corroboration window check uses _last_estop_mono
    mesh._estop_replay_cache[float(envelope_t)] = (
        "operator-A",
        time.monotonic(),
        _OPERATOR_A_ZID,
    )

    # Second estop from operator B on a DIFFERENT session zid, same t.
    mesh._on_safety_estop(
        _envelope_with_wire_zid(
            t=envelope_t,
            peer_id="operator-B",
            wire_zid=_OPERATOR_B_ZID,
            reason="Operator B emergency",
        )
    )

    assert len(audit_calls) == 1, f"Expected 1 audit call, got {len(audit_calls)}"
    assert audit_calls[0]["event_type"] == "estop_corroborated", (
        f"Expected estop_corroborated, got {audit_calls[0]['event_type']}"
    )
    assert audit_calls[0]["severity"] == "info"
    assert audit_calls[0]["payload"]["issuer"] == "operator-B"
    # New: corroboration audit names the cross-session wire zids so an
    # operator dashboard can prove independence post-hoc.
    assert audit_calls[0]["payload"]["wire_zid"] == _OPERATOR_B_ZID
    assert audit_calls[0]["payload"]["corroborates_wire_zid"] == _OPERATOR_A_ZID


def test_same_wire_zid_mutated_peer_id_audited_as_replay_rejected():
    """
    R7 regression pin: a same-session attacker who captures a legitimate
    estop and republishes within 200 ms with a mutated body ``peer_id``
    must audit as ``estop_replay_rejected`` (severity ``warning``,
    operator-dashboard-visible) -- NOT ``estop_corroborated`` (severity
    ``info``, false-positive forensic).

    Pre-fix: the corroboration heuristic was "lockout active + within
    0.2 s of last estop", which classified this attacker case as
    corroboration because the cache key (``float(t)``) was peer_id-blind
    and the heuristic did not consult the wire_zid.

    Post-fix: corroboration requires both wire_zids to be non-None AND
    distinct. A same-zid replay -- regardless of body peer_id -- audits
    as replay_rejected.
    """
    mesh = _stub_mesh()
    audit_calls = []

    def capture_audit(**kwargs):
        audit_calls.append(kwargs)

    mesh.publish_safety_event = capture_audit  # type: ignore[method-assign]

    # Legitimate first estop establishes the cache slot bound to wire_zid A.
    envelope_t = time.time()
    mesh._estop_lockout.set()
    mesh._last_estop_ts = envelope_t
    mesh._estop_replay_cache[float(envelope_t)] = (
        "legit-operator",
        time.monotonic(),
        _OPERATOR_A_ZID,
    )

    # Attacker on the SAME wire_zid republishes with a mutated body
    # peer_id, hoping to earn an ``estop_corroborated`` (severity info)
    # attribution they did not provide.
    mesh._on_safety_estop(
        _envelope_with_wire_zid(
            t=envelope_t,
            peer_id="attacker-claims-corroboration",
            wire_zid=_OPERATOR_A_ZID,  # same wire_zid as the cached slot
            reason="forged corroboration attempt",
        )
    )

    assert len(audit_calls) == 1, f"Expected 1 audit call, got {len(audit_calls)}"
    # Critical: the audit must classify this as REPLAY, not corroboration.
    assert audit_calls[0]["event_type"] == "estop_replay_rejected", (
        f"R7 regression: same-wire-zid mutated-peer_id replay must audit as "
        f"estop_replay_rejected, got {audit_calls[0]['event_type']}"
    )
    assert audit_calls[0]["severity"] == "warning"
    # The replay-rejection payload preserves the (forged) attacker peer_id
    # and the original t for forensics, but the severity/event_type
    # signals are correct.
    assert audit_calls[0]["payload"]["issuer"] == "attacker-claims-corroboration"
    assert audit_calls[0]["payload"]["issuer_t"] == envelope_t


def test_attribution_less_transport_same_t_audited_as_replay_rejected():
    """
    Bridge / IoT transports legitimately have no SourceInfo, so wire_zid
    is ``None`` on either or both sides. Without a TLS-bound attribution
    anchor, corroboration cannot be proven -- the second envelope MUST
    be classified as replay (the conservative, security-preserving
    default), not silently downgraded to ``info``.
    """
    mesh = _stub_mesh()
    audit_calls = []

    def capture_audit(**kwargs):
        audit_calls.append(kwargs)

    mesh.publish_safety_event = capture_audit  # type: ignore[method-assign]

    envelope_t = time.time()
    mesh._estop_lockout.set()
    mesh._last_estop_ts = envelope_t
    # Cached slot has wire_zid=None (bridge publisher).
    mesh._estop_replay_cache[float(envelope_t)] = (
        "bridge-operator",
        time.monotonic(),
        None,
    )

    # Incoming envelope ALSO has wire_zid=None (no source_info).
    body = {"peer_id": "second-bridge-operator", "t": envelope_t}
    raw = json.dumps(body).encode()
    sample = SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda r=raw: r),
        source_info=None,
    )
    mesh._on_safety_estop(sample)

    assert len(audit_calls) == 1
    assert audit_calls[0]["event_type"] == "estop_replay_rejected", (
        "attribution-less transports cannot prove corroboration; must audit as replay_rejected"
    )


class TestEstopRedundantAudit:
    """When a second-operator estop arrives while lockout is already
    engaged, an audit event must be emitted (forensic preservation).
    """

    def test_redundant_estop_emits_audit_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        # Reset audit state for isolated test (PR4 audit-tamper-evident
        # adds these globals; on PR6 standalone they may not exist).
        if hasattr(audit_mod, "_AUDIT_STATE"):
            audit_mod._AUDIT_STATE.psk_fingerprint = None
            audit_mod._AUDIT_STATE.seq_loaded = False
        if hasattr(audit_mod, "_SEQ_COUNTERS"):
            audit_mod._SEQ_COUNTERS.clear()

        class StubRobot:
            pass

        m = core.Mesh(robot=StubRobot(), peer_id="robot-r")
        # publish_safety_event is gated on self._running; flip it on
        # without calling start() (which does network I/O). Stub publish()
        # since we only care about the audit-log side-effect.
        m._running = True
        m.publish = lambda key, data: None
        # First estop engages
        e1 = {"peer_id": "op-1", "t": time.time(), "type": "estop"}

        class S:
            def __init__(self, e):
                self.payload = type("P", (), {"to_bytes": lambda self: json.dumps(e).encode()})()

        m._on_safety_estop(S(e1))
        assert m._estop_lockout.is_set()

        # Second-operator estop, fresh `t`, lockout already engaged
        e2 = {"peer_id": "op-2", "t": time.time() + 0.5, "type": "estop"}
        m._on_safety_estop(S(e2))

        # Walk the audit log
        records = audit_mod.read_audit_log()
        events = [r["event"] for r in records]
        assert "remote_estop_engaged" in events, f"first engagement missing: {events}"
        assert "remote_estop_redundant" in events, f"second-operator audit missing: {events}"


# ---------------------------------------------------------------------
# the prior fix-1: _PSK_STATE_LOCK exists and protects fingerprint snapshot
# ---------------------------------------------------------------------


# === per-issuer fairness bound on the estop replay cache ===


class TestEstopPerIssuerFairnessBound:
    """The the prior float-only key closes peer_id-permutation replay but
    opened a denial-of-estop surface where one attacker pre-publishing
    at ``t = now + skew - eps`` could occupy float slots.

    the prior fix adds a per-issuer slot cap: each issuer may occupy at most
    ``RESUME_REPLAY_CACHE_MAX // 4`` slots before their newer entries
    are refused. This means at least 4 distinct issuers always have
    working slots regardless of attacker volume.
    """

    def test_one_issuer_cannot_exceed_cap(self, monkeypatch):
        # Tighten the cache size via env (prior made this lazy-resolved)
        monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")
        # 8 / 4 = 2 slots per issuer

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._running = True
        m.publish = lambda key, data: None
        m.publish_safety_event = lambda **kw: None  # don't audit in this test

        now = time.time()

        # Attacker fires 5 envelopes at distinct fresh `t` values
        for i in range(5):
            envelope = {"peer_id": "attacker-1", "t": now + 0.001 * i, "type": "estop"}

            class S:
                payload = type("P", (), {"to_bytes": lambda self, e=envelope: json.dumps(e).encode()})()

            m._on_safety_estop(S())

        # per-issuer count is derived from cache contents, not a
        # separate dict. Count entries owned by attacker-1; the
        # attacker is capped at per_issuer_cap = MAX // 4 = 2 slots.
        attacker_slots = sum(1 for issuer, _mono, _zid in m._estop_replay_cache.values() if issuer == "attacker-1")
        assert attacker_slots <= 2, (
            f"attacker should be capped at 2 slots, got {attacker_slots} (cache: {m._estop_replay_cache})"
        )


# === per-issuer count derived from cache contents ===


class TestPerIssuerCountFromCache:
    """the per-issuer fairness bound counts entries
    by issuer from cache contents, not a separate dict that drifts after
    eviction. After eviction, an attacker who flooded their cap legitimately
    has fewer entries and can reclaim slots -- the dynamic-attacker rate
    limit. A sustained attacker is bounded by ``per_issuer_cap`` AT EVERY
    INSTANT, not just between eviction windows.
    """

    def test_cache_carries_issuer_attribution_in_value(self):
        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._running = True
        m.publish = lambda key, data: None
        m.publish_safety_event = lambda **kw: None

        envelope = {"peer_id": "alice", "t": time.time(), "type": "estop"}

        class S:
            payload = type("P", (), {"to_bytes": lambda self: json.dumps(envelope).encode()})()

        m._on_safety_estop(S())

        assert len(m._estop_replay_cache) == 1
        # Value is (issuer_id, mono_ts, wire_zid_or_None) 3-tuple.
        # The third element captures the TLS-bound publisher zid in
        # effect when the slot was first populated; it gates the
        # corroboration-vs-replay classification on later same-t hits.
        value = next(iter(m._estop_replay_cache.values()))
        assert isinstance(value, tuple), "cache value must be (issuer_id, mono_ts, wire_zid) tuple"
        assert len(value) == 3, f"cache value must be 3-tuple, got len={len(value)}"
        issuer, mono_ts, wire_zid = value
        assert issuer == "alice"
        assert isinstance(mono_ts, float)
        # No source_info on the test envelope -> wire_zid is None.
        # This is the bridge/IoT path; corroboration is unavailable but
        # the cache entry is still valid for replay-rejection.
        assert wire_zid is None, f"expected wire_zid=None for source_info-less stub, got {wire_zid!r}"


class TestF14OverCapStillEngagesLockout:
    """Issue #263/#270: per-issuer cap MUST NOT deny lockout-engagement.
    The cap protects the bounded replay-cache resource; the lockout itself
    is idempotent boolean state with no resource cost. Suppressing the
    lockout for an over-cap issuer is a denial-of-estop on legitimate
    operators (e.g. fault-handling loops re-emitting estop on every
    detection cycle, or multi-robot consoles fanning repeated stops).

    The cap-exceeded audit event still fires; only the cache-slot
    insertion is gated. Safety-over-DoS priority is preserved.
    """

    def test_at_cap_envelope_still_engages_lockout(self, monkeypatch):
        # Tighten cap for fast test
        monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")
        # 8 / 4 = 2 slots per issuer

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._last_estop_mono = 0.0
        m._running = True
        m.publish = lambda key, data: None
        m.publish_safety_event = lambda **kw: None

        now = time.time()

        # First two envelopes from operator fill the cap and engage lockout
        for i in range(2):
            envelope = {"peer_id": "operator", "t": now + 0.001 * i, "type": "estop"}

            class S:
                payload = type("P", (), {"to_bytes": lambda self, e=envelope: json.dumps(e).encode()})()

            m._on_safety_estop(S())

        assert m._estop_lockout.is_set(), "first 2 should engage lockout"

        # Clear lockout and try a 3rd envelope (over cap) -- it MUST re-engage
        # for safety-over-DoS reasons. The cache-slot is still gated.
        m._estop_lockout.clear()
        cache_size_before = len(m._estop_replay_cache)
        envelope = {"peer_id": "operator", "t": now + 0.005, "type": "estop"}

        class S2:
            payload = type("P", (), {"to_bytes": lambda self: json.dumps(envelope).encode()})()

        m._on_safety_estop(S2())

        # Per issue #263: the lockout MUST engage even when the issuer is
        # at-cap. The cap only suppresses the cache-slot insert (DoS bound).
        assert m._estop_lockout.is_set(), (
            f"#263: over-cap envelope MUST still engage lockout (safety-over-DoS). Cache: {m._estop_replay_cache}"
        )
        # Cache slot was NOT consumed (cap held)
        assert len(m._estop_replay_cache) == cache_size_before, "cap-exceeded envelope should not consume a cache slot"


def test_per_issuer_cap_exceeded_still_engages_lockout(monkeypatch):
    """An issuer that exceeds the per-issuer cache cap still engages lockout.

    Regression for issue #263: when issuer_slots >= per_issuer_cap the
    cache slot is refused (resource fairness) but the lockout primitive
    must still engage so a legitimate operator's safety event is honored.
    """
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4")
    mesh = _stub_mesh()
    mesh.publish_safety_event = lambda **kwargs: None  # type: ignore[method-assign]

    # per_issuer_cap == max(1, 4 // 4) == 1. Pre-fill the issuer's single
    # slot at an older-but-still-fresh t so the new estop exceeds the cap.
    now = time.time()
    mesh._estop_replay_cache[float(now - 0.001)] = ("issuer", time.monotonic(), None)

    assert not mesh._estop_lockout.is_set()
    mesh._on_safety_estop(_envelope(t=now, peer_id="issuer"))

    assert mesh._estop_lockout.is_set(), "lockout must engage even when the per-issuer cache cap is exceeded"
    # The cache slot for the new t was refused (fairness preserved).
    assert float(now) not in mesh._estop_replay_cache


def test_low_cache_max_does_not_deny_safety(monkeypatch):
    """With RESUME_REPLAY_CACHE_MAX=4 a second estop from the same issuer
    still engages lockout (issue #263 embedded-peer scenario)."""
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4")
    mesh = _stub_mesh()
    mesh.publish_safety_event = lambda **kwargs: None  # type: ignore[method-assign]

    base = time.time()
    # First estop populates the single per-issuer slot and engages lockout.
    mesh._on_safety_estop(_envelope(t=base, peer_id="issuer"))
    assert mesh._estop_lockout.is_set()
    mesh._estop_lockout.clear()  # isolate the second-estop assertion

    # Second estop from the same issuer at a fresh t exceeds cap==1 but
    # must still re-engage the lockout rather than be silently denied.
    mesh._on_safety_estop(_envelope(t=base + 0.001, peer_id="issuer"))
    assert mesh._estop_lockout.is_set(), "a tight cache_max must not deny a legitimate issuer's safety primitive"


def test_cap_exceeded_audit_plus_lockout(monkeypatch):
    """Both the estop_per_issuer_cap_exceeded audit AND the lockout are
    produced when the cap is exceeded -- not one or the other (issue #263)."""
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "4")
    mesh = _stub_mesh()
    audit_calls = []
    mesh.publish_safety_event = lambda **kwargs: audit_calls.append(kwargs)  # type: ignore[method-assign]

    now = time.time()
    mesh._estop_replay_cache[float(now - 0.001)] = ("issuer", time.monotonic(), None)

    mesh._on_safety_estop(_envelope(t=now, peer_id="issuer"))

    event_types = [c["event_type"] for c in audit_calls]
    assert "estop_per_issuer_cap_exceeded" in event_types, f"expected cap-exceeded audit, got {event_types}"
    assert "remote_estop_engaged" in event_types, (
        f"expected lockout-engaged audit alongside cap audit, got {event_types}"
    )
    assert mesh._estop_lockout.is_set()
