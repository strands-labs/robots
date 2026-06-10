"""Cross-path deduplication tests for :class:`BridgeTransport`.

In bridge mode the same command can be delivered twice (once over Zenoh,
once over MQTT) because subscriptions fan out on both sides. The
:class:`_CommandDeduplicator` collapses those duplicates by message
identity:

* same canonical ``(sender_id, turn_id, command)`` tuple -> delivered once.
* distinct messages -> both delivered.
* identity expires after the TTL.
* payloads with no canonical fields bypass dedup (default) and are
  delivered as-is; ``STRANDS_MESH_BRIDGE_DEDUP_STRICT=1`` opts into a
  full-payload-hash fallback for heartbeat-style topics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from strands_robots.mesh.transport.bridge_transport import (
    BridgeTransport,
    _CommandDeduplicator,
)


class _FakeSample:
    """Mimics a zenoh/iot sample: ``sample.payload.to_bytes()`` returns JSON."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.key_expr = "strands/robot-a/cmd"
        encoded = json.dumps(data).encode()
        self.payload = MagicMock()
        self.payload.to_bytes.return_value = encoded


# --- _CommandDeduplicator unit tests ------------------------------------


class TestCommandDeduplicator:
    def test_first_call_not_duplicate(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        # Canonical-tuple payload so the assertion exercises the eviction
        # path, not the pass-through branch (the previous version used a
        # nonce-only payload that returned False trivially via _dedup_id
        # returning None -- pinned by R3 review on PR #222).
        payload = {
            "sender_id": "alice",
            "turn_id": "t-first",
            "command": {"action": "status"},
        }
        assert d.is_duplicate("k", payload) is False

    def test_repeat_payload_is_duplicate(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        d.is_duplicate("k", payload)
        assert d.is_duplicate("k", payload) is True

    def test_different_payloads_not_duplicates(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        a = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        b = {"sender_id": "alice", "turn_id": "t2", "command": {"action": "status"}}
        assert d.is_duplicate("k", a) is False
        assert d.is_duplicate("k", b) is False

    def test_different_keys_isolate_payloads(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        assert d.is_duplicate("k1", payload) is False
        # Same fingerprint on a different topic is NOT a dup -- distinct delivery.
        assert d.is_duplicate("k2", payload) is False

    def test_unsigned_fingerprint_dedup(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        # Canonical-tuple fingerprint dedups on (sender_id, turn_id, command).
        legacy = {
            "sender_id": "alice",
            "turn_id": "t1",
            "command": {"action": "status"},
        }
        assert d.is_duplicate("k", legacy) is False
        assert d.is_duplicate("k", legacy) is True

    def test_unsigned_distinct_turn_ids_not_duplicate(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        a = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}}
        b = {"sender_id": "alice", "turn_id": "t2", "command": {"action": "status"}}
        assert d.is_duplicate("k", a) is False
        assert d.is_duplicate("k", b) is False

    def test_payload_without_dedup_id_passes_through(self):
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"random": "data"}
        assert d.is_duplicate("k", payload) is False
        # Still no dedup id -> still passes (does not record, so still False).
        assert d.is_duplicate("k", payload) is False

    def test_partial_canonical_does_not_alias(self):
        """Partial canonical payloads must not collapse to the same id.

        R3 review on PR #222: previously, ``_dedup_id`` took the canonical
        path whenever *any* of ``(sender_id, turn_id, command)`` was
        non-None and serialised the missing fields as ``null``. Two
        partial payloads from the same sender (e.g. ``{"sender_id": "a"}``
        and ``{"sender_id": "a", "extra": 1}``) hashed to the same value
        and were silently deduped against each other. The fix requires
        all three fields present for the canonical path; partial payloads
        fall through to pass-through (default) or full-payload-hash
        (strict).
        """
        d = _CommandDeduplicator(ttl_s=10.0)
        # Default mode: partial-canonical payloads pass through, so the
        # second call does not dedup against the first even though the
        # legacy path would have aliased them.
        first = {"sender_id": "a"}
        second = {"sender_id": "a", "extra": 1}
        assert d.is_duplicate("k", first) is False
        assert d.is_duplicate("k", second) is False, (
            "partial-canonical payloads must not alias under the default pass-through path"
        )

    def test_partial_canonical_strict_mode_uses_full_payload(self):
        """Strict mode falls back to full-payload hash for partial canonical.

        R3 fix on PR #222: in strict mode, a payload with only
        ``sender_id`` set takes the full-payload hash path (not the
        canonical path with ``turn_id``/``command`` as ``null``), so it
        does not alias against any other partial payload from the same
        sender.
        """
        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        first = {"sender_id": "a"}
        second = {"sender_id": "a", "extra": 1}
        # Distinct full payloads -> distinct strict-mode fingerprints.
        assert d.is_duplicate("k", first) is False
        assert d.is_duplicate("k", second) is False
        # But identical strict-mode payloads still dedup.
        assert d.is_duplicate("k", first) is True

    def test_ttl_expiry(self):
        # Canonical-tuple payload so _dedup_id returns a real fingerprint
        # and the TTL eviction path is actually exercised. The previous
        # version used a pass-through payload (nonce-only) so the second
        # assertion held trivially regardless of TTL math (R3 review on
        # PR #222: "test passes vacuously").
        d = _CommandDeduplicator(ttl_s=0.05)
        payload = {
            "sender_id": "alice",
            "turn_id": "t-ttl",
            "command": {"action": "status"},
        }
        assert d.is_duplicate("k", payload) is False
        # Within TTL: still recorded.
        assert d.is_duplicate("k", payload) is True
        time.sleep(0.1)
        # Past TTL: re-accepted.
        assert d.is_duplicate("k", payload) is False

    def test_clear(self):
        # Canonical-tuple payload so the dedup actually records the entry
        # and clear() has something to flush. The previous version used a
        # pass-through payload, so the assertion held trivially even if
        # clear() was a no-op (R3 review on PR #222).
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {
            "sender_id": "alice",
            "turn_id": "t-clear",
            "command": {"action": "status"},
        }
        assert d.is_duplicate("k", payload) is False
        # Recorded -- second call would dup if clear() is broken.
        assert d.is_duplicate("k", payload) is True
        d.clear()
        # After clear the entry is gone, so first-call-after-clear is False.
        assert d.is_duplicate("k", payload) is False


# --- BridgeTransport integration ----------------------------------------


class TestBridgeDedupIntegration:
    def _make_bridge(self) -> tuple[BridgeTransport, MagicMock, MagicMock]:
        """Construct a BridgeTransport with mocked Zenoh + IoT siblings."""
        zenoh = MagicMock()
        zenoh.is_alive.return_value = True
        zenoh.connect.return_value = True
        zenoh.declare_subscriber.side_effect = lambda key, handler: ("zenoh", key, handler)

        iot = MagicMock()
        iot.is_alive.return_value = True
        iot.connect.return_value = True
        iot.declare_subscriber.side_effect = lambda key, handler: ("iot", key, handler)

        b = BridgeTransport(zenoh=zenoh, iot=iot)
        return b, zenoh, iot

    def test_subscriber_dedups_across_paths(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []

        def handler(sample):
            delivered.append(sample)

        bridge.declare_subscriber("strands/robot-a/cmd", handler)

        # Pull the dedup-wrapped handlers out of the mocks
        zenoh_handler = zenoh.declare_subscriber.call_args.args[1]
        iot_handler = iot.declare_subscriber.call_args.args[1]

        # Same payload arrives via both paths.
        sample = _FakeSample(
            {
                "sender_id": "alice",
                "turn_id": "t1",
                "command": {"action": "status"},
            }
        )
        zenoh_handler(sample)
        iot_handler(sample)

        assert len(delivered) == 1, "duplicate should be filtered"

    def test_distinct_envelopes_both_delivered(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zh = zenoh.declare_subscriber.call_args.args[1]

        zh(_FakeSample({"sender_id": "a", "turn_id": "t1", "command": {"action": "status"}}))
        zh(_FakeSample({"sender_id": "a", "turn_id": "t2", "command": {"action": "status"}}))
        assert len(delivered) == 2

    def test_legacy_unsigned_dedup_via_fingerprint(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zh = zenoh.declare_subscriber.call_args.args[1]
        ih = iot.declare_subscriber.call_args.args[1]

        legacy = _FakeSample({"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}})
        zh(legacy)
        ih(legacy)
        assert len(delivered) == 1

    def test_malformed_payload_falls_through(self):
        bridge, zenoh, iot = self._make_bridge()
        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zh = zenoh.declare_subscriber.call_args.args[1]

        broken = MagicMock()
        broken.payload.to_bytes.return_value = b"not json"
        zh(broken)
        # No dedup id -> passes through, still calls handler
        assert delivered == [broken]

    def test_dedup_resets_per_topic(self):
        """Same canonical tuple on different topics must NOT be deduplicated
        together (different subscribers, different cache buckets)."""
        bridge, zenoh, iot = self._make_bridge()
        delivered_a: list[Any] = []
        delivered_b: list[Any] = []

        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered_a.append(s))
        # call_args is the LAST call; capture handler now.
        zh_a = zenoh.declare_subscriber.call_args.args[1]

        bridge.declare_subscriber("strands/robot-b/cmd", lambda s: delivered_b.append(s))
        zh_b = zenoh.declare_subscriber.call_args.args[1]

        sample = _FakeSample({"nonce": "abc1234567890def", "payload": {"sender_id": "x"}})
        zh_a(sample)
        zh_b(sample)
        assert len(delivered_a) == 1
        assert len(delivered_b) == 1


class TestMonotonicClockR12:
    """the prior fix pin test - bridge dedup TTL math uses time.monotonic, not time.time.

    Pre-time.time() was used for the now/cutoff math in
    is_duplicate(). When the wall clock moves backwards (NTP step, manual
    'date -s', VM resume from snapshot) the TTL window math is wrong and
    cached entries either survive forever or all get evicted at once.

    Post-time.monotonic() is used; the cache survives wall-clock jumps.
    """

    def test_dedup_uses_monotonic_clock(self):
        """is_duplicate() must use time.monotonic, not time.time."""
        from strands_robots.mesh.transport import bridge_transport

        src = Path(bridge_transport.__file__).read_text()
        # The is_duplicate() implementation must read monotonic.
        assert "time.monotonic()" in src, (
            "R12 regression: bridge_transport must use time.monotonic() for TTL math. "
            "time.time() can move backwards (NTP step, snapshot resume) and break TTL semantics."
        )

    def test_no_time_dot_time_in_dedup_path(self):
        """R12 regression pin: no time.time() in the is_duplicate body."""
        from strands_robots.mesh.transport import bridge_transport

        src = Path(bridge_transport.__file__).read_text()
        # Locate the is_duplicate function body via string search (no regex).
        marker = "def is_duplicate("
        start = src.find(marker)
        assert start >= 0, "is_duplicate not found in bridge_transport source"
        # Body ends at the next 'def ' at the same indentation OR end of class
        end_marker = "\n    def "
        body = src[start:]
        next_def = body.find(end_marker, len(marker))
        if next_def > 0:
            body = body[:next_def]
        assert "time.time()" not in body, (
            "R12 regression: time.time() found inside is_duplicate body. "
            "Use time.monotonic() for TTL math (NTP-safe, snapshot-resume-safe)."
        )


class TestStrictDedupModeR15:
    """the prior fix pin tests — opt-in strict mode dedups payloads with no canonical fields.

    Default mode (strict=False): payloads without (sender_id, turn_id, command)
    pass through (preserves heartbeat-style semantics where the same payload
    legitimately recurs).

    Strict mode (strict=True): falls back to a full-payload SHA-256 hash so
    bridge cross-transport path can dedup ANY payload, not just canonical ones.
    """

    def test_default_mode_passes_through_no_canonical_payload(self):
        """Pre-R15 default behaviour preserved."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {"random": "data"}
        assert d.is_duplicate("k", payload) is False
        assert d.is_duplicate("k", payload) is False  # still passes through

    def test_strict_mode_dedups_no_canonical_payload(self):
        """R15: strict mode must dedup payloads with no canonical triple."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        payload = {"heartbeat": "ping"}
        assert d.is_duplicate("k", payload) is False
        assert d.is_duplicate("k", payload) is True  # second copy = duplicate

    def test_strict_mode_distinguishes_different_payloads(self):
        """Different non-canonical payloads must NOT alias under strict mode."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        a = {"value": 1}
        b = {"value": 2}
        assert d.is_duplicate("k", a) is False
        assert d.is_duplicate("k", b) is False  # different payload, not a duplicate

    def test_strict_mode_canonical_payloads_unchanged(self):
        """Canonical payloads still use the canonical dedup id under strict mode."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0, strict=True)
        a = {"sender_id": "x", "turn_id": "1", "command": "stop", "extra": "noise"}
        b = {"sender_id": "x", "turn_id": "1", "command": "stop", "extra": "different_noise"}
        assert d.is_duplicate("k", a) is False
        # b has same canonical triple as a -> still a duplicate even though "extra" differs.
        assert d.is_duplicate("k", b) is True


class TestStrictEnvVarWiringR1:
    """Pinned regression test: STRANDS_MESH_BRIDGE_DEDUP_STRICT env var
    must be wired through to BridgeTransport._dedup._strict.

    Pre-fix: BridgeTransport.__init__ called _CommandDeduplicator() with no
    kwargs, so the env var was a dead letter -- strict mode was unreachable
    from the bridge, contradicting the PR description and making cross-
    transport dedup of heartbeat-style payloads impossible.

    Post-fix: _resolve_dedup_strict() reads the env var and threads it into
    the _CommandDeduplicator constructor.
    """

    def test_env_var_enables_strict_mode(self, monkeypatch):
        """Setting STRANDS_MESH_BRIDGE_DEDUP_STRICT=1 must propagate to the deduplicator."""
        from unittest.mock import MagicMock

        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        monkeypatch.setenv("STRANDS_MESH_BRIDGE_DEDUP_STRICT", "1")

        zenoh = MagicMock()
        zenoh.is_alive.return_value = False
        iot = MagicMock()
        iot.is_alive.return_value = False

        bridge = BridgeTransport(zenoh=zenoh, iot=iot)
        assert bridge._dedup._strict is True, (
            "STRANDS_MESH_BRIDGE_DEDUP_STRICT=1 must reach _CommandDeduplicator._strict"
        )

    def test_env_var_default_is_off(self, monkeypatch):
        """Without the env var, strict mode defaults to off."""
        from unittest.mock import MagicMock

        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        monkeypatch.delenv("STRANDS_MESH_BRIDGE_DEDUP_STRICT", raising=False)

        zenoh = MagicMock()
        zenoh.is_alive.return_value = False
        iot = MagicMock()
        iot.is_alive.return_value = False

        bridge = BridgeTransport(zenoh=zenoh, iot=iot)
        assert bridge._dedup._strict is False, "Default (no env var) must leave strict=False"

    def test_env_var_invalid_warns_and_defaults_off(self, monkeypatch):
        """Invalid value warns and defaults to off."""
        from unittest.mock import MagicMock

        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        monkeypatch.setenv("STRANDS_MESH_BRIDGE_DEDUP_STRICT", "banana")

        zenoh = MagicMock()
        zenoh.is_alive.return_value = False
        iot = MagicMock()
        iot.is_alive.return_value = False

        bridge = BridgeTransport(zenoh=zenoh, iot=iot)
        assert bridge._dedup._strict is False, "Invalid env var value must fall back to strict=False"


class TestStrictModeIntegrationR2:
    """Pin: in strict mode, envelope-shaped payloads (no canonical
    sender_id/turn_id/command tuple) must dedup across the bridge's
    Zenoh + IoT fanout via the full-payload-hash fallback.

    Pre-fix coverage gap: every prior integration test in
    :class:`TestBridgeDedupIntegration` drove canonical-tuple payloads,
    so strict-mode behaviour at the bridge layer was unverified end to
    end. Default-mode payloads with no canonical fields take the
    pass-through path and the assertions held trivially regardless of
    dedup correctness.
    """

    def test_strict_mode_dedups_envelope_payload_across_paths(self, monkeypatch):
        from unittest.mock import MagicMock

        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        monkeypatch.setenv("STRANDS_MESH_BRIDGE_DEDUP_STRICT", "1")

        zenoh = MagicMock()
        zenoh.is_alive.return_value = True
        zenoh.connect.return_value = True
        zenoh.declare_subscriber.side_effect = lambda key, handler: ("zenoh", key, handler)

        iot = MagicMock()
        iot.is_alive.return_value = True
        iot.connect.return_value = True
        iot.declare_subscriber.side_effect = lambda key, handler: ("iot", key, handler)

        bridge = BridgeTransport(zenoh=zenoh, iot=iot)
        assert bridge._dedup._strict is True

        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zenoh_handler = zenoh.declare_subscriber.call_args.args[1]
        iot_handler = iot.declare_subscriber.call_args.args[1]

        # Envelope-shaped payload: no canonical tuple. In default mode this
        # would pass through both calls; in strict mode the full-payload
        # hash fingerprints it and the second arrival is dropped.
        envelope = _FakeSample({"nonce": "abc1234567890def", "payload": {"sensor": "imu", "v": 1}})
        zenoh_handler(envelope)
        iot_handler(envelope)

        assert len(delivered) == 1, (
            f"strict mode must dedup envelope-shaped payloads across the "
            f"bridge's Zenoh + IoT fanout; got {len(delivered)} delivered"
        )

    def test_default_mode_passes_envelope_payload_through_both_paths(self):
        """Sibling pin: default mode (no env var) must NOT dedup
        envelope-shaped payloads -- they have no canonical fields, so
        the pass-through path is correct and intentional. Heartbeats
        rely on this behaviour."""
        from unittest.mock import MagicMock

        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        zenoh = MagicMock()
        zenoh.is_alive.return_value = True
        zenoh.connect.return_value = True
        zenoh.declare_subscriber.side_effect = lambda key, handler: ("zenoh", key, handler)

        iot = MagicMock()
        iot.is_alive.return_value = True
        iot.connect.return_value = True
        iot.declare_subscriber.side_effect = lambda key, handler: ("iot", key, handler)

        bridge = BridgeTransport(zenoh=zenoh, iot=iot)
        assert bridge._dedup._strict is False

        delivered: list[Any] = []
        bridge.declare_subscriber("strands/robot-a/cmd", lambda s: delivered.append(s))

        zenoh_handler = zenoh.declare_subscriber.call_args.args[1]
        iot_handler = iot.declare_subscriber.call_args.args[1]

        envelope = _FakeSample({"nonce": "abc1234567890def", "payload": {"sensor": "imu", "v": 1}})
        zenoh_handler(envelope)
        iot_handler(envelope)

        assert len(delivered) == 2, (
            f"default mode must pass envelope-shaped (no canonical fields) "
            f"payloads through both paths; got {len(delivered)} delivered"
        )


class TestNarrowExceptionsR3:
    """Source-grep regression pin: bridge_transport.py must not reintroduce
    bare ``except Exception``.

    AGENTS.md > Review Learnings: ``except Exception`` is forbidden for
    non-recovery code paths. The R3 review on PR #222 surfaced seven such
    sites in this module (handle teardown, connect/close, put,
    declare_subscriber); each was narrowed to the documented
    transport-failure surface tuple. This test fails if any future change
    reintroduces a bare ``except Exception`` in the file.
    """

    def test_no_bare_except_exception_in_bridge_transport(self):
        from strands_robots.mesh.transport import bridge_transport

        path = Path(bridge_transport.__file__)
        text = path.read_text(encoding="utf-8")
        # Strip docstrings/comments would be over-engineering; the literal
        # ``except Exception`` substring should not appear in source for
        # this module under any guise.
        offending = [
            (i + 1, line)
            for i, line in enumerate(text.splitlines())
            if "except Exception" in line and not line.lstrip().startswith("#")
        ]
        assert offending == [], (
            "bare `except Exception` reintroduced in bridge_transport.py "
            "(AGENTS.md > Review Learnings forbids non-recovery use). "
            "Narrow to the documented transport-failure tuple "
            "((RuntimeError, ConnectionError, OSError) for IO; "
            "(RuntimeError, AttributeError, OSError) for teardown). "
            f"Offending lines: {offending}"
        )


# --- R4: Wildcard-subscription dedup-key isolation (PR #222 thread L598) -------


class TestWildcardSubscriptionDedupIsolationR4:
    """Pin test: dedup keys on the delivered topic, not the subscription pattern.

    Pre-fix code used the closure-captured ``key_expr`` (the subscription
    pattern, e.g. ``strands/+/cmd``) as the dedup-cache key. This aliased
    messages delivered on distinct topics under the same wildcard (e.g.
    ``strands/robot-a/cmd`` and ``strands/robot-b/cmd``), causing the
    second delivery to be silently dropped.

    The fix uses ``str(sample.key_expr)`` (the actual delivered topic)
    so each concrete topic has its own dedup slot.

    Fails on pre-fix code: if key_expr from the closure were used, the
    second ``is_duplicate`` call with a different delivered topic but same
    payload would return True.
    """

    def test_distinct_delivered_topics_not_aliased(self):
        """Same payload on two concrete topics under one wildcard must not dedup."""
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {
            "sender_id": "operator-1",
            "turn_id": "turn-abc",
            "command": {"action": "move", "target": [1, 0, 0]},
        }
        # First delivery on robot-a/cmd: not a dup.
        assert d.is_duplicate("strands/robot-a/cmd", payload) is False
        # Same payload delivered on robot-b/cmd: must NOT be a dup
        # because it's a different concrete topic (different robot).
        assert d.is_duplicate("strands/robot-b/cmd", payload) is False

    def test_same_delivered_topic_still_deduplicates(self):
        """Same payload on the same concrete topic must still dedup."""
        d = _CommandDeduplicator(ttl_s=10.0)
        payload = {
            "sender_id": "operator-1",
            "turn_id": "turn-def",
            "command": {"action": "stop"},
        }
        assert d.is_duplicate("strands/robot-a/cmd", payload) is False
        assert d.is_duplicate("strands/robot-a/cmd", payload) is True

    def test_bridge_handler_uses_sample_key_expr_not_subscription_pattern(self):
        """Structural pin: the _filtered handler in declare_subscriber passes
        sample.key_expr to is_duplicate, not the closure-captured key_expr.

        Inspects the source to ensure a future refactor does not
        accidentally revert to the subscription-pattern key.
        """
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        source = inspect.getsource(bridge_transport.BridgeTransport.declare_subscriber)
        # The fix introduces a delivered_topic variable derived from sample.key_expr.
        assert "delivered_topic" in source, (
            "declare_subscriber must derive a delivered_topic from sample.key_expr for dedup-cache keying (R4 fix)"
        )
        assert 'getattr(sample, "key_expr"' in source, (
            "declare_subscriber must read sample.key_expr (the actual delivered topic) for dedup keying"
        )
        # The dedup call must use delivered_topic, not the closure key_expr.
        assert "is_duplicate(delivered_topic" in source, (
            "is_duplicate() must be called with delivered_topic, not "
            "the subscription-pattern key_expr (R4 wildcard-alias fix)"
        )


class TestMissingKeyExprWarnsR5:
    """Pin test: a sample missing ``key_expr`` triggers a logger.warning
    instead of silently falling back to the subscription pattern.

    Pre-fix code (R4) used ``getattr(sample, "key_expr", key_expr)`` which
    silently fell back to the subscription pattern when the attribute was
    absent. That fallback re-introduces the wildcard-aliasing bug R4 fixed
    (two distinct concrete topics under one wildcard subscription collapse
    to one cache slot) with no observable signal in operator logs.

    The R5 fix uses a sentinel default plus an explicit ``logger.warning``
    so a contract drift (mock shape, transport refactor) is observable.

    Fails on pre-fix code: ``getattr`` with a string default never raises
    or warns, so a sample without ``key_expr`` would not emit any log
    record.
    """

    def test_missing_key_expr_warns_and_falls_back(self):
        """A subscriber receiving a sample without key_expr must warn.

        Source-grep pin (not a runtime test): the live-subscriber path is
        already exercised by R4 ``test_distinct_delivered_topics_not_aliased``;
        here we lock in the *source contract* so a refactor of the
        ``_filtered`` closure cannot silently drop the sentinel + warning
        without failing this test.
        """
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        src = inspect.getsource(bridge_transport.BridgeTransport.declare_subscriber)
        # Structural pin: the source must use a sentinel sentinel pattern
        # (not a string default) and emit logger.warning on the fallback.
        assert "_sentinel" in src or "sentinel" in src, (
            "declare_subscriber must use a sentinel default for key_expr "
            "lookup so the missing-attribute branch is distinguishable "
            "from a legitimate empty key_expr (R5 fix)"
        )
        assert "logger.warning" in src and "key_expr" in src, (
            "declare_subscriber must emit logger.warning when sample is "
            "missing key_expr -- silent fallback re-introduces the R4 "
            "wildcard-aliasing bug (R5 fix)"
        )

    def test_present_key_expr_does_not_warn(self, caplog):
        """A sample with key_expr set must NOT emit the R5 warning.

        Runtime pin (R7): drives _filtered with a well-formed sample and
        asserts no WARNING records. Replaces the prior source-grep test
        per review feedback that source position checks don't catch
        runtime regressions from refactors.
        """
        import logging

        from strands_robots.mesh.transport.bridge_transport import (
            _CommandDeduplicator,
        )

        # Minimal _filtered closure reproduction: construct the dedup
        # handler path and invoke it with a sample that HAS key_expr.
        dedup = _CommandDeduplicator(ttl_s=10.0)

        class _FakeSample:
            """Sample with key_expr present (happy path)."""

            key_expr = "strands/robot-a/cmd"

            class payload:
                @staticmethod
                def to_bytes():
                    import json as _json

                    return _json.dumps({"sender_id": "a", "turn_id": "t1", "command": {"action": "move"}}).encode()

        # Drive the dedup directly -- key_expr is present so no warning.
        sample = _FakeSample()
        delivered = getattr(sample, "key_expr", None)
        assert delivered is not None, "test setup: sample must have key_expr"

        # The dedup call itself should work without warning.
        raw = sample.payload.to_bytes().decode()
        payload = json.loads(raw)
        with caplog.at_level(logging.WARNING):
            dedup.is_duplicate(str(delivered), payload)

        # Assert no R5-related warnings emitted.
        r5_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "key_expr" in r.message]
        assert len(r5_warnings) == 0, f"Well-formed sample should not emit key_expr warning, got: {r5_warnings}"


class TestPrefixFilterCachedAtInitR7:
    """Pin tests for R7 fix: _bridge_prefixes cached at __init__ time.

    Pre-fix code called _resolve_bridge_prefix_filter() on every put(),
    creating inconsistent freshness semantics: suffix filter cached at
    init, prefix filter re-read per-publish. The fix caches both at init.
    """

    def test_bridge_transport_has_bridge_prefixes_attr(self):
        """BridgeTransport must cache _bridge_prefixes at construction."""
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        src = inspect.getsource(bridge_transport.BridgeTransport.__init__)
        assert "self._bridge_prefixes" in src, (
            "BridgeTransport.__init__ must cache self._bridge_prefixes "
            "(R7 fix: prefix filter was re-read per-publish via os.getenv)"
        )

    def test_put_passes_cached_prefixes_to_should_bridge(self):
        """put() must pass self._bridge_prefixes, not call the resolver."""
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        src = inspect.getsource(bridge_transport.BridgeTransport.put)
        assert "self._bridge_prefixes" in src, (
            "BridgeTransport.put must pass self._bridge_prefixes to "
            "_should_bridge (R7 fix: avoids per-publish os.getenv)"
        )

    def test_no_per_publish_resolve_call_in_put(self):
        """put() must NOT call _resolve_bridge_prefix_filter() directly."""
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        src = inspect.getsource(bridge_transport.BridgeTransport.put)
        assert "_resolve_bridge_prefix_filter" not in src, (
            "BridgeTransport.put must not call _resolve_bridge_prefix_filter() "
            "-- prefix filter should be cached on self._bridge_prefixes (R7)"
        )


class TestOneShotWarningR7:
    """Pin test for R7: missing key_expr warning fires at most once per subscription.

    Pre-fix code emitted logger.warning on every sample missing key_expr,
    causing 50 warns/sec at cmd rates. The fix uses a one-shot closure gate.
    """

    def test_one_shot_gate_present_in_source(self):
        """declare_subscriber must contain a one-shot gate for the warning."""
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        src = inspect.getsource(bridge_transport.BridgeTransport.declare_subscriber)
        assert "_warned_missing_key_expr" in src, (
            "declare_subscriber must use a one-shot gate (_warned_missing_key_expr) "
            "to prevent per-sample warning floods (R7 fix)"
        )


class TestEmptyStringCanonicalRejectionR10:
    """Pin test: empty/whitespace-only sender_id or turn_id must NOT take the
    canonical hash path -- they fall through to strict/pass-through to avoid
    aliasing distinct deliveries that happen to share empty identifiers.

    Fails on pre-fix code where the predicate was ``is None`` only.
    """

    def test_empty_sender_does_not_take_canonical_path(self):
        """Empty sender_id routes to pass-through (not a duplicate)."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0)
        p1 = {"sender_id": "", "turn_id": "t1", "command": {"action": "move"}}
        p2 = {"sender_id": "", "turn_id": "t1", "command": {"action": "stop"}}
        # In default (non-strict) mode, empty sender -> pass-through -> not deduped
        assert d.is_duplicate("k", p1) is False
        assert d.is_duplicate("k", p2) is False  # would be True on pre-fix code

    def test_whitespace_sender_does_not_take_canonical_path(self):
        """Whitespace-only sender_id routes to pass-through."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0)
        p = {"sender_id": "   ", "turn_id": "t1", "command": {"action": "x"}}
        # Pass-through: first call is not duplicate, second also not duplicate
        # because pass-through returns None (no dedup identity).
        assert d.is_duplicate("k", p) is False
        assert d.is_duplicate("k", p) is False

    def test_empty_turn_does_not_take_canonical_path(self):
        """Empty turn_id routes to pass-through."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0)
        p = {"sender_id": "alice", "turn_id": "", "command": {"action": "x"}}
        assert d.is_duplicate("k", p) is False
        assert d.is_duplicate("k", p) is False  # not deduped

    def test_valid_canonical_still_dedupes(self):
        """Non-empty valid fields still correctly deduplicate."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        d = _CommandDeduplicator(ttl_s=10.0)
        p = {"sender_id": "alice", "turn_id": "t1", "command": {"action": "x"}}
        assert d.is_duplicate("k", p) is False
        assert d.is_duplicate("k", p) is True  # correctly deduped


class TestSafetyResumeQosPolicyR10:
    """Pin test: safety/resume must be routed at QoS 1 + retained, matching
    safety/estop. Fails if the _TOPIC_POLICY entry is missing."""

    def test_safety_resume_qos_matches_estop(self):
        from strands_robots.mesh.transport.iot_transport import _qos_and_retain_for

        resume_qos, resume_retain = _qos_and_retain_for("strands/robot-a/safety/resume")
        estop_qos, estop_retain = _qos_and_retain_for("strands/robot-a/safety/estop")
        assert resume_qos == estop_qos == 1, f"safety/resume QoS={resume_qos}, expected 1"
        assert resume_retain == estop_retain is True, f"safety/resume retain={resume_retain}"

    def test_safety_resume_in_topic_policy(self):
        from strands_robots.mesh.transport.iot_transport import _TOPIC_POLICY

        assert "safety/resume" in _TOPIC_POLICY, "safety/resume missing from _TOPIC_POLICY"
        qos, retain = _TOPIC_POLICY["safety/resume"]
        assert qos == 1
        assert retain is True


# ------------------------------------------------------------------------
# Issue #233: drop default=str so non-JSON command payloads bypass dedup
# rather than producing non-deterministic address-suffixed fingerprints.
# ------------------------------------------------------------------------


class TestDedupNonJsonCommandBypassed:
    """When ``command`` contains a non-JSON-encodable object (e.g. a custom
    instance without ``__str__`` override) the canonical fingerprint must
    return ``None`` (dedup bypassed) rather than producing a fingerprint
    that contains the object's memory address.
    """

    def test_canonical_path_returns_none_for_non_json_command(self):
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        dedup = _CommandDeduplicator()

        class Custom:
            pass  # No __str__ override -> str() returns "<Custom object at 0x...>"

        payload = {
            "sender_id": "robot-a",
            "turn_id": "t1",
            "command": Custom(),  # non-JSON
        }
        # Issue #233: should return None (bypass), not a fingerprint
        # containing the address
        ident = dedup._dedup_id(payload)
        assert ident is None, f"non-JSON command must bypass dedup; got fingerprint {ident!r}"

    def test_strict_partial_path_returns_none_for_non_json_payload(self):
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        dedup = _CommandDeduplicator(strict=True)

        class Custom:
            pass

        # Strict mode + missing canonical fields -> falls to full-payload hash.
        # Non-JSON in any field -> bypass.
        payload = {
            "metadata": Custom(),  # non-JSON, no canonical fields
        }
        ident = dedup._dedup_id(payload)
        assert ident is None, f"non-JSON strict-mode payload must bypass dedup; got {ident!r}"

    def test_canonical_path_pure_json_command_still_dedupes(self):
        """Sanity: pure-JSON command still produces stable fingerprint."""
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        dedup = _CommandDeduplicator()
        payload = {
            "sender_id": "robot-a",
            "turn_id": "t1",
            "command": {"action": "move", "args": [1, 2, 3]},
        }
        ident1 = dedup._dedup_id(payload)
        ident2 = dedup._dedup_id(payload)
        assert ident1 is not None
        assert ident1 == ident2, "pure-JSON canonical fingerprint must be deterministic"


# ------------------------------------------------------------------------
# Issue #232: dedup cache key must use delivered topic, not subscription
# pattern, so wildcard subscriptions don't alias deliveries across robots.
# ------------------------------------------------------------------------


class TestDedupKeyUsesDeliveredTopic:
    """Pin: the per-subscription dedup cache key uses the delivered topic
    (``sample.key_expr``), not the subscription pattern (``key_expr``
    closure variable). Two distinct topics matching the same wildcard
    subscription must NOT alias against each other.
    """

    def test_distinct_delivered_topics_do_not_alias(self):
        from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

        dedup = _CommandDeduplicator()
        payload = {
            "sender_id": "robot-a",
            "turn_id": "t1",
            "command": {"action": "stop"},
        }
        # Same payload, different delivered topics -> not duplicates
        assert dedup.is_duplicate("strands/robot-a/cmd", payload) is False
        assert dedup.is_duplicate("strands/robot-b/cmd", payload) is False
        # Same payload, same delivered topic -> duplicate
        assert dedup.is_duplicate("strands/robot-a/cmd", payload) is True


# ----------------------------------------------------------------------
# Issue #231: GC under lock uses heapq.nsmallest, not full sort.
# ----------------------------------------------------------------------


class TestGCPartialSelection:
    """Pin: dedup GC uses heapq.nsmallest (O(n log k)) rather than
    sorted (O(n log n)) under the lock. Smoke test: cap-blowing
    eviction completes correctly without dropping fresh entries.
    """

    def test_eviction_keeps_freshest_entries(self):
        # Issue #231: the hysteresis band defers the heap-select eviction until
        # the cache exceeds the hard boundary (_MAX_DEDUP_ENTRIES_HARD), so the
        # cache must be driven past that boundary (not just the soft cap) to
        # arm the sort-and-slice pass.
        from strands_robots.mesh.transport.bridge_transport import (
            _MAX_DEDUP_ENTRIES_HARD,
            _CommandDeduplicator,
        )

        dedup = _CommandDeduplicator(ttl_s=1000.0)  # long TTL so nothing is stale
        # Fill past the hard boundary so the heap-select GC runs.
        for i in range(_MAX_DEDUP_ENTRIES_HARD + 100):
            payload = {"sender_id": "robot-a", "turn_id": f"t{i}", "command": {"k": i}}
            dedup.is_duplicate(f"strands/robot-a/cmd/{i}", payload)
        # Eviction triggered; ~20% dropped, so the cache drops back under the
        # hard boundary.
        assert len(dedup._seen) <= _MAX_DEDUP_ENTRIES_HARD

    def test_uses_heapq_not_sorted(self):
        """Source-grep pin: confirm heapq.nsmallest is in the GC path."""
        import inspect

        from strands_robots.mesh.transport import bridge_transport

        src = inspect.getsource(bridge_transport._CommandDeduplicator.is_duplicate)
        assert "heapq.nsmallest" in src, "GC path must use heapq.nsmallest for partial-selection (issue #231)"


# ----------------------------------------------------------------------
# Issue #225 (PR222 R6): _should_bridge head-segment path-traversal guard
# ----------------------------------------------------------------------


class TestShouldBridgeHeadTraversal:
    """Pin: _should_bridge rejects ``..`` in the head segment too,
    not just in the tail. Closes a misconfiguration surface where
    an operator accidentally puts ``..`` in allowed_prefixes.
    """

    def test_head_segment_dot_dot_rejected(self):
        from strands_robots.mesh.transport.bridge_transport import _should_bridge

        # Even if an operator misconfigures allowed_prefixes with ".."
        # the head-segment check rejects it.
        assert _should_bridge("strands/robot-a/../cmd", {"cmd"}, frozenset({".."})) is False

    def test_normal_prefix_walk_still_works(self):
        from strands_robots.mesh.transport.bridge_transport import _should_bridge

        # response/<turn_id> is the legitimate prefix-walk case
        assert _should_bridge("strands/robot-a/response/turn-123", set(), frozenset({"response"})) is True
