"""Pin: estop replay cache resists peer_id permutation.

Background: the prior fix review flagged that the prior cache key
``(issuer_peer_id, float(envelope_t))`` could be bypassed by an attacker
who captures one valid estop envelope and replays it with a single-byte
mutation to the payload ``peer_id`` field. ``peer_id`` is untrusted (it
comes from the JSON body, not the TLS cert CN), so the cache key was
attacker-controlled and the replay-defence claim in the prior docstring was
overstated.

Fix: cache key narrowed to ``float(envelope_t)`` alone, and
envelopes with missing/empty ``peer_id`` are rejected outright.

These tests pin both behaviours.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as core_module


class _FakeSample:
    """Minimal Zenoh-sample stand-in that carries a JSON payload."""

    def __init__(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.payload = SimpleNamespace(to_bytes=lambda: body)


@pytest.fixture
def receiver():
    """A bare ``Mesh`` with the bits ``_on_safety_estop`` actually touches."""
    m = core_module.Mesh.__new__(core_module.Mesh)
    m.peer_id = "receiver-1"
    m._estop_lockout = core_module.threading.Event()
    m._estop_replay_cache = {}
    m._estop_replay_lock = core_module.threading.Lock()
    m._last_estop_ts = 0.0
    # Stub publish_safety_event so we don't need a live transport.
    m.publish_safety_event = MagicMock()
    return m


def test_peer_id_permutation_cannot_replay(receiver, caplog):
    """A captured envelope replayed with a different ``peer_id`` is dropped.

    Pre-fix the cache key was ``(issuer_peer_id, t)`` so flipping
    ``peer_id`` from ``"op-1"`` to ``"op-2"`` yielded a fresh key and the
    replay was accepted. Post-fix the cache key is ``float(t)`` alone.
    """
    now = time.time()
    captured = {"t": now, "peer_id": "op-1", "trigger": "remote"}

    # First receipt -- accepted, lockout engages.
    receiver._on_safety_estop(_FakeSample(captured))
    assert receiver._estop_lockout.is_set(), "first receipt must engage lockout"
    assert len(receiver._estop_replay_cache) == 1
    receiver._estop_lockout.clear()  # reset for the replay observation

    # Replay with permuted peer_id but same ``t``.
    permuted = {"t": now, "peer_id": "op-2", "trigger": "remote"}
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_estop(_FakeSample(permuted))

    # The replay must be REJECTED -- lockout must NOT re-engage and
    # cache size stays at 1.
    assert not receiver._estop_lockout.is_set(), (
        "permuted peer_id replayed within freshness window must NOT engage lockout"
    )
    assert len(receiver._estop_replay_cache) == 1, (
        f"cache must not grow on permuted-peer_id replay; got {receiver._estop_replay_cache}"
    )
    assert any("REJECTED" in rec.message for rec in caplog.records), "expected a REJECTED log for the permuted replay"


def test_missing_peer_id_envelope_rejected(receiver, caplog):
    """Envelopes with missing/empty ``peer_id`` are rejected outright."""
    now = time.time()
    for bad_payload in (
        {"t": now},  # missing peer_id
        {"t": now, "peer_id": ""},  # empty string
        {"t": now, "peer_id": None},  # not a string
        {"t": now, "peer_id": 123},  # not a string (int)
    ):
        receiver._estop_lockout.clear()
        receiver._estop_replay_cache.clear()
        with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
            receiver._on_safety_estop(_FakeSample(bad_payload))
        assert not receiver._estop_lockout.is_set(), (
            f"envelope with bad peer_id ({bad_payload}) must NOT engage lockout"
        )
        assert len(receiver._estop_replay_cache) == 0, f"envelope with bad peer_id ({bad_payload}) must NOT enter cache"
    # At least one WARNING line per bad payload.
    msgs = [rec.message for rec in caplog.records]
    assert sum("invalid ``peer_id``" in m for m in msgs) >= 4, f"expected >=4 invalid-peer_id warnings, got: {msgs}"


def test_legitimate_distinct_t_envelopes_both_accepted(receiver):
    """Two distinct-``t`` envelopes from the same issuer are both accepted."""
    base = time.time()
    # Two envelopes 100ms apart -- both within freshness window.
    for offset in (0.0, 0.1):
        receiver._estop_lockout.clear()
        receiver._on_safety_estop(_FakeSample({"t": base + offset, "peer_id": "op-1", "trigger": "remote"}))
    assert len(receiver._estop_replay_cache) == 2, (
        f"two distinct-t envelopes must populate two cache entries; got {receiver._estop_replay_cache}"
    )
