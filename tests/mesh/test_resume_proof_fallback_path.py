"""Regression: remote lockout resume must verify on the SourceInfo-less
fallback publish path.

The resume override proof is an ``HMAC(override_code, <envelope fields>)``.
When a Zenoh build lacks ``SourceInfo`` (or no session/publisher is available)
the safety envelope is published on the fallback ``put()`` path, which strips
``source_zid`` from the body. If the issuer binds ``source_zid`` into the MAC
while the published body has it stripped, every receiver recomputes the proof
over a different byte string -- the proof never verifies and the fleet stays
e-stopped forever.

This pins the round trip: an issuer that publishes on the fallback path
produces an envelope that a receiver accepts and that clears the lockout.
"""

import json
import types
from unittest.mock import MagicMock

from strands_robots.mesh import core


def _fallback_sample(payload: dict) -> object:
    """A wire sample with NO source_info (the fallback / bridge transport):
    _extract_sample_source_zid() returns None for it."""
    sample = types.SimpleNamespace()
    sample.payload = types.SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    # deliberately no ``source_info`` attribute -> wire_zid resolves to None
    return sample


def test_resume_proof_verifies_when_published_on_fallback_path(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "operator-secret")

    # --- Issuer: an open session (so _local_session_zid resolves a real zid)
    # but the native SourceInfo path is unavailable, so the envelope is
    # published on the fallback path that strips source_zid. ---
    issuer = core.Mesh(robot=object(), peer_id="issuer")
    issuer.publish_safety_event = MagicMock()
    monkeypatch.setattr(issuer, "_local_session_zid", lambda: "deadbeefdeadbeef")
    # No native publisher available -> fallback path (and _safety_wire_zid None).
    monkeypatch.setattr(issuer, "_safety_publisher_for", lambda key: None)

    published: dict = {}

    def capture_put(key, payload):
        published["key"] = key
        published["payload"] = payload

    monkeypatch.setattr(core, "put", capture_put)

    # Engage the local lockout, then resume with the correct override code.
    issuer._estop_lockout.set()
    issuer._last_estop_ts = core.time.time()
    issuer._last_estop_mono = core.time.monotonic()
    result = issuer._resume_lockout("operator-secret")
    assert result == {"status": "ok"}

    assert published["key"] == "strands/safety/resume"
    envelope = published["payload"]
    # Fallback path stripped source_zid from the body...
    assert "source_zid" not in envelope
    # ...and the proof is bound to that exact (zid-less) body.
    assert "override_proof" in envelope

    # --- Receiver on the same fallback transport (no wire source_zid). ---
    receiver = core.Mesh(robot=object(), peer_id="receiver")
    receiver.publish_safety_event = MagicMock()
    receiver._estop_lockout.set()
    assert receiver._estop_lockout.is_set()

    receiver._on_safety_resume(_fallback_sample(envelope))

    # The proof verified against the published body -> lockout cleared.
    assert receiver._estop_lockout.is_set() is False


def test_safety_wire_zid_none_when_source_info_unavailable(monkeypatch):
    """_safety_wire_zid returns None on a zenoh build lacking SourceInfo, so
    the proof is bound to the zid-less body that the fallback path publishes."""
    import sys

    m = core.Mesh(robot=object(), peer_id="t1")
    monkeypatch.setattr(m, "_local_session_zid", lambda: "abc123")
    monkeypatch.setattr(m, "_safety_publisher_for", lambda key: object())
    fake_zenoh = types.ModuleType("zenoh")  # no SourceInfo attribute
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    assert m._safety_wire_zid("strands/safety/resume") is None


def test_safety_wire_zid_returns_zid_on_native_path(monkeypatch):
    import sys

    m = core.Mesh(robot=object(), peer_id="t1")
    monkeypatch.setattr(m, "_local_session_zid", lambda: "abc123")
    monkeypatch.setattr(m, "_safety_publisher_for", lambda key: object())
    fake_zenoh = types.ModuleType("zenoh")
    fake_zenoh.SourceInfo = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    assert m._safety_wire_zid("strands/safety/resume") == "abc123"
