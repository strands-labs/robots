"""Regression tests for the wire-level ``source_zid`` binding on
safety estop / resume envelopes.

Threat model addressed
----------------------
Body-level HMAC binding (peer_id, t, lockout_elapsed_s, proof_nonce)
catches an attacker who mutates body fields of a captured envelope. It
does NOT catch an attacker on a *different* mTLS-authenticated session
who also holds the override code: that attacker can mint a fresh
envelope with a body of their choosing and a freshly-computed MAC. The
receiver-side compare passes because the attacker's body matches their
MAC.

Fix: bind the publisher's TLS-authenticated Zenoh session ID
(``sample.source_info.source_id.zid``) into both:

1. The body of every safety envelope (``source_zid`` field).
2. The HMAC input for resume envelopes.

The receiver:

- extracts ``sample.source_info.source_id.zid`` (set by Zenoh during
  the mTLS-bootstrapped session handshake; ``ZenohId`` has no public
  Python constructor),
- requires body ``source_zid`` to equal wire ``source_zid`` when both
  are present,
- requires both-present-or-both-absent (no silent downgrade),
- re-derives the resume MAC using the wire-level zid.

These tests pin all four behaviours.
"""

from __future__ import annotations

import hmac
import json
import time
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as core_module
from strands_robots.mesh.core import Mesh, _extract_sample_source_zid

# Real-format zid: 32 lowercase hex chars
_LEGIT_ZID = "0123456789abcdef0123456789abcdef"
_ATTACKER_ZID = "fedcba9876543210fedcba9876543210"


def _zid_obj(zid_str: str) -> Any:
    """Stand-in for ``zenoh.ZenohId`` whose ``str()`` returns the hex digest."""

    class _Zid:
        def __init__(self, s: str) -> None:
            self._s = s

        def __str__(self) -> str:
            return self._s

    return _Zid(zid_str)


def _make_sample(payload: dict, source_zid: str | None = None) -> SimpleNamespace:
    """Construct a minimal Zenoh-sample fake.

    When *source_zid* is provided the sample's
    ``source_info.source_id.zid`` returns it; otherwise ``source_info``
    is None (mirroring a transport that did not attach SourceInfo).
    """
    body = json.dumps(payload).encode("utf-8")
    if source_zid is None:
        return SimpleNamespace(
            payload=SimpleNamespace(to_bytes=lambda: body),
            source_info=None,
        )
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: body),
        source_info=SimpleNamespace(
            source_id=SimpleNamespace(zid=_zid_obj(source_zid)),
            source_sn=1,
        ),
    )


@pytest.fixture
def receiver():
    """Bare ``Mesh`` instance with the bits ``_on_safety_*`` touch."""
    m = Mesh.__new__(Mesh)
    Mesh.__init__(m, MagicMock(), peer_id="receiver-1")
    m.publish_safety_event = MagicMock()
    return m


# Extractor ---------------------------------------------------------------


def test_extract_zid_from_well_formed_sample():
    """A sample carrying a 32-char hex zid returns that string."""
    sample = _make_sample({"hello": "world"}, source_zid=_LEGIT_ZID)
    assert _extract_sample_source_zid(sample) == _LEGIT_ZID


def test_extract_zid_returns_none_when_source_info_absent():
    """Bridge / IoT transports do not propagate source_info."""
    sample = _make_sample({"hello": "world"}, source_zid=None)
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_rejects_non_hex_stand_ins():
    """A ``MagicMock`` whose source_id.zid stringifies to a Mock repr
    must NOT be accepted as a real wire-bound zid."""
    sample = MagicMock()
    # MagicMock auto-creates source_info / source_id / zid; the str()
    # of a MagicMock is the well-known ``<MagicMock id=...>`` form,
    # which does not match the strict hex pattern.
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_rejects_uppercase_hex():
    """ZenohId stringifies to lowercase hex; uppercase is rejected.

    Defence-in-depth: the format pin is part of the binding contract.
    A future zenoh-python that emitted uppercase would force a code
    review (this test fails) rather than silently changing the wire
    invariant.
    """
    sample = _make_sample({}, source_zid=_LEGIT_ZID.upper())
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_rejects_overlong_string():
    """Strings longer than 32 hex chars are rejected as malformed."""
    sample = _make_sample({}, source_zid="0" * 33)
    assert _extract_sample_source_zid(sample) is None


# Receiver-side: estop ----------------------------------------------------


def test_estop_body_zid_matches_wire_zid_accepted(receiver):
    """When wire and body source_zid agree, the envelope is accepted."""
    now = time.time()
    payload = {
        "peer_id": "op-1",
        "t": now,
        "source_zid": _LEGIT_ZID,
        "trigger": "remote",
    }
    receiver._on_safety_estop(_make_sample(payload, source_zid=_LEGIT_ZID))
    assert receiver._estop_lockout.is_set()


def test_estop_body_zid_disagreeing_with_wire_rejected(receiver, caplog):
    """An attacker on a different session whose body claims a peer's
    session zid is rejected: the wire zid (set by Zenoh, attacker
    cannot choose) does not match the body claim."""
    now = time.time()
    payload = {
        "peer_id": "op-1",
        "t": now,
        "source_zid": _LEGIT_ZID,  # attacker's body claims legit zid
        "trigger": "remote",
    }
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        # ...but wire carries the attacker's actual zid.
        receiver._on_safety_estop(_make_sample(payload, source_zid=_ATTACKER_ZID))
    assert not receiver._estop_lockout.is_set()
    assert any("cross-session forgery rejected" in rec.message for rec in caplog.records), (
        "expected an explicit cross-session forgery rejection log"
    )


def test_estop_body_zid_present_wire_zid_absent_rejected(receiver, caplog):
    """A publisher that advertises body source_zid but failed to attach
    SourceInfo on the wire is rejected (no silent downgrade)."""
    now = time.time()
    payload = {
        "peer_id": "op-1",
        "t": now,
        "source_zid": _LEGIT_ZID,
    }
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_estop(_make_sample(payload, source_zid=None))
    assert not receiver._estop_lockout.is_set()
    assert any("body source_zid present but wire" in rec.message for rec in caplog.records)


def test_estop_wire_zid_present_body_zid_absent_rejected(receiver, caplog):
    """A pre-binding publisher (no body source_zid) on a Zenoh session
    that DOES propagate source_info is rejected: operators must
    upgrade all peers together so the binding is never silently
    downgraded."""
    now = time.time()
    payload = {"peer_id": "op-1", "t": now}
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_estop(_make_sample(payload, source_zid=_LEGIT_ZID))
    assert not receiver._estop_lockout.is_set()
    assert any("wire source_zid present but body" in rec.message for rec in caplog.records)


def test_estop_neither_wire_nor_body_zid_accepted(receiver):
    """Bridge / IoT transports where neither wire nor body carry a zid:
    the envelope is accepted via the body-level HMAC defences alone."""
    now = time.time()
    payload = {"peer_id": "op-1", "t": now, "trigger": "remote"}
    receiver._on_safety_estop(_make_sample(payload, source_zid=None))
    assert receiver._estop_lockout.is_set()


# Receiver-side: resume ----------------------------------------------------


def _make_resume_envelope(
    *,
    override_code: str,
    peer_id: str = "op-1",
    t: float | None = None,
    elapsed: float = 1.0,
    proof_nonce: str | None = None,
    source_zid: str | None = None,
) -> dict:
    """Mint a valid resume envelope, optionally including ``source_zid``
    in both the body and the HMAC input."""
    if t is None:
        t = time.time()
    if proof_nonce is None:
        proof_nonce = uuid.uuid4().hex
    mac_fields = {
        "peer_id": peer_id,
        "t": t,
        "lockout_elapsed_s": elapsed,
        "proof_nonce": proof_nonce,
    }
    if source_zid is not None:
        mac_fields["source_zid"] = source_zid
    mac_input = json.dumps(mac_fields, sort_keys=True, separators=(",", ":")).encode()
    proof = hmac.new(override_code.encode(), mac_input, "sha256").hexdigest()
    env = {
        "peer_id": peer_id,
        "t": t,
        "lockout_elapsed_s": elapsed,
        "proof_nonce": proof_nonce,
        "override_proof": proof,
    }
    if source_zid is not None:
        env["source_zid"] = source_zid
    return env


def test_resume_body_zid_matches_wire_zid_clears_lockout(receiver, monkeypatch):
    """Happy path: wire and body source_zid agree, MAC verifies."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    receiver._estop_lockout.set()

    env = _make_resume_envelope(override_code="secret", source_zid=_LEGIT_ZID)
    receiver._on_safety_resume(_make_sample(env, source_zid=_LEGIT_ZID))

    assert not receiver._estop_lockout.is_set()


def test_resume_cross_session_forgery_rejected(receiver, monkeypatch, caplog):
    """An attacker on a DIFFERENT mTLS session who also holds the
    override code mints a body whose source_zid claims to be the
    legit session, recomputes the MAC against that body, and publishes.

    The receiver:

    1. Sees ``sample.source_info.source_id.zid`` is the attacker's
       (Zenoh sets it, attacker cannot choose),
    2. Sees the body claims a different ``source_zid``,
    3. Rejects on the body!=wire mismatch BEFORE running the MAC
       compare.

    Even if step (3) were bypassed, step (4) would catch it: the
    receiver re-derives the MAC using the WIRE zid (attacker's), so
    the precomputed body MAC (built against the legit zid) fails
    constant-time compare.
    """
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    receiver._estop_lockout.set()

    # Attacker has legit override code and forges a body claiming legit zid.
    env = _make_resume_envelope(override_code="secret", source_zid=_LEGIT_ZID)
    # ...but wire carries attacker's actual zid.
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_resume(_make_sample(env, source_zid=_ATTACKER_ZID))

    assert receiver._estop_lockout.is_set(), "cross-session forgery must NOT clear lockout"
    assert any("cross-session forgery rejected" in rec.message for rec in caplog.records)


def test_resume_mac_binds_wire_zid_not_body_zid(receiver, monkeypatch, caplog):
    """Defence-in-depth: even if the body!=wire shape check were
    bypassed, the MAC re-derivation uses the wire zid. We simulate
    "body matches wire" but the MAC was built against a third zid.

    Construction:
      - Body source_zid = _ATTACKER_ZID
      - Wire source_zid = _ATTACKER_ZID  (so shape check passes)
      - MAC built against _LEGIT_ZID  (attacker tried to bind a
        different zid into the MAC than they're publishing under)

    Result: receiver re-derives MAC against _ATTACKER_ZID, compare
    fails, envelope is rejected.
    """
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    receiver._estop_lockout.set()

    # Manual construction: body source_zid = attacker, but MAC was
    # computed binding source_zid=legit (a captured-and-mutated MAC).
    proof_nonce = uuid.uuid4().hex
    envelope_t = time.time()
    elapsed = 1.0
    mac_fields = {
        "peer_id": "op-1",
        "t": envelope_t,
        "lockout_elapsed_s": elapsed,
        "proof_nonce": proof_nonce,
        "source_zid": _LEGIT_ZID,  # MAC binds legit
    }
    mac_input = json.dumps(mac_fields, sort_keys=True, separators=(",", ":")).encode()
    proof = hmac.new(b"secret", mac_input, "sha256").hexdigest()
    env = {
        "peer_id": "op-1",
        "t": envelope_t,
        "lockout_elapsed_s": elapsed,
        "proof_nonce": proof_nonce,
        "override_proof": proof,
        "source_zid": _ATTACKER_ZID,  # body publishes attacker
    }
    # NOTE: shape check rejects on body!=wire (legit vs attacker),
    # so we set wire == body to specifically test the MAC binding.
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_resume(_make_sample(env, source_zid=_ATTACKER_ZID))

    assert receiver._estop_lockout.is_set()
    assert any("override_proof mismatch" in rec.message for rec in caplog.records)


def test_resume_replay_cache_keyed_on_wire_zid(receiver, monkeypatch):
    """Same proof_nonce from two distinct sessions must NOT collide.

    Pre-fix: cache key was ``(issuer_peer_id, proof_nonce)``. An attacker
    could replay one session's nonce from a different session by
    setting body ``peer_id`` to match. Post-fix the cache key uses the
    wire-bound zid when available.
    """
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    receiver._estop_lockout.set()

    nonce = uuid.uuid4().hex
    # Session A
    env_a = _make_resume_envelope(
        override_code="secret",
        peer_id="op-1",
        proof_nonce=nonce,
        source_zid=_LEGIT_ZID,
    )
    receiver._on_safety_resume(_make_sample(env_a, source_zid=_LEGIT_ZID))
    assert not receiver._estop_lockout.is_set()
    receiver._estop_lockout.set()  # re-lock for observation

    # Session B publishes the SAME nonce + same body peer_id but its
    # OWN session zid. Pre-fix this would have hit the cache (key was
    # peer_id+nonce); post-fix the wire zid differs so it's a cache
    # miss and proceeds through MAC verification (which it passes
    # because MAC includes session B's zid).
    env_b = _make_resume_envelope(
        override_code="secret",
        peer_id="op-1",
        proof_nonce=nonce,
        source_zid=_ATTACKER_ZID,
    )
    receiver._on_safety_resume(_make_sample(env_b, source_zid=_ATTACKER_ZID))

    # Two distinct cache entries -- one per session.
    assert len(receiver._resume_replay_cache) == 2, (
        f"expected 2 cache entries (one per session zid); got {receiver._resume_replay_cache}"
    )


def test_resume_pre_binding_publisher_rejected_when_wire_has_zid(receiver, monkeypatch, caplog):
    """A pre-binding peer (no body source_zid) communicating over a
    Zenoh session that DOES propagate source_info is rejected so the
    fleet upgrade is atomic."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    receiver._estop_lockout.set()

    # Old envelope, no source_zid in body or in MAC.
    env = _make_resume_envelope(override_code="secret", source_zid=None)
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_resume(_make_sample(env, source_zid=_LEGIT_ZID))

    assert receiver._estop_lockout.is_set()
    assert any("publisher predates source_zid binding" in rec.message for rec in caplog.records)


def test_resume_bridge_transport_no_zid_either_side_accepted(receiver, monkeypatch):
    """Bridge / IoT transport: neither wire nor body carries a zid.
    The envelope is accepted via the body-level HMAC binding alone --
    cross-session-forgery defence is Zenoh-specific."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    receiver._estop_lockout.set()

    env = _make_resume_envelope(override_code="secret", source_zid=None)
    receiver._on_safety_resume(_make_sample(env, source_zid=None))

    assert not receiver._estop_lockout.is_set()


# Publisher-side helpers --------------------------------------------------


def test_local_session_zid_returns_none_without_zenoh_session(receiver):
    """When no Zenoh session is open, ``_local_session_zid`` returns
    ``None`` and the safety publisher path falls back to body-only
    binding."""
    # Default fixture has no live session; the helper must return None
    # rather than raising.
    assert receiver._local_session_zid() is None


def test_safety_publisher_for_returns_none_without_session(receiver):
    """``_safety_publisher_for`` returns None when no session is open;
    callers fall back to ``put()``."""
    assert receiver._safety_publisher_for("strands/safety/estop") is None


def test_next_safety_sn_is_monotonic_per_topic(receiver):
    """Sequence numbers increment per-topic and never repeat."""
    a1 = receiver._next_safety_sn("strands/safety/estop")
    a2 = receiver._next_safety_sn("strands/safety/estop")
    a3 = receiver._next_safety_sn("strands/safety/estop")
    assert a1 < a2 < a3

    b1 = receiver._next_safety_sn("strands/safety/resume")
    b2 = receiver._next_safety_sn("strands/safety/resume")
    assert b1 < b2

    # Per-topic counters are independent.
    assert b1 == 1, "resume topic counter starts at 1 independently of the estop topic"


def test_publish_safety_envelope_falls_back_to_put_without_session(receiver, monkeypatch):
    """Without a Zenoh session ``_publish_safety_envelope`` MUST call
    ``put()`` so the bridge / IoT transport path still delivers the
    envelope (just without TLS-bound source attribution)."""
    calls = []

    def fake_put(key, payload):
        calls.append((key, dict(payload)))

    monkeypatch.setattr(core_module, "put", fake_put)
    receiver._publish_safety_envelope("strands/safety/estop", {"hello": "world"})

    assert calls == [("strands/safety/estop", {"hello": "world"})]
