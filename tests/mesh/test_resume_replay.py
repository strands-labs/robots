"""Regression tests for resume-replay defenses in Mesh._on_safety_resume.

This test suite validates the fix for this PR reviewer concern:
the cryptographic shape `HMAC(override_code, proof_nonce)` with the issuer
choosing the nonce originally gave no replay defense between fellow
operator-class peers.

The fix in core.py adds:
1. Freshness window (RESUME_FRESHNESS_WINDOW_S, default 60s)
2. Forward-skew bound (RESUME_FORWARD_SKEW_S, default 5s)
3. Per-receiver replay cache ((issuer_peer_id, proof_nonce) tuple)
4. Bounded cache (RESUME_REPLAY_CACHE_MAX, default 4096) with stale-entry
   sweep + oldest-20%-drop fallback
"""

import hmac
import json
import logging
import time
import uuid
from unittest.mock import MagicMock

from strands_robots.mesh.core import Mesh


def _make_mesh(peer_id="r-test"):
    """Construct a minimally-instantiated Mesh without calling init_mesh."""
    robot = MagicMock()
    m = Mesh.__new__(Mesh)  # bypass __init__ side-effects
    Mesh.__init__(m, robot, peer_id)
    return m


def _sample(payload_dict):
    """Make a fake zenoh sample with a JSON payload."""
    s = MagicMock()
    s.payload.to_bytes.return_value = json.dumps(payload_dict).encode()
    return s


def _make_envelope(override_code, *, t=None, peer_id="op-1", proof_nonce=None, lockout_elapsed_s=1.0):
    """Mint a valid resume envelope keyed off a specific override code.

    the HMAC binds (peer_id, t, lockout_elapsed_s, proof_nonce)
    via a deterministic JSON encoding -- we mirror that on the issuing
    fixture side so the receiver-side compare passes.
    """
    import json as _json

    proof_nonce = proof_nonce or uuid.uuid4().hex
    envelope_t = t if t is not None else time.time()
    mac_input = _json.dumps(
        {
            "peer_id": peer_id,
            "t": envelope_t,
            "lockout_elapsed_s": lockout_elapsed_s,
            "proof_nonce": proof_nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    proof = hmac.new(override_code.encode(), mac_input, "sha256").hexdigest()
    return {
        "peer_id": peer_id,
        "t": envelope_t,
        "lockout_elapsed_s": lockout_elapsed_s,
        "proof_nonce": proof_nonce,
        "override_proof": proof,
    }


def test_first_legitimate_resume_clears_lockout(monkeypatch):
    """Sanity / happy path: a valid fresh envelope clears the lockout."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()  # stub out audit publishing

    # Set lockout
    m._estop_lockout.set()
    assert m._estop_lockout.is_set()

    # Send valid resume
    env = _make_envelope("secret")
    m._on_safety_resume(_sample(env))

    # Lockout should be cleared
    assert m._estop_lockout.is_set() is False


def test_replay_of_same_envelope_is_rejected(monkeypatch):
    """Replay of the same envelope is rejected via cache."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint ONE envelope
    env = _make_envelope("secret", peer_id="op-1")

    # First resume: accepted
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is False

    # Re-arm lockout
    m._estop_lockout.set()

    # Second resume with SAME envelope: rejected (replay)
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is True  # lockout stays set

    # Verify cache contains the (issuer, proof_nonce) tuple
    # issue #264: domain-tagged cache key prevents wire_zid/issuer_id namespace collision
    cache_key = (("body", env["peer_id"]), env["proof_nonce"])
    assert cache_key in m._resume_replay_cache

    # Verify audit event was emitted
    calls = [c for c in m.publish_safety_event.call_args_list if c[1].get("event_type") == "resume_replay_rejected"]
    assert len(calls) == 1


def test_stale_envelope_rejected(monkeypatch):
    """Envelope older than RESUME_FRESHNESS_WINDOW_S is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with t = 1 hour ago
    env = _make_envelope("secret", t=time.time() - 3600)

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set (envelope rejected for staleness)
    assert m._estop_lockout.is_set() is True


def test_future_envelope_rejected_beyond_skew(monkeypatch):
    """Envelope with t beyond RESUME_FORWARD_SKEW_S is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with t = 60s in future (beyond default 5s skew)
    env = _make_envelope("secret", t=time.time() + 60)

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set
    assert m._estop_lockout.is_set() is True


def test_envelope_within_forward_skew_accepted(monkeypatch):
    """Envelope with t within RESUME_FORWARD_SKEW_S is accepted."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with t = 1s in future (within default 5s skew)
    env = _make_envelope("secret", t=time.time() + 1.0)

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should be cleared
    assert m._estop_lockout.is_set() is False


def test_replay_cache_bounded(monkeypatch):
    """Replay cache is bounded at RESUME_REPLAY_CACHE_MAX.

    hot paths now re-read the env var via lazy resolver, so the
    test sets the env var directly rather than monkeypatching the
    module-level constant (which is now only the import-time default).
    """
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")

    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Drive 20 distinct nonces through the resume handler
    for i in range(20):
        env = _make_envelope("secret", peer_id=f"op-{i}", proof_nonce=uuid.uuid4().hex)
        m._estop_lockout.set()
        m._on_safety_resume(_sample(env))

    # Cache should be bounded
    assert len(m._resume_replay_cache) <= 8


def test_envelope_missing_t_field_rejected(monkeypatch):
    """Envelope missing the t field is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope then delete t field
    env = _make_envelope("secret")
    del env["t"]

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set
    assert m._estop_lockout.is_set() is True


def test_envelope_invalid_t_type_rejected(monkeypatch):
    """Envelope with invalid t type is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with invalid t type
    env = _make_envelope("secret")
    env["t"] = "not-a-number"

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set
    assert m._estop_lockout.is_set() is True


def test_replay_emits_audit_event(monkeypatch):
    """Replay rejection emits resume_replay_rejected audit event."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint ONE envelope
    env = _make_envelope("secret", peer_id="op-attacker")

    # First resume: accepted
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is False

    # Check that resume_clear event was emitted
    clear_calls = [
        c for c in m.publish_safety_event.call_args_list if c[1].get("event_type") == "remote_resume_applied"
    ]
    assert len(clear_calls) == 1

    # Re-arm lockout
    m._estop_lockout.set()

    # Second resume with SAME envelope: replay rejected
    m._on_safety_resume(_sample(env))

    # Check that resume_replay_rejected event was emitted
    replay_calls = [
        c for c in m.publish_safety_event.call_args_list if c[1].get("event_type") == "resume_replay_rejected"
    ]
    assert len(replay_calls) == 1

    # Verify payload contains issuer
    replay_payload = replay_calls[0][1]["payload"]
    assert replay_payload["issuer"] == "op-attacker"
    assert "proof_nonce_prefix" in replay_payload


class TestResumeStrictPeerId:
    """The estop handler rejects envelopes with empty/missing
    peer_id outright. Resume must mirror that posture.
    """

    def test_resume_with_empty_peer_id_rejected(self, caplog):
        class StubRobot:
            pass

        m = Mesh(robot=StubRobot(), peer_id="robot-test")
        m._estop_lockout.set()  # lockout is engaged so resume would normally clear it

        # Forge a resume envelope that's otherwise well-formed but has empty peer_id
        envelope = {
            "peer_id": "",
            "t": time.time(),
            "proof_nonce": "n" * 32,
            "override_proof": "x" * 64,
        }

        class FakeSample:
            payload = type("P", (), {"to_bytes": lambda self: json.dumps(envelope).encode()})()

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            m._on_safety_resume(FakeSample())

        # Lockout MUST still be engaged (resume rejected)
        assert m._estop_lockout.is_set(), "resume with empty peer_id should NOT clear lockout"
        # Cache should NOT have a polluting entry
        assert len(m._resume_replay_cache) == 0, "no cache entry should be created for invalid peer_id"


# ---------------------------------------------------------------------
# the prior fix-4: remote_estop_redundant audit on second-operator estop
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# HMAC binds envelope routing fields (peer_id, t,
# lockout_elapsed_s, proof_nonce). A captured envelope mutated on ANY
# of those four fields by an attacker is rejected at the MAC layer
# regardless of the cache state.
# ---------------------------------------------------------------------


def test_f18a_captured_envelope_with_mutated_peer_id_rejected(monkeypatch):
    """Captured legitimate envelope, attacker rewrites peer_id only:
    pre-the prior fix the MAC compare passed (covered nonce only), and the
    cache key (issuer_id, proof_nonce) became (NEW_id, proof_nonce)
    -- a cache miss -- so the lockout cleared on every replay.
    Post-the prior fix the MAC compare fails because peer_id is now bound."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()

    # 1. Legitimate envelope minted by the canonical issuer.
    env = _make_envelope("secret", peer_id="op-legit")

    # 2. Attacker captures the envelope and mutates peer_id only.
    env["peer_id"] = "op-attacker-impersonating"

    # 3. Lockout is engaged; receiver sees the mutated envelope.
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # The lockout MUST stay engaged -- the prior fix rejects mutated envelopes
    # at the MAC layer.
    assert m._estop_lockout.is_set() is True


def test_f18a_captured_envelope_with_mutated_t_rejected(monkeypatch):
    """Captured legitimate envelope, attacker rewrites t to bypass
    the freshness check (push it forward) -- post-the prior fix the MAC compare
    fails because t is now bound to the signature."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()

    # Mint at t=now.
    original_t = time.time()
    env = _make_envelope("secret", t=original_t, peer_id="op-legit")

    # Attacker forwards t by 1 second.
    env["t"] = original_t + 1.0

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    assert m._estop_lockout.is_set() is True


def test_f18a_captured_envelope_with_mutated_lockout_elapsed_rejected(monkeypatch):
    """Mutating lockout_elapsed_s (forensic noise field) also breaks
    the MAC -- the field is bound, so the receiver cannot trust any
    of these wire fields without the issuer's cooperation."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()

    env = _make_envelope("secret", peer_id="op-legit", lockout_elapsed_s=2.5)
    env["lockout_elapsed_s"] = 9999.0  # attacker rewrites

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    assert m._estop_lockout.is_set() is True


def test_f18a_envelope_without_lockout_elapsed_s_rejected(monkeypatch):
    """A malformed envelope missing lockout_elapsed_s is rejected
    outright before MAC compare (prior shape gate)."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()

    env = _make_envelope("secret", peer_id="op-legit")
    del env["lockout_elapsed_s"]

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    assert m._estop_lockout.is_set() is True


def test_resume_cache_per_issuer_cap_enforced(monkeypatch):
    """One issuer cannot exceed per_issuer_cap slots in the resume cache.

    Mirrors the estop per-issuer fairness bound: with CACHE_MAX=8 the cap
    is max(1, 8//4) == 2, so the third distinct-nonce resume from the same
    issuer is refused with a resume_per_issuer_cap_exceeded audit and the
    cache holds at most the cap for that issuer.
    """
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")

    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # cap = max(1, 8 // 4) == 2
    for _ in range(3):
        env = _make_envelope("secret", peer_id="op-flooder", proof_nonce=uuid.uuid4().hex)
        m._estop_lockout.set()
        m._on_safety_resume(_sample(env))

    flooder_slots = sum(1 for k in m._resume_replay_cache if k[0] == ("body", "op-flooder"))
    assert flooder_slots == 2

    audit_types = [c.kwargs.get("event_type") for c in m.publish_safety_event.call_args_list]
    assert "resume_per_issuer_cap_exceeded" in audit_types


def test_resume_cache_other_issuer_entries_not_evicted(monkeypatch):
    """A flooding issuer at cap cannot evict another issuer's cached slot.

    Issuer A fills its cap, issuer B records one legitimate slot, then A
    floods again (refused, no eviction). Replaying B's original envelope
    is still detected as a replay -- B's slot survived A's churn.
    """
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")

    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # A fills its cap of 2.
    for _ in range(2):
        env_a = _make_envelope("secret", peer_id="op-A", proof_nonce=uuid.uuid4().hex)
        m._estop_lockout.set()
        m._on_safety_resume(_sample(env_a))

    # B records one legitimate slot.
    env_b = _make_envelope("secret", peer_id="op-B", proof_nonce=uuid.uuid4().hex)
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env_b))
    assert ("body", "op-B") in {k[0] for k in m._resume_replay_cache}

    # A keeps flooding -- every attempt is refused, none evicts B.
    for _ in range(5):
        env_a = _make_envelope("secret", peer_id="op-A", proof_nonce=uuid.uuid4().hex)
        m._estop_lockout.set()
        m._on_safety_resume(_sample(env_a))

    # B's slot survives: replay of B's exact envelope is still rejected.
    m.publish_safety_event.reset_mock()
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env_b))
    audit_types = [c.kwargs.get("event_type") for c in m.publish_safety_event.call_args_list]
    assert "resume_replay_rejected" in audit_types


def test_resume_and_estop_per_issuer_cap_use_same_expression():
    """Structural symmetry pin: both replay-cache handlers must derive
    per_issuer_cap from the identical expression so the two fairness
    defenses cannot silently diverge."""
    import inspect

    resume_src = inspect.getsource(Mesh._on_safety_resume)
    estop_src = inspect.getsource(Mesh._on_safety_estop)
    cap_expr = "per_issuer_cap = max(1, _resume_replay_cache_max() // 4)"
    assert cap_expr in resume_src
    assert cap_expr in estop_src
