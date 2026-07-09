"""Audit-publish failure must never break the remote-resume safety path.

``_on_safety_resume`` emits a forensic audit event when it refuses a remote
resume: ``resume_replay_rejected`` when the ``(issuer, proof_nonce)`` envelope
was already accepted, and ``resume_per_issuer_cap_exceeded`` when one issuer
tries to hold more than its fair share of replay-cache slots. Both audit
publishes are best-effort: AGENTS.md mandates that a failing audit sink must
NOT propagate out of the safety path, since a flaky or full-disk audit backend
must never abort a rejection and let the fleet slip into a half-state.

Both calls are wrapped in ``except (TypeError, ValueError, OSError)`` so a
malformed payload (TypeError/ValueError) or a disk failure (OSError) is
swallowed at DEBUG. These tests pin that contract for both resume audit
branches - the estop path already has the equivalent coverage in
``test_estop_audit_publish_failure_nonblocking``. When ``publish_safety_event``
raises, the handler must return cleanly and leave the lockout in the safe
state: a rejected replay keeps the lockout engaged, and an over-cap resume is
refused (the resume never clears the lockout).
"""

import hmac
import json
import time
import uuid

from strands_robots.mesh.core import Mesh


def _make_mesh(peer_id="r-test"):
    """Construct a minimally-instantiated Mesh (mirrors test_resume_replay)."""
    from unittest.mock import MagicMock

    robot = MagicMock()
    m = Mesh.__new__(Mesh)
    Mesh.__init__(m, robot, peer_id)
    return m


def _sample(payload_dict):
    """Wrap a JSON payload in a fake zenoh sample."""
    from unittest.mock import MagicMock

    s = MagicMock()
    s.payload.to_bytes.return_value = json.dumps(payload_dict).encode()
    return s


def _make_envelope(override_code, *, t=None, peer_id="op-1", proof_nonce=None, lockout_elapsed_s=1.0):
    """Mint a valid resume envelope whose HMAC binds the routing fields."""
    proof_nonce = proof_nonce or uuid.uuid4().hex
    envelope_t = t if t is not None else time.time()
    mac_input = json.dumps(
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


def test_replay_rejected_audit_oserror_is_swallowed_and_lockout_preserved(monkeypatch):
    """A disk failure while auditing a resume_replay_rejected event must not
    propagate out of the safety handler, and the rejected replay must leave
    the lockout engaged."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()

    calls = []

    def audit(**kwargs):
        calls.append(kwargs)
        if kwargs.get("event_type") == "resume_replay_rejected":
            raise OSError("audit log volume full")

    m.publish_safety_event = audit

    env = _make_envelope("secret", peer_id="op-1")

    # First resume: accepted, clears the lockout.
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is False

    # Re-arm and replay the SAME envelope: rejected via the replay cache.
    # The rejection audit raises OSError, which must be swallowed.
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))  # must not raise

    # Safety intact: the failing audit did not clear the re-armed lockout.
    assert m._estop_lockout.is_set() is True
    rejected = [c for c in calls if c.get("event_type") == "resume_replay_rejected"]
    assert len(rejected) == 1


def test_per_issuer_cap_audit_valueerror_is_swallowed_and_cap_enforced(monkeypatch):
    """A malformed-payload (ValueError) failure while auditing a
    resume_per_issuer_cap_exceeded event must not propagate, and the
    per-issuer fairness bound must still be enforced (the refused resume
    adds no cache slot and never clears the lockout)."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")  # cap = max(1, 8 // 4) == 2
    m = _make_mesh()

    calls = []

    def audit(**kwargs):
        calls.append(kwargs)
        if kwargs.get("event_type") == "resume_per_issuer_cap_exceeded":
            raise ValueError("bad audit payload shape")

    m.publish_safety_event = audit

    # Two accepted resumes fill the issuer's cap; the third trips it.
    for _ in range(3):
        env = _make_envelope("secret", peer_id="op-flooder", proof_nonce=uuid.uuid4().hex)
        m._estop_lockout.set()
        m._on_safety_resume(_sample(env))  # third must not raise despite audit ValueError

    cap_calls = [c for c in calls if c.get("event_type") == "resume_per_issuer_cap_exceeded"]
    assert len(cap_calls) == 1
    # Cap enforced despite the failing audit sink: issuer holds at most the cap.
    flooder_slots = sum(1 for k in m._resume_replay_cache if k[0] == ("body", "op-flooder"))
    assert flooder_slots == 2
    # The refused (third) resume must NOT clear the lockout.
    assert m._estop_lockout.is_set() is True
