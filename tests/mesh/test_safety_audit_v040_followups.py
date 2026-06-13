"""v0.4.0 mesh safety+audit follow-up pins (#386), from the #221/#225 R12 trail.

- #324: a non-symlink _next_seq failure writes a NEXT_SEQ_DEGRADED poison
  record (symmetry with SEQ_LOCK_DEGRADED) instead of dropping the record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolate_audit_state(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._SEQ_COUNTERS.clear()
    yield
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._SEQ_COUNTERS.clear()


# --------------------------------------------------------------------------- #
# #324 — NEXT_SEQ_DEGRADED poison record                                       #
# --------------------------------------------------------------------------- #
def test_next_seq_non_symlink_failure_writes_poison_record(tmp_path, monkeypatch, caplog):
    """A non-SeqLockSymlinkError failure inside _next_seq must NOT drop the
    record silently -- it writes a NEXT_SEQ_DEGRADED poison record so a verifier
    can attribute the seq gap. Pre-fix the record was dropped (return)."""
    import logging

    def _boom(_peer_id):
        raise OSError("simulated seq sidecar I/O failure")

    monkeypatch.setattr(audit, "_next_seq", _boom)

    with caplog.at_level(logging.ERROR, logger="strands_robots.mesh.audit"):
        audit.log_safety_event("emergency_stop", "peerA", {"reason": "test"})

    # The audit log file must contain a NEXT_SEQ_DEGRADED poison record.
    log_path = Path(tmp_path) / "mesh_audit.jsonl"
    assert log_path.exists(), "a poison record must still be written (not dropped)"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    poison = [r for r in records if r.get("sig") == "NEXT_SEQ_DEGRADED"]
    assert poison, f"expected a NEXT_SEQ_DEGRADED poison record; got {records}"
    assert poison[0]["peer_id"] == "peerA"
    assert poison[0]["seq"] == 0


def test_seqlock_symlink_still_writes_seq_lock_degraded(tmp_path, monkeypatch):
    """The pre-existing SEQ_LOCK_DEGRADED path must be unchanged by #324."""
    from strands_robots.mesh.audit import SeqLockSymlinkError

    def _symlink_boom(_peer_id):
        raise SeqLockSymlinkError("symlinked seq lockfile")

    monkeypatch.setattr(audit, "_next_seq", _symlink_boom)
    audit.log_safety_event("emergency_stop", "peerB", {"reason": "test"})

    log_path = Path(tmp_path) / "mesh_audit.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert any(r.get("sig") == "SEQ_LOCK_DEGRADED" for r in records), (
        f"SEQ_LOCK_DEGRADED path must be preserved; got {records}"
    )
    assert not any(r.get("sig") == "NEXT_SEQ_DEGRADED" for r in records)


# --------------------------------------------------------------------------- #
# #324 — NEXT_SEQ_DEGRADED poison survives PSK signing gate                    #
# --------------------------------------------------------------------------- #
def test_next_seq_degraded_poison_survives_psk_signing(tmp_path, monkeypatch, caplog):
    """Regression pin: when STRANDS_MESH_AUDIT_PSK is configured, the
    NEXT_SEQ_DEGRADED poison marker must NOT be overwritten by _sign_record.

    The signing-skip gate must cover both seq_lock_degraded_reason AND
    next_seq_degraded_reason. Without the fix, _sign_record runs on the
    poison record and clobbers record['sig'] with a valid HMAC, making
    the seq-counter integrity gap invisible to verifiers."""
    import logging

    # Configure a real PSK (32-byte hex = 64 hex chars).
    psk_hex = "a" * 64
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", psk_hex)

    # Force a non-symlink _next_seq failure.
    def _boom(_peer_id):
        raise OSError("simulated seq sidecar permissions failure")

    monkeypatch.setattr(audit, "_next_seq", _boom)

    with caplog.at_level(logging.ERROR, logger="strands_robots.mesh.audit"):
        audit.log_safety_event("emergency_stop", "peerC", {"reason": "psk_regression"})

    # The poison marker must survive -- NOT be overwritten by a real HMAC.
    log_path = Path(tmp_path) / "mesh_audit.jsonl"
    assert log_path.exists(), "poison record must be written even with PSK configured"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    poison = [r for r in records if r.get("peer_id") == "peerC"]
    assert poison, f"expected a record for peerC; got {records}"
    assert poison[0]["sig"] == "NEXT_SEQ_DEGRADED", (
        f"NEXT_SEQ_DEGRADED poison must survive PSK signing gate; "
        f"got sig={poison[0].get('sig')!r} (likely overwritten by HMAC)"
    )
    assert poison[0]["seq"] == 0
