"""the prior fix pin tests: ``verify_audit_integrity`` must fail closed when the
verifier lacks the PSK but signed records exist.

Pre-fix bug: ``ok=True`` on a signed log that the local verifier
cannot actually verify -- a forensic walker who has lost track of the
PSK reads a green light on an unverifiable log. This is the inverse
of the prior attack (writer briefly cleared PSK to forge unsigned
records) and is closed by the same fail-closed posture.

Per AGENTS.md > Review Learnings (#85) > "Pin regression tests for
reviewed fixes." Each test fails on pre-fix HEAD and passes on the prior fix+.
"""

from __future__ import annotations

import pytest

import strands_robots.mesh.audit as audit


@pytest.fixture(autouse=True)
def _reset_audit_state(monkeypatch, tmp_path):
    """Isolate each test from process-global PSK and audit dir state."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)
    yield
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)


def _signed_record_under_psk(monkeypatch, psk_value: str) -> dict:
    """Write one signed record under ``psk_value`` and return the dict."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", psk_value)
    audit._AUDIT_STATE.psk_fingerprint = None
    audit.log_safety_event(
        event_type="estop",
        peer_id="op-1",
        payload={"sender_id": "op-1"},
    )
    records = audit.read_audit_log()
    assert len(records) == 1
    assert records[0].get("sig") is not None, "fixture: record should be signed"
    return records[0]


def test_verify_returns_not_ok_when_signed_records_present_but_verifier_has_no_psk(
    monkeypatch,
):
    """R21: a signed log + verifier-without-PSK MUST report ok=False."""
    record = _signed_record_under_psk(monkeypatch, "the-fleet-key")

    # Verifier loses the PSK (different process / shell / sidecar).
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")

    report = audit.verify_audit_integrity([record])

    assert report["signed"] == 1
    assert report["bad_signature"] == 0
    assert report["missing_sig"] == 0
    assert report["psk_present"] is False
    assert report["unverifiable_signed"] == 1, "verifier without PSK must count signed records as unverifiable"
    assert report["ok"] is False, "verifier without PSK on a signed log MUST fail closed"


def test_verify_unverifiable_signed_count_matches_signed_record_count(monkeypatch):
    """When the verifier lacks the PSK, every signed record contributes one
    to ``unverifiable_signed`` -- the inverse of ``verified``."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "k")
    audit._AUDIT_STATE.psk_fingerprint = None
    for i in range(5):
        audit.log_safety_event(
            event_type="estop",
            peer_id=f"op-{i}",
            payload={"i": i},
        )
    records = audit.read_audit_log()
    assert len(records) == 5
    assert all(r.get("sig") is not None for r in records)

    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")
    report = audit.verify_audit_integrity(records)

    assert report["signed"] == 5
    assert report["verified"] == 0
    assert report["unverifiable_signed"] == 5
    assert report["ok"] is False


def test_verify_ok_true_when_psk_present_and_signed_records_verify(monkeypatch):
    """Sanity: with the right PSK at verify time, a signed log is ok=True
    and ``unverifiable_signed`` is zero. Confirms the new field doesn't
    over-fire on the happy path."""
    record = _signed_record_under_psk(monkeypatch, "k")
    # Same PSK at verify time: still set from the fixture above.
    report = audit.verify_audit_integrity([record])

    assert report["signed"] == 1
    assert report["verified"] == 1
    assert report["unverifiable_signed"] == 0
    assert report["ok"] is True


def test_verify_unsigned_records_with_no_psk_either_side_reports_ok(monkeypatch):
    """When neither writer nor verifier has a PSK, an all-unsigned log is
    legitimately ok=True (this is the documented fully-unsigned mode).
    Sanity check -- the prior fix targets the verifier-lacks-PSK-on-signed-log
    case ONLY, not the all-unsigned case."""
    audit.log_safety_event(
        event_type="estop",
        peer_id="op-1",
        payload={"sender_id": "op-1"},
    )
    records = audit.read_audit_log()
    assert len(records) == 1
    assert records[0].get("sig") is None

    report = audit.verify_audit_integrity(records)
    assert report["signed"] == 0
    assert report["missing_sig"] == 0
    assert report["unverifiable_signed"] == 0
    assert report["psk_present"] is False
    assert report["ok"] is True


def test_verify_returns_not_ok_when_mixed_signed_unsigned_with_verifier_lacking_psk(
    monkeypatch,
):
    """mix of signed (writer had PSK) and unsigned records -- verifier
    lacking PSK must still report ok=False because of the signed prefix
    it cannot verify."""
    # First write some signed records.
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "k")
    audit._AUDIT_STATE.psk_fingerprint = None
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})
    signed_records = audit.read_audit_log()
    assert len(signed_records) == 2

    # Synthesise an unsigned record alongside (different writer process).
    unsigned = dict(signed_records[0])
    unsigned["sig"] = None
    unsigned["seq"] = 99

    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")
    report = audit.verify_audit_integrity([*signed_records, unsigned])

    # No PSK on the verifier; the unsigned record is fine (consistent with
    # all-unsigned mode), but the signed records are unverifiable.
    assert report["signed"] == 2
    assert report["unverifiable_signed"] == 2
    assert report["missing_sig"] == 0  # no PSK on verifier; not flagged
    assert report["ok"] is False
