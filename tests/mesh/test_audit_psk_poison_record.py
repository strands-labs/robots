"""Pin test for the prior audit-poison-record on PSK degrade.

this PR review thread PRRT_kwDORUMiZs6ER6_2: silent drop on PSK degrade
contradicts the module docstring. The docstring says the record is
"rejected"; in practice the entire safety event vanishes from the
audit log with only an ``error``-level log line. ``verify_audit_integrity``
sees a clean log post-incident and an operator concludes nothing
happened.

the prior fix fix: write a poison record with ``sig="PSK_DEGRADED"`` and a
``psk_degraded`` reason field instead of dropping. A signed-record
verifier (PSK present) reports it as ``bad_signature`` (the literal
string is not a valid HMAC hex) which forces ``ok=False`` and
preserves the forensic trail.

Pin: simulate a signed -> unsigned PSK transition and assert (a) the
log file has a record, (b) ``verify_audit_integrity`` reports
``ok=False``, (c) the poison sentinel is present.
"""

from __future__ import annotations

import importlib

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolated_audit(monkeypatch, tmp_path):
    """Run each test against a fresh audit dir with a clean per-process state."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    importlib.reload(audit)
    yield
    importlib.reload(audit)


def test_psk_degrade_writes_poison_record(monkeypatch, tmp_path):
    """Signed -> unsigned mid-run produces a poison record, not a drop.

    Pre-fix: record dropped silently; verify_audit_integrity reports ok=True
              (only the first signed record exists, which verifies fine).
    Post-fix: poison record present; verify_audit_integrity reports ok=False
              (poison record's sig != valid HMAC -> bad_signature += 1).
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "first-key-very-secret")
    audit.log_safety_event("first_event", "peer-A", {"action": "estop"})

    # Now clear the PSK -- a downgrade. The next write must NOT silently drop.
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")
    audit.log_safety_event("second_event_under_degrade", "peer-A", {"action": "stop"})

    # Read the records.
    records = list(audit.read_audit_log())
    assert len(records) == 2, f"expected poison record + 1 signed; got {records}"

    # Find the poison record.
    poison = [r for r in records if r.get("event") == "second_event_under_degrade"]
    assert len(poison) == 1
    assert poison[0].get("sig") == "PSK_DEGRADED", f"expected poison sentinel; got sig={poison[0].get('sig')!r}"
    assert "psk_degraded" in poison[0], "expected psk_degraded reason field for forensic traceability"

    # verify_audit_integrity must report failure.
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "first-key-very-secret")
    report = audit.verify_audit_integrity()
    assert report["ok"] is False, f"poison record must surface as ok=False (preserves forensic trail); got {report}"
    assert report["bad_signature"] >= 1, (
        f"poison sentinel sig=PSK_DEGRADED must count as bad_signature (literal != hex HMAC); got {report}"
    )


def test_psk_degrade_unsigned_to_signed_writes_poison(monkeypatch, tmp_path):
    """Unsigned -> signed mid-run is also refused symmetrically.

    Same poison-record discipline: instead of dropping, write a record
    with sig=PSK_DEGRADED so the unsigned prefix transition is visible
    in the log.
    """
    # First write WITHOUT PSK.
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit.log_safety_event("first_unsigned", "peer-A", {"action": "estop"})

    # Now ROTATE IN a PSK -- direction the symmetric the prior fix pin documents.
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "appeared-mid-run")
    audit.log_safety_event("second_under_psk_appearance", "peer-A", {"action": "stop"})

    records = list(audit.read_audit_log())
    assert len(records) == 2

    poison = [r for r in records if r.get("event") == "second_under_psk_appearance"]
    assert len(poison) == 1
    assert poison[0].get("sig") == "PSK_DEGRADED"
    assert "psk_degraded" in poison[0]
