"""the prior fix pin tests for PSK rotation detection.

Yin's review on audit.py:582 -- the original snapshot was presence-only
(``bool | None``). An operator who rotated the PSK *value* mid-run
(e.g. via secret-manager rollover without restart) hit ``snapshot is
True and psk is not None`` -> falls through, no error, records
continue to be signed under the new key. A verifier holding the OLD
PSK fails signature on the post-rotation segment; a verifier holding
the NEW PSK fails on the pre-rotation segment. There is no
record-internal signal of which PSK was active for which records.

the prior fix fix: snapshot a fingerprint (``sha256(psk)[:16]``) and refuse any
mid-run transition (set->unset, unset->set, OR rotated value) by
writing a poison record (``sig="PSK_DEGRADED"``) -- mirroring the prior
poison record path.

Per AGENTS.md > Review Learnings (#85) > "Pin regression tests for
reviewed fixes."
"""

from __future__ import annotations

import pytest

import strands_robots.mesh.audit as audit


@pytest.fixture(autouse=True)
def _reset_audit_state(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)
    yield
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)


def test_psk_value_rotation_drops_record(monkeypatch, caplog):
    """rotating the PSK value mid-run must drop the next write
    with a poison record, not silently sign under the new key."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "key-A")
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={"sender_id": "op-1"})

    # Operator rotates the PSK value mid-run (no restart).
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "key-B")

    with caplog.at_level("ERROR", logger="strands_robots.mesh.audit"):
        audit.log_safety_event(
            event_type="estop",
            peer_id="op-1",
            payload={"sender_id": "op-1", "after_rotation": True},
        )

    records = audit.read_audit_log()
    # First (key-A signed), second (poison record from rotation).
    assert len(records) == 2

    sig0 = records[0].get("sig")
    sig1 = records[1].get("sig")
    assert sig0 is not None and sig0 != "PSK_DEGRADED"
    assert sig1 == "PSK_DEGRADED", "rotated-value record should be replaced by a poison record"

    # Operator-facing forensic signal.
    assert any("PSK" in rec.message and "rotat" in rec.message.lower() for rec in caplog.records), (
        "rotation should emit an ERROR mentioning rotation"
    )


def test_psk_fingerprint_snapshot_uses_value_not_just_presence(monkeypatch):
    """the snapshot must encode the PSK *value* fingerprint, not
    just presence -- otherwise rotation goes undetected."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "key-A")
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})

    snap_a = audit._AUDIT_STATE.psk_fingerprint
    assert isinstance(snap_a, bytes) and len(snap_a) == 16

    # Sanity: the fingerprint differs from a different value's.
    assert snap_a != audit._psk_fingerprint(b"key-B")
    assert snap_a == audit._psk_fingerprint(b"key-A")


def test_psk_unset_to_set_still_drops(monkeypatch):
    """Carry-over the prior fix contract: unset->set transition still refused
    (would create unverifiable unsigned prefix)."""
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})  # unsigned

    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "k")
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})

    records = audit.read_audit_log()
    assert len(records) == 2
    assert records[0].get("sig") is None  # the unsigned first record
    assert records[1].get("sig") == "PSK_DEGRADED"  # poison on transition


def test_psk_set_to_unset_still_drops(monkeypatch):
    """Carry-over R4-2 contract: set->unset transition still refused."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "k")
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})

    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})

    records = audit.read_audit_log()
    assert len(records) == 2
    assert records[0].get("sig") not in (None, "PSK_DEGRADED")
    assert records[1].get("sig") == "PSK_DEGRADED"


def test_psk_same_value_no_rotation_no_poison(monkeypatch):
    """Sanity: writing many records under the same PSK does NOT trip
    the rotation detector. Confirms the new fingerprint check doesn't
    over-fire."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "k")
    for _ in range(5):
        audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})

    records = audit.read_audit_log()
    assert len(records) == 5
    assert all(r.get("sig") not in (None, "PSK_DEGRADED") for r in records)


def test_psk_fingerprint_does_not_leak_psk(monkeypatch):
    """Defensive: the stored fingerprint must NOT contain the PSK
    bytes verbatim (it is sha256-truncated). Without this property,
    leaking ``_AUDIT_STATE.psk_fingerprint`` to a debugger / dump
    would leak the PSK."""
    secret = b"super-secret-key-do-not-leak"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", secret.decode())
    audit.log_safety_event(event_type="estop", peer_id="op-1", payload={})

    fp = audit._AUDIT_STATE.psk_fingerprint
    assert fp is not None
    assert secret not in fp
    assert len(fp) == 16  # sha256 truncated to 128 bits
