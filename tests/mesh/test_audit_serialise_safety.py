"""Pin test: log_safety_event must not raise on non-JSON-serialisable payload.

Regression test for the P0 hold on PR #221 R5: json.dumps at the
serialisation site (formerly line 1103) is outside the signing and
write try blocks, so a TypeError from a non-serialisable payload
escaped log_safety_event and crashed the safety code path.

The contract at the function header says:
    Raises: Nothing -- audit-log failure must never propagate up into
    the safety code path that called this function.

The fix wraps json.dumps in try/except (TypeError, ValueError) and
drops the record with a WARNING, matching the existing fail-soft
discipline for signing failures.
"""

import logging

import pytest


@pytest.fixture(autouse=True)
def _audit_tmp_dir(monkeypatch, tmp_path):
    """Point the audit log at a temp directory and reset module state."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    # Reset module-level state so each test starts clean.
    from strands_robots.mesh import audit

    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._SEQ_COUNTERS.clear()


class TestLogSafetyEventNonSerialisablePayload:
    """log_safety_event must not raise on non-JSON-serialisable payload."""

    def test_non_serialisable_object_does_not_raise(self):
        """A payload containing a non-serialisable object must not crash."""
        from strands_robots.mesh.audit import log_safety_event

        # object() is not JSON-serialisable -- triggers TypeError in json.dumps
        log_safety_event("test_event", "peer1", {"obj": object()})
        # If we reach here, the contract holds: no exception escaped.

    def test_non_serialisable_set_does_not_raise(self):
        """A payload containing a set must not crash."""
        from strands_robots.mesh.audit import log_safety_event

        log_safety_event("test_event", "peer1", {"items": {1, 2, 3}})

    def test_non_serialisable_bytes_does_not_raise(self):
        """A payload containing raw bytes must not crash."""
        from strands_robots.mesh.audit import log_safety_event

        log_safety_event("test_event", "peer1", {"raw": b"\x00\x01\x02"})

    def test_warning_logged_on_non_serialisable(self, caplog):
        """A WARNING must be logged when serialisation fails."""
        from strands_robots.mesh.audit import log_safety_event

        with caplog.at_level(logging.WARNING):
            log_safety_event("test_event", "peer1", {"obj": object()})

        assert any("could not serialise record" in rec.message for rec in caplog.records), (
            f"Expected 'could not serialise record' warning, got: {[r.message for r in caplog.records]}"
        )

    def test_valid_payload_still_writes(self, tmp_path):
        """A valid payload must still write successfully."""
        from strands_robots.mesh.audit import audit_log_path, log_safety_event

        log_safety_event("test_event", "peer1", {"key": "value"})

        path = audit_log_path()
        assert path.exists(), "Audit log should have been created for valid payload"
        content = path.read_text()
        assert "test_event" in content
        assert "peer1" in content

    def test_nan_infinity_does_not_raise(self):
        """NaN/Infinity values (ValueError in strict JSON) must not crash."""
        from strands_robots.mesh.audit import log_safety_event

        # float('nan') and float('inf') raise ValueError with
        # json.dumps(..., allow_nan=False) but are accepted by default.
        # Still, ensure the contract holds for edge cases.
        log_safety_event("test_event", "peer1", {"val": float("nan")})
        log_safety_event("test_event", "peer1", {"val": float("inf")})
