"""A payload the JSON encoder cannot represent is poisoned, never dropped.

Two contracts meet at the serialisation site in
:func:`strands_robots.mesh.audit.log_safety_event`.

The first is fail-soft: the function header promises it raises nothing,
because an audit-log failure must never propagate up into the safety code
path that called it. A ``TypeError`` from a non-serialisable payload used to
escape and crash that path.

The second is poison-record symmetry: every degraded audit path writes a
record carrying a discriminating ``sig`` rather than returning early, so a
:func:`strands_robots.mesh.audit.verify_audit_integrity` walker can attribute
a stream anomaly to a failure class. ``_next_seq`` has already consumed and
persisted this peer's sequence number before serialisation is attempted, so a
silent drop leaves behind exactly what the module header documents as
"records were deleted" -- a payload mistake wearing the forensic signature of
tampering. ``SEQ_LOCK_DEGRADED``, ``NEXT_SEQ_DEGRADED``, ``PSK_DEGRADED`` and
``SIGN_FAILED`` each keep the second contract; the serialisation branch kept
only the first.

The tests below hold both: nothing escapes, and the unrepresentable payload is
replaced by a bounded diagnostic under ``sig="SERIALISE_FAILED"`` so the
consumed sequence number still has a record to name it.
"""

import datetime
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

    def test_the_failure_is_reported_under_its_discriminator(self, caplog):
        """The failure must be reported, not silent.

        It is reported at ERROR naming ``SERIALISE_FAILED``, the level and shape
        the other four poison paths use, because a record that reaches the log
        with a poison ``sig`` is an integrity event rather than a lost write.
        """
        from strands_robots.mesh.audit import log_safety_event

        with caplog.at_level(logging.DEBUG):
            log_safety_event("test_event", "peer1", {"obj": object()})

        reported = [r for r in caplog.records if "SERIALISE_FAILED" in r.message]
        assert reported, f"the failure was not reported: {[r.message for r in caplog.records]}"
        assert reported[0].levelno == logging.ERROR, reported[0].levelname

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


#: The degraded paths of ``log_safety_event``, each with the ``sig`` it must
#: leave behind. Driven from one table so a path added later is graded here
#: instead of quietly becoming the next silent drop.
_DEGRADED_PATHS = (
    ("seq lockfile is a symlink", "_next_seq", "SeqLockSymlinkError", "SEQ_LOCK_DEGRADED"),
    ("seq sidecar unreadable", "_next_seq", "OSError", "NEXT_SEQ_DEGRADED"),
    ("PSK flipped mid-run", "_sign_record", "AuditPSKDegradedError", "PSK_DEGRADED"),
    ("signing raised", "_sign_record", "RuntimeError", "SIGN_FAILED"),
    ("payload not representable", None, None, "SERIALISE_FAILED"),
)

#: Payload values a robot safety event plausibly carries that the JSON encoder
#: cannot represent. A numpy pose or a raw fingerprint is the ordinary mistake
#: here, not an exotic one.
_UNREPRESENTABLE = (
    ("bytes fingerprint", {"fingerprint": b"\x01\x02\x03"}),
    ("set of zone ids", {"zones": {"dock-a", "dock-b"}}),
    ("datetime", {"at": datetime.datetime(2026, 1, 1)}),
    ("bare object", {"handle": object()}),
)


def _degraded_record(monkeypatch, *, attr, exc_name, payload):
    """Drive one degraded path between two clean records, return the audit log.

    Returns the record list so a caller can assert on the middle record and on
    the two that bracket it.
    """
    from strands_robots.mesh import audit

    audit.log_safety_event("clean_before", "peer-d", {"ok": 1})
    if attr is not None:
        exc: BaseException
        if exc_name == "SeqLockSymlinkError":
            exc = audit.SeqLockSymlinkError("the seq lockfile is a symlink")
        elif exc_name == "AuditPSKDegradedError":
            exc = audit.AuditPSKDegradedError("signed -> unsigned mid-run")
        elif exc_name == "OSError":
            exc = OSError("seq sidecar unreadable")
        else:
            exc = RuntimeError("hmac backend unavailable")
        if exc_name == "RuntimeError":
            # SIGN_FAILED is only poisoned when a PSK is configured; without
            # one the unsigned write is the documented dev-mode posture.
            monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "a-test-key")

        def _raise(*_args, **_kwargs):
            raise exc

        monkeypatch.setattr(audit, attr, _raise)
    audit.log_safety_event("degraded", "peer-d", payload)
    # Deliberately no monkeypatch.undo() here: it would also revert the autouse
    # fixture's STRANDS_MESH_AUDIT_DIR and point read_audit_log at the real
    # default directory. pytest tears both down at test exit.
    return audit


class TestAnUnrepresentablePayloadLeavesAnAttributableRecord:
    """The consumed sequence number still has a record naming the failure."""

    def test_the_record_is_written_under_the_serialise_failed_discriminator(self, monkeypatch):
        """Pre-fix: no record at all, so the seq gap named no failure class."""
        import numpy as np

        from strands_robots.mesh import audit

        audit.log_safety_event("clean_before", "peer-s", {"ok": 1})
        audit.log_safety_event("collision_imminent", "peer-s", {"pose": np.zeros(3)})

        records = audit.read_audit_log(since=0)
        poisoned = [r for r in records if r.get("event") == "collision_imminent"]
        assert poisoned, (
            "the unrepresentable payload left no record, so the sequence number "
            f"_next_seq already consumed names no failure class: {records}"
        )
        assert poisoned[0].get("sig") == "SERIALISE_FAILED", poisoned[0]

    def test_the_record_keeps_the_sequence_number_the_writer_consumed(self, monkeypatch):
        """The poison record occupies the seq, so the anomaly has an owner."""
        import numpy as np

        from strands_robots.mesh import audit

        audit.log_safety_event("first", "peer-q", {"ok": 1})
        audit.log_safety_event("second", "peer-q", {"pose": np.zeros(2)})
        audit.log_safety_event("third", "peer-q", {"ok": 1})

        by_event = {r["event"]: r for r in audit.read_audit_log(since=0)}
        assert set(by_event) == {"first", "second", "third"}, by_event
        assert by_event["first"]["seq"] == 1
        assert by_event["second"]["seq"] == 2, "the consumed seq must be the poison record's"
        assert by_event["third"]["seq"] == 3

    def test_the_record_keeps_the_envelope_forensics_reads(self, monkeypatch):
        """event / peer_id / ts survive; only the payload could not be written."""
        import numpy as np

        from strands_robots.mesh import audit

        audit.log_safety_event("estop_engaged", "arm-7", {"joints": np.zeros(6)})

        record = audit.read_audit_log(since=0)[-1]
        assert record["event"] == "estop_engaged"
        assert record["peer_id"] == "arm-7"
        assert isinstance(record["ts"], float)

    def test_the_unrepresentable_value_is_replaced_by_a_bounded_diagnostic(self, monkeypatch):
        """The payload is not written verbatim; the reason names the encoder error."""
        from strands_robots.mesh import audit

        audit.log_safety_event("estop_engaged", "arm-8", {"handle": object()})

        record = audit.read_audit_log(since=0)[-1]
        assert record["payload"] == {"unrepresentable": True}, record["payload"]
        assert "TypeError" in record["payload_error"], record["payload_error"]
        assert len(record["payload_error"]) <= audit._MAX_POISON_REASON_CHARS

    @pytest.mark.parametrize(("label", "payload"), _UNREPRESENTABLE, ids=[c[0] for c in _UNREPRESENTABLE])
    def test_every_unrepresentable_shape_is_poisoned(self, label, payload, monkeypatch):
        """A robot payload carries poses and fingerprints, not just JSON scalars."""
        from strands_robots.mesh import audit

        audit.log_safety_event("probe", "peer-p", payload)

        records = audit.read_audit_log(since=0)
        assert records, f"{label} left no record"
        assert records[-1].get("sig") == "SERIALISE_FAILED", f"{label}: {records[-1]}"


class TestEveryDegradedPathWritesARecord:
    """One rule across the whole function, not four paths and an exception."""

    @pytest.mark.parametrize(
        ("label", "attr", "exc_name", "expected_sig"),
        _DEGRADED_PATHS,
        ids=[c[0] for c in _DEGRADED_PATHS],
    )
    def test_the_path_leaves_its_discriminator(self, label, attr, exc_name, expected_sig, monkeypatch):
        """Pre-fix the payload row wrote nothing while the other four did."""
        import numpy as np

        payload = {"pose": np.zeros(3)} if attr is None else {"ok": 1}
        audit = _degraded_record(monkeypatch, attr=attr, exc_name=exc_name, payload=payload)

        degraded = [r for r in audit.read_audit_log(since=0) if r.get("event") == "degraded"]
        assert degraded, f"{label} wrote no record, so the anomaly names no failure class"
        assert degraded[0].get("sig") == expected_sig, f"{label}: {degraded[0]}"

    def test_the_verifier_still_fails_closed_on_a_poisoned_stream(self, monkeypatch):
        """A degraded audit must never read as ok; the poison record is not a pass."""
        import numpy as np

        from strands_robots.mesh import audit

        audit.log_safety_event("clean_before", "peer-v", {"ok": 1})
        audit.log_safety_event("degraded", "peer-v", {"pose": np.zeros(3)})
        audit.log_safety_event("clean_after", "peer-v", {"ok": 1})

        assert audit.verify_audit_integrity()["ok"] is False


class TestTheGoodPathAndTheLastResortAreUnchanged:
    """Controls: a representable payload is untouched, and nothing escapes."""

    def test_a_representable_payload_carries_no_poison_fields(self):
        """The write path a caller normally takes is byte-identical."""
        from strands_robots.mesh import audit

        audit.log_safety_event("estop_engaged", "peer-g", {"reason": "operator"})

        record = audit.read_audit_log(since=0)[-1]
        assert "sig" not in record, "an unsigned good record must stay unsigned"
        assert "payload_error" not in record
        assert record["payload"] == {"reason": "operator"}

    def test_an_unrepresentable_envelope_still_raises_nothing(self, caplog):
        """When even the envelope cannot be written there is nothing to poison."""
        from strands_robots.mesh import audit

        with caplog.at_level(logging.WARNING):
            audit.log_safety_event(object(), "peer-e", {"ok": 1})  # type: ignore[arg-type]

        assert any("record dropped" in rec.message for rec in caplog.records), [r.message for r in caplog.records]
