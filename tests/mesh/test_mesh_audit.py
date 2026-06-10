"""Tests for strands_robots.mesh_audit — append-only safety audit log."""

from __future__ import annotations

import json
import os
import time

import pytest

from strands_robots.mesh import audit as mesh_audit


@pytest.fixture
def audit_env(tmp_path, monkeypatch):
    """Redirect the audit directory to a temp path for the duration of a test."""
    target = tmp_path / "audit"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(target))
    return target


def test_audit_log_path_uses_env_override(audit_env):
    path = mesh_audit.audit_log_path()
    assert path == audit_env / "mesh_audit.jsonl"


def test_audit_log_path_default(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_AUDIT_DIR", raising=False)
    path = mesh_audit.audit_log_path()
    assert path.name == "mesh_audit.jsonl"
    # Default lives under home directory.
    assert ".strands_robots" in str(path)


def test_log_safety_event_creates_file_with_0600(audit_env):
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"reason": "test"})
    log_file = audit_env / "mesh_audit.jsonl"
    assert log_file.exists()
    mode = log_file.stat().st_mode & 0o777
    assert mode == 0o600


def test_log_safety_event_creates_dir_with_0700(audit_env):
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {})
    mode = audit_env.stat().st_mode & 0o777
    assert mode == 0o700


def test_log_safety_event_appends_one_line_per_event(audit_env):
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 1})
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 2})
    mesh_audit.log_safety_event("emergency_stop", "peer-b", {"i": 3})

    log_file = audit_env / "mesh_audit.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert [p["payload"]["i"] for p in parsed] == [1, 2, 3]
    assert [p["peer_id"] for p in parsed] == ["peer-a", "peer-a", "peer-b"]


def test_log_safety_event_record_shape(audit_env):
    mesh_audit.log_safety_event(
        event_type="emergency_stop",
        peer_id="peer-x",
        payload={"sender_id": "peer-x", "responses_received": 5},
    )
    record = json.loads((audit_env / "mesh_audit.jsonl").read_text().strip())
    # The record carries a process-monotonic "seq" plus an optional "sig"
    # (HMAC) when STRANDS_MESH_AUDIT_PSK is configured.
    expected = {"ts", "event", "peer_id", "payload", "seq"}
    assert expected.issubset(set(record.keys()))
    assert set(record.keys()) - expected <= {"sig"}
    assert isinstance(record["ts"], float)
    assert record["event"] == "emergency_stop"
    assert record["peer_id"] == "peer-x"
    assert record["payload"]["responses_received"] == 5


def test_read_audit_log_returns_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "nope"))
    assert mesh_audit.read_audit_log() == []


def test_read_audit_log_returns_all_records(audit_env):
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 1})
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 2})

    records = mesh_audit.read_audit_log()
    assert len(records) == 2
    assert [r["payload"]["i"] for r in records] == [1, 2]


def test_read_audit_log_filters_by_since(audit_env):
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 1})
    cutoff = time.time() + 0.01
    time.sleep(0.05)
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 2})

    records = mesh_audit.read_audit_log(since=cutoff)
    assert len(records) == 1
    assert records[0]["payload"]["i"] == 2


def test_read_audit_log_skips_malformed_lines(audit_env):
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 1})
    log_file = audit_env / "mesh_audit.jsonl"
    with open(log_file, "a") as fh:
        fh.write("this is not json\n")
        fh.write('{"event": "x", "peer_id": "p", "payload": {}, "ts": 1.0}\n')

    records = mesh_audit.read_audit_log()
    # Both well-formed lines are returned; the corrupt one is skipped.
    assert len(records) == 2
    events = [r["event"] for r in records]
    assert events == ["emergency_stop", "x"]


def test_log_safety_event_handles_write_error(audit_env, monkeypatch):
    """A write failure must not propagate."""
    real_open = open

    def boom(path, *a, **kw):
        # Allow ensure_paths to work (touch + chmod) but break the append.
        if "a" in (a[0] if a else kw.get("mode", "")):
            raise OSError("disk full")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", boom)
    # Should not raise.
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {})


def test_log_safety_event_re_chmods_file_each_call(audit_env):
    """Even if the file mode is tampered, log_safety_event re-applies 0o600."""
    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 1})
    log_file = audit_env / "mesh_audit.jsonl"
    os.chmod(log_file, 0o644)  # someone widens it

    mesh_audit.log_safety_event("emergency_stop", "peer-a", {"i": 2})
    mode = log_file.stat().st_mode & 0o777
    assert mode == 0o600


# -----------------------------------------------------------------------
# Issue #238: _seq_flock hard-fails on symlinked lockfile (no silent
# downgrade of cross-process serialisation).
# -----------------------------------------------------------------------


class TestSeqFlockSymlinkRejection:
    """Pin: pre-creating ``mesh_audit.seq.lock`` as a symlink must
    raise ``SeqLockSymlinkError`` rather than silently yielding without
    a lock. Closes the gap that defeated the per-peer-monotonic-seq
    guarantee under attacker control of the audit dir.
    """

    def test_symlinked_lockfile_raises_hard_fail(self, tmp_path, monkeypatch):
        import os

        from strands_robots.mesh import audit as audit_mod

        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        # Pre-create the lockfile as a symlink
        lockfile = tmp_path / "mesh_audit.seq.lock"
        target = tmp_path / "innocent_target"
        target.touch()
        os.symlink(str(target), str(lockfile))
        assert lockfile.is_symlink()

        # Try to acquire the seq flock -- must raise SeqLockSymlinkError
        # not silently yield without a lock
        with pytest.raises(audit_mod.SeqLockSymlinkError):
            with audit_mod._seq_flock():
                pass  # should never reach here

    def test_normal_lockfile_still_works(self, tmp_path, monkeypatch):
        """Sanity: regular (non-symlink) lockfile path is untouched."""
        from strands_robots.mesh import audit as audit_mod

        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        # No pre-created lockfile -- _seq_flock creates it normally
        with audit_mod._seq_flock():
            pass  # acquires lock successfully

        # Lockfile should now exist as a regular file (not a symlink)
        lockfile = tmp_path / "mesh_audit.seq.lock"
        assert lockfile.exists()
        assert not lockfile.is_symlink()


# -----------------------------------------------------------------------
# Issue #238 R2: SEQ_LOCK_DEGRADED poison record discipline
# -----------------------------------------------------------------------


class TestSeqLockDegradedPoisonRecord:
    """Pin: when ``_next_seq`` raises ``SeqLockSymlinkError``,
    ``log_safety_event`` writes a poison record with
    ``sig="SEQ_LOCK_DEGRADED"`` rather than silently dropping the
    record. Mirrors the PSK_DEGRADED discipline.
    """

    def test_symlinked_seq_lockfile_writes_poison_record(self, tmp_path, monkeypatch):
        import os

        from strands_robots.mesh import audit as audit_mod

        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        # Create the audit dir and the lockfile-as-symlink
        tmp_path.mkdir(exist_ok=True)
        lockfile = tmp_path / "mesh_audit.seq.lock"
        if lockfile.exists():
            lockfile.unlink()
        target = tmp_path / "innocent"
        target.touch()
        os.symlink(str(target), str(lockfile))
        assert lockfile.is_symlink()

        # Reset state
        if hasattr(audit_mod, "_AUDIT_STATE"):
            audit_mod._AUDIT_STATE.psk_fingerprint = None
            audit_mod._AUDIT_STATE.seq_loaded = False
            audit_mod._AUDIT_STATE.audit_log_seeded = True
        if hasattr(audit_mod, "_SEQ_COUNTERS"):
            audit_mod._SEQ_COUNTERS.clear()

        # log_safety_event must NOT crash; instead it writes a poison record
        audit_mod.log_safety_event("test_event", "victim-peer", {"k": "v"})

        # Walk the audit log -- the record must be present with the
        # SEQ_LOCK_DEGRADED poison sig
        records = audit_mod.read_audit_log()
        assert len(records) >= 1
        seq_lock_records = [r for r in records if r.get("sig") == "SEQ_LOCK_DEGRADED"]
        assert len(seq_lock_records) == 1, (
            f"expected 1 SEQ_LOCK_DEGRADED poison record; got: {[r.get('sig') for r in records]}"
        )
        rec = seq_lock_records[0]
        assert rec["event"] == "test_event"
        assert rec["peer_id"] == "victim-peer"
        assert "seq_lock_degraded" in rec  # contains the reason
