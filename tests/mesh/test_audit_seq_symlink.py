"""Pin: ``_load_seq_counters`` refuses to follow a symlinked sidecar.

Background: the prior fix review flagged that ``_persist_seq_counters`` checks
``is_symlink()`` and uses ``O_NOFOLLOW`` (defence in depth against a swap-and-
write attack), but ``_load_seq_counters`` opened the sidecar with plain
``open()`` -- asymmetric defence (anti-pattern #25 in our system prompt).

A symlink-swap attacker who can drop a symlink at ``mesh_audit.seq.json``
between writer crashes and the next reader start can redirect the counter
restore to attacker-controlled state -- e.g. a sidecar from a different
audit dir, or ``/dev/null`` returning zero counters that roll the per-peer
cursor back. The inter-process ``_seq_flock`` does NOT defend against this:
it serialises *writers*, not the inode the reader is about to open.

These tests pin the symmetric defence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolate_audit_state(tmp_path, monkeypatch):
    """Each test starts with a fresh audit dir + reset module state."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    # Reset the module-level seq state (one-shot loaded flag + counters).
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)
    audit._SEQ_COUNTERS.clear()
    yield
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)
    audit._SEQ_COUNTERS.clear()


def _write_real_sidecar(audit_dir: Path, payload: dict) -> Path:
    sidecar = audit_dir / "mesh_audit.seq.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


@pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt",
    reason="symlink semantics differ on Windows; O_NOFOLLOW is 0 there",
)
def test_load_seq_counters_refuses_symlinked_sidecar(tmp_path, caplog) -> None:
    """A symlink at the sidecar path must not be followed.

    Pre-fix, the open() call would happily follow the symlink and load
    attacker-chosen counters (or no counters from a /dev/null target),
    silently rolling the per-peer cursor backward.
    """
    # Attacker-controlled file lives outside the audit dir.
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    attacker_sidecar = attacker_dir / "evil.json"
    attacker_sidecar.write_text(json.dumps({"victim_peer": 1}), encoding="utf-8")

    # Symlink the real sidecar path at the attacker's file.
    audit_dir = tmp_path
    sidecar = audit_dir / "mesh_audit.seq.json"
    os.symlink(attacker_sidecar, sidecar)

    # Pre-populate _SEQ_COUNTERS at a higher value than the attacker's.
    audit._SEQ_COUNTERS["victim_peer"] = 1000

    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit._load_seq_counters()

    # Counter must NOT be rolled backward by the attacker's value.
    assert audit._SEQ_COUNTERS["victim_peer"] == 1000, (
        "_load_seq_counters must not follow a symlinked sidecar -- "
        "the attacker payload was loaded and rolled the counter back"
    )
    # And a WARNING must surface so operators can attribute the no-op.
    assert any("SYMLINK" in rec.message or "symlink" in rec.message.lower() for rec in caplog.records), (
        "expected a WARNING line about the symlinked sidecar"
    )


def test_load_seq_counters_loads_real_sidecar(tmp_path) -> None:
    """The non-symlink happy path still restores counters as before."""
    _write_real_sidecar(tmp_path, {"peerA": 42, "peerB": 100})
    audit._load_seq_counters()
    assert audit._SEQ_COUNTERS.get("peerA") == 42
    assert audit._SEQ_COUNTERS.get("peerB") == 100


def test_load_seq_counters_does_not_roll_counter_backwards(tmp_path) -> None:
    """The monotonic-restore guard from the prior implementation is preserved."""
    audit._SEQ_COUNTERS["peerA"] = 500
    _write_real_sidecar(tmp_path, {"peerA": 1})  # disk says 1, memory says 500
    audit._load_seq_counters()
    assert audit._SEQ_COUNTERS["peerA"] == 500, "load must never roll a counter backward even from a stale sidecar"
