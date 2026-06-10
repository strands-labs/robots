"""
Pin tests for audit sequence-counter recovery from corrupt sidecar.

When the sidecar file (~/.strands_robots/mesh_audit.seq.json) is corrupt
or rejected as a symlink, the system should seed _SEQ_COUNTERS by walking
the audit log instead of failing open (resetting all counters to 0).

Without this defense, an attacker writing garbage to the sidecar can
reset every peer's sequence counter on next boot.
"""

import json
from pathlib import Path

import pytest

from strands_robots.mesh import audit


@pytest.fixture
def isolated_audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide isolated audit directory via STRANDS_MESH_AUDIT_DIR."""
    audit_dir = tmp_path / "mesh_audit"
    audit_dir.mkdir()
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(audit_dir))
    # Reset audit state
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)
    audit._SEQ_COUNTERS.clear()
    return audit_dir


def test_corrupt_sidecar_seeds_from_log(isolated_audit_dir: Path) -> None:
    """
    When sidecar is corrupt, _load_seq_counters should seed from audit log.

    Pre-fix: corrupt sidecar -> _SEQ_COUNTERS empty -> _next_seq returns 1.
    Post-fix: corrupt sidecar -> walk log -> seed from max(seq) -> _next_seq returns 6.
    """
    peer_id = "test-peer-001"

    # 1. Write audit log with seq=5 for peer_id
    log_path = audit.audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for seq in range(1, 6):
            record_dict = {
                "ts": 1000.0 + seq,
                "peer_id": peer_id,
                "seq": seq,
                "event": "test_event",
            }
            f.write(json.dumps(record_dict) + "\n")

    # 2. Write corrupt sidecar (garbage JSON)
    sidecar_path = audit._seq_sidecar_path()
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write("{corrupt_json_not_valid")

    # 3. Load seq counters - should seed from log
    audit._load_seq_counters()

    # 4. Assert _next_seq returns 6 (not 1)
    next_val = audit._next_seq(peer_id)

    # Pre-fix: returns 1 (fail-open)
    # Post-fix: returns 6 (seeded from log)
    assert next_val == 6, f"Expected _next_seq to return 6 after seeding from log, got {next_val}"
