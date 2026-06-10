"""Pin tests for the audit-log seed fallback gating (R3 perf concern).

Reviewed concern (R3 thread on ``strands_robots/mesh/audit.py:702``):
``_next_seq`` resets ``_AUDIT_STATE.seq_loaded = False`` on every call,
which is correct for the cheap sidecar path (peer-process increments
need to be merged inside the flock). But when the sidecar is degraded
(missing, corrupt, symlinked, or non-dict payload after fix 1) the
fallback path walks the entire audit-log rotation set and verifies the
HMAC on every record. Pre-fix, that walk ran on EVERY ``_next_seq``
call -- O(events_logged_so_far) per event -- so a 100 MiB rotation set
at 20 events/sec produced seconds-per-event latency on the safety code
path.

Post-fix: a separate ``_AUDIT_STATE.audit_log_seeded`` flag gates the
expensive walk. It is set to True after the walk runs once (regardless
of whether records were found) and persists across ``seq_loaded``
resets. The walk only repeats on a fresh process (where ``__init__``
sets the flag back to False). The cheap sidecar path still runs on
every call.

This test pins the behaviour by:
1. Constructing an audit log with N records and NO sidecar (the
   degraded-sidecar entry condition).
2. Calling ``_load_seq_counters`` repeatedly.
3. Counting how many times ``read_audit_log`` is invoked across all
   calls.

Pre-fix: every call walks the log -> ``read_audit_log`` is called N
times for N ``_load_seq_counters`` invocations. Post-fix:
``read_audit_log`` is called exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from strands_robots.mesh import audit


@pytest.fixture
def isolated_audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated audit directory and reset audit module state."""
    audit_dir = tmp_path / "mesh_audit"
    audit_dir.mkdir()
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(audit_dir))
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._SEQ_COUNTERS.clear()
    return audit_dir


def _seed_log_with_records(peer_id: str, count: int) -> None:
    log_path = audit.audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        for seq in range(1, count + 1):
            fh.write(
                json.dumps(
                    {
                        "ts": 1000.0 + seq,
                        "peer_id": peer_id,
                        "seq": seq,
                        "event": "test_event",
                    }
                )
                + "\n"
            )


def test_degraded_sidecar_walks_audit_log_only_once_across_many_calls(
    isolated_audit_dir: Path,
) -> None:
    """The audit-log seed fallback must not repeat per ``_next_seq``.

    Pre-fix: ``_load_seq_counters`` re-walked the full audit log on every
    call when the sidecar was unusable. With N=100 invocations,
    ``read_audit_log`` was called 100 times.

    Post-fix: ``audit_log_seeded`` flag gates the walk to once per
    process. ``read_audit_log`` is called exactly once across the 100
    invocations; subsequent calls take the cheap fast path.
    """
    peer_id = "perf-test-peer"
    _seed_log_with_records(peer_id, count=5)

    # No sidecar exists -> every _load_seq_counters call hits the fallback
    # path. We spy on read_audit_log to count walks.
    walk_count = 0
    real_read_audit_log = audit.read_audit_log

    def counting_read_audit_log(*args, **kwargs):
        nonlocal walk_count
        walk_count += 1
        return real_read_audit_log(*args, **kwargs)

    with patch.object(audit, "read_audit_log", side_effect=counting_read_audit_log):
        for _ in range(100):
            audit._AUDIT_STATE.seq_loaded = False  # mimic _next_seq
            audit._load_seq_counters()

    # Cheap sidecar fast path runs on every call (it has to, for cross-
    # process peer increments). The expensive audit-log walk runs once.
    assert walk_count == 1, (
        f"Expected exactly 1 audit-log walk across 100 _load_seq_counters "
        f"invocations on a degraded-sidecar scenario, got {walk_count}. "
        "Pre-fix, this returned 100 (every safety event re-walked the full "
        "rotation set, producing O(events_logged_so_far) latency per event). "
        "The audit_log_seeded flag must persist across seq_loaded resets."
    )

    # Confirm the seed itself worked: counter still reflects max(seq)=5.
    assert audit._SEQ_COUNTERS.get(peer_id) == 5
    assert audit._next_seq(peer_id) == 6


def test_audit_log_seeded_flag_clears_per_process(
    isolated_audit_dir: Path,
) -> None:
    """A fresh process (simulated by clearing both flags) re-walks once.

    The flag protects against in-process re-walks; it does NOT prevent
    a process restart from re-establishing the floor from the audit log.
    """
    peer_id = "fresh-process-peer"
    _seed_log_with_records(peer_id, count=3)

    walk_count = 0
    real_read_audit_log = audit.read_audit_log

    def counting_read_audit_log(*args, **kwargs):
        nonlocal walk_count
        walk_count += 1
        return real_read_audit_log(*args, **kwargs)

    with patch.object(audit, "read_audit_log", side_effect=counting_read_audit_log):
        # First "process" walks once.
        audit._load_seq_counters()
        assert walk_count == 1
        assert audit._SEQ_COUNTERS.get(peer_id) == 3

        # Simulate a fresh process: clear ALL audit state (this mimics
        # what happens at module import time).
        audit._AUDIT_STATE.seq_loaded = False
        audit._AUDIT_STATE.audit_log_seeded = False
        audit._SEQ_COUNTERS.clear()

        audit._load_seq_counters()
        assert walk_count == 2
        assert audit._SEQ_COUNTERS.get(peer_id) == 3
