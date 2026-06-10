"""Pin: sidecar seq values are capped on healthy load.

The :func:`_load_seq_counters` healthy-sidecar path applies the same
``_MAX_SEED_SEQ`` cap that the audit-log fallback path uses. Without
the cap, an attacker with audit-dir write access could drop a
syntactically valid sidecar like ``{"victim": 999999999}``; the next
process restart would seed the seq counter at ~10**9 and the
legitimate writer's first event would silently jump by a billion with
no upper bound and no operator-visible signal.

Note: the sidecar file is unsigned (the per-record HMAC defence
covers the audit-log only), so this cap is the only fail-loud surface
on the healthy-sidecar code path. The audit-log walk has both the cap
and the prior HMAC-verify defence.

These tests pin the cap, the WARNING, and the symmetric behaviour
between the sidecar and audit-log seed paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from strands_robots.mesh import audit


@pytest.fixture
def isolated_audit(tmp_path, monkeypatch):
    """Point the audit module at an empty per-test temp dir."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    # Reset module-level state so each test starts from a clean slate.
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)
    yield tmp_path
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False  # R3: gate the audit-log walk fallback (PR #221)


def _write_sidecar(tmp_path: Path, payload: dict) -> Path:
    sidecar = tmp_path / "mesh_audit.seq.json"
    sidecar.write_text(json.dumps(payload))
    return sidecar


def test_sidecar_value_above_cap_is_refused(isolated_audit, caplog):
    """Sidecar value > ``_MAX_SEED_SEQ`` is dropped with a WARNING."""
    bad = audit._MAX_SEED_SEQ + 1
    _write_sidecar(isolated_audit, {"victim": bad, "ok": 100})
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
        audit._load_seq_counters()
    assert audit._SEQ_COUNTERS.get("victim", 0) == 0, "over-cap sidecar value must NOT seed the counter"
    assert audit._SEQ_COUNTERS.get("ok") == 100, "well-formed sidecar entry next to a tampered one must still seed"
    msgs = [r.getMessage() for r in caplog.records if "victim" in r.getMessage() and "tampered" in r.getMessage()]
    assert msgs, f"expected a WARNING naming the victim peer; got {[r.getMessage() for r in caplog.records]}"


def test_sidecar_value_at_cap_accepted(isolated_audit):
    """Boundary: exactly ``_MAX_SEED_SEQ`` is accepted (cap is inclusive)."""
    _write_sidecar(isolated_audit, {"peer": audit._MAX_SEED_SEQ})
    audit._load_seq_counters()
    assert audit._SEQ_COUNTERS["peer"] == audit._MAX_SEED_SEQ


def test_sidecar_value_below_cap_accepted(isolated_audit):
    """Sanity: realistic values pass."""
    _write_sidecar(isolated_audit, {"peer-a": 12345, "peer-b": 67890})
    audit._load_seq_counters()
    assert audit._SEQ_COUNTERS == {"peer-a": 12345, "peer-b": 67890}


def test_max_seed_seq_is_module_level_constant():
    """The cap is a module-level constant (so the audit-log fallback
    references the same value, keeping the two paths symmetric)."""
    assert isinstance(audit._MAX_SEED_SEQ, int)
    assert audit._MAX_SEED_SEQ > 0
    # Sanity: a 100-million-events-per-peer-per-year cap is the documented
    # value; this also pins the constant against an accidental change.
    assert audit._MAX_SEED_SEQ == 100_000_000


def test_negative_value_silently_rejected(isolated_audit):
    """Negative seq values fall through the existing isinstance gate."""
    _write_sidecar(isolated_audit, {"bad": -1, "ok": 5})
    audit._load_seq_counters()
    assert "bad" not in audit._SEQ_COUNTERS
    assert audit._SEQ_COUNTERS["ok"] == 5
