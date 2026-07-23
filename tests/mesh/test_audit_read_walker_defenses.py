"""Forensic-walker defenses for :mod:`strands_robots.mesh.audit`.

``read_audit_log`` is both the operator-facing forensic reader and the
seed source for ``_load_seq_counters`` when the seq sidecar is corrupt.
That dual role makes its file-handling discipline security-relevant:

* A rotated log file swapped for a SYMLINK must be refused, not followed
  (mirrors the O_NOFOLLOW discipline applied to every other open in the
  module). Following it would let an attacker redirect the forensic read
  to ``/dev/null`` (fail-open seq reset) or to forged content.
* ``_audit_log_files_in_order`` must only treat files whose suffix after
  the active-log name is purely numeric as rotated copies; an unrelated
  sibling such as ``mesh_audit.jsonl.bak`` must be ignored so its bytes
  never enter the forensic stream.
* The ``since=`` cutoff must drop records strictly older than the
  timestamp (and records with a non-numeric ``ts``) while keeping newer
  ones, so a time-bounded forensic query returns only the window asked
  for.
* A corrupt line that is not valid JSON (a torn write, or a
  forward-compatibility line from a newer writer) must be skipped so
  the whole read is not aborted, while a DEBUG breadcrumb is emitted --
  the skipped line is invisible to the ``_load_seq_counters`` seq-seed
  walk, so the breadcrumb is the operator's only signal the seed may be
  incomplete.

These behaviors had no direct regression coverage; this file pins them.
"""

from __future__ import annotations

import os

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolated_audit(monkeypatch, tmp_path):
    """Fresh audit dir + reset module state for each test."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None
    yield
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform without symlink support")
def test_read_audit_log_refuses_symlinked_rotated_file(tmp_path, monkeypatch, caplog):
    """A rotated log file that is a SYMLINK must be skipped, not followed.

    The active log is read normally; the symlinked rotated copy's target
    bytes must never appear in the returned records.
    """
    # Write a legitimate record to the active log.
    audit.log_safety_event("real_event", "peer-a", {"index": 1})
    active = audit.audit_log_path()
    assert active.exists()

    # Plant attacker-controlled bytes outside the audit dir, then point a
    # rotated-log path at them via a symlink.
    secret = tmp_path / "attacker_payload.jsonl"
    secret.write_text('{"event": "forged", "payload": {"index": 999}}\n', encoding="utf-8")
    rotated = active.with_suffix(active.suffix + ".1")
    os.symlink(secret, rotated)
    assert rotated.is_symlink()

    with caplog.at_level("WARNING"):
        records = audit.read_audit_log()

    events = [r.get("event") for r in records]
    assert "real_event" in events, "active log record should still be read"
    assert "forged" not in events, "symlinked rotated log must not be followed"
    assert any("refusing to read" in m and "SYMLINK" in m for m in caplog.messages), (
        f"expected a symlink-refusal warning, got {caplog.messages}"
    )


def test_files_in_order_ignores_non_numeric_suffix_siblings(tmp_path, monkeypatch):
    """Only ``<active>.<digits>`` siblings count as rotated copies.

    A sibling like ``mesh_audit.jsonl.bak`` must be excluded so its bytes
    never enter the forensic read stream.
    """
    audit.log_safety_event("real_event", "peer-a", {"index": 1})
    active = audit.audit_log_path()

    # A genuine rotated copy (numeric suffix) and a decoy (non-numeric).
    rotated = active.with_suffix(active.suffix + ".1")
    rotated.write_text('{"event": "rotated_event", "payload": {"index": 0}}\n', encoding="utf-8")
    decoy = active.with_suffix(active.suffix + ".bak")
    decoy.write_text('{"event": "decoy_event", "payload": {"index": -1}}\n', encoding="utf-8")

    ordered = audit._audit_log_files_in_order()
    names = [p.name for p in ordered]
    assert rotated.name in names, "numeric-suffix rotated copy must be included"
    assert decoy.name not in names, "non-numeric-suffix sibling must be excluded"
    # Chronological order: oldest rotated copy first, active log last.
    assert names[-1] == active.name

    events = [r.get("event") for r in audit.read_audit_log()]
    assert "rotated_event" in events
    assert "real_event" in events
    assert "decoy_event" not in events


def test_read_audit_log_since_filters_old_and_non_numeric_ts(tmp_path, monkeypatch):
    """``since=`` keeps records at/after the cutoff and drops older ones.

    A record whose ``ts`` is missing or non-numeric is also dropped, since
    it cannot be placed in the requested time window.
    """
    active = audit.audit_log_path()
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "\n".join(
            [
                '{"event": "old", "ts": 100.0, "payload": {}}',
                '{"event": "new", "ts": 200.0, "payload": {}}',
                '{"event": "no_ts", "payload": {}}',
                '{"event": "bad_ts", "ts": "not-a-number", "payload": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = audit.read_audit_log(since=150.0)
    events = [r.get("event") for r in records]
    assert events == ["new"], f"since= should keep only records at/after the cutoff, got {events}"


def test_read_audit_log_skips_malformed_json_line(tmp_path, monkeypatch, caplog):
    """A corrupt (non-JSON) line is skipped; surrounding valid records survive.

    A torn write (process killed mid-append) or a forward-compat line from a
    newer writer can leave a line that is not valid JSON. The forensic walker
    must skip that single line rather than abort the entire read, so the valid
    records around it remain recoverable. A DEBUG breadcrumb is emitted because
    a skipped line is invisible to the ``_load_seq_counters`` seq-seed walk,
    which would otherwise silently under-seed the per-peer counter.
    """
    active = audit.audit_log_path()
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "\n".join(
            [
                '{"event": "before", "payload": {"i": 1}}',
                "{ this is not valid json",
                '{"event": "after", "payload": {"i": 2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level("DEBUG", logger="strands_robots.mesh.audit"):
        records = audit.read_audit_log()

    events = [r.get("event") for r in records]
    assert events == ["before", "after"], f"malformed line must be skipped while valid records are kept, got {events}"
    assert any("skipping malformed line" in m for m in caplog.messages), (
        f"expected a DEBUG breadcrumb for the skipped line, got {caplog.messages}"
    )


def test_read_audit_log_skips_blank_lines_without_a_corruption_breadcrumb(tmp_path, monkeypatch, caplog):
    """Blank / whitespace-only lines are skipped silently, not as corruption.

    An append-only JSONL audit log routinely carries blank lines: the trailing
    newline after the final record yields an empty last line, and a torn write
    (process killed between the newline and the next record) can leave a
    whitespace-only line. The reader guards these with an explicit empty-line
    skip *before* the ``json.loads`` attempt, so a blank line never reaches the
    malformed-JSON branch. That distinction matters: the "skipping malformed
    line" DEBUG breadcrumb is the operator's only signal that a real corrupt
    record may have under-seeded the ``_load_seq_counters`` seq walk. Emitting
    it for an ordinary trailing newline would drown that signal in
    false positives, so a blank line must be dropped without any breadcrumb.
    """
    active = audit.audit_log_path()
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "\n".join(
            [
                '{"event": "before", "payload": {"i": 1}}',
                "",
                "   ",
                '{"event": "after", "payload": {"i": 2}}',
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level("DEBUG", logger="strands_robots.mesh.audit"):
        records = audit.read_audit_log()

    events = [r.get("event") for r in records]
    assert events == ["before", "after"], f"blank lines must be skipped while valid records are kept, got {events}"
    assert not any("skipping malformed line" in m for m in caplog.messages), (
        f"a blank line is not corruption; it must not emit a malformed-line breadcrumb, got {caplog.messages}"
    )
