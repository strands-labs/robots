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
* Bytes that are not valid UTF-8 must cost their own line and nothing
  more. ``UnicodeDecodeError`` is a ``ValueError``, so a strict decode
  escapes the walker's ``except OSError``, aborts the read, discards the
  records already collected from earlier rotated copies, and takes
  ``verify_audit_integrity`` down with it -- one undecodable byte would
  silence the tamper report it is evidence for.

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


@pytest.mark.parametrize(
    ("damaged_line", "expected_events"),
    [
        # Damage inside a JSON string value: the line still parses after the
        # bytes are replaced, so the record is returned (with the damage
        # visible in it) rather than dropped.
        (b'{"event": "damaged", "payload": {"note": "\xff\xfe"}}', ["before", "damaged", "after"]),
        # Damage that breaks the structure: the replaced line is no longer
        # valid JSON, so it lands on the malformed-line skip.
        (b'{"event": "damaged", "payload\xff', ["before", "after"]),
    ],
    ids=["damage_inside_a_string", "damage_breaks_the_json"],
)
def test_read_audit_log_keeps_records_around_undecodable_bytes(damaged_line, expected_events, caplog):
    """Bytes that are not UTF-8 cost their own line, not the whole read.

    A torn write, failing media, or forged content in a rotated copy can
    leave bytes the UTF-8 codec cannot decode. Decoding strictly raises
    ``UnicodeDecodeError`` -- a ``ValueError``, not an ``OSError`` -- which
    escapes the walker's ``except OSError`` and aborts the entire read, so
    every record in the file (and every record already collected from an
    earlier rotated copy) is lost to one bad byte. The walker replaces
    undecodable bytes instead, which confines the damage to the line that
    holds it: the surrounding records survive, and the damaged line either
    still parses or falls to the malformed-JSON skip. Either way a DEBUG
    breadcrumb reports it, so the substitution is never silent.
    """
    active = audit.audit_log_path()
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(
        b'{"event": "before", "payload": {"i": 1}}\n' + damaged_line + b'\n{"event": "after", "payload": {"i": 2}}\n'
    )

    with caplog.at_level("DEBUG", logger="strands_robots.mesh.audit"):
        records = audit.read_audit_log()

    assert [r.get("event") for r in records] == expected_events
    assert any("undecodable bytes" in m for m in caplog.messages), (
        f"expected a DEBUG breadcrumb naming the undecodable bytes, got {caplog.messages}"
    )


def test_read_audit_log_reaches_the_active_log_past_a_damaged_rotated_copy():
    """Damage in an older rotated copy must not hide the newer records.

    The walker reads rotated copies before the active log so verification
    spans the whole retained window. An abort on the first file therefore
    costs more than that file: the records already appended from it are
    discarded with the exception, and every later file is never opened. The
    active log's records must survive damage in a copy that precedes them.
    """
    active = audit.audit_log_path()
    active.parent.mkdir(parents=True, exist_ok=True)
    rotated = active.with_suffix(active.suffix + ".1")
    rotated.write_bytes(b'{"event": "rotated", "payload": {}}\n\xff\xfe not utf-8\n')
    active.write_bytes(b'{"event": "active", "payload": {}}\n')

    events = [r.get("event") for r in audit.read_audit_log()]
    assert events == ["rotated", "active"], (
        f"damage in a rotated copy must not stop the walk reaching the active log, got {events}"
    )


def test_verify_audit_integrity_reports_damaged_bytes_instead_of_raising(monkeypatch):
    """A damaged record is tamper evidence the walker reports, not a crash.

    ``verify_audit_integrity`` exists to attest the trail, so the one thing
    it must not do on a damaged log is raise: an attacker who cannot forge
    an HMAC could otherwise silence the whole report by writing a single
    undecodable byte. With a PSK configured the damaged record's bytes no
    longer match its signature, which is exactly the verdict wanted --
    counted as ``bad_signature`` with ``ok`` False, while the intact
    records around it still verify.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "unit-test-psk")
    audit._AUDIT_STATE.psk_fingerprint = None
    for index in range(3):
        audit.log_safety_event("emergency_stop", "peer-a", {"index": index})

    active = audit.audit_log_path()
    lines = active.read_bytes().splitlines()
    assert len(lines) == 3, f"expected three signed records, got {len(lines)}"
    # Corrupt one byte of the middle record in place: same length, so the
    # damage is a decode failure and nothing else.
    middle = bytearray(lines[1])
    middle[middle.index(b"emergency_stop")] = 0xFF
    active.write_bytes(b"\n".join([lines[0], bytes(middle), lines[2]]) + b"\n")

    report = audit.verify_audit_integrity()

    assert report["total"] == 3, f"every record must still be examined, got {report}"
    assert report["verified"] == 2, f"the intact records must still verify, got {report}"
    assert report["bad_signature"] == 1, f"the damaged record must be reported, got {report}"
    assert report["ok"] is False


def test_read_audit_log_leaves_an_undamaged_log_without_a_damage_breadcrumb(caplog):
    """The damage breadcrumb fires on damage only.

    Every record this module writes is ASCII, so a healthy log can never
    contain the replacement character. A breadcrumb on an intact read would
    make the signal useless for the forensic question it answers.
    """
    audit.log_safety_event("emergency_stop", "peer-a", {"index": 1})

    with caplog.at_level("DEBUG", logger="strands_robots.mesh.audit"):
        records = audit.read_audit_log()

    assert len(records) == 1
    assert not any("undecodable bytes" in m for m in caplog.messages), (
        f"an intact log must not report damage, got {caplog.messages}"
    )
