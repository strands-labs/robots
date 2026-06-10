"""Pin tests for audit sequence-counter recovery from non-dict sidecar payloads.

Reviewed concern (R3 thread on ``strands_robots/mesh/audit.py:488``):
``sidecar_loaded = True`` was set unconditionally inside the
``elif sidecar.exists():`` branch, including when the parsed JSON payload
was ``null``, a list, a string, or a number. The ``isinstance(payload,
dict)`` check above gated the for-loop body but did NOT gate the flag.

Attack: an attacker with audit-dir write access drops
``mesh_audit.seq.json`` containing the bytes ``null``. ``json.load``
succeeds, ``isinstance(payload, dict)`` is False, the loop body is
skipped, ``sidecar_loaded = True`` still fires, the audit-log seed
fallback is bypassed, ``_SEQ_COUNTERS`` ends up empty, and the next
``_next_seq`` returns 1 -- exactly the fail-open the rest of
``_load_seq_counters`` was designed to defend against.

Pre-fix behaviour: ``null``/``[]``/``"oops"`` payloads silently bypass
the audit-log seed fallback and return ``_next_seq=1`` despite the audit
log carrying records up to ``seq=5`` for the same peer.

Post-fix behaviour: any non-dict payload falls through to the audit-log
seed fallback (existing ``test_audit_seq_recovery.py`` already covers the
JSON-decode-error fallback path) and ``_next_seq`` returns ``max(seq)+1``.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _seed_audit_log_with_seq(peer_id: str, max_seq: int) -> None:
    """Write ``max_seq`` records for ``peer_id`` into the audit log."""
    log_path = audit.audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        for seq in range(1, max_seq + 1):
            record = {
                "ts": 1000.0 + seq,
                "peer_id": peer_id,
                "seq": seq,
                "event": "test_event",
            }
            fh.write(json.dumps(record) + "\n")


@pytest.mark.parametrize(
    "non_dict_payload",
    [
        pytest.param("null", id="json-null"),
        pytest.param("[]", id="json-empty-list"),
        pytest.param("[1, 2, 3]", id="json-non-empty-list"),
        pytest.param('"oops"', id="json-string"),
        pytest.param("42", id="json-number"),
        pytest.param("true", id="json-bool"),
    ],
)
def test_non_dict_sidecar_payload_falls_through_to_audit_log_seed(
    isolated_audit_dir: Path, non_dict_payload: str
) -> None:
    """Each non-dict payload must NOT short-circuit the audit-log fallback.

    Pre-fix: ``sidecar_loaded = True`` ran unconditionally; the audit-log
    seed was skipped, ``_SEQ_COUNTERS`` was empty, ``_next_seq`` returned 1.

    Post-fix: ``sidecar_loaded`` only flips inside the ``isinstance(payload,
    dict)`` branch; non-dict payloads fall through, the audit-log walk
    seeds ``_SEQ_COUNTERS["test-peer"] = 5``, and ``_next_seq`` returns 6.
    """
    peer_id = "test-peer-non-dict"
    _seed_audit_log_with_seq(peer_id, max_seq=5)

    sidecar_path = audit._seq_sidecar_path()
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(non_dict_payload, encoding="utf-8")

    audit._load_seq_counters()

    assert audit._SEQ_COUNTERS.get(peer_id) == 5, (
        f"Non-dict sidecar payload {non_dict_payload!r} should fall through to "
        f"audit-log seed (max(seq)=5 for peer_id={peer_id!r}), but counter is "
        f"{audit._SEQ_COUNTERS.get(peer_id)!r}. Pre-fix, this test exposed the "
        "silent-mask: sidecar_loaded=True ran unconditionally and the fallback "
        "walk was skipped."
    )

    next_seq = audit._next_seq(peer_id)
    assert next_seq == 6, (
        f"_next_seq({peer_id!r}) must return 6 after the audit-log fallback "
        f"seeds the counter from max(seq)=5; got {next_seq}. Pre-fix this "
        "returned 1 because the non-dict sidecar was treated as a successful "
        "(zero-counter) load."
    )


def test_dict_sidecar_payload_still_short_circuits_audit_log_walk(
    isolated_audit_dir: Path,
) -> None:
    """Symmetric pin: a healthy dict sidecar must NOT trigger the walk.

    Without this assertion, fix 1 could over-correct by always falling
    through to the audit-log walk -- which would defeat the perf point of
    fix 2 (the cheap sidecar path is the common path).
    """
    peer_id = "test-peer-dict"
    _seed_audit_log_with_seq(peer_id, max_seq=5)

    sidecar_path = audit._seq_sidecar_path()
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps({peer_id: 7}), encoding="utf-8")

    audit._load_seq_counters()

    # Sidecar value 7 trumps the audit-log max of 5 (the sidecar is the
    # authoritative source for the in-memory floor on the healthy path).
    assert audit._SEQ_COUNTERS.get(peer_id) == 7
    assert audit._next_seq(peer_id) == 8
