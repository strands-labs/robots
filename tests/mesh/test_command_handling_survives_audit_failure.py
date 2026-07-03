"""Audit-log unavailability must never break inbound command handling.

Every rejection and successful-execution branch of ``Mesh._exec_cmd`` records a
forensic event via :func:`strands_robots.mesh.audit.log_safety_event`. That call
is best-effort: the audit log is file-backed and can be transiently unwritable
(disk full, permissions, a symlink refusal), but a wired peer must still get its
structured response and the robot must still act. Each call site therefore wraps
``log_safety_event`` in a narrow ``except (TypeError, ValueError, OSError)`` that
degrades to a debug log instead of propagating.

These pin that fail-soft contract on all four ``_exec_cmd`` audit call sites --
non-dict envelope, validation rejection, replay rejection, and successful
non-readonly execution -- by forcing ``log_safety_event`` to raise ``OSError``
and asserting the observable wire behaviour is unchanged and no exception
escapes the handler.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh import core
from strands_robots.mesh.core import Mesh


class _FakeRobot:
    """Minimal robot adapter exposing the readonly + actuating surface used here."""

    def status(self) -> dict[str, Any]:
        return {"status": "idle"}

    def stop_task(self) -> dict[str, Any]:
        return {"ok": True}


def _mesh_with_unwritable_audit(monkeypatch: pytest.MonkeyPatch) -> tuple[Mesh, list[tuple[str, dict]]]:
    """A Mesh whose ``publish`` is captured and whose audit log always raises OSError."""
    m = Mesh(_FakeRobot(), peer_id="robot-a")
    puts: list[tuple[str, dict]] = []
    monkeypatch.setattr(m, "publish", lambda key, payload, **kw: puts.append((key, payload)))

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("audit log unavailable")

    monkeypatch.setattr(core, "log_safety_event", _boom)
    return m, puts


def _response(puts: list[tuple[str, dict]], sender: str, turn: str) -> dict[str, Any]:
    key = f"strands/{sender}/response/robot-a/{turn}"
    matches = [payload for k, payload in puts if k == key]
    assert matches, f"no response published on {key}: {puts}"
    return matches[0]


def test_non_dict_command_rejection_survives_audit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-dict command still gets a wire error even when the audit write raises."""
    m, puts = _mesh_with_unwritable_audit(monkeypatch)

    m._exec_cmd({"sender_id": "op1", "turn_id": "t1", "command": "not-a-dict"})

    err = _response(puts, "op1", "t1")
    assert err["type"] == "error"
    assert "validation" in err["error"]


def test_validation_rejection_survives_audit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown-action command still gets a wire error when the audit write raises."""
    m, puts = _mesh_with_unwritable_audit(monkeypatch)

    m._exec_cmd({"sender_id": "op1", "turn_id": "t2", "command": {"action": "bogus"}})

    err = _response(puts, "op1", "t2")
    assert err["type"] == "error"
    assert "validation" in err["error"]


def test_successful_execution_and_replay_survive_audit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid actuating command dispatches (success audit) and its replay is rejected,
    both while the audit write raises OSError -- neither path may propagate.
    """
    m, puts = _mesh_with_unwritable_audit(monkeypatch)

    # First send: dispatches to stop_task -> success-audit branch (fail-soft).
    m._exec_cmd({"sender_id": "op1", "turn_id": "t3", "command": {"action": "stop"}})
    ok = _response(puts, "op1", "t3")
    assert ok["type"] == "response"
    assert ok["result"] == {"ok": True}

    # Second send with the same (sender, turn_id): replay-rejection branch.
    m._exec_cmd({"sender_id": "op1", "turn_id": "t3", "command": {"action": "stop"}})
    replay = [p for k, p in puts if k == "strands/op1/response/robot-a/t3"]
    assert len(replay) == 2
    assert replay[1]["type"] == "error"
    assert "replay" in replay[1]["error"]
