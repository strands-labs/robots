"""``robot_mesh(action="stop")`` over the built-in mesh reads the envelope ``Mesh.send`` returns.

``Mesh.send`` answers a single-target stop with one of three shapes: the peer's
``{"type": "response", "result": ...}``, the peer's own ``{"type": "error",
...}`` (a lockout, a replay or an authorization rejection, carrying no
``result`` at all), or ``send``'s own verdict - ``{"status": "timeout"}`` when
nothing answered inside the budget, or ``{"status": "error", ...}`` for a
precondition refused before publishing. The branch used to read none of them:
only a *raised* ``send`` reached the error path, so every one of those
envelopes came back as ``status="success"`` with an ``ok=True`` audit row,
while the receiving peer's own audit had already recorded ``command_refused``
for the same turn.

Hardware-free: the local mesh is a ``MagicMock`` behind ``get_local_robots``,
the audit row is captured rather than muted so its verdict is graded too, and
the CRITICAL record is read off ``caplog``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rm
from strands_robots.tools.robot_mesh import robot_mesh

TARGET = "so100-lab-1"

REFUSED_OK_FALSE = {"type": "response", "result": {"ok": False, "error": "no running task"}}
REFUSED_TOOL_ENVELOPE = {
    "type": "response",
    "result": {"status": "error", "content": [{"text": "stop_policy: no rollout claim for so100"}]},
}
REJECTED_BY_PEER = {"type": "error", "error": "rejected during lockout", "responder_id": TARGET}
TIMED_OUT = {"status": "timeout"}
SEND_REFUSED = {"status": "error", "error": "mesh not running"}
HALTED = {"type": "response", "result": {"ok": True, "stopped": True}}


@pytest.fixture
def stop_over_mesh(monkeypatch):
    """Drive ``robot_mesh(action="stop")`` at a scripted local mesh; return (call, audit rows)."""
    mesh = MagicMock(name="LocalMesh")
    mesh.peer_id = "local-a"
    mesh.peer_type = "sim"
    mesh.inbox = {}
    rows: list[tuple[Any, ...]] = []
    monkeypatch.setattr(rm, "_audit_tool_action", lambda *a, **k: rows.append(a))
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    fn = getattr(robot_mesh, "original", robot_mesh)

    def call(envelope: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
        mesh.send.return_value = envelope
        with (
            patch("strands_robots.mesh.get_local_robots", return_value={"local-a": mesh}),
            patch("strands_robots.mesh.session.get_peers", return_value=[]),
        ):
            return fn(tool_context=ctx, action="stop", target=TARGET, timeout=timeout)

    return call, rows


@pytest.mark.parametrize(
    ("envelope", "names"),
    [
        pytest.param(REFUSED_OK_FALSE, "no running task", id="result-ok-false"),
        pytest.param(REFUSED_TOOL_ENVELOPE, "no rollout claim", id="result-tool-envelope-error"),
        pytest.param(REJECTED_BY_PEER, "rejected during lockout", id="peer-type-error"),
        pytest.param(TIMED_OUT, "no answer within 5s", id="send-timeout"),
        pytest.param(SEND_REFUSED, "mesh not running", id="send-precondition-error"),
    ],
)
def test_an_envelope_that_does_not_confirm_the_stop_is_an_error_naming_the_answer(stop_over_mesh, envelope, names):
    call, rows = stop_over_mesh

    out = call(envelope)

    text = out["content"][0]["text"]
    assert out["status"] == "error", text
    assert TARGET in text
    assert "did NOT stop" in text
    assert names in text


@pytest.mark.parametrize(
    "envelope",
    [REFUSED_OK_FALSE, REFUSED_TOOL_ENVELOPE, REJECTED_BY_PEER, TIMED_OUT, SEND_REFUSED],
    ids=["result-ok-false", "result-tool-envelope-error", "peer-type-error", "send-timeout", "send-precondition"],
)
def test_the_audit_row_records_the_unconfirmed_stop_as_a_failure(stop_over_mesh, envelope):
    call, rows = stop_over_mesh

    call(envelope)

    assert len(rows) == 1, rows
    action, target, ok, detail = rows[0]
    assert (action, target, ok) == ("stop", TARGET, False)
    assert "did not stop" in detail


def test_a_refusal_is_logged_once_at_critical_naming_the_target_and_the_answer(stop_over_mesh, caplog):
    call, _rows = stop_over_mesh

    with caplog.at_level(logging.CRITICAL, logger=rm.logger.name):
        call(REFUSED_OK_FALSE)

    records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    message = records[0].getMessage()
    assert TARGET in message
    assert "no running task" in message


def test_the_timeout_names_the_capped_budget_the_send_was_given(stop_over_mesh):
    call, _rows = stop_over_mesh

    out = call(TIMED_OUT, timeout=2.5)

    assert "no answer within 2.5s" in out["content"][0]["text"]


def test_a_peer_that_reports_it_halted_is_still_a_success(stop_over_mesh):
    call, rows = stop_over_mesh

    out = call(HALTED)

    assert out["status"] == "success", out
    assert rows == [("stop", TARGET, True, "")]


def test_a_response_carrying_no_verdict_either_way_is_not_read_as_a_refusal(stop_over_mesh):
    """The conservative reading the shared rule documents: silence is not a failure report."""
    call, rows = stop_over_mesh

    out = call({"type": "response", "result": {"stopped": True}})

    assert out["status"] == "success", out
    assert rows[0][2] is True
