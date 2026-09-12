"""A single-target mesh ``stop`` is graded by the peer's answer, not by delivery.

``robot_mesh(action="stop", target=...)`` over the built-in mesh path hands the
command to :meth:`~strands_robots.mesh.core.Mesh.send`, which answers with the
peer's first response envelope, with ``{"status": "timeout"}``, or with one of
its own precondition errors. Only a *raised* ``send`` is a dispatch error, so
every one of those otherwise reported ``status="success"`` and audited
``ok=True`` -- the same shape #3560 and #3566 corrected on the two Device
Connect stop branches, on the path every non-Device-Connect deployment takes.

The conservative direction is pinned too: an answer that reports the fact
neither way is not read as a failure.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.robot_mesh import robot_mesh

TARGET = "peer-b"

# One row per answer ``Mesh.send`` can hand this branch, with the reason the
# stop is not confirmed. The last two rows are answers that DO confirm a halt.
NOT_CONFIRMED: list[tuple[str, dict[str, Any], str]] = [
    (
        "handler answered ok=false",
        {"type": "response", "responder_id": TARGET, "result": {"ok": False, "error": "no running task"}},
        "did NOT stop",
    ),
    (
        "handler answered the tool envelope status=error",
        {"type": "response", "responder_id": TARGET, "result": {"status": "error", "content": [{"text": "no"}]}},
        "did NOT stop",
    ),
    (
        # The rejection response has no ``result`` at all, so the shared rule
        # finds neither key. Replay is one of the sources reachable for a stop;
        # the estop lockout is not, since ``_dispatch`` admits stop while locked.
        "mesh-level rejection carries no result at all",
        {"type": "error", "responder_id": TARGET, "error": "duplicate command rejected (replay)"},
        "rejected the stop",
    ),
    ("no answer inside the budget", {"status": "timeout"}, "UNCONFIRMED"),
    ("send refused its own precondition", {"status": "error", "error": "mesh not running"}, "was not sent"),
]

CONFIRMED: list[tuple[str, dict[str, Any]]] = [
    ("the peer answered that it stopped", {"type": "response", "responder_id": TARGET, "result": {"ok": True}}),
    # The shape ``test_stop_sends_stop_action`` sends: neither key, so it
    # reports the fact neither way and stays a success by design.
    ("an answer that is not a graded envelope", {"stopped": True}),
]


@pytest.fixture
def audit_rows(monkeypatch):
    """Capture, rather than mute, every ``(action, target, ok, detail)`` audited."""
    rows: list[tuple[str, str, bool, str]] = []
    monkeypatch.setattr(
        "strands_robots.tools.robot_mesh._audit_tool_action",
        lambda action, target, success, detail: rows.append((action, target, success, detail)),
    )
    return rows


def _stop(answer: Any) -> dict[str, Any]:
    """Call the tool's ``stop`` branch with a mesh whose ``send`` returns *answer*."""
    fake = MagicMock(name="LocalMesh")
    fake.peer_id, fake.peer_type, fake.inbox = "local-a", "sim", {}
    fake.send.return_value = answer
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    fn = getattr(robot_mesh, "original", robot_mesh)
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={"local-a": fake}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        return fn(tool_context=ctx, action="stop", target=TARGET)


@pytest.mark.parametrize(("label", "answer", "reason"), NOT_CONFIRMED, ids=[r[0] for r in NOT_CONFIRMED])
def test_an_unconfirmed_halt_is_reported_as_an_error_and_audited_false(label, answer, reason, audit_rows, caplog):
    with caplog.at_level(logging.CRITICAL, logger="strands_robots.tools.robot_mesh"):
        out = _stop(answer)

    text = out["content"][0]["text"]
    assert out["status"] == "error", label
    assert TARGET in text and reason in text
    # The answer itself is carried through so the reader sees what the peer said.
    assert str(next(iter(answer.values()))) in text or "answer:" in text

    assert audit_rows == [("stop", TARGET, False, audit_rows[0][3])]
    assert reason in audit_rows[0][3]

    critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert len(critical) == 1
    assert TARGET in critical[0].getMessage()


@pytest.mark.parametrize(("label", "answer"), CONFIRMED, ids=[r[0] for r in CONFIRMED])
def test_a_confirmed_halt_stays_a_success(label, answer, audit_rows, caplog):
    with caplog.at_level(logging.CRITICAL, logger="strands_robots.tools.robot_mesh"):
        out = _stop(answer)

    assert out["status"] == "success", label
    assert audit_rows == [("stop", TARGET, True, "")]
    assert not [r for r in caplog.records if r.levelno >= logging.CRITICAL]


def test_a_raised_send_is_still_a_dispatch_error():
    """The pre-existing contract: only a raised ``send`` reads as a dispatch error."""
    fake = MagicMock(name="LocalMesh")
    fake.peer_id, fake.peer_type, fake.inbox = "local-a", "sim", {}
    fake.send.side_effect = RuntimeError("link reset")
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    fn = getattr(robot_mesh, "original", robot_mesh)
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={"local-a": fake}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        out = fn(tool_context=ctx, action="stop", target=TARGET)

    assert out["status"] == "error"
    assert "dispatch error" in out["content"][0]["text"]
