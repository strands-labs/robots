"""The operator-response audit row every human-in-the-loop gate owes.

Three tools stop and ask a human before an agent-issued command reaches a robot
or a training run, and each owes that reply two things that pull in opposite
directions. It must not reach the model - a flat sentinel goes back instead, so
an agent that authors the approval reason cannot make the operator's typed answer
carry data into the context. And it must reach the LOCAL audit log, because every
gate accepts a canonical affirmative only: a reply that carries a reason is always
a decline, and the audit row is the one place that reason survives.

The echo half was already pinned per tool. The record half was not, and only the
mesh tool wrote it - a ``use_ros`` publish to ``/cmd_vel`` and a ``lerobot_train``
``output_dir`` override that an operator declined left no audit row, no log record,
and so no trace that a gate had fired at all. These tests grade the observable (a
row exists, carrying the reply) rather than which function writes it, and derive
the set of gates from the ``interrupt()`` call sites so a fourth gate is graded on
arrival instead of inheriting the silence.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots
import strands_robots.tools.lerobot_train as train_mod
import strands_robots.tools.robot_mesh as mesh_mod
import strands_robots.tools.use_ros as ros_mod
from strands_robots.mesh.audit import read_audit_log

# A reply that carries a reason. Every gate accepts a canonical affirmative only,
# so this is always a decline - which is exactly why the audit row is the only
# place the reason can survive.
REASONED_DECLINE = "n - not while the cell door is open"
APPROVAL = "y"

_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent


def _ctx(response: object) -> MagicMock:
    """Stand-in ToolContext whose interrupt() returns *response*."""
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


def _drive_use_ros(response: object) -> dict[str, Any] | None:
    """A publish aimed at a blocklisted drive topic."""
    return ros_mod._gate_command("publish", "/cmd_vel", _ctx(response))


def _drive_lerobot_train(response: object) -> dict[str, Any] | None:
    """A training run overriding the blocked ``output_dir`` flag."""
    return train_mod._gate_extra_flags({"output_dir": "/tmp/elsewhere"}, _ctx(response))


def _drive_robot_mesh(response: object) -> dict[str, Any] | None:
    """A fleet-wide emergency_stop through the tool's own dispatch."""
    fn = getattr(mesh_mod.robot_mesh, "__wrapped__", None) or mesh_mod.robot_mesh
    mesh = MagicMock()
    mesh.emergency_stop.return_value = [{"status": "ok"}]
    mesh_mod._reset_rate_limits()
    with (
        patch.object(mesh_mod, "_gateway_mesh", lambda: None),
        patch.object(mesh_mod, "_resolve_mesh", return_value=mesh),
    ):
        return fn(action="emergency_stop", tool_context=_ctx(response))


class _Gate:
    """One HITL gate: how to drive it, and where its interrupt lives."""

    def __init__(
        self,
        label: str,
        source: str,
        action: str,
        target: str,
        drive: Callable[[object], dict[str, Any] | None],
        module: Any,
        function: str,
    ) -> None:
        self.label = label
        self.source = source
        self.action = action
        self.target = target
        self.drive = drive
        self.module = module
        self.function = function


# The target each drive above aims at. ``emergency_stop`` is fleet-wide, so no
# single peer is named and its row's target is legitimately empty - the verb is
# what identifies it. Pinning the expected value per gate keeps that deliberate
# rather than letting an empty target pass everywhere.
_GATES: tuple[_Gate, ...] = (
    _Gate("use_ros", "use_ros_tool", "publish", "/cmd_vel", _drive_use_ros, ros_mod, "_gate_command"),
    _Gate(
        "lerobot_train",
        "lerobot_train_tool",
        "train",
        "output_dir",
        _drive_lerobot_train,
        train_mod,
        "_gate_extra_flags",
    ),
    _Gate("robot_mesh", "robot_mesh_tool", "emergency_stop", "", _drive_robot_mesh, mesh_mod, "robot_mesh"),
)

_GATE_IDS = tuple(gate.label for gate in _GATES)


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither pre-approval nor bypass, so every gate reaches the operator."""
    for name in ("BYPASS_TOOL_CONSENT", "STRANDS_ROS2_COMMAND_ALLOW", "STRANDS_TRAIN_EXTRA_FLAGS_ALLOW"):
        monkeypatch.delenv(name, raising=False)


def _operator_rows(recorded: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """The audit rows a forensic reader looking for operator verdicts would find."""
    return [
        (source, payload)
        for _event, source, payload in recorded
        if str(payload.get("detail", "")).startswith("operator ")
    ]


@pytest.fixture
def audit_rows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    """Capture what each gate hands the audit log, without touching the filesystem."""
    recorded: list[tuple[str, str, dict[str, Any]]] = []

    def _record(event_type: str, source: str, payload: dict[str, Any]) -> None:
        recorded.append((event_type, source, payload))

    monkeypatch.setattr("strands_robots.mesh.audit.log_safety_event", _record)
    return recorded


class TestEveryGateRecordsTheOperatorReply:
    """A decline and an approval each leave a row carrying the literal reply."""

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    def test_a_reasoned_decline_is_recorded_with_its_reason(
        self, gate: _Gate, audit_rows: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        result = gate.drive(REASONED_DECLINE)

        assert result is not None and result.get("status") == "error", (
            f"premise: {gate.label} did not treat a reasoned reply as a decline: {result}"
        )
        rows = _operator_rows(audit_rows)
        assert len(rows) == 1, f"{gate.label} wrote {len(rows)} operator rows for a decline, want 1: {audit_rows}"
        source, payload = rows[0]
        assert source == gate.source
        assert payload["success"] is False
        assert repr(REASONED_DECLINE) in payload["detail"], (
            f"{gate.label} recorded no reason for the decline, so nothing says why the "
            f"command did not happen: {payload['detail']!r}"
        )

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    def test_an_approval_is_recorded_too(self, gate: _Gate, audit_rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        """A human authorising an agent to reach a physical surface is the row an
        incident audit reads first, so the approved path must record as well."""
        gate.drive(APPROVAL)

        rows = _operator_rows(audit_rows)
        assert len(rows) == 1, f"{gate.label} wrote {len(rows)} operator rows for an approval, want 1: {audit_rows}"
        source, payload = rows[0]
        assert source == gate.source
        assert payload["success"] is True
        assert repr(APPROVAL) in payload["detail"]

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    def test_the_row_names_the_verb_and_the_target(
        self, gate: _Gate, audit_rows: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """A row saying only that someone declined does not say what they declined."""
        gate.drive(REASONED_DECLINE)

        _source, payload = _operator_rows(audit_rows)[0]
        assert payload["action"] == gate.action
        assert payload["target"] == gate.target

    def test_a_declined_use_ros_publish_lands_on_disk(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """End to end through the real audit writer, not a captured call."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        _drive_use_ros(REASONED_DECLINE)

        details = [str(rec.get("payload", {}).get("detail", "")) for rec in read_audit_log(since=0)]
        assert any(repr(REASONED_DECLINE) in detail for detail in details), (
            f"the declined publish left no readable audit row: {details}"
        )


class TestTheReplyStillNeverReachesTheModel:
    """Control: recording the reply must not start echoing it."""

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    def test_a_declined_reply_is_not_in_the_model_visible_text(
        self, gate: _Gate, audit_rows: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        secret = "no, and the door code is hunter2"

        result = gate.drive(secret)

        assert result is not None
        text = " ".join(block.get("text", "") for block in result["content"])
        assert "hunter2" not in text, f"{gate.label} echoed the operator's reply to the model: {text!r}"
        # ... and the reply the model never saw is the one the audit row keeps.
        assert repr(secret) in _operator_rows(audit_rows)[0][1]["detail"]


class TestTheGatedSetIsDerivedFromTheInterruptSites:
    """The graded set comes from the tree, so a fourth gate cannot inherit silence."""

    @staticmethod
    def _own_calls(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
        """Calls in *fn* itself, not in a function nested inside it."""
        nested = {
            id(node)
            for child in ast.iter_child_nodes(fn)
            for node in ast.walk(child)
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
        }
        return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and id(n) not in nested]

    @classmethod
    def _interrupt_sites(cls) -> dict[str, str]:
        """``{module dotted path: function name}`` for every ``interrupt()`` caller."""
        sites: dict[str, str] = {}
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if any(ast.unparse(call.func).endswith(".interrupt") for call in cls._own_calls(node)):
                    dotted = ".".join(path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts)
                    sites[dotted] = node.name
        return sites

    def test_the_table_covers_every_interrupt_site(self) -> None:
        found = self._interrupt_sites()
        assert found, "premise: the scan found no interrupt() call site to grade"

        covered = {gate.module.__name__: gate.function for gate in _GATES}
        assert found == covered, (
            "a human-in-the-loop gate is not graded for its operator-response audit row. "
            f"Found {found}, graded {covered}. Add it to _GATES and make it record."
        )

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    def test_each_gate_records_through_the_shared_owner(self, gate: _Gate) -> None:
        """One owner for the row, so its wording cannot differ between two gates.

        A reader greps the audit log for one phrasing; a gate that spelled the row
        itself could drift to another and become invisible to that search.
        """
        source = textwrap.dedent(inspect.getsource(getattr(gate.module, gate.function)))
        calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
        assert "log_operator_response" in calls, (
            f"{gate.label}'s gate does not record the operator's reply through the shared owner"
        )


class TestTheMeshToolRowIsUnchanged:
    """Control: the tool that already recorded keeps writing the same row."""

    def test_the_recorded_payload_is_byte_identical(self, audit_rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        _drive_robot_mesh(REASONED_DECLINE)

        rows = _operator_rows(audit_rows)
        assert rows == [
            (
                "robot_mesh_tool",
                {
                    "action": "emergency_stop",
                    "target": "",
                    "success": False,
                    "detail": f"operator declined: {REASONED_DECLINE!r}",
                },
            )
        ]


class TestAnUnwritableAuditLogDoesNotChangeTheVerdict:
    """Control: the record is owed, but a safety gate may not fail on it."""

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    @pytest.mark.parametrize("response", [REASONED_DECLINE, APPROVAL], ids=["decline", "approve"])
    def test_the_gate_reaches_the_same_verdict(
        self, gate: _Gate, response: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> None:
            raise OSError("audit log unavailable")

        monkeypatch.setattr("strands_robots.mesh.audit.log_safety_event", _boom)

        result = gate.drive(response)

        declined = response is REASONED_DECLINE
        assert (result is not None and result.get("status") == "error") is declined, (
            f"{gate.label} changed its verdict when the audit write failed: {result}"
        )
