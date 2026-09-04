"""The operator-response audit row every human-in-the-loop gate owes.

Four gates stop and ask a human before an agent-issued command reaches a robot
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
the set of gates from the ``interrupt()`` call sites so a further gate is graded
on arrival instead of inheriting the silence.

That derivation has since done its job once: the dashboard's motion gate arrived as
a fourth site and this file failed until it was graded. It is a ``BeforeToolCallEvent``
hook rather than a tool body, so it is the first gate whose interrupt lives on a
METHOD and which signals a decline by setting ``event.cancel_tool`` instead of
returning a result - see ``_Gate.owner`` and ``_drive_dashboard_agent_hitl``.
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
import strands_robots.dashboard.agent_hitl as dash_hitl_mod
import strands_robots.tools._command_gate as gate_mod
import strands_robots.tools.lerobot_train as train_mod
import strands_robots.tools.robot_mesh as mesh_mod
import strands_robots.tools.use_ros as ros_mod
from strands_robots.mesh.audit import audit_log_path, read_audit_log

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


def _drive_dashboard_agent_hitl(response: object) -> dict[str, Any] | None:
    """A fleet ``task`` aimed at a peer that reports real hardware.

    This gate is a ``BeforeToolCallEvent`` hook, not a tool body, so it says "no"
    by setting ``event.cancel_tool`` rather than by returning a result. The SDK
    turns exactly that into the result the model sees -- a truthy ``cancel_tool``
    becomes ``{"status": "error", "content": [{"text": cancel_tool}]}`` in
    ``strands.tools.executors._executor`` -- so this drive performs the SDK's own
    translation and the table's shared cells grade one shape across all four gates.

    The peer states ``hw`` rather than relying on ``peer_is_physical``'s
    fall-through, so the drive keeps reaching the operator even if the default for
    an unclassified peer ever changes.
    """
    tool_input = {"action": "task", "target": "arm-1", "instruction": "wave"}
    hook = dash_hitl_mod.MotionInterruptHook(lambda: {"arm-1": {"presence": {"hw": "so101"}}})

    event = MagicMock(name="BeforeToolCallEvent")
    event.tool_use = {"name": "fleet", "input": tool_input}
    event.interrupt.return_value = response
    event.cancel_tool = False  # the real event's default, so an approval reads as "not cancelled"

    try:
        hook._gate(event)
    finally:
        # A yes deposits a one-shot grant in process-global state; do not leak it
        # into another cell (or another file) that reads the same set.
        dash_hitl_mod.consume_grant("fleet", tool_input)

    if not event.cancel_tool:
        return None
    return {"status": "error", "content": [{"text": str(event.cancel_tool)}]}


class _Gate:
    """One HITL gate: how to drive it, and where its interrupt lives.

    ``module`` and ``function`` are the coordinates the tree scan reports, so they
    are the pair ``test_the_table_covers_every_interrupt_site`` compares against.
    ``owner`` is the object that function is reachable on, for the cells that read
    its source. The two coincide for a module-level gate and default that way; the
    dashboard's gate is a method, so its function hangs off the class while its
    module stays the coordinate the scan yields.
    """

    def __init__(
        self,
        label: str,
        source: str,
        action: str,
        target: str,
        drive: Callable[[object], dict[str, Any] | None],
        module: Any,
        function: str,
        owner: Any = None,
    ) -> None:
        self.label = label
        self.source = source
        self.action = action
        self.target = target
        self.drive = drive
        self.module = module
        self.function = function
        self.owner = module if owner is None else owner


# The module/function columns name where the interrupt is raised, which for the
# ROS 2 command gate is the owner shared by all three graph transports rather
# than any one tool - one interrupt site, one audit row, whichever tool asked.
# The target each drive above aims at. ``emergency_stop`` is fleet-wide, so no
# single peer is named and its row's target is legitimately empty - the verb is
# what identifies it. Pinning the expected value per gate keeps that deliberate
# rather than letting an empty target pass everywhere.
_GATES: tuple[_Gate, ...] = (
    _Gate("use_ros", "use_ros_tool", "publish", "/cmd_vel", _drive_use_ros, gate_mod, "gate_command"),
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
    _Gate(
        "dashboard_agent_hitl",
        "dashboard_agent_hitl",
        "task",
        "arm-1",
        _drive_dashboard_agent_hitl,
        dash_hitl_mod,
        "_gate",
        owner=dash_hitl_mod.MotionInterruptHook,
    ),
)

_GATE_IDS = tuple(gate.label for gate in _GATES)


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither pre-approval nor bypass, so every gate reaches the operator."""
    for name in (
        "BYPASS_TOOL_CONSENT",
        "STRANDS_ROS2_COMMAND_ALLOW",
        "STRANDS_TRAIN_EXTRA_FLAGS_ALLOW",
        dash_hitl_mod.MOTION_ENV,
    ):
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


@pytest.fixture(autouse=True)
def audit_dir_is_this_tests_own(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every gate driven in this file off the developer's real audit trail.

    ``audit_log_path()`` falls back to ``~/.strands_robots`` when
    ``STRANDS_MESH_AUDIT_DIR`` is unset, so a test that drives a real gate
    reaches the real writer. Measured before this fixture existed: one run of
    the rate-limit-race control appended an ``operator approved: 'y'`` row for
    action ``tell`` with ``success: true`` to the developer's own
    ``mesh_audit.jsonl`` and advanced the signed ``mesh_audit.seq.json``
    sidecar beside it. That row is indistinguishable from a genuine human
    authorisation to exactly the incident reader this file exists to serve, and
    it accumulates on every local run.

    Autouse rather than one line in the offending test, because the hole is
    structural: capturing ``log_safety_event`` is a choice each test makes, so a
    gate-driving test added later re-grows the same contamination in silence and
    nothing fails. A test that wants a directory it can read back still names one
    itself - its own ``setenv`` runs after this one and wins.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))


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
        source = textwrap.dedent(inspect.getsource(getattr(gate.owner, gate.function)))
        calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
        assert "log_operator_response" in calls, (
            f"{gate.label}'s gate does not record the operator's reply through the shared owner"
        )

    @pytest.mark.parametrize("gate", _GATES, ids=_GATE_IDS)
    def test_each_gate_records_from_one_site_no_later_refusal_can_return_past(self, gate: _Gate) -> None:
        """One site, reached as soon as the verdict is known.

        Two per-branch sites are what let a third exit skip both: the mesh tool
        recorded ``approved=False`` in its decline branch and ``approved=True``
        after its post-approval rate-limit re-check, so the raced refusal between
        them returned with the operator's verdict unrecorded. A single site the
        gate reaches before it can return again cannot be bypassed that way, and
        it is the shape the other two gates already had.
        """
        fn_src = textwrap.dedent(inspect.getsource(getattr(gate.owner, gate.function)))
        fn = next(
            node
            for node in ast.walk(ast.parse(fn_src))
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == gate.function
        )
        calls = self._own_calls(fn)

        sites = [call.lineno for call in calls if ast.unparse(call.func) == "log_operator_response"]
        assert len(sites) == 1, (
            f"{gate.label} records the operator's verdict at {len(sites)} sites {sites}; want one that every "
            "path reaches once the verdict is known, so a later refusal cannot return past all of them"
        )

        verdicts = [
            call
            for call in calls
            if "approve" in ast.unparse(call.func) and ast.unparse(call.func) != "log_operator_response"
        ]
        assert len(verdicts) == 1, (
            f"{gate.label} computes the operator verdict at {len(verdicts)} sites; the rule below reads the first"
        )

        escapes = [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Return) and verdicts[0].lineno < node.lineno < sites[0]
        ]
        assert not escapes, (
            f"{gate.label} can return at lines {escapes} between computing the operator's verdict and recording "
            "it, so that path leaves no row saying whether a human authorised the action"
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


def _drive_robot_mesh_racing_the_rate_limit(response: object) -> tuple[dict[str, Any] | None, MagicMock]:
    """A ``tell`` that a concurrent invocation out-races while the operator decides.

    The pre-interrupt rate-limit check deliberately does not consume a slot, so a
    second invocation can take the last one while the human is deciding; the
    re-check under the lock then refuses an action the operator APPROVED. Draining
    the bucket from inside ``interrupt()`` is that concurrency, made deterministic
    without threads: the slot goes while the gate is blocked on the human.
    """
    fn = getattr(mesh_mod.robot_mesh, "__wrapped__", None) or mesh_mod.robot_mesh
    mesh = MagicMock()
    mesh.tell.return_value = {"status": "ok"}
    mesh_mod._reset_rate_limits()
    mesh_mod._reset_interrupt_actions_cache()

    taken = 0

    def _interrupt(*_a: object, **_k: object) -> object:
        nonlocal taken
        while mesh_mod._rate_limit_check_and_record("tell") is None and taken < 500:
            taken += 1
        return response

    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.side_effect = _interrupt
    with (
        patch.object(mesh_mod, "_gateway_mesh", lambda: None),
        patch.object(mesh_mod, "_resolve_mesh", return_value=mesh),
    ):
        result = fn(action="tell", tool_context=ctx, target="peer-a", instruction="go")

    assert taken, "premise: no slot was taken while the operator was deciding, so nothing raced"
    return result, mesh


class TestAnApprovedActionRefusedForAnotherReasonStillRecordsTheHuman:
    """The verdict is recorded even when the gate then refuses the action itself.

    A gate can refuse an action the operator approved: the mesh tool re-checks its
    rate limit under the lock, and a concurrent invocation can take the last slot
    while the human is deciding. Recording per-branch after that check left the
    raced path with no operator row at all - the audit log carried only
    ``rate_limit_race``, which says why the action was refused and nothing about
    who authorised it, so an incident audit asking "did a human approve this?"
    found no answer for the one gate that reaches physical actuation.
    """

    def test_the_human_verdict_is_recorded_even_though_the_action_was_refused(
        self, audit_rows: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        result, mesh = _drive_robot_mesh_racing_the_rate_limit(APPROVAL)

        assert result is not None and result["status"] == "error", f"premise: the race did not refuse: {result}"
        assert "rate limit" in result["content"][0]["text"].lower(), (
            f"premise: the refusal was not the rate-limit race: {result['content'][0]['text']}"
        )
        mesh.tell.assert_not_called()

        rows = _operator_rows(audit_rows)
        assert len(rows) == 1, (
            "a human approved a physical actuation and the audit log has no row saying so; "
            f"rows a forensic reader would find: {audit_rows}"
        )
        source, payload = rows[0]
        assert source == "robot_mesh_tool"
        assert payload["success"] is True, f"the operator's approval was recorded as a non-approval: {payload}"
        assert repr(APPROVAL) in str(payload["detail"]), f"the recorded row does not carry the reply: {payload}"

    def test_the_reason_the_action_was_refused_is_recorded_alongside_it(
        self, audit_rows: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Two rows, two facts: a human authorised it, and it was refused anyway."""
        _drive_robot_mesh_racing_the_rate_limit(APPROVAL)

        raced = [p for _e, _s, p in audit_rows if "rate_limit_race" in str(p.get("detail", ""))]
        assert len(raced) == 1, f"the refusal reason is not recorded: {audit_rows}"
        assert raced[0]["success"] is False

    def test_the_model_is_told_the_rate_limit_refused_it_not_the_operator(self) -> None:
        """Control: the operator approved, so the text must not blame a human."""
        result, _mesh = _drive_robot_mesh_racing_the_rate_limit(APPROVAL)

        assert result is not None
        text = result["content"][0]["text"].lower()
        assert "declined by the operator" not in text, (
            f"an approved action was reported to the model as an operator decline: {text}"
        )

    def test_an_unraced_approval_records_the_same_row_it_always_did(
        self, audit_rows: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Control: moving the site changes nothing on the path that already worked."""
        _drive_robot_mesh(APPROVAL)

        assert _operator_rows(audit_rows) == [
            (
                "robot_mesh_tool",
                {
                    "action": "emergency_stop",
                    "target": "",
                    "success": True,
                    "detail": f"operator approved: {APPROVAL!r}",
                },
            )
        ]


class TestTheAuditTrailATestWritesIsItsOwn:
    """A gate driven by this suite must not write where an incident audit reads.

    AGENTS.md item 16 exempts ``tests/`` from whole-log attestation *because* a
    test redirects the audit directory. That exemption is only sound while the
    redirect actually holds, so it is pinned here rather than assumed.
    """

    def test_the_resolved_audit_log_is_not_the_real_home(self, tmp_path: Path) -> None:
        """The fallback in ``audit_log_path()`` is never the path under test."""
        resolved = audit_log_path()

        assert resolved.parent != Path.home() / ".strands_robots", (
            "a gate driven by this suite would write to the developer's real audit trail at "
            f"{resolved}, where a fabricated 'operator approved' row is indistinguishable "
            "from a genuine human authorisation"
        )
        assert tmp_path in resolved.parents, f"the audit log is not this test's own: {resolved}"

    def test_driving_a_real_gate_writes_only_under_this_tests_directory(self, tmp_path: Path) -> None:
        """The row the rate-limit-race control used to plant lands in tmp_path."""
        _drive_robot_mesh_racing_the_rate_limit(APPROVAL)

        details = [str(rec.get("payload", {}).get("detail", "")) for rec in read_audit_log(since=0)]
        assert any(detail.startswith("operator approved") for detail in details), (
            f"premise: driving the gate left no operator row to locate: {details}"
        )
        assert tmp_path in audit_log_path().parents, "the operator row was written outside this test"
