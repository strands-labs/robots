"""Regression tests for the use_ros safety-critical command blocklist + HIL gate.

The gate exists so a prompt-injected agent cannot drive a robot through
``use_ros``. A robot is reachable through three different verbs - a topic
publish, a service call and an action goal - so these tests assert the gate from
the tool's public dispatch as well as from the helper, because a helper that
refuses correctly is worthless if a verb never consults it.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.use_ros as ros_mod
from strands_robots.tools.use_ros import (
    _approve_response,
    _canonical_command_name,
    _gate_command,
    _is_command_blocked,
    use_ros,
)

# The verbs that carry a command to a robot, with the parameter naming the
# surface and the module-level helper each one reaches once the gate allows it.
_COMMAND_VERBS: tuple[tuple[str, str, str], ...] = (
    ("publish", "topic", "_publish"),
    ("service_call", "service", "_service_call"),
    ("action_send_goal", "action_name", "_action_send_goal"),
)

_TYPE_FOR_VERB = {
    "publish": "geometry_msgs/msg/Twist",
    "service_call": "std_srvs/srv/Trigger",
    "action_send_goal": "nav2_msgs/action/NavigateToPose",
}


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


class TestCommandBlocklist:
    """Pin the blocklist contract: safety-critical surfaces blocked, others pass."""

    @pytest.mark.parametrize(
        "name",
        [
            "/cmd_vel",
            "/cmd_vel_unstamped",
            "/joint_command",
            "/joint_trajectory",
            "/emergency_stop",
            "/e_stop",
            "/motor_enable",
            "/enable_motor",
            "/disable_motor",
            "/navigate_to_pose",
            "/follow_path",
        ],
    )
    def test_safety_critical_surfaces_blocked(self, name: str) -> None:
        err = _is_command_blocked("publish", name)
        assert err is not None
        assert "blocked" in err

    @pytest.mark.parametrize(
        "name",
        [
            "/my_robot/cmd_vel",
            "/ns1/ns2/cmd_vel",
            "/robot_arm/joint_command",
            "/fleet/robot1/emergency_stop",
        ],
    )
    def test_namespaced_surfaces_blocked(self, name: str) -> None:
        """Namespace-prefixed forms of blocked surfaces must also be caught."""
        assert _is_command_blocked("publish", name) is not None

    @pytest.mark.parametrize(
        "name",
        ["/my_custom_topic", "/robot/status", "/diagnostics", "/tf", "/rosout"],
    )
    def test_non_blocked_surfaces_pass(self, name: str) -> None:
        assert _is_command_blocked("publish", name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "/cmd_vel_evil",
            "/my_robot/cmd_vel_evil",
            "/not_cmd_vel",
            "/foo/notcmd_vel",
            "/joint_trajectory_status",
            "/emergency_stop_status",
        ],
    )
    def test_substring_does_not_match(self, name: str) -> None:
        """Blocklist must be exact final-segment match, not substring."""
        assert _is_command_blocked("publish", name) is None

    def test_multi_segment_blocklist_entry(self) -> None:
        """/joint_trajectory_controller/joint_trajectory is in the default list."""
        assert _is_command_blocked("publish", "/joint_trajectory_controller/joint_trajectory") is not None

    def test_message_names_the_verb_that_was_attempted(self) -> None:
        """The operator needs to know which verb reached the surface."""
        assert "service_call" in (_is_command_blocked("service_call", "/emergency_stop") or "")


class TestCanonicalNameSpellings:
    """Spellings rclpy resolves to one surface must resolve to one verdict.

    ``cmd_vel`` and ``/cmd_vel/`` reach the same robot as ``/cmd_vel`` once rclpy
    has resolved them, so comparing the caller's literal spelling against the
    blocklist lets a caller pick a spelling that misses it.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "cmd_vel",
            "cmd_vel/",
            "/cmd_vel/",
            "/cmd_vel//",
            "robot/cmd_vel",
            "~/cmd_vel",
            "emergency_stop",
            "navigate_to_pose",
        ],
    )
    def test_unrooted_and_trailing_separator_spellings_are_blocked(self, name: str) -> None:
        assert _is_command_blocked("publish", name) is not None, f"{name!r} reached the robot ungated"

    @pytest.mark.parametrize("name", ["/CMD_VEL", "/Cmd_Vel", "/EMERGENCY_STOP"])
    def test_case_is_preserved_because_graph_names_are_case_sensitive(self, name: str) -> None:
        """A differently-cased name is a different ROS 2 topic, not a bypass.

        No ``/cmd_vel`` subscriber receives ``/CMD_VEL``, so case-folding would
        refuse a legitimate surface without closing a path to the robot.
        """
        assert _is_command_blocked("publish", name) is None

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("cmd_vel", "/cmd_vel"),
            ("/cmd_vel/", "/cmd_vel"),
            ("/cmd_vel", "/cmd_vel"),
            ("robot/cmd_vel", "/robot/cmd_vel"),
            ("/", "/"),
        ],
    )
    def test_canonical_form(self, spelling: str, expected: str) -> None:
        assert _canonical_command_name(spelling) == expected

    def test_allowlist_accepts_an_unrooted_spelling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator allowlisting ``cmd_vel`` means the same surface as ``/cmd_vel``."""
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", "cmd_vel")
        assert _gate_command("publish", "/cmd_vel", None) is None


class TestGateCommand:
    """Pin the HIL gate contract: allowlist, bypass, interrupt, decline."""

    @pytest.fixture(autouse=True)
    def _hermetic_gate_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neutralize ambient env that short-circuits the gate.

        Both BYPASS_TOOL_CONSENT and STRANDS_ROS2_COMMAND_ALLOW cause the gate to
        allow blocked surfaces without prompting. A developer or CI shell that
        exports BYPASS_TOOL_CONSENT=true (common in agent/automation contexts)
        would otherwise make the no-context, allowlist and interrupt cases pass
        silently and fail their assertions. Clearing both per-test makes each case
        deterministic regardless of the ambient environment; tests that exercise
        those paths opt in explicitly via monkeypatch.setenv.
        """
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)

    @pytest.mark.parametrize("kind", ["publish", "service_call", "action_send_goal"])
    def test_non_blocked_surface_passes(self, kind: str) -> None:
        assert _gate_command(kind, "/my_topic", None) is None

    @pytest.mark.parametrize("kind", ["publish", "service_call", "action_send_goal"])
    def test_blocked_surface_no_context_returns_error(self, kind: str) -> None:
        result = _gate_command(kind, "/cmd_vel", None)
        assert result is not None
        assert result["status"] == "error"
        assert "approval" in _texts(result).lower()

    def test_no_context_error_names_the_env_vars_that_lift_it(self) -> None:
        text = _texts(_gate_command("publish", "/cmd_vel", None) or {"content": []})
        assert "STRANDS_ROS2_COMMAND_ALLOW" in text
        assert "BYPASS_TOOL_CONSENT" in text

    def test_allowlist_skips_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", "/cmd_vel")
        assert _gate_command("publish", "/cmd_vel", None) is None

    def test_allowlist_namespaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", "/cmd_vel")
        assert _gate_command("publish", "/my_robot/cmd_vel", None) is None

    def test_allowlist_does_not_cover_other_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", "/cmd_vel")
        result = _gate_command("publish", "/emergency_stop", None)
        assert result is not None
        assert result["status"] == "error"

    def test_bypass_consent_allows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        assert _gate_command("publish", "/cmd_vel", None) is None

    @pytest.mark.parametrize("kind", ["publish", "service_call", "action_send_goal"])
    def test_interrupt_approved(self, kind: str) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        assert _gate_command(kind, "/cmd_vel", ctx) is None
        ctx.interrupt.assert_called_once()
        reason = ctx.interrupt.call_args[1]["reason"]
        assert reason["action"] == kind
        assert reason["target"] == "/cmd_vel"

    def test_interrupt_declined(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "no"
        result = _gate_command("publish", "/cmd_vel", ctx)
        assert result is not None
        assert result["status"] == "error"
        assert "declined" in _texts(result)

    def test_interrupt_runtime_error_fails_closed(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.side_effect = RuntimeError("no agent loop")
        result = _gate_command("publish", "/cmd_vel", ctx)
        assert result is not None
        assert result["status"] == "error"

    def test_operator_reply_is_never_echoed_back(self) -> None:
        """The refusal must not carry the operator's free-text reply."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "no, and my token is hunter2"
        result = _gate_command("publish", "/cmd_vel", ctx)
        assert result is not None
        assert "hunter2" not in _texts(result)

    @pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", "approve", "Approved"])
    def test_approve_response_affirmative(self, response: str) -> None:
        assert _approve_response(response) is True

    @pytest.mark.parametrize("response", ["n", "no", "nope", "", 42, None])
    def test_approve_response_negative(self, response: object) -> None:
        assert _approve_response(response) is False


class TestEveryCommandVerbConsultsTheGate:
    """Drive the public tool: every verb that commands a robot must be gated.

    The blocklist names ROS 2 actions (``/navigate_to_pose``, ``/follow_path``)
    and surfaces that ship as services (``/emergency_stop``, ``/motor_enable``),
    so a gate wired only into ``publish`` leaves those entries unenforceable -
    an agent asked to drive somewhere reaches for ``action_send_goal``.

    The rclpy transport is substituted (the same boundary
    ``tests/tools/test_use_ros.py`` doubles) so the assertions run without a
    sourced ROS 2 distro and no message can reach a real DDS graph. The gate and
    the dispatch wiring under test run unmodified.
    """

    calls: dict[str, list[tuple[Any, ...]]]

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
        monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
        self.calls = {name: [] for _, _, name in _COMMAND_VERBS}

        def _recorder(key: str, outcome: Any) -> Callable[..., Any]:
            def _fake(*args: Any) -> Any:
                self.calls[key].append(args)
                return outcome

            return _fake

        monkeypatch.setattr(ros_mod, "_publish", _recorder("_publish", None))
        monkeypatch.setattr(ros_mod, "_service_call", _recorder("_service_call", {"ok": True}))
        monkeypatch.setattr(ros_mod, "_action_send_goal", _recorder("_action_send_goal", {"goal_status": "SUCCEEDED"}))

    def _invoke(self, verb: str, param: str, name: str, ctx: MagicMock) -> dict[str, Any]:
        kwargs: dict[str, Any] = {param: name, "type": _TYPE_FOR_VERB[verb]}
        return use_ros(action=verb, tool_context=ctx, **kwargs)

    @pytest.mark.parametrize(("verb", "param", "transport"), _COMMAND_VERBS)
    @pytest.mark.parametrize("name", ["/cmd_vel", "/emergency_stop", "/navigate_to_pose"])
    def test_a_declined_command_never_reaches_the_transport(
        self, verb: str, param: str, transport: str, name: str
    ) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = self._invoke(verb, param, name, ctx)
        assert ctx.interrupt.called, f"{verb} to {name} was never gated"
        assert result["status"] == "error"
        assert "declined" in _texts(result)
        assert self.calls[transport] == [], f"{verb} reached rclpy despite a declined approval"

    @pytest.mark.parametrize(("verb", "param", "transport"), _COMMAND_VERBS)
    def test_an_approved_command_reaches_the_transport(self, verb: str, param: str, transport: str) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = self._invoke(verb, param, "/cmd_vel", ctx)
        assert result["status"] == "success"
        assert len(self.calls[transport]) == 1

    @pytest.mark.parametrize(("verb", "param", "transport"), _COMMAND_VERBS)
    def test_an_unrooted_spelling_is_gated_too(self, verb: str, param: str, transport: str) -> None:
        """`cmd_vel` resolves to `/cmd_vel`, so it must not slip past the gate."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = self._invoke(verb, param, "cmd_vel", ctx)
        assert ctx.interrupt.called, f"{verb} to 'cmd_vel' was never gated"
        assert result["status"] == "error"
        assert self.calls[transport] == []

    @pytest.mark.parametrize(("verb", "param", "transport"), _COMMAND_VERBS)
    def test_a_non_blocked_surface_is_not_gated(self, verb: str, param: str, transport: str) -> None:
        ctx = MagicMock()
        result = self._invoke(verb, param, "/my_custom_topic", ctx)
        assert not ctx.interrupt.called
        assert result["status"] == "success"
        assert len(self.calls[transport]) == 1

    def test_reading_a_blocked_surface_is_never_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Telemetry must stay readable - over-blocking the read path is a defect."""
        monkeypatch.setattr(ros_mod, "_echo", lambda *a: [{"linear": {"x": 0.0}}])
        ctx = MagicMock()
        result = use_ros(action="echo", tool_context=ctx, topic="/cmd_vel", type="geometry_msgs/msg/Twist")
        assert not ctx.interrupt.called
        assert result["status"] == "success"

    def test_inspecting_a_blocked_surface_is_never_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ros_mod, "_info", lambda target: f"topic info {target}")
        ctx = MagicMock()
        result = use_ros(action="info", tool_context=ctx, topic="/cmd_vel")
        assert not ctx.interrupt.called
        assert result["status"] == "success"


class TestGateRunsAfterArgumentValidation:
    """An operator must never be asked to approve a command that cannot run.

    The gate fires on the surface name alone, so gating before the required
    arguments are checked prompts the operator and then fails with "requires
    ... and type" whatever they answer - approval spent on a no-op trains the
    operator to approve without reading.
    """

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
        monkeypatch.setattr(ros_mod._backend, "available", lambda: True)

    @pytest.mark.parametrize(
        ("verb", "param", "name"),
        [
            ("publish", "topic", "/cmd_vel"),
            ("service_call", "service", "/emergency_stop"),
            ("action_send_goal", "action_name", "/navigate_to_pose"),
        ],
    )
    def test_a_missing_type_is_reported_without_asking_the_operator(self, verb: str, param: str, name: str) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        kwargs: dict[str, Any] = {param: name}
        result = use_ros(action=verb, tool_context=ctx, **kwargs)
        assert result["status"] == "error"
        assert "requires" in _texts(result)
        assert not ctx.interrupt.called, "operator was asked to approve a command that cannot run"


def test_every_command_verb_branch_calls_the_shared_gate() -> None:
    """Structural guard: a verb added later must not ship without the gate.

    Reads the dispatch source rather than a behaviour, so a new command verb
    wired straight to rclpy is reported even if no test drives it yet.
    """
    module = ast.parse(Path(ros_mod.__file__).read_text(encoding="utf-8"))
    dispatch = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "use_ros")
    gated: set[str] = set()
    for node in ast.walk(dispatch):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "action"
            and isinstance(test.comparators[0], ast.Constant)
        ):
            continue
        verb = test.comparators[0].value
        if not isinstance(verb, str):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "_gate_command":
                gated.add(verb)
    assert gated == {verb for verb, _, _ in _COMMAND_VERBS}, (
        f"command verbs consulting _gate_command: {sorted(gated)}; "
        f"expected {sorted(verb for verb, _, _ in _COMMAND_VERBS)}"
    )


def test_the_blocklist_is_documented_where_operators_look() -> None:
    """Every blocklisted surface and both env vars appear in the ROS 2 docs."""
    docs = Path(__file__).resolve().parents[1] / "docs" / "ros2-integration.md"
    text = docs.read_text(encoding="utf-8")
    for entry in ros_mod._DEFAULT_COMMAND_BLOCKLIST:
        assert entry in text, f"{entry} is blocked but undocumented"
    assert "STRANDS_ROS2_COMMAND_ALLOW" in text


# Files an operator reads to decide what to pre-approve before a headless run.
# The README Configuration table is the single source of truth for env vars, so a
# wrong contract there is the one that gets scaffolded from.
_OPERATOR_FACING_DOCS: tuple[str, ...] = (
    "README.md",
    "docs/ros2-integration.md",
    "docs/security.md",
)

# A clause that names the halt and denies that it is gated claims an exemption.
_HALT_PHRASES: tuple[str, ...] = ("zero-velocity", "zero velocity", "zero `twist`", "halt", "stop()")
_NOT_GATED_PHRASES: tuple[str, ...] = (
    "never gated",
    "not gated",
    "ungated",
    "un-gated",
    "never prompt",
    "no approval",
    "without approval",
    "never asks",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _readme_allow_row() -> str:
    """Return the README Configuration row documenting the pre-approval variable."""
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    for line in readme.splitlines():
        if line.startswith("|") and f"`{ros_mod._COMMAND_ALLOW_ENV}`" in line:
            return line
    return ""


# Clauses asserting a pre-approval covers only the surfaces the operator listed.
# An allowlist is the one direction where the gate's namespace-stripped matching
# is fail-OPEN, so a document that claims exclusivity understates the blast radius.
_EXCLUSIVE_REACH_PHRASES: tuple[str, ...] = (
    "everything else stays gated",
    "everything else on the safety-critical blocklist stays gated",
    "individual surfaces",
    "those surfaces",
    "only the surfaces",
)

# Prose stating the reach the gate actually has: an entry also pre-approves every
# namespaced surface sharing its base name.
_BASE_NAME_REACH_PHRASES: tuple[str, ...] = (
    "same base name",
    "namespaced",
    "any namespace",
    "every namespace",
)


def _surfaces_documenting_the_allowlist() -> list[tuple[str, str]]:
    """Every surface an operator or maintainer reads to size a pre-approval.

    The markdown files contribute the lines naming the variable; the gate helper
    contributes its own docstring, which enumerates the four ways through the
    gate and is what a maintainer reads before changing one.

    Returns:
        ``(surface_label, text)`` pairs, text lowercased for phrase matching.
    """
    surfaces: list[tuple[str, str]] = []
    for name in _OPERATOR_FACING_DOCS:
        lines = [
            line
            for line in (_repo_root() / name).read_text(encoding="utf-8").splitlines()
            if ros_mod._COMMAND_ALLOW_ENV in line
        ]
        if lines:
            surfaces.append((name, "\n".join(lines).lower()))
    gate_doc = inspect.getdoc(_gate_command) or ""
    if ros_mod._COMMAND_ALLOW_ENV in gate_doc:
        surfaces.append(("_gate_command docstring", gate_doc.lower()))
    return surfaces


def _measured_allowlist_reach() -> tuple[str, list[str]]:
    """Measure what one bare pre-approval entry actually un-gates.

    Sets the variable to a single bare blocklist entry and asks the gate about
    namespaced siblings the entry never names. Nothing is published: the gate is
    consulted with ``tool_context=None``, so a surface that is still gated comes
    back as the fail-closed error rather than reaching a transport.

    Returns:
        ``("base" | "exact", extra_surfaces)`` - the reach, and the surfaces the
        entry un-gated without naming them.
    """
    siblings = ["/robot_a/cmd_vel", "/robot_b/cmd_vel", "/mobile_base/cmd_vel"]
    previous = os.environ.get(ros_mod._COMMAND_ALLOW_ENV)
    previous_bypass = os.environ.get("BYPASS_TOOL_CONSENT")
    os.environ[ros_mod._COMMAND_ALLOW_ENV] = "/cmd_vel"
    os.environ.pop("BYPASS_TOOL_CONSENT", None)
    try:
        extra = [name for name in siblings if _gate_command("publish", name, None) is None]
    finally:
        if previous is None:
            os.environ.pop(ros_mod._COMMAND_ALLOW_ENV, None)
        else:
            os.environ[ros_mod._COMMAND_ALLOW_ENV] = previous
        if previous_bypass is not None:
            os.environ["BYPASS_TOOL_CONSENT"] = previous_bypass
    return ("base" if extra else "exact"), extra


def _documented_halt_exemptions() -> list[tuple[str, int, str]]:
    """Every operator-facing clause asserting the halt is exempt from the gate."""
    found: list[tuple[str, int, str]] = []
    for name in _OPERATOR_FACING_DOCS:
        text = (_repo_root() / name).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for clause in re.split(r"(?<=[.;])\s+|\s*\|\s*", line):
                low = clause.lower()
                if any(h in low for h in _HALT_PHRASES) and any(u in low for u in _NOT_GATED_PHRASES):
                    found.append((name, lineno, clause.strip()))
    return found


class TestTheDocumentedExemptionsAreTheRealOnes:
    """An operator-facing document may only claim an exemption the gate makes.

    ``_gate_command`` is handed the verb and the surface name and never the
    payload, so an exemption that depends on what is being sent cannot be
    implemented. An operator who believes the halt is exempt leaves ``cmd_vel``
    out of ``STRANDS_ROS2_COMMAND_ALLOW`` and discovers in the field that
    ``stop()`` is refused - the unreachable-halt hazard, reintroduced through
    documentation rather than through code.

    The documented claims are graded against the running gate rather than banned
    outright, so a gate that one day does exempt a payload makes the claim
    permissible instead of failing here.
    """

    calls: dict[str, list[tuple[Any, ...]]]

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv(ros_mod._COMMAND_ALLOW_ENV, raising=False)
        monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
        self.calls = {"_publish": [], "_action_send_goal": []}

        def _recorder(key: str, outcome: Any) -> Callable[..., Any]:
            def _fake(*args: Any) -> Any:
                self.calls[key].append(args)
                return outcome

            return _fake

        monkeypatch.setattr(ros_mod, "_publish", _recorder("_publish", None))
        monkeypatch.setattr(ros_mod, "_action_send_goal", _recorder("_action_send_goal", {"goal_status": "SUCCEEDED"}))

    @staticmethod
    def _halt_fields() -> dict[str, Any]:
        return {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}

    def _publish_twist(self, fields: dict[str, Any], ctx: MagicMock) -> dict[str, Any]:
        return use_ros(
            action="publish",
            tool_context=ctx,
            topic="/cmd_vel",
            type="geometry_msgs/msg/Twist",
            fields=fields,
        )

    def _halt_is_gated(self) -> bool:
        """Measure whether a zero-velocity halt is subject to the gate."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        self._publish_twist(self._halt_fields(), ctx)
        return bool(ctx.interrupt.called)

    def test_the_gate_never_receives_the_payload(self) -> None:
        """The reason a payload-conditional exemption cannot be documented."""
        params = tuple(inspect.signature(_gate_command).parameters)
        assert params == ("kind", "name", "tool_context"), (
            f"_gate_command takes {params}; a documented payload exemption is only "
            "implementable if the payload is passed in"
        )

    def test_a_halt_is_refused_exactly_like_a_full_speed_drive(self) -> None:
        """Same surface, same verb, same outcome - the payload changes nothing."""
        outcomes = []
        for fields in (self._halt_fields(), {"linear": {"x": 1.0}, "angular": {"z": 0.5}}):
            ctx = MagicMock()
            ctx.interrupt.return_value = "n"
            result = self._publish_twist(fields, ctx)
            outcomes.append((ctx.interrupt.called, result["status"]))
        assert outcomes == [(True, "error"), (True, "error")], (
            f"halt vs drive outcomes differ: {outcomes}; the gate is keyed on the surface"
        )
        assert self.calls["_publish"] == [], "a declined publish reached the transport"

    def test_the_documented_halt_contract_matches_the_gate(self) -> None:
        """The docs and the gate must agree on whether the halt is exempt."""
        row = _readme_allow_row()
        assert row, (
            f"no README Configuration row documents {ros_mod._COMMAND_ALLOW_ENV}; a clean sweep would prove nothing"
        )
        claims = _documented_halt_exemptions()
        measured = "exempt" if not self._halt_is_gated() else "gated"
        documented = "exempt" if claims else "gated"
        detail = "\n".join(f"  {name}:{lineno}: {clause}" for name, lineno, clause in claims) or "  (none)"
        assert measured == documented, (
            f"the gate treats a zero-velocity halt to a blocklisted surface as {measured}, "
            f"but the operator-facing documentation describes it as {documented}.\n"
            f"clauses claiming an exemption:\n{detail}"
        )

    def test_the_documented_read_exemption_is_real(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The row's other clause: reads of a blocked surface are never gated."""
        assert re.search(r"read[s]? are never gated", _readme_allow_row().lower()), (
            "the row no longer states the read exemption that does hold"
        )
        monkeypatch.setattr(ros_mod, "_echo", lambda *a: [{"linear": {"x": 0.0}}])
        ctx = MagicMock()
        result = use_ros(action="echo", tool_context=ctx, topic="/cmd_vel", type="geometry_msgs/msg/Twist")
        assert not ctx.interrupt.called
        assert result["status"] == "success"

    def test_the_pre_approval_example_in_the_row_lifts_the_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Follow the row's own example and the halt must then go through.

        Correcting the default must not remove the documented way to make the
        halt reachable unattended, which is the whole point of the variable.
        """
        row = _readme_allow_row()
        example = re.search(r"e\.g\.\s*`([^`]+)`", row)
        assert example, f"the row no longer shows a pre-approval example: {row}"
        surfaces = example.group(1)
        assert "/cmd_vel" in surfaces, f"the example stopped naming the halt surface: {surfaces}"
        monkeypatch.setenv(ros_mod._COMMAND_ALLOW_ENV, surfaces)

        ctx = MagicMock()
        result = self._publish_twist(self._halt_fields(), ctx)
        assert result["status"] == "success", f"the documented pre-approval did not lift the gate: {_texts(result)}"
        assert not ctx.interrupt.called, "a pre-approved surface still prompted the operator"
        assert len(self.calls["_publish"]) == 1, "the halt never reached the transport"

        goal = use_ros(
            action="action_send_goal",
            tool_context=ctx,
            action_name="/navigate_to_pose",
            type="nav2_msgs/action/NavigateToPose",
        )
        assert goal["status"] == "success", "the example's second surface was not pre-approved"


class TestTheDocumentedAllowlistReachIsTheRealReach:
    """A pre-approval's documented scope must be the scope the gate grants.

    ``_match_blocklist`` reduces both sides to a canonical form and then also
    compares base names, so a bare ``/cmd_vel`` matches ``/robot_b/cmd_vel``.
    That breadth is deliberate and load-bearing for the *blocklist*, where it is
    fail-safe: one entry has to catch every namespaced drive topic in the graph.
    The same matcher serves the operator's pre-approval list, where the identical
    breadth is fail-OPEN - one entry lifts the gate on surfaces the operator
    never named.

    Sizing that reach is the operator's job and the documentation is their only
    input, so a document may not describe the pre-approval as exclusive while the
    gate grants it by base name. The claims are graded against the running gate
    rather than banned outright, so a gate that one day matches exactly makes the
    exclusivity claim permissible instead of failing here.
    """

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv(ros_mod._COMMAND_ALLOW_ENV, raising=False)

    def test_every_surface_documenting_the_variable_states_its_real_reach(self) -> None:
        """The headline, graded in both directions against the running gate.

        Every surface that names the variable is telling an operator how to size
        a safety exemption, so each one has to carry the reach - a passing
        mention that leaves it out is how the fleet-wide case gets discovered in
        the field. The converse arm matters just as much: a gate narrowed to
        exact matching must not leave a namespaced-reach claim behind.
        """
        reach, extra = _measured_allowlist_reach()
        surfaces = _surfaces_documenting_the_allowlist()
        offenders = []
        for label, text in surfaces:
            states_reach = any(phrase in text for phrase in _BASE_NAME_REACH_PHRASES)
            claims_exclusive = any(phrase in text for phrase in _EXCLUSIVE_REACH_PHRASES)
            if reach == "base" and not states_reach:
                detail = " (and calls the pre-approval exclusive)" if claims_exclusive else ""
                offenders.append(f"{label}{detail}")
            if reach == "exact" and states_reach:
                offenders.append(f"{label} (claims a namespaced reach the gate no longer grants)")
        assert not offenders, (
            f"a single {ros_mod._COMMAND_ALLOW_ENV}=/cmd_vel entry pre-approves {extra} - surfaces it "
            f"never names - so the gate's reach is {reach!r}, which these do not state: {offenders}"
        )

    def test_one_bare_entry_pre_approves_every_robots_drive_topic(self) -> None:
        """The measured fact the documentation has to carry."""
        reach, extra = _measured_allowlist_reach()
        assert reach == "base", "the gate now matches pre-approvals exactly; update the documented reach"
        assert len(extra) >= 3, f"expected several namespaced siblings un-gated, got {extra}"

    def test_a_namespaced_entry_scopes_the_approval_to_one_robot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The remedy the corrected documentation points at has to work.

        Naming the namespace is what keeps a fleet-wide pre-approval from being
        the only option, so it is pinned here rather than left to prose.
        """
        monkeypatch.setenv(ros_mod._COMMAND_ALLOW_ENV, "/robot_a/cmd_vel")
        assert _gate_command("publish", "/robot_a/cmd_vel", None) is None
        for other in ("/robot_b/cmd_vel", "/cmd_vel"):
            result = _gate_command("publish", other, None)
            assert result is not None and result["status"] == "error", (
                f"a pre-approval naming /robot_a/cmd_vel also lifted the gate on {other}"
            )

    def test_the_sweep_reaches_every_operator_facing_surface(self) -> None:
        """Non-vacuity: a doc reflow that hides the variable must fail loudly."""
        labels = [label for label, _ in _surfaces_documenting_the_allowlist()]
        assert set(_OPERATOR_FACING_DOCS) <= set(labels), (
            f"only {labels} document {ros_mod._COMMAND_ALLOW_ENV}; a clean sweep would prove nothing"
        )
        assert "_gate_command docstring" in labels, "the gate helper stopped naming the variable it reads"
