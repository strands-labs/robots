"""The mesh ROS 2 bridge must reach the ``use_ros`` operator gate, not fail closed.

Every :class:`RosBridgedRobot` command forwards to ``use_ros``, whose command
gate refuses a safety-critical surface when no operator context is reachable.
A bridge that never forwards a context therefore turns its whole command
surface - including the ``stop`` halt - into a per-call refusal, and
``tests/mesh/test_ros_bridge.py`` cannot see it because it patches the
``use_ros`` symbol at the boundary the gate lives behind.

These tests keep the real ``use_ros`` and substitute the rclpy transport
instead (the same boundary ``tests/tools/test_use_ros.py`` doubles), so the gate
and the bridge wiring under test both run unmodified while no message can reach
a real DDS graph.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.use_ros as ros_mod
from strands_robots.mesh import RosBridgedRobot

_COMMAND_METHODS = frozenset({"drive", "stop", "navigate_to"})
_COMMAND_ACTIONS = frozenset({"publish", "service_call", "action_send_goal"})
_BRIDGE_SOURCE = Path(ros_mod.__file__).parent.parent / "mesh" / "ros_bridge.py"


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _turtle() -> RosBridgedRobot:
    """A bridge whose cmd_vel and nav action are both blocklisted surfaces.

    ``/turtle1/cmd_vel`` matches ``/cmd_vel`` on the final-segment rule and
    ``/navigate_to_pose`` matches exactly, so every command this bridge sends is
    gated - which is what makes it the right instance to test the gate with.
    """
    return RosBridgedRobot.from_ros(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
        odom_topic="/turtle1/pose",
        odom_type="turtlesim/msg/Pose",
        nav_action="/navigate_to_pose",
    )


def _tool(robot: RosBridgedRobot, name: str) -> Any:
    return next(t for t in robot.tools if t.tool_name == name)


@pytest.fixture(scope="module")
def bridge_ast() -> ast.Module:
    """Parse the bridge source once for the structural guards below."""
    return ast.parse(_BRIDGE_SOURCE.read_text(encoding="utf-8"))


class TestBridgeCommandsReachTheGate:
    """The bridge's agent tools must prompt the operator, not refuse outright."""

    published: list[tuple[Any, ...]]
    goals: list[tuple[Any, ...]]
    echoed: list[tuple[Any, ...]]

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both env vars short-circuit the gate, so an ambient BYPASS_TOOL_CONSENT
        # (common in agent/automation shells) would make these assertions pass
        # without the gate ever running. Cases that need them opt in explicitly.
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
        monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
        self.published, self.goals, self.echoed = [], [], []

        def _record(sink: list[tuple[Any, ...]], outcome: Any) -> Any:
            def _fake(*args: Any) -> Any:
                sink.append(args)
                return outcome

            return _fake

        monkeypatch.setattr(ros_mod, "_publish", _record(self.published, None))
        monkeypatch.setattr(ros_mod, "_action_send_goal", _record(self.goals, {"goal_status": "SUCCEEDED"}))
        monkeypatch.setattr(ros_mod, "_echo", _record(self.echoed, [{"x": 1.0, "y": 2.0, "theta": 0.0}]))

    def test_drive_tool_prompts_the_operator_and_publishes_on_approval(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert ctx.interrupt.called, "the drive tool never reached the operator gate"
        assert ctx.interrupt.call_args[1]["reason"]["target"] == "/turtle1/cmd_vel"
        assert result["status"] == "success"
        assert len(self.published) == 1

    def test_drive_tool_declined_publishes_nothing(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert result["status"] == "error"
        assert "declined" in _texts(result)
        assert self.published == []

    def test_stop_tool_halts_the_robot_once_the_operator_approves(self) -> None:
        """The halt is gated like any other cmd_vel publish, and must be reachable.

        A bridge that cannot forward an operator context makes ``stop`` an
        unconditional refusal - removing the one control the ``tools`` contract
        guarantees ("a caller that can start motion must be able to end it").
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "stop_turtlesim")(tool_context=ctx)
        assert ctx.interrupt.called, "the stop tool never reached the operator gate"
        assert result["status"] == "success"
        topic, _type, fields = self.published[0][:3]
        assert topic == "/turtle1/cmd_vel"
        assert fields == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}

    def test_navigate_tool_gates_the_nav_action_goal(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = _tool(_turtle(), "navigate_turtlesim")(x=1.0, y=2.0, tool_context=ctx)
        assert ctx.interrupt.called, "the navigate tool never reached the operator gate"
        assert ctx.interrupt.call_args[1]["reason"]["action"] == "action_send_goal"
        assert result["status"] == "error"
        assert self.goals == []

    def test_navigate_tool_sends_the_goal_once_approved(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "navigate_turtlesim")(x=1.0, y=2.0, tool_context=ctx)
        assert result["status"] == "success"
        assert len(self.goals) == 1

    def test_read_tools_need_no_context_because_reads_are_never_gated(self) -> None:
        result = _tool(_turtle(), "get_pose_turtlesim")()
        assert result["status"] == "success"
        assert len(self.echoed) == 1

    def test_programmatic_command_without_a_context_names_the_headless_variables(self) -> None:
        """The documented decision: no operator context means the gate refuses.

        A programmatic ``turtle.drive(...)`` has nobody to prompt, so the refusal
        has to say how an operator lifts it rather than reading as a broken API.
        """
        result = _turtle().drive(linear=1.0)
        assert result["status"] == "error"
        assert "STRANDS_ROS2_COMMAND_ALLOW" in _texts(result)
        assert "BYPASS_TOOL_CONSENT" in _texts(result)
        assert self.published == []

    @pytest.mark.parametrize("allow", ["/turtle1/cmd_vel", "cmd_vel"])
    def test_programmatic_drive_and_stop_run_under_the_headless_allowlist(
        self, monkeypatch: pytest.MonkeyPatch, allow: str
    ) -> None:
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", allow)
        robot = _turtle()
        assert robot.drive(linear=1.0)["status"] == "success"
        assert robot.stop()["status"] == "success"
        assert len(self.published) == 2

    def test_programmatic_read_is_unaffected_by_the_gate(self) -> None:
        assert _turtle().get_pose()["status"] == "success"
        assert len(self.echoed) == 1


class TestCommandToolsDeclareTheOperatorContext:
    """Structural guards, so a command tool added later cannot ship contextless.

    The behavioural tests above cover the three command tools that exist today.
    A fourth one added without ``context=True`` would fail closed on every call
    with no test noticing, so the wiring itself is pinned here.
    """

    @staticmethod
    def _decorated_tools(tree: ast.Module) -> list[ast.FunctionDef]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "tool" for dec in node.decorator_list
            )
        ]

    @staticmethod
    def _forwards_a_command_method(func: ast.FunctionDef) -> bool:
        return any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _COMMAND_METHODS
            for node in ast.walk(func)
        )

    def test_every_command_tool_is_context_enabled_and_forwards_it(self, bridge_ast: ast.Module) -> None:
        command_tools = [f for f in self._decorated_tools(bridge_ast) if self._forwards_a_command_method(f)]
        assert {f.name for f in command_tools} == {"drive", "stop", "navigate"}
        for func in command_tools:
            decorator = next(d for d in func.decorator_list if isinstance(d, ast.Call))
            context_kwarg = next((kw for kw in decorator.keywords if kw.arg == "context"), None)
            assert context_kwarg is not None and getattr(context_kwarg.value, "value", None) is True, (
                f"bridge command tool {func.name!r} is not declared @tool(context=True), "
                "so it can never reach the use_ros operator gate"
            )
            params = [a.arg for a in func.args.args] + [a.arg for a in func.args.kwonlyargs]
            assert "tool_context" in params, f"{func.name!r} does not receive the injected operator context"
            forwarded = [
                kw.arg
                for node in ast.walk(func)
                if isinstance(node, ast.Call)
                for kw in node.keywords
                if kw.arg == "tool_context"
            ]
            assert forwarded, f"{func.name!r} receives the operator context but does not forward it"

    def test_read_only_tools_stay_contextless(self, bridge_ast: ast.Module) -> None:
        """``echo`` is never gated, so a read tool must not ask for an operator."""
        read_tools = [f for f in self._decorated_tools(bridge_ast) if not self._forwards_a_command_method(f)]
        assert {f.name for f in read_tools} == {"get_pose", "get_scan"}
        for func in read_tools:
            params = [a.arg for a in func.args.args]
            assert "tool_context" not in params, f"read-only tool {func.name!r} should not require an operator context"

    def test_every_bridge_command_call_forwards_the_context_to_use_ros(self, bridge_ast: ast.Module) -> None:
        """A bridge method that carries a command must not drop the context.

        This is the failure this suite exists for: the gate lives inside
        ``use_ros``, so a call site that omits ``tool_context`` silently becomes
        a fail-closed refusal for its whole method.
        """
        checked = 0
        for node in ast.walk(bridge_ast):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "use_ros"):
                continue
            action = next((kw.value for kw in node.keywords if kw.arg == "action"), None)
            if not (isinstance(action, ast.Constant) and action.value in _COMMAND_ACTIONS):
                continue
            checked += 1
            assert any(kw.arg == "tool_context" for kw in node.keywords), (
                f"use_ros(action={action.value!r}) at ros_bridge.py:{node.lineno} does not forward tool_context"
            )
        assert checked == 2, f"expected the publish and action_send_goal command call sites, found {checked}"

    def test_command_tools_do_not_expose_the_context_in_their_input_schema(self) -> None:
        """The operator context is injected, never a parameter the model fills."""
        for name in ("drive_turtlesim", "stop_turtlesim", "navigate_turtlesim"):
            spec = _tool(_turtle(), name).tool_spec
            assert "tool_context" not in spec["inputSchema"]["json"].get("properties", {})
