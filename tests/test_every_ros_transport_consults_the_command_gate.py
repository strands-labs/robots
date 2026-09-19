"""The operator-approval gate covers every transport onto a ROS 2 graph, not one.

Three tools reach a ROS 2 graph - :mod:`~strands_robots.tools.use_ros` over
in-process rclpy, :mod:`~strands_robots.tools.use_rtps` over raw RTPS, and
:mod:`~strands_robots.tools.use_rosbridge` over a rosbridge WebSocket - and all
three are agent-callable and can carry a command to a physical robot. A ``Twist``
on ``/cmd_vel`` moves the same base whichever of them wrote it.

The gate shipped in ``use_ros`` alone. Measured on the two siblings before this
suite existed, against the same blocklisted surface and with the operator
declining on ``use_ros``::

    use_ros   publish /probe_only/cmd_vel -> prompted, error: declined by the operator
    use_rtps  publish /probe_only/cmd_vel -> success: published 1 message(s)

``use_rtps`` had no ``tool_context`` parameter at all, so there was nothing to
decline with: an agent refused at a drive topic could re-issue the identical
command under another tool name and it went out with no prompt, no allowlist
check and no audit row. The tool name was the whole difference.

The sibling suite ``tests/test_use_ros_command_blocklist.py`` states the same
argument one level down - "a helper that refuses correctly is worthless if a verb
never consults it". This file is that sentence with *transport* for *verb*, so
the two together grade the gate over both axes it has to cover: every verb of
every transport.

The inventory the structural half grades is DERIVED from the tree rather than
listed here, so a fourth transport is graded on arrival instead of shipping
un-gated the way these two did.
"""

from __future__ import annotations

import ast
import inspect
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.ros as ros_transport_mod
import strands_robots.rosbridge as rosbridge_transport_mod
import strands_robots.rtps.participant as rtps_participant_mod
import strands_robots.tools.use_ros as ros_mod
from strands_robots.mesh import RosBridgedRobot, RosbridgeRobot, RtpsRobot
from strands_robots.tools.use_ros import use_ros
from strands_robots.tools.use_rosbridge import use_rosbridge
from strands_robots.tools.use_rtps import use_rtps

# A surface the blocklist has carried since the gate was introduced, and the
# namespaced spelling a bare entry also has to cover.
_BLOCKED = "/cmd_vel"
_BLOCKED_NAMESPACED = "/robot_a/cmd_vel"

# The verbs that carry a command to a robot. Read from the dispatch of each tool
# below rather than assumed for all three: only ``use_ros`` speaks all three
# protocols.
_COMMAND_VERBS = frozenset({"publish", "service_call", "action_send_goal"})

# Each agent-callable transport, with the module holding its backend probe and
# the interface type spelling that transport accepts. rosbridge speaks ROS 1
# two-segment types; the other two speak ROS 2 three-segment types. All three
# are envelopes over transports a layer down, which the mesh robots publish
# through as well, so each probe is that transport's rather than the tool's.
_TRANSPORTS: tuple[tuple[str, Any, Any, str], ...] = (
    ("use_ros", use_ros, ros_transport_mod, "geometry_msgs/msg/Twist"),
    ("use_rtps", use_rtps, rtps_participant_mod, "geometry_msgs/msg/Twist"),
    ("use_rosbridge", use_rosbridge, rosbridge_transport_mod, "geometry_msgs/Twist"),
)

_TOOLS_DIR = Path(ros_mod.__file__).resolve().parent
#: The package root, so the single-owner pin below reads every module rather
#: than one directory: the gate itself sits in ``core``, and a second copy
#: anywhere would make two transports disagree just as surely.
_PACKAGE_DIR = _TOOLS_DIR.parent


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient pre-approval, and every transport reports itself reachable.

    The backend probes are forced so the gate is exercised on a runner that has
    neither rclpy, cyclonedds nor roslibpy: the gate is consulted before any of
    them is touched, which is the property under test.
    """
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
    for module in (ros_transport_mod, rtps_participant_mod, rosbridge_transport_mod):
        monkeypatch.setattr(module._backend, "available", lambda: True)


# Kept short so a case that proceeds PAST the gate does not sit on a real dial:
# rosbridge reads it for every action, and a refusal happens before the connect,
# so the gated cases never wait at all.
_DIAL_TIMEOUT = 0.1


def _publish(tool: Any, msg_type: str, ctx: Any, topic: str = _BLOCKED) -> dict[str, Any]:
    return tool(action="publish", topic=topic, type=msg_type, count=1, timeout=_DIAL_TIMEOUT, tool_context=ctx)


class TestEveryTransportRefusesTheSameSurface:
    """The headline: one blocklisted surface, one verdict, whichever tool asks."""

    @pytest.mark.parametrize(("label", "tool", "_module", "msg_type"), _TRANSPORTS)
    def test_a_declined_command_is_refused_on_every_transport(
        self, label: str, tool: Any, _module: Any, msg_type: str
    ) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = _publish(tool, msg_type, ctx)
        assert ctx.interrupt.called, f"{label} published to {_BLOCKED} without asking the operator"
        assert result["status"] == "error", f"{label} reported success for a declined command"
        assert "declined" in _texts(result)

    @pytest.mark.parametrize(("label", "tool", "_module", "msg_type"), _TRANSPORTS)
    def test_a_blocked_publish_is_refused_without_passing_any_context(
        self, label: str, tool: Any, _module: Any, msg_type: str
    ) -> None:
        """The bypass, stated in the one call shape every version of these tools accepts.

        No ``tool_context`` keyword is passed at all, so this is exactly the call an
        operator-less caller makes and it is expressible against a tool that has no
        such parameter. The command has to be refused; a ``success`` here is a
        ``Twist`` on a drive topic that nobody approved.
        """
        result = tool(action="publish", topic=_BLOCKED, type=msg_type, count=1, timeout=_DIAL_TIMEOUT)
        assert result["status"] == "error", (
            f"{label} published to {_BLOCKED} with no operator approval: {_texts(result)}"
        )
        assert "safety-critical command surface" in _texts(result)

    @pytest.mark.parametrize(("label", "tool", "_module", "msg_type"), _TRANSPORTS)
    def test_a_namespaced_instance_is_gated_on_every_transport(
        self, label: str, tool: Any, _module: Any, msg_type: str
    ) -> None:
        """A bare blocklist entry has to cover the namespaced deployment too."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = _publish(tool, msg_type, ctx, topic=_BLOCKED_NAMESPACED)
        assert ctx.interrupt.called, f"{label} published to {_BLOCKED_NAMESPACED} ungated"
        assert result["status"] == "error"

    @pytest.mark.parametrize(("label", "tool", "_module", "msg_type"), _TRANSPORTS)
    def test_headless_fails_closed_naming_both_env_vars(
        self, label: str, tool: Any, _module: Any, msg_type: str
    ) -> None:
        """With no interrupt reachable the command is refused, not sent."""
        result = _publish(tool, msg_type, None)
        assert result["status"] == "error", f"{label} sent a blocked command with no operator reachable"
        text = _texts(result)
        assert "STRANDS_ROS2_COMMAND_ALLOW" in text
        assert "BYPASS_TOOL_CONSENT" in text

    @pytest.mark.parametrize(("label", "tool", "_module", "msg_type"), _TRANSPORTS)
    def test_the_allowlist_pre_approves_on_every_transport(
        self, label: str, tool: Any, _module: Any, msg_type: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented headless escape works on every transport, not just one.

        Without this the fix would trade one broken posture for another: a gate
        an operator cannot pre-approve is one they will disable entirely.
        """
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", _BLOCKED)
        ctx = MagicMock()
        result = _publish(tool, msg_type, ctx)
        assert not ctx.interrupt.called, f"{label} prompted for a pre-approved surface"
        assert "safety-critical command surface" not in _texts(result)

    @pytest.mark.parametrize(("label", "tool", "_module", "msg_type"), _TRANSPORTS)
    def test_an_unblocked_surface_is_never_gated(self, label: str, tool: Any, _module: Any, msg_type: str) -> None:
        """The gate is keyed on the surface, so ordinary topics are untouched."""
        ctx = MagicMock()
        result = _publish(tool, msg_type, ctx, topic="/demo/twist")
        assert not ctx.interrupt.called, f"{label} gated an unblocked topic"
        assert "safety-critical command surface" not in _texts(result)


class TestReadingIsNeverGated:
    """Every transport keeps its read path ungated, and RTPS keeps ``advertise``."""

    @pytest.mark.parametrize(
        ("label", "tool", "kwargs"),
        (
            ("use_ros", use_ros, {"action": "echo", "topic": _BLOCKED, "type": "geometry_msgs/msg/Twist"}),
            ("use_rtps", use_rtps, {"action": "echo", "topic": _BLOCKED, "type": "geometry_msgs/msg/Twist"}),
            ("use_rosbridge", use_rosbridge, {"action": "echo", "topic": _BLOCKED, "type": "geometry_msgs/Twist"}),
        ),
    )
    def test_reading_a_blocked_surface_never_prompts(self, label: str, tool: Any, kwargs: dict[str, Any]) -> None:
        ctx = MagicMock()
        tool(tool_context=ctx, timeout=_DIAL_TIMEOUT, **kwargs)
        assert not ctx.interrupt.called, f"{label} asked the operator to read {_BLOCKED}"

    def test_rtps_advertise_is_not_gated_because_it_writes_no_sample(self) -> None:
        """Creating a publisher is not commanding: no sample reaches the robot.

        The boundary is deliberate, so it is pinned - a later widening that gates
        ``advertise`` would prompt an operator for an action that moves nothing.
        """
        ctx = MagicMock()
        use_rtps(action="advertise", topic=_BLOCKED, type="geometry_msgs/msg/Twist", tool_context=ctx)
        assert not ctx.interrupt.called


class TestTheGateRunsAfterArgumentValidation:
    """A caller's own mistake is reported without bothering an operator."""

    @pytest.mark.parametrize(("label", "tool", "_module", "_msg_type"), _TRANSPORTS)
    def test_a_missing_type_is_reported_without_asking_the_operator(
        self, label: str, tool: Any, _module: Any, _msg_type: str
    ) -> None:
        ctx = MagicMock()
        result = tool(action="publish", topic=_BLOCKED, timeout=_DIAL_TIMEOUT, tool_context=ctx)
        assert result["status"] == "error"
        assert not ctx.interrupt.called, f"{label} asked the operator about an incomplete call"


#: A transport whose mechanics are shared by an agent tool and a library class:
#: the label the operator decision is keyed with, the entry point both callers
#: reach it through, a factory for the class, the tool, and the interface type
#: that transport speaks. The structural pin below reads the tool package, so it
#: cannot see the second caller at all - these rows are the same argument one
#: layer down.
_SHARED_TRANSPORTS: tuple[tuple[str, Any, Any, Any, str], ...] = (
    (
        "ros",
        ros_transport_mod.ros_action,
        lambda: RosBridgedRobot("turtle", _BLOCKED, "/odom"),
        use_ros,
        "geometry_msgs/msg/Twist",
    ),
    (
        "rtps",
        rtps_participant_mod.rtps_action,
        lambda: RtpsRobot.from_rtps(node_name="rover", cmd_vel_topic=_BLOCKED),
        use_rtps,
        "geometry_msgs/msg/Twist",
    ),
    (
        "rosbridge",
        rosbridge_transport_mod.rosbridge_action,
        lambda: RosbridgeRobot("rover", _BLOCKED, "/odom"),
        use_rosbridge,
        "geometry_msgs/Twist",
    ),
)


class TestEveryCallerOfOneTransportAsksTheSameQuestion:
    """A transport is not always a tool: every one of them has a second caller.

    :mod:`strands_robots.ros` carries the in-process ``rclpy`` mechanics for the
    ``use_ros`` tool *and* for :class:`~strands_robots.mesh.RosBridgedRobot` and
    :class:`~strands_robots.mesh.AckermannRosRobot`;
    :mod:`strands_robots.rtps.participant` carries the DDS mechanics for the
    ``use_rtps`` tool *and* for :class:`~strands_robots.mesh.RtpsRobot`; and
    :mod:`strands_robots.rosbridge` carries the WebSocket mechanics for the
    ``use_rosbridge`` tool *and* for
    :class:`~strands_robots.mesh.RosbridgeRobot`. Each of those robots reaches
    the same physical ``cmd_vel`` without going through an agent tool at all,
    and the structural pin above reads the tool package, so it cannot see that
    second caller. Two pins per shared transport: a transport nobody can command
    through without deciding about the operator, and one label for the decision
    however it was reached.
    """

    @pytest.mark.parametrize(("label", "entry_point", "_factory", "_tool", "_msg_type"), _SHARED_TRANSPORTS)
    def test_the_transport_refuses_to_command_without_a_gate_argument(
        self, label: str, entry_point: Any, _factory: Any, _tool: Any, _msg_type: str
    ) -> None:
        """The gate is a required argument, so a caller cannot omit it silently.

        A default would be the un-gated sibling all over again: whichever value
        it took, a new caller would inherit it by writing nothing.
        """
        gate = inspect.signature(entry_point).parameters["gate"]
        assert gate.default is inspect.Parameter.empty, f"{label}: a defaulted gate is a gate a caller can forget"
        assert gate.kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize(("label", "_entry_point", "factory", "tool", "msg_type"), _SHARED_TRANSPORTS)
    def test_a_robot_and_the_tool_prompt_the_operator_identically(
        self, label: str, _entry_point: Any, factory: Any, tool: Any, msg_type: str
    ) -> None:
        """One blocklisted topic, one question - whichever caller reached it.

        The label keys the interrupt id ``<tool>-command-approval`` and the audit
        source ``<tool>_tool``, so two spellings would file one incident's rows
        under two names and an operator would be asked the same thing twice over.
        Both callers decline here, so the assertion is made before any socket is
        dialed or anything joins a graph.
        """
        tool_ctx, robot_ctx = MagicMock(), MagicMock()
        tool_ctx.interrupt.return_value = "n"
        robot_ctx.interrupt.return_value = "n"

        assert _publish(tool, msg_type, tool_ctx)["status"] == "error"
        assert factory().drive(linear=1.0, tool_context=robot_ctx)["status"] == "error"

        assert robot_ctx.interrupt.call_args == tool_ctx.interrupt.call_args, (
            f"{label}: the two callers ask the operator different questions about one surface"
        )


def _commanding_transport_modules() -> dict[str, set[str]]:
    """Derive the tool modules that dispatch a command verb onto a ROS graph.

    Read from the tree rather than listed, so a transport added later is graded
    on arrival. A module qualifies when its dispatch compares ``action`` against
    one of the verbs that carries a command; the read-only actions never do.

    Returns:
        ``{module name: the command verbs it dispatches}``.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(_TOOLS_DIR.glob("*.py")):
        verbs: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "action"):
                continue
            for comparator in node.comparators:
                candidates = comparator.elts if isinstance(comparator, ast.Tuple) else [comparator]
                for element in candidates:
                    if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                        continue
                    if element.value in _COMMAND_VERBS:
                        verbs.add(element.value)
        if verbs:
            found[path.stem] = verbs
    return found


class TestEveryCommandingTransportConsultsTheGate:
    """Structural half: a transport added later cannot ship un-gated in silence."""

    def test_the_scan_finds_the_transports_it_is_meant_to_cover(self) -> None:
        """Non-vacuity: an empty or mis-rooted scan must not read as compliant."""
        modules = _commanding_transport_modules()
        assert set(modules) >= {"use_ros", "use_rtps", "use_rosbridge"}, (
            f"the scan lost a known commanding transport: found {sorted(modules)}"
        )

    def test_every_commanding_transport_consults_the_shared_gate(self) -> None:
        """The rule, over the derived inventory rather than a hand-written list."""
        offenders = []
        for name in _commanding_transport_modules():
            source = (_TOOLS_DIR / f"{name}.py").read_text(encoding="utf-8")
            if "gate_command(" not in source:
                offenders.append(name)
        assert not offenders, (
            f"these tools dispatch a command verb onto a ROS graph without consulting the "
            f"operator gate, so the same surface is refused on one transport and sent on "
            f"another: {offenders}"
        )

    @pytest.mark.parametrize(("label", "tool", "_module", "_msg_type"), _TRANSPORTS)
    def test_every_commanding_tool_accepts_the_operator_context(
        self, label: str, tool: Any, _module: Any, _msg_type: str
    ) -> None:
        """Without ``context=True`` the gate can only ever fail closed.

        A tool that cannot receive the context inherits the headless refusal for
        every command, so coverage alone would make the surface unusable rather
        than approvable.
        """
        source = (_TOOLS_DIR / f"{label}.py").read_text(encoding="utf-8")
        assert "@tool(context=True)" in source, f"{label} cannot receive an operator context"
        params = inspect.signature(getattr(tool, "__wrapped__", tool)).parameters
        assert "tool_context" in params, f"{label} takes no tool_context"

    def test_the_gate_has_exactly_one_owner(self) -> None:
        """One blocklist. A second copy is how two transports come to disagree."""
        owners = [
            str(path.relative_to(_PACKAGE_DIR))
            for path in sorted(_PACKAGE_DIR.rglob("*.py"))
            if "COMMAND_BLOCKLIST = frozenset(" in path.read_text(encoding="utf-8")
        ]
        assert owners == ["_command_gate.py"], f"the command blocklist is defined in {owners}"


def _lock_is_free(backend: Any) -> bool:
    """Whether a second thread can take ``backend.lock`` right now.

    The lock is an ``RLock``, so the thread already holding it re-enters freely -
    the question is only answerable from another thread, which is also the thread
    the harm lands on: every other caller of the transport.
    """
    taken: list[bool] = []

    def probe() -> None:
        acquired = backend.lock.acquire(timeout=_LOCK_PROBE_TIMEOUT)
        taken.append(acquired)
        if acquired:
            backend.lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return taken[0]


#: Long enough that a held lock is not mistaken for scheduler noise, short enough
#: that the failing case costs a fraction of a second per row.
_LOCK_PROBE_TIMEOUT = 0.5


class TestNoOperatorIsAskedUnderTheTransportLock:
    """A human thinking must not stall every other caller of the transport.

    Each transport serialises its callers through one process-wide lock - the
    rclpy executor is not re-entrant, DDS access is serialised, and one
    ``roslibpy.Ros`` is shared per ``(host, port)``. ``ctx.interrupt()`` blocks
    for as long as the operator takes to answer, so consulting the gate under
    that lock hands a human the transport: an unrelated ``echo`` on the same
    graph - the odometry read of a second robot, a scan - waits out the decision.

    Measured on the three transports before this pin existed, with the operator
    being asked about a ``publish`` onto a blocklisted topic::

        use_ros        lock free while asking: False
        use_rtps       lock free while asking: True
        use_rosbridge  lock free while asking: True

    ``use_ros`` consulted the gate inside ``with _backend.lock:``; the two
    siblings hoisted it above. Both halves below are table-driven over every
    transport rather than written against the one that diverged, so the next one
    to grow a lock is graded on arrival.
    """

    @pytest.mark.parametrize(("label", "tool", "module", "msg_type"), _TRANSPORTS)
    def test_the_transport_lock_is_free_while_the_operator_is_asked(
        self, label: str, tool: Any, module: Any, msg_type: str
    ) -> None:
        observed: list[bool] = []
        ctx = MagicMock()

        def ask(*_args: Any, **_kwargs: Any) -> str:
            observed.append(_lock_is_free(module._backend))
            return "n"

        ctx.interrupt.side_effect = ask
        result = _publish(tool, msg_type, ctx)

        assert observed, f"{label} did not ask the operator about {_BLOCKED}"
        assert result["status"] == "error"
        assert observed == [True], (
            f"{label} asked the operator while holding its transport lock: every other "
            f"caller of this transport - a read, a second robot on the same graph - waits "
            f"out the human decision"
        )

    @pytest.mark.parametrize(("label", "entry_point", "_factory", "_tool", "_msg_type"), _SHARED_TRANSPORTS)
    def test_no_gate_call_sits_inside_a_lock_block(
        self, label: str, entry_point: Any, _factory: Any, _tool: Any, _msg_type: str
    ) -> None:
        """The same rule structurally, so a new verb cannot reintroduce it.

        The behavioural half above exercises ``publish``; this reads the whole
        dispatch, so a ``service_call`` or an ``action_send_goal`` that consults
        the gate under the lock is caught without a row of its own.
        """
        source = Path(inspect.getsourcefile(entry_point) or "").read_text(encoding="utf-8")
        dispatch = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == entry_point.__name__
        )
        lock_blocks = [
            node
            for node in ast.walk(dispatch)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == "lock" for item in node.items
            )
        ]
        assert lock_blocks, f"{label}: the scan found no lock block in {entry_point.__name__} to grade"
        gated_under_lock = [
            node.lineno
            for block in lock_blocks
            for node in ast.walk(block)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "gate"
        ]
        assert not gated_under_lock, (
            f"{label}: the operator gate is consulted under the transport lock at "
            f"line(s) {gated_under_lock}, so a human deciding holds the lock every "
            f"other caller of this transport has to take"
        )
