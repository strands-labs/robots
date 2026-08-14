"""One failure on the inbound command path costs exactly one message.

Both hardware bridges deliver ``/<robot>/joint_command`` into
:meth:`~strands_robots.ros_telemetry.RosTelemetryBase._command_action`, whose
contract is an action dict or ``None`` - never an exception. A position it could
not read used to escape as ``ValueError``/``TypeError``, and what that cost
depended on the transport:

* rclpy delivers one subscription callback per ``spin_once``, so the raise was
  absorbed by ``_spin_loop``'s tolerance and cost that one message.
* cyclonedds has no executor: ``_poll_loop`` calls ``take(N=10)``, which has
  already consumed the batch by the time a sample is parsed. The raise aborted
  the ``for``, so every sample *behind* the malformed one was discarded with it -
  valid commands that had already arrived, dropped with a DEBUG line.

The parser now reports the position and returns ``None``, so a malformed command
costs itself on either transport. The classes below pin that, the values that
were always readable (including ``nan``, whose finiteness ``send_action`` owns),
and the three contracts of the cyclonedds poll loop that its rclpy sibling has
pinned all along in ``test_hardware_ros_bridge.py``: a sample taken by the loop
is dispatched, a reader failure costs only that tick, and a second start spawns
no second thread (that last one lives beside its siblings in
``test_hardware_rtps_bridge.py``).

The loop is driven synchronously against a recording stop event: the command
thread is a background daemon coverage cannot trace, which is the same reason
the rclpy module gives for driving ``_spin_loop`` in the test thread.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
from strands_robots.ros_telemetry import RosTelemetryBase
from tests.test_hardware_rtps_bridge import _FakeReader, _FakeRobot, _JointState
from tests.test_wait_budget_domain import _RecordingStop

_LOGGER = "strands_robots.ros_telemetry"

#: Positions ``float()`` cannot read. Exactly the set that used to raise.
_UNREADABLE = [
    pytest.param("not-a-number", id="non-numeric-string"),
    pytest.param(None, id="None"),
    pytest.param([0.1], id="list"),
    pytest.param({"j": 0.1}, id="dict"),
    pytest.param(object(), id="object"),
]

#: Positions that were readable before this change and must stay readable.
_READABLE = [
    pytest.param(0.5, 0.5, id="float"),
    pytest.param(1, 1.0, id="int"),
    pytest.param("0.5", 0.5, id="numeric-string"),
    pytest.param(True, 1.0, id="bool"),
]


def _poll_skeleton(reader: Any, robot: Any, *, iterations: int) -> Any:
    """A bridge whose poll loop can be driven in the traced test thread.

    ``__new__`` skips ``__init__``, so no DDS participant is built and no poll
    thread is started; the loop body then runs synchronously with no race
    against a live daemon. Typed ``Any`` because the skeleton is deliberately
    partial - only what ``_poll_loop`` reads is wired.
    """
    bridge: Any = HardwareRtpsBridge.__new__(HardwareRtpsBridge)
    bridge._robot = robot
    bridge._command_reader = reader
    bridge._poll_period = 0.001
    bridge._joint_limits = None
    bridge._stop = _RecordingStop(iterations)
    return bridge


class TestAPositionThatCannotBeReadRejectsTheMessage:
    """The parser reports it and returns ``None`` - it does not raise."""

    @pytest.mark.parametrize("position", _UNREADABLE)
    def test_an_unreadable_position_returns_none(self, position: Any) -> None:
        base = RosTelemetryBase()
        assert base._command_action(_JointState(name=["j0"], position=[position])) is None

    @pytest.mark.parametrize("position", _UNREADABLE)
    def test_an_unreadable_position_is_reported(self, position: Any, caplog: pytest.LogCaptureFixture) -> None:
        base = RosTelemetryBase()
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            base._command_action(_JointState(name=["j0"], position=[position]))
        assert [r for r in caplog.records if r.levelno >= logging.WARNING], "the refusal was not reported at all"
        assert any("non-numeric position" in r.getMessage() for r in caplog.records)

    def test_the_report_names_the_joint_and_the_value(self, caplog: pytest.LogCaptureFixture) -> None:
        base = RosTelemetryBase()
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            base._command_action(_JointState(name=["ok", "wrist"], position=[0.5, "spin"]))
        assert caplog.records, "the refusal was not reported at all"
        message = caplog.records[-1].getMessage()
        assert "'wrist'" in message, message
        assert "'spin'" in message, message

    def test_the_whole_command_is_rejected_not_partially_applied(self) -> None:
        # The readable joint precedes the unreadable one, so a parser that
        # applied what it could would move "ok" and leave "wrist" behind.
        robot = _FakeRobot()
        RosTelemetryBase()._drive_from_command(robot, _JointState(name=["ok", "wrist"], position=[0.5, "spin"]))
        assert robot.sent_actions == []


class TestEveryReadablePositionStaysReadable:
    """The refused set is exactly what used to raise - nothing wider."""

    @pytest.mark.parametrize(("position", "expected"), _READABLE)
    def test_a_readable_position_is_still_accepted(self, position: Any, expected: float) -> None:
        base = RosTelemetryBase()
        assert base._command_action(_JointState(name=["j0"], position=[position])) == {"j0": expected}

    def test_a_nan_position_is_still_read(self) -> None:
        """Finiteness is not this parser's job.

        ``nan`` is a readable number, and ``send_action`` already refuses a
        non-finite action value naming the joint - so checking it here would
        move that report away from the surface that owns it.
        """
        base = RosTelemetryBase()
        action = base._command_action(_JointState(name=["j0"], position=[float("nan")]))
        assert action is not None
        assert action["j0"] != action["j0"]  # nan


class TestBothTransportsGetTheSameRefusal:
    """The parser is one inherited function, so neither transport can drift."""

    def test_the_parser_is_shared_by_both_bridges(self) -> None:
        assert HardwareRosBridge._command_action is RosTelemetryBase._command_action
        assert HardwareRtpsBridge._command_action is RosTelemetryBase._command_action

    @pytest.mark.parametrize("bridge_cls", [HardwareRosBridge, HardwareRtpsBridge])
    def test_each_bridge_refuses_the_same_message(self, bridge_cls: Any) -> None:
        # ``Any`` so ``__new__`` is reachable: skipping ``__init__`` is the point
        # here - the parser reads only ``type(self).__name__``.
        bridge: Any = bridge_cls.__new__(bridge_cls)
        assert bridge._command_action(_JointState(name=["j0"], position=["nope"])) is None


class TestTheCyclonedddsLoopKeepsTheRestOfTheBatch:
    """``take(N=10)`` consumes a batch, so one bad sample must not cost the rest."""

    def test_the_samples_behind_an_unreadable_one_are_still_applied(self) -> None:
        robot = _FakeRobot()
        reader = _FakeReader("/arm/joint_command")
        reader.feed(_JointState(name=["j0"], position=[0.1]))
        reader.feed(_JointState(name=["j1"], position=["not-a-number"]))
        reader.feed(_JointState(name=["j2"], position=[0.3]))

        _poll_skeleton(reader, robot, iterations=1)._poll_loop()

        # Both valid commands land; only the malformed one is dropped.
        assert robot.sent_actions == [{"j0": 0.1}, {"j2": 0.3}]

    def test_the_loop_dispatches_every_sample_of_a_batch(self) -> None:
        robot = _FakeRobot()
        reader = _FakeReader("/arm/joint_command")
        for i in range(3):
            reader.feed(_JointState(name=[f"j{i}"], position=[float(i)]))

        _poll_skeleton(reader, robot, iterations=1)._poll_loop()

        assert robot.sent_actions == [{"j0": 0.0}, {"j1": 1.0}, {"j2": 2.0}]


class TestAReaderFailureCostsOnlyThatTick:
    """The cyclonedds mirror of ``test_command_spin_loop_survives_a_transient_spin_once_failure``."""

    def test_a_take_that_raises_does_not_kill_the_loop(self) -> None:
        class _FlakyReader:
            def __init__(self) -> None:
                self.takes = 0

            def take(self, N: int = 10) -> list[Any]:
                self.takes += 1
                if self.takes == 1:
                    raise RuntimeError("cyclonedds take failed")
                # Reached only if the loop survived the failure above.
                return [_JointState(name=["j0"], position=[0.75])]

        robot = _FakeRobot()
        reader = _FlakyReader()

        _poll_skeleton(reader, robot, iterations=2)._poll_loop()  # must not propagate

        assert reader.takes == 2, "the loop stopped at the failing tick"
        assert robot.sent_actions == [{"j0": 0.75}]
