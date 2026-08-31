"""An inbound ``joint_command`` position must be finite, not merely readable.

:meth:`~strands_robots.ros_telemetry.RosTelemetryBase._command_action` is the one
inherited parser both hardware bridges and the simulation bridge deliver
``/<robot>/joint_command`` into. It refused a position ``float()`` could not read
and deliberately let a ``nan`` through, on this stated ground:

    Finiteness is deliberately not checked here: a nan position is a readable
    number that ``send_action`` refuses naming the joint.

That is a claim about ``send_action``, and the parser is shared by subclasses
whose ``send_action`` is a different function. It held for one of them:

* :class:`~strands_robots.simulation.ros_bridge.SimRosBridge` drives a
  simulation host, and ``MuJoCoSimEngine.send_action`` does refuse a non-finite
  action value, naming the key: ``send_action: action value for key '1' must be
  finite (no nan/inf), got nan``.
* :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge` drives a real
  arm through ``bus_access.write_action``, which takes the bus lock and
  delegates, and lerobot's ``SOFollower.send_action`` checks nothing before
  ``bus.sync_write("Goal_Position", ...)``.

lerobot bounds a normalized position in ``MotorsBus._unnormalize`` with
``min(100.0, max(-100.0, val))`` (and ``min(100.0, max(0.0, val))`` for the
gripper's ``RANGE_0_100``). ``nan`` compares false against both bounds, so
``max`` keeps its first argument and the clamp resolves the joint to an end
stop instead of refusing it. Measured through the real
``FeetechMotorsBus._unnormalize`` on an SO-101's own default norm modes, with a
shoulder calibrated ``800..3200``:

===============  =========================  =========
position          wire value                 warnings
===============  =========================  =========
``10.0``          ``2120``                   0
``nan``           ``800``  (``range_min``)   0
``inf``           ``3200`` (``range_max``)   0
===============  =========================  =========

So a ``joint_command`` carrying a ``nan`` drove a real arm to an end stop under
a ``success`` envelope with nothing logged: ``_drive_from_command`` warns only
when ``send_action`` raises or answers ``status="error"``, and it did neither.
``joint_limits`` catches it when supplied - ``not (low <= nan <= high)`` is true -
but it is optional and defaults to ``None`` on both bridges, so the default
``ros2_bridge=True`` configuration had nothing between the wire and the clamp.

The parser now refuses a non-finite position WHOLE, which is the disposition its
two siblings already have (a length mismatch and an out-of-range value are both
rejected entire, never partially applied), on the same
:func:`~strands_robots.utils.finite_number_error` domain the ``joint_limits``
bounds above it already use.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import textwrap
from typing import Any

import pytest

from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
from strands_robots.ros_telemetry import RosTelemetryBase
from tests.test_hardware_rtps_bridge import _FakeRobot, _JointState

_LOGGER = "strands_robots.ros_telemetry"

#: Readable numbers that are not usable positions.
_NON_FINITE = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="negative-inf"),
]

#: Positions that were accepted before this change and must stay accepted.
#: ``True`` is here because the domain rejects ``bool`` and the parser asks it
#: about the *coerced* value, so ``float(True)`` is judged as ``1.0``.
_STILL_ACCEPTED = [
    pytest.param(0.5, 0.5, id="float"),
    pytest.param(1, 1.0, id="int"),
    pytest.param("0.5", 0.5, id="numeric-string"),
    pytest.param(True, 1.0, id="bool"),
    pytest.param(0.0, 0.0, id="zero"),
    pytest.param(-99.5, -99.5, id="negative"),
]


def _base() -> RosTelemetryBase:
    return RosTelemetryBase()


class TestANonFinitePositionRefusesTheMessage:
    """The regression: a readable-but-non-finite position drops the command."""

    @pytest.mark.parametrize("position", _NON_FINITE)
    def test_a_non_finite_position_returns_none(self, position: float) -> None:
        assert _base()._command_action(_JointState(name=["j0"], position=[position])) is None

    @pytest.mark.parametrize("position", _NON_FINITE)
    def test_a_non_finite_position_is_reported(self, position: float, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _base()._command_action(_JointState(name=["j0"], position=[position]))
        assert len(caplog.records) == 1
        assert "j0" in caplog.text
        assert "finite" in caplog.text

    @pytest.mark.parametrize("position", _NON_FINITE)
    def test_no_part_of_the_message_is_applied(self, position: float) -> None:
        """Whole-message rejection, matching the length-mismatch sibling."""
        action = _base()._command_action(
            _JointState(name=["j0", "j1"], position=[0.25, position]),
        )
        assert action is None

    @pytest.mark.parametrize("position", _NON_FINITE)
    def test_a_bad_first_joint_does_not_let_the_rest_through(self, position: float) -> None:
        """The check is per-position, not a single look at the last one.

        Ordering is the whole difference between a per-position refusal and one
        asked once after the loop: with the bad value last, an after-the-loop
        check still catches it, so only this case distinguishes them.
        """
        action = _base()._command_action(
            _JointState(name=["j0", "j1"], position=[position, 0.25]),
        )
        assert action is None

    def test_nothing_reaches_send_action(self) -> None:
        """The refusal is upstream of the actuator write, so the arm never moves."""
        bridge: Any = HardwareRosBridge.__new__(HardwareRosBridge)
        bridge._joint_limits = None
        robot = _FakeRobot()
        bridge._drive_from_command(robot, _JointState(name=["j0"], position=[float("nan")]))
        assert robot.sent_actions == []


class TestBothBridgesShareTheRefusal:
    """One inherited parser, so no transport can drift from another."""

    def test_the_parser_is_shared(self) -> None:
        assert HardwareRosBridge._command_action is RosTelemetryBase._command_action
        assert HardwareRtpsBridge._command_action is RosTelemetryBase._command_action

    @pytest.mark.parametrize("bridge_cls", [HardwareRosBridge, HardwareRtpsBridge])
    def test_each_bridge_refuses_a_non_finite_position(self, bridge_cls: Any) -> None:
        # ``__new__`` on purpose: the parser reads only ``type(self).__name__``.
        bridge: Any = bridge_cls.__new__(bridge_cls)
        assert bridge._command_action(_JointState(name=["j0"], position=[float("nan")])) is None

    def test_the_refusal_names_the_bridge(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge: Any = HardwareRosBridge.__new__(HardwareRosBridge)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            bridge._command_action(_JointState(name=["j0"], position=[float("nan")]))
        assert "HardwareRosBridge" in caplog.text


class TestTheRefusalConsultsTheSharedDomain:
    """Single-sourced with the ``joint_limits`` bounds in the same module."""

    def test_the_parser_calls_the_shared_domain(self) -> None:
        source = textwrap.dedent(inspect.getsource(RosTelemetryBase._command_action))
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "finite_number_error" in called

    def test_the_domain_is_the_one_the_bounds_use(self) -> None:
        """``_validate_joint_limits`` has always used it; the position now does."""
        bounds_source = inspect.getsource(RosTelemetryBase._validate_joint_limits)
        assert "finite_number_error" in bounds_source


class TestWhatIsUnchanged:
    """The refused set is exactly the non-finite one - nothing wider."""

    @pytest.mark.parametrize(("position", "expected"), _STILL_ACCEPTED)
    def test_an_accepted_position_is_still_accepted(self, position: Any, expected: float) -> None:
        assert _base()._command_action(_JointState(name=["j0"], position=[position])) == {"j0": expected}

    @pytest.mark.parametrize(("position", "expected"), _STILL_ACCEPTED)
    def test_an_accepted_position_is_reported_silently(
        self, position: Any, expected: float, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _base()._command_action(_JointState(name=["j0"], position=[position]))
        assert caplog.records == []

    def test_a_healthy_command_still_reaches_send_action(self) -> None:
        bridge: Any = HardwareRosBridge.__new__(HardwareRosBridge)
        bridge._joint_limits = None
        robot = _FakeRobot()
        bridge._drive_from_command(robot, _JointState(name=["j0"], position=[0.25]))
        assert robot.sent_actions == [{"j0": 0.25}]

    def test_an_unreadable_position_keeps_its_own_refusal(self, caplog: pytest.LogCaptureFixture) -> None:
        """The non-numeric branch still reports non-numeric, not non-finite."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert _base()._command_action(_JointState(name=["j0"], position=["nope"])) is None
        assert "non-numeric" in caplog.text

    def test_an_empty_keepalive_is_still_dropped_silently(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert _base()._command_action(_JointState(name=[], position=[]), skip_empty=True) is None
        assert caplog.records == []


class TestTheOptInJointLimitsPathIsUnchanged:
    """``joint_limits`` caught a ``nan`` already; it still does, and still bounds."""

    def test_a_non_finite_position_is_still_refused_with_limits_declared(self) -> None:
        assert (
            _base()._command_action(
                _JointState(name=["j0"], position=[float("nan")]),
                joint_limits={"j0": (-100.0, 100.0)},
            )
            is None
        )

    def test_an_out_of_range_position_keeps_its_own_refusal(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert (
                _base()._command_action(
                    _JointState(name=["j0"], position=[150.0]),
                    joint_limits={"j0": (-100.0, 100.0)},
                )
                is None
            )
        assert "outside declared range" in caplog.text

    def test_an_in_range_position_is_still_accepted(self) -> None:
        assert _base()._command_action(
            _JointState(name=["j0"], position=[50.0]),
            joint_limits={"j0": (-100.0, 100.0)},
        ) == {"j0": 50.0}

    def test_joint_limits_defaults_to_none_on_both_bridges(self) -> None:
        """Why the clamp was reachable: the mitigation is opt-in."""
        for cls in (HardwareRosBridge, HardwareRtpsBridge):
            assert inspect.signature(cls.__init__).parameters["joint_limits"].default is None


class TestPremises:
    """Why the old reason was unsafe, measured rather than asserted."""

    def test_python_min_max_do_not_bound_a_nan(self) -> None:
        """The arithmetic lerobot's clamp is built from."""
        assert min(100.0, max(-100.0, float("nan"))) == -100.0
        assert min(100.0, max(0.0, float("nan"))) == 0.0
        assert min(100.0, max(-100.0, float("inf"))) == 100.0

    def test_a_non_finite_position_survives_float(self) -> None:
        """Which is why the non-numeric branch could never have caught it."""
        assert math.isnan(float(float("nan")))
        assert math.isinf(float(float("inf")))

    def test_lerobot_resolves_a_non_finite_position_to_an_end_stop(self) -> None:
        """The refusal the old reason relied on does not exist on this path."""
        feetech = pytest.importorskip("lerobot.motors.feetech")
        motors_mod = pytest.importorskip("lerobot.motors")
        motors = {
            "shoulder_pan": motors_mod.Motor(1, "sts3215", motors_mod.MotorNormMode.RANGE_M100_100),
        }
        calibration = {
            "shoulder_pan": motors_mod.MotorCalibration(
                id=1, drive_mode=0, homing_offset=0, range_min=800, range_max=3200
            ),
        }
        bus = feetech.FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calibration)
        assert bus._unnormalize({1: 10.0})[1] == 2120
        assert bus._unnormalize({1: float("nan")})[1] == 800
        assert bus._unnormalize({1: float("inf")})[1] == 3200

    def test_the_hardware_write_path_adds_no_finiteness_check(self) -> None:
        """``write_action`` takes the bus lock and delegates - nothing else."""
        from strands_robots.bus_access import write_action

        source = inspect.getsource(write_action)
        assert "isfinite" not in source
        assert "finite_number_error" not in source
