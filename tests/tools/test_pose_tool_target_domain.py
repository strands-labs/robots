# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the degree-valued targets ``pose_tool`` drives a joint to.

``position``, ``delta`` and the values of ``positions`` all reach a servo the
same way: ``MotorController.degrees_to_position`` clamps them into the motor's
configured ``range`` and scales the result onto the 12-bit ``Goal_Position``
register. The clamp is what makes an off-domain target dangerous rather than
merely wrong, and it is why this module measures the refusals against that
conversion instead of only asserting on them:

* **Every refused value shared an encoding with a mechanical limit.**
  ``TestWhyTheTargetsAreRefused`` shows ``nan``, ``inf`` and an out-of-range
  target all converting to ``Goal_Position`` 0 or 4095 - a full-travel command
  to an end stop. ``nan`` gets there because ``min(max_deg, nan)`` returns
  ``max_deg``, so the guard that looks like a safety net is the thing that
  fabricates the command.

* **And the caller was told it went where it asked.** The success text echoes
  the *requested* value, so the pre-guard behaviour reported
  ``"Moved shoulder_pan to nan deg"`` for a move to +180. That is the property
  ``TestTheBusIsNotTouched`` pins: a refused target produces no write at all.

The two deferrals on the ``delta`` path are pinned here as well, because each is
only sound if the thing it defers TO refuses. An unknown motor has no travel to
bound a displacement against, and ``incremental_move``'s own position read is
what refuses it; a displacement *inside* the travel can still compute an absolute
target outside the range, and ``degrees_to_position``'s clamp is what bounds that.

The domain itself is delegated to :func:`~strands_robots.utils.finite_number_error`
so an off-type or non-finite target is reported in the words every other surface
uses; only the per-joint bounds are decided in this module, because they are a
property of the arm it drives. That split is asserted in
``TestTheBoundsHaveOneAuthority`` rather than left to convention.

Every test that reaches the motor path takes ``fake_serial`` and passes an
explicit fake ``port``: ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``,
so a test that omits it drives whatever arm is plugged into the machine running
the suite.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytest
import serial

from strands_robots.tools.pose_tool import (
    _DEFAULT_MOTOR_CONFIGS,
    _TARGET_OPTION_BY_ACTION,
    MotorController,
    _joint_delta_error,
    _joint_target_error,
    _pose_target_error,
    pose_tool,
)
from strands_robots.utils import finite_number_error

from .conftest import FakeSerial

_PORT = "/dev/fake-pose-target"

# A joint with a symmetric range, and the gripper, which is configured 0-100 and
# is the one motor whose targets are a percentage rather than degrees.
_JOINT = "shoulder_pan"
_JOINT_RANGE = (-180, 180)

# Targets no joint can be driven to. Each is refused for one of two reasons -
# it is not a finite number, or it is outside the joint's configured travel -
# and ``TestWhyTheTargetsAreRefused`` measures both against the conversion.
_UNUSABLE_TARGETS: tuple[Any, ...] = (
    math.nan,
    math.inf,
    -math.inf,
    5000,
    -5000,
    180.5,
    True,
    False,
    "90",
    [90],  # a list is not a scalar target
    10**400,
)

# Targets inside the joint's travel, which must keep working unchanged.
_USABLE_TARGETS: tuple[Any, ...] = (0, 0.0, 90.0, -90.0, 180, -180, 12.5)


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool through one funnel.

    Several tests deliberately supply values outside the declared types (that is
    the contract under test), and a ``**dict[str, Any]`` splat is not narrowed,
    so routing every call through here states the intent once instead of
    scattering per-call suppressions.

    Args:
        **kwargs: Forwarded verbatim to :func:`pose_tool`.

    Returns:
        The tool result dict.
    """
    return pose_tool(**kwargs)


def _texts(result: dict[str, Any]) -> str:
    """Concatenate every ``text`` field of a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _goal_position(motor_name: str, degrees: Any) -> int:
    """The ``Goal_Position`` ``degrees`` would be written as, clamp included."""
    return MotorController(_PORT).degrees_to_position(motor_name, degrees)


class TestWhyTheTargetsAreRefused:
    """The domain is justified by what the conversion does with the value."""

    @pytest.mark.parametrize("target", [math.nan, math.inf, 5000, 180.5])
    def test_an_over_range_target_encodes_as_the_upper_end_stop(self, target):
        """Each collides with the *same* command: full travel to the limit.

        This is the whole reason a clamp cannot stand in for a refusal - the
        encoding is lossy in the one direction that matters, so ``nan`` and a
        deliberate ``180`` are indistinguishable once on the wire.
        """
        assert _goal_position(_JOINT, target) == _goal_position(_JOINT, _JOINT_RANGE[1])
        assert _goal_position(_JOINT, target) == 4095

    @pytest.mark.parametrize("target", [-math.inf, -5000, -180.5])
    def test_an_under_range_target_encodes_as_the_lower_end_stop(self, target):
        assert _goal_position(_JOINT, target) == _goal_position(_JOINT, _JOINT_RANGE[0])
        assert _goal_position(_JOINT, target) == 0

    def test_nan_reaches_the_limit_through_the_clamp_itself(self):
        """``min(max_deg, nan)`` returns ``max_deg``, so the clamp fabricates it."""
        assert min(_JOINT_RANGE[1], math.nan) == _JOINT_RANGE[1]
        assert _goal_position(_JOINT, math.nan) == 4095

    def test_a_bool_would_have_been_read_as_one_degree(self):
        """``True`` is an ``int`` subclass, so it encodes as a real 1-degree move."""
        assert _goal_position(_JOINT, True) == _goal_position(_JOINT, 1)

    @pytest.mark.parametrize("target", _USABLE_TARGETS)
    def test_an_in_range_target_is_not_the_end_stop_it_is_distinguishable_from(self, target):
        """An accepted target keeps its own encoding, which is the point."""
        encoded = _goal_position(_JOINT, target)
        assert 0 <= encoded <= 4095
        if target not in _JOINT_RANGE:
            assert encoded not in (0, 4095)


class TestTheBusIsNotTouched:
    """A refused target produces no serial write, on any of the three actions."""

    @pytest.mark.parametrize("target", _UNUSABLE_TARGETS)
    def test_move_motor_refuses_without_writing(self, target, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT, position=target)
        assert result["status"] == "error"
        assert "position" in _texts(result)
        assert fake_serial == [], "the port was opened for a target that cannot be honored"

    @pytest.mark.parametrize("target", _UNUSABLE_TARGETS)
    def test_move_multiple_refuses_without_writing(self, target, fake_serial, cwd_tmp):
        result = _call(action="move_multiple", port=_PORT, positions={_JOINT: target})
        assert result["status"] == "error"
        assert f"positions[{_JOINT!r}]" in _texts(result)
        assert fake_serial == []

    @pytest.mark.parametrize("target", [math.nan, math.inf, True, "90", [90], 10**400])
    def test_incremental_move_refuses_a_non_finite_delta_without_writing(self, target, fake_serial, cwd_tmp):
        result = _call(action="incremental_move", port=_PORT, motor_name=_JOINT, delta=target)
        assert result["status"] == "error"
        assert "delta" in _texts(result)
        assert fake_serial == []

    def test_a_delta_larger_than_the_full_travel_is_refused(self, fake_serial, cwd_tmp):
        """No starting position could honor it, so it is unhonorable by construction."""
        span = _JOINT_RANGE[1] - _JOINT_RANGE[0]
        result = _call(action="incremental_move", port=_PORT, motor_name=_JOINT, delta=span + 1)
        assert result["status"] == "error"
        assert f"at most {span} degrees in magnitude" in _texts(result)
        assert fake_serial == []

    def test_the_refusal_names_the_motor_and_its_travel(self, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT, position=5000)
        text = _texts(result)
        assert "[-180, 180] degrees" in text
        assert f"'{_JOINT}'" in text
        assert "5000" in text

    def test_the_gripper_is_quoted_as_a_percentage(self, fake_serial, cwd_tmp):
        """Its configured range is 0-100, so "degrees" would be the wrong word."""
        result = _call(action="move_motor", port=_PORT, motor_name="gripper", position=200)
        assert "[0, 100] percent" in _texts(result)


class TestUsableTargetsStillReachTheServo:
    """The guard refuses; it must not start refusing what already worked."""

    @pytest.mark.parametrize("target", _USABLE_TARGETS)
    def test_move_motor_still_writes_the_goal_position(self, target, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT, position=target)
        assert result["status"] == "success"
        assert len(fake_serial) == 1
        written = fake_serial[0].writes
        assert len(written) == 1
        # Feetech INST_WRITE to Goal_Position (0x2A), little-endian payload.
        encoded = _goal_position(_JOINT, target)
        assert written[0][4] == 0x03
        assert written[0][5] == 0x2A
        assert written[0][6] == encoded & 0xFF
        assert written[0][7] == (encoded >> 8) & 0xFF

    def test_move_multiple_still_drives_every_motor(self, fake_serial, cwd_tmp):
        result = _call(
            action="move_multiple",
            port=_PORT,
            positions={_JOINT: 10.0, "elbow_flex": -20.0},
            smooth=False,
        )
        assert result["status"] == "success"
        assert len(fake_serial[0].writes) == 2

    def test_a_target_exactly_on_each_bound_is_accepted(self, fake_serial, cwd_tmp):
        """The bounds are inclusive - they are reachable positions, not limits."""
        for bound in _JOINT_RANGE:
            assert (
                _pose_target_error("move_motor", motor_name=_JOINT, position=bound, delta=None, positions=None) is None
            )


class TestOnlyTheTargetTheActionReadsIsChecked:
    """A caller is never refused for a value the requested action never looks at."""

    @pytest.mark.parametrize("action", ["read_all", "list_poses", "connect", "read_position"])
    def test_an_action_reading_no_target_is_not_refused(self, action):
        assert (
            _pose_target_error(
                action, motor_name=_JOINT, position=math.nan, delta=math.nan, positions={_JOINT: math.nan}
            )
            is None
        )

    def test_move_motor_ignores_an_unusable_delta(self):
        """It commands an absolute position; ``delta`` is not its parameter."""
        assert _pose_target_error("move_motor", motor_name=_JOINT, position=0.0, delta=math.nan, positions=None) is None

    def test_incremental_move_ignores_an_unusable_position(self):
        assert (
            _pose_target_error("incremental_move", motor_name=_JOINT, position=math.nan, delta=0.0, positions=None)
            is None
        )

    @pytest.mark.parametrize("action", list(_TARGET_OPTION_BY_ACTION))
    def test_an_absent_target_is_left_to_the_required_check(self, action):
        """The action reports the whole missing pair; this guard must not pre-empt it."""
        assert _pose_target_error(action, motor_name=_JOINT, position=None, delta=None, positions=None) is None

    def test_a_missing_position_still_reports_the_required_pair(self, fake_serial, cwd_tmp):
        result = _call(action="move_motor", port=_PORT, motor_name=_JOINT)
        assert result["status"] == "error"
        assert "required" in _texts(result)


class TestTheBoundsHaveOneAuthority:
    """The servo and the guard must not read two copies of the same range."""

    def test_the_controller_is_configured_from_the_module_table(self):
        controller = MotorController(_PORT)
        assert {name: cfg["range"] for name, cfg in controller.motor_configs.items()} == {
            name: cfg["range"] for name, cfg in _DEFAULT_MOTOR_CONFIGS.items()
        }

    def test_each_controller_gets_its_own_copy(self):
        """Hoisting the table must not make one instance's edit global."""
        first = MotorController(_PORT)
        first.motor_configs[_JOINT]["range"] = (-1, 1)
        assert MotorController(_PORT).motor_configs[_JOINT]["range"] == _JOINT_RANGE
        assert _DEFAULT_MOTOR_CONFIGS[_JOINT]["range"] == _JOINT_RANGE

    def test_the_shared_domain_owns_finiteness_and_type(self):
        """Only the per-joint bounds are decided here; the rest is delegated.

        Asserted as an equality with the shared helper's own text so the two
        cannot drift into reporting the same value in different words.

        ``None`` is excluded deliberately: it is the omitted-target spelling,
        which the action's own required check reports as a missing pair. That
        exception is pinned by
        ``TestOnlyTheTargetTheActionReadsIsChecked::test_an_absent_target_is_left_to_the_required_check``.
        """
        for value in (math.nan, math.inf, True, "90", [90], 10**400):
            expected = finite_number_error(value, "position", "move_motor")
            if expected is None:
                continue
            assert (
                _pose_target_error("move_motor", motor_name=_JOINT, position=value, delta=None, positions=None)
                == expected
            )


class TestNoTargetSurfaceDrifts:
    """A target parameter added to the tool cannot skip the guard silently."""

    def test_every_declared_target_parameter_is_routed(self):
        """Each ``float | None`` / ``dict[str, float] | None`` option is covered.

        ``steps`` and ``step_delay`` are excluded by annotation rather than by
        name: they are ``int`` and ``float`` with defaults, never ``| None``,
        because they are interpolation options with their own guard.
        """
        routed = set(_TARGET_OPTION_BY_ACTION.values())
        declared = {
            name
            for name, param in inspect.signature(pose_tool.__wrapped__).parameters.items()
            if str(param.annotation) in ("float | None", "dict[str, float] | None")
        }
        assert declared, "the signature probe matched nothing - it has stopped testing anything"
        assert declared == routed, f"unrouted joint-target parameters: {sorted(declared - routed)}"

    def test_every_routed_action_is_a_real_action(self):
        """A typo in the map would silently disable the guard for that action."""
        doc = pose_tool.__wrapped__.__doc__ or ""
        for action in _TARGET_OPTION_BY_ACTION:
            assert f'"{action}"' in doc


class TestNeighbouringTargetProducersStayOutOfScope:
    """The boundary of this change, pinned so it narrows deliberately.

    Both remaining producers of a clamped target come from somewhere other than
    the caller's arguments, so neither is reachable from the guard above:

    * a **stored pose** is validated by ``PoseManager.validate_pose``, which
      already refuses an out-of-bounds position - but only when the pose file
      carries ``safety_bounds``;
    * ``reset_to_home`` supplies its own literal targets, which are in range.

    ``degrees_to_position`` therefore keeps its clamp. It is unreachable from
    ``move_motor`` / ``move_multiple``, whose targets are absolute and are held
    to the joint's endpoints - but NOT from ``incremental_move``, whose delta is
    held to the full travel instead, so a displacement inside that travel can
    still compute an absolute target outside the range.
    ``TestTheComputedTargetDeferralHolds`` measures that path.
    """

    def test_the_clamp_is_still_present_for_the_paths_that_rely_on_it(self):
        assert _goal_position(_JOINT, 5000) == 4095

    def test_load_pose_is_not_routed_through_the_target_guard(self):
        assert "load_pose" not in _TARGET_OPTION_BY_ACTION
        assert "reset_to_home" not in _TARGET_OPTION_BY_ACTION

    def test_an_unknown_motor_is_left_to_the_existing_path(self, fake_serial, cwd_tmp):
        """No configured range means no bounds to check it against."""
        assert (
            _pose_target_error("move_motor", motor_name="no_such_joint", position=5000, delta=None, positions=None)
            is None
        )

    def test_an_unknown_motor_with_a_non_finite_target_is_still_refused(self):
        """Finiteness needs no range, so the shared domain still applies."""
        assert (
            _pose_target_error("move_motor", motor_name="no_such_joint", position=math.nan, delta=None, positions=None)
            is not None
        )


def _position_packet(raw: int) -> bytes:
    """A Feetech read response encoding ``raw`` at bytes 5/6."""
    return bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, raw & 0xFF, (raw >> 8) & 0xFF, 0, 0, 0])


# A joint parked near the upper end of its travel, and a displacement inside the
# full span but *larger than either endpoint*. That is the value only the travel
# rule accepts, so it distinguishes this domain from one written against the
# endpoints; together they also compute an absolute target outside the range,
# which is the case the delta domain deliberately does not bound.
_NEAR_UPPER_RAW = 3980
_INSIDE_TRAVEL_DELTA = 300


class _ReadingSerial(FakeSerial):
    """A ``FakeSerial`` that always answers a read with a decodable position.

    ``incremental_move`` reads the current position before commanding anything,
    so a source that never answers refuses every motor - configured or not - and
    an assertion that an unknown motor reached no servo would hold for the wrong
    reason.
    """

    def read(self, n: int = 1) -> bytes:
        return _position_packet(_NEAR_UPPER_RAW)


@pytest.fixture
def reading_serial(monkeypatch: pytest.MonkeyPatch) -> list[_ReadingSerial]:
    """Patch ``serial.Serial`` with an always-answering position source."""
    instances: list[_ReadingSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _ReadingSerial:
        fs = _ReadingSerial(port, baudrate, timeout)
        instances.append(fs)
        return fs

    monkeypatch.setattr(serial, "Serial", _ctor)
    return instances


def _goal_positions(instances: list[_ReadingSerial]) -> list[int]:
    """Every ``Goal_Position`` value that reached the bus, decoded from the packets.

    A goal write is ``INST_WRITE`` (``0x03``) whose first parameter is the
    ``Goal_Position`` address (``0x2A``), with the value little-endian after it.
    Reads share the bus, so the payload is what distinguishes a command from a
    query.

    Args:
        instances: The recording serial stand-ins the fixture handed out.

    Returns:
        The commanded goal positions, in the order they were written.
    """
    goals: list[int] = []
    for fake in instances:
        for packet in fake.writes:
            if len(packet) >= 9 and packet[4] == 0x03 and packet[5] == 0x2A:
                goals.append(packet[6] | (packet[7] << 8))
    return goals


class TestTheUnknownMotorDeferralHolds:
    """A displacement for a motor with no configured travel, and what refuses it.

    :func:`_joint_delta_error` returns ``None`` for a motor absent from
    ``_DEFAULT_MOTOR_CONFIGS``: there is no travel to bound a displacement
    against, so it has nothing to say and defers - exactly as its sibling
    :func:`_joint_target_error` does for an absolute target.

    A deferral is only sound if the thing it defers TO refuses, and that half
    was unasserted. The branch itself was unexecuted by the whole suite, so a
    change making an unconfigured motor commandable through the delta path would
    have left every test green.
    """

    def test_an_unknown_motor_has_no_travel_to_bound_the_delta_against(self):
        """The domain defers rather than inventing a bound it cannot know."""
        assert _joint_delta_error("incremental_move", "no_such_joint", 5000) is None

    def test_both_helpers_defer_for_the_same_absent_configuration(self):
        """Whatever the domain does here it does for the absolute target too."""
        assert _joint_target_error("move_motor", "position", "no_such_joint", 5000) is None
        assert _joint_delta_error("incremental_move", "no_such_joint", 5000) is None

    def test_finiteness_still_applies_without_a_configured_range(self):
        """Only the per-joint bound needs a configuration; the shared domain does not."""
        assert _joint_delta_error("incremental_move", "no_such_joint", math.nan) is not None

    def test_the_action_refuses_the_unknown_motor_without_commanding_it(self, reading_serial, cwd_tmp):
        """The deferral's target: a read that cannot address an unconfigured motor."""
        result = _call(action="incremental_move", motor_name="no_such_joint", delta=5000, port=_PORT)
        assert result["status"] == "error"
        assert "no_such_joint" in _texts(result)
        assert _goal_positions(reading_serial) == []

    def test_a_configured_motor_takes_the_same_call_to_the_servo(self, reading_serial, cwd_tmp):
        """The refusal above is about the motor, not about the reading source."""
        result = _call(action="incremental_move", motor_name=_JOINT, delta=-90, port=_PORT)
        assert result["status"] == "success"
        assert _goal_positions(reading_serial) != []


class TestTheComputedTargetDeferralHolds:
    """A displacement inside the travel can still compute a target outside the range.

    The delta is bounded by the joint's *full travel* rather than by its
    endpoints, because a displacement is relative and the endpoints are not. So
    ``current + delta`` can leave the configured range for a delta this domain
    accepts, and ``degrees_to_position``'s clamp is what bounds it - making
    ``incremental_move`` the one caller-driven path from which that clamp is
    still reachable.
    """

    def test_a_displacement_inside_the_full_travel_is_accepted(self):
        """The premise: this delta is one the domain has no reason to refuse."""
        span = _JOINT_RANGE[1] - _JOINT_RANGE[0]
        assert abs(_INSIDE_TRAVEL_DELTA) < span
        # And larger than either endpoint, so a domain written against those
        # would refuse it: this is what makes the travel rule observable.
        assert abs(_INSIDE_TRAVEL_DELTA) > _JOINT_RANGE[1]
        assert _joint_delta_error("incremental_move", _JOINT, _INSIDE_TRAVEL_DELTA) is None

    def test_the_computed_absolute_target_leaves_the_configured_range(self):
        """And the premise for the clamp: the sum is outside the endpoints."""
        start = MotorController(_PORT).position_to_degrees(_JOINT, _NEAR_UPPER_RAW)
        assert start + _INSIDE_TRAVEL_DELTA > _JOINT_RANGE[1]

    def test_the_clamp_bounds_it_while_the_caller_is_told_it_moved(self, reading_serial, cwd_tmp):
        """So the end stop is commanded, and the text still echoes the request."""
        result = _call(action="incremental_move", motor_name=_JOINT, delta=_INSIDE_TRAVEL_DELTA, port=_PORT)
        assert result["status"] == "success"
        assert _goal_positions(reading_serial) == [_goal_position(_JOINT, _JOINT_RANGE[1])]
        assert f"+{_INSIDE_TRAVEL_DELTA}" in _texts(result)
