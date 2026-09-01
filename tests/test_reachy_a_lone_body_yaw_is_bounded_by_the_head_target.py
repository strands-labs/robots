"""A body-only turn is held to the head-body coupling limit, not just to its axis.

:data:`~strands_robots.tools.reachy.HEAD_BODY_YAW_DELTA_LIMIT_DEG` bounds
``head_yaw - body_yaw``, and ``envelope_error`` only reached it when one action
carried both names. Every motion verb that sends one member routed around it:
``reachy_body_turn`` sends ``body_yaw`` alone, ``reachy_look`` omits ``body_yaw``
at its ``None`` default, and an action naming a head axis other than the yaw
(``{"head_pitch": 10, "body_yaw": 160}``) commands the head yaw to zero without
spelling it, so the gate did not see that pair either.

What the limit is. The daemon's default kinematics solves a head pose through
``inverse_kinematics_safe(pose, body_yaw, max_relative_yaw=65 deg,
max_body_yaw=160 deg)`` - the same two figures the envelope carries. It never
refuses an over-twist and it cannot mechanically make one: it keeps the twist
inside the limit by moving the body, holding the head pose as the primary task.
Driving that solver over its own shipped kinematics data:

    requested head_yaw   requested body_yaw   body joint out   twist
                   180                    0           115.00   65.00
                    90                    0            25.00   65.00
                     0                  160            65.00  -65.00
                     0                   66            65.00  -65.00
                    65                    0            -0.00   65.00

So the two directions of a lone value are not the same event, and that asymmetry
is what this file pins:

* A lone ``head_yaw`` of 180 is honored - the body turns to 115 under it. The
  caller named no body yaw, so nothing of theirs is substituted, and refusing it
  would refuse the head verb its own range. Accepted, and pinned as accepted.
* A lone ``body_yaw`` of 160 against a head target of 0 reaches 65 and stops.
  The caller's own explicit value is replaced by one 95 degrees short of it,
  which is the silent substitution ``_reachy_common`` exists to refuse, and the
  same event as an out-of-limit pair. Refused.

Which head yaw to check against. Not a sensor reading - no telemetry carries
body yaw, and the head IMU measures the head's orientation rather than the twist.
It is the head pose the driver last commanded, which is exactly what the daemon
is still targeting, and it is known exactly rather than estimated because
``_wire_commands`` sends a whole pose every time. When the driver has not
commanded one, or a path that re-pins the daemon's own target has run since
(``play_move``, ``wake_up``, ``goto_sleep``, ``set_motors`` - the vendor's
``enable_motors`` re-pins ``target_head_pose`` to wherever the head physically
is), the target is unknown and the coupling is skipped rather than guessed: a
turn refused against a stale target would be a turn the robot could have made.
``TestAnUnknownTargetRefusesNothing`` holds that direction, because it is the
one a fix to this defect could plausibly break.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import pytest

from strands_robots.drivers.reachy import ReachyDriver, _head_yaw_of
from strands_robots.tools.reachy import HEAD_BODY_YAW_DELTA_LIMIT_DEG, MOTION_ENVELOPE_DEG, envelope_error

#: Comfortably outside the coupling limit while inside the body axis itself, so
#: a refusal here can only be the coupling and never per-axis travel.
_BEYOND_COUPLING = MOTION_ENVELOPE_DEG["body_yaw"]

#: Comfortably inside both.
_WITHIN_COUPLING = HEAD_BODY_YAW_DELTA_LIMIT_DEG - 25.0


def _native() -> tuple[ReachyDriver, list[dict[str, Any]]]:
    """A connected driver that records what it would put on the wire."""
    driver = ReachyDriver.__new__(ReachyDriver)
    driver._tool_name = "reachy_mini"
    driver._connected = True
    driver._cache_lock = threading.Lock()
    driver._head_yaw_target = None
    sent: list[dict[str, Any]] = []

    def _send(command: dict[str, Any]) -> str | None:
        sent.append(command)
        return None

    driver._send_cmd = _send  # type: ignore[method-assign]
    driver._daemon_post = lambda *a, **k: {}  # type: ignore[method-assign]
    return driver, sent


def _head_pose(**overrides: float) -> dict[str, Any]:
    """A whole head pose action, as ``reachy_look`` builds one."""
    action = {"head_pitch": 0.0, "head_roll": 0.0, "head_yaw": 0.0, "head_x": 0.0, "head_y": 0.0, "head_z": 0.0}
    action.update(overrides)
    return action


def _body_on_wire(sent: list[dict[str, Any]]) -> float | None:
    """The body yaw that reached the link, in degrees, or ``None`` if none did."""
    for command in sent:
        if "body_yaw" in command:
            return math.degrees(command["body_yaw"])
    return None


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in envelope.get("content", []) if "text" in block)


class TestALoneBodyTurnBeyondTheCouplingIsRefused:
    """The defect: a body-only turn the daemon would serve only in part."""

    def test_a_body_turn_beyond_the_limit_from_the_head_target_is_refused(self) -> None:
        driver, sent = _native()
        driver.send_action(_head_pose())
        sent.clear()

        result = driver.send_action({"body_yaw": _BEYOND_COUPLING})

        assert result["status"] == "error"
        assert _body_on_wire(sent) is None

    def test_the_refusal_names_the_limit_the_head_target_and_the_way_to_ask(self) -> None:
        """A refusal teaches the envelope, per this module's own contract."""
        driver, _sent = _native()
        driver.send_action(_head_pose())

        reason = _text(driver.send_action({"body_yaw": _BEYOND_COUPLING}))

        assert f"{HEAD_BODY_YAW_DELTA_LIMIT_DEG:g} deg" in reason
        assert "head yaw the daemon is targeting (0 deg)" in reason
        assert "name head_yaw in the same action" in reason

    def test_a_body_turn_within_the_limit_still_reaches_the_robot(self) -> None:
        """The control: this is a bound, not a ban on turning the body."""
        driver, sent = _native()
        driver.send_action(_head_pose())
        sent.clear()

        result = driver.send_action({"body_yaw": _WITHIN_COUPLING})

        assert result["status"] == "success"
        assert _body_on_wire(sent) == pytest.approx(_WITHIN_COUPLING)

    def test_the_bound_follows_the_head_target_rather_than_being_a_fixed_range(self) -> None:
        """With the head already round at 100, a body turn to 160 is 60 away - legal.

        The case a stateless rule would get wrong. Refusing every lone body yaw
        beyond the coupling limit would refuse this, and the robot can make it.
        """
        driver, sent = _native()
        driver.send_action(_head_pose(head_yaw=100.0))
        sent.clear()

        result = driver.send_action({"body_yaw": _BEYOND_COUPLING})

        assert result["status"] == "success"
        assert _body_on_wire(sent) == pytest.approx(_BEYOND_COUPLING)

    def test_a_body_yaw_beside_a_head_axis_that_is_not_the_yaw_is_checked_too(self) -> None:
        """No state needed here: naming any head key commands the yaw to zero."""
        driver, sent = _native()

        result = driver.send_action({"head_pitch": 10.0, "body_yaw": _BEYOND_COUPLING})

        assert result["status"] == "error"
        assert sent == [], "the head pose must not go out either when the pair is refused"

    def test_the_head_yaw_an_action_commands_is_read_from_the_one_rule(self) -> None:
        """``_head_yaw_of`` is what makes the previous case state-free."""
        assert _head_yaw_of({"head_pitch": 10.0}) == 0.0
        assert _head_yaw_of({"head_yaw": 33.0}) == 33.0
        assert _head_yaw_of({"body_yaw": 33.0}) is None
        assert _head_yaw_of({"antenna_left": 10.0}) is None


class TestALoneHeadYawIsHonoredNotRefused:
    """The other direction, which measurement says is legal motion."""

    def test_a_full_turn_of_the_head_alone_is_accepted(self) -> None:
        driver, sent = _native()

        result = driver.send_action(_head_pose(head_yaw=MOTION_ENVELOPE_DEG["head_yaw"]))

        assert result["status"] == "success"
        assert [sorted(command) for command in sent] == [["head_pose"]]

    def test_a_head_pose_records_the_target_the_next_body_turn_is_checked_against(self) -> None:
        """The pair (180, 0) is refused whether it arrives in one action or two.

        Two actions is the shape the tool surface produces: ``reachy_look``
        followed by ``reachy_body_turn``.
        """
        one_action, _sent = _native()
        together = one_action.send_action({"head_yaw": 180.0, "body_yaw": 0.0})

        two_actions, sent = _native()
        two_actions.send_action(_head_pose(head_yaw=180.0))
        sent.clear()
        apart = two_actions.send_action({"body_yaw": 0.0})

        assert together["status"] == "error"
        assert apart["status"] == "error"
        assert _body_on_wire(sent) is None


class TestAnUnknownTargetRefusesNothing:
    """Forgetting is the safe direction, so nothing is refused against a guess."""

    def test_a_body_turn_before_any_head_pose_is_not_refused(self) -> None:
        driver, sent = _native()

        result = driver.send_action({"body_yaw": _BEYOND_COUPLING})

        assert result["status"] == "success"
        assert _body_on_wire(sent) == pytest.approx(_BEYOND_COUPLING)

    @pytest.mark.parametrize(
        ("verb", "args"),
        [
            ("play_move", ("happy",)),
            ("wake_up", ()),
            ("goto_sleep", ()),
            ("set_motors", ("enabled",)),
        ],
    )
    def test_a_path_that_moves_the_head_without_a_pose_forgets_the_target(
        self, verb: str, args: tuple[Any, ...]
    ) -> None:
        driver, sent = _native()
        driver.send_action(_head_pose())
        assert driver.send_action({"body_yaw": _BEYOND_COUPLING})["status"] == "error"

        assert getattr(driver, verb)(*args)["status"] == "success"
        sent.clear()

        assert driver.send_action({"body_yaw": _BEYOND_COUPLING})["status"] == "success"
        assert _body_on_wire(sent) == pytest.approx(_BEYOND_COUPLING)

    def test_a_lone_body_yaw_the_envelope_is_given_no_target_for_is_unchecked(self) -> None:
        """The shared envelope's own half of that decision."""
        assert envelope_error({"body_yaw": _BEYOND_COUPLING}, "look") is None
        assert envelope_error({"body_yaw": _BEYOND_COUPLING}, "look", head_yaw_target=0.0) is not None

    def test_a_target_that_is_not_a_usable_number_is_named_rather_than_compared(self) -> None:
        """``abs(nan) <= 65`` is ``False``, so an unordered target must not read as travel."""
        reason = envelope_error({"body_yaw": 10.0}, "look", head_yaw_target=float("nan"))

        assert reason is not None
        assert "head_yaw_target" in reason


class TestTheActionsOwnHeadYawWinsOverTheRecordedTarget:
    """An action that says where the head is going is the stronger answer."""

    def test_a_pair_is_judged_on_its_own_two_values(self) -> None:
        driver, sent = _native()
        driver.send_action(_head_pose(head_yaw=180.0))
        sent.clear()

        result = driver.send_action({"head_yaw": 60.0, "body_yaw": 30.0})

        assert result["status"] == "success", "the action moves the head too, so the old target is stale"
        assert _body_on_wire(sent) == pytest.approx(30.0)

    def test_a_recorded_target_is_ignored_when_the_action_carries_a_head_yaw(self) -> None:
        assert envelope_error({"head_yaw": 10.0, "body_yaw": 0.0}, "look", head_yaw_target=180.0) is None
