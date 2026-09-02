"""The driver against a controller double: what reaches the wire, and what does not.

Every cell here drives :class:`~strands_robots.drivers.ur.URDriver` through the
same resolution path it uses on hardware - the doubles are installed as the
``rtde_control`` / ``rtde_receive`` importables (see ``conftest.py``) - so the
assertions are about the driver's own decisions rather than about a patch.

Two of them are the reason this driver exists in the shape it has:

* :meth:`TestConnecting.test_a_protective_stopped_controller_is_refused_before_control_opens`
  - a UR controller in a stop *accepts* an RTDE control connection and then
  performs no motion, so connecting must interrogate the receive side first.
* :meth:`TestCommanding.test_a_setpoint_the_controller_declines_is_not_reported_as_success`
  - ``servoJ`` returns a boolean, and a driver that ignored it would report
  success for motion that never happened.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import pytest

from strands_robots.drivers.ur import (
    JOINT_NAMES,
    SERVOJ_GAIN,
    SERVOJ_LOOKAHEAD_TIME,
    URDriver,
)
from tests.mocks.ur_rtde import (
    MEASURED_Q,
    MEASURED_TCP_POSE,
    MEASURED_WRENCH,
    FakeRTDE,
    json_of,
    text_of,
)

HOST = "192.168.1.10"


def _connected(fake: FakeRTDE, **kwargs: object) -> URDriver:
    """Build a driver on the doubles and assert it connected."""
    driver = URDriver(tool_name="ur5e", port=HOST, **kwargs)  # type: ignore[arg-type]
    assert driver.connect_eagerly() is None
    return driver


class TestConnecting:
    """Reachability, and the mode interrogation that decides usability."""

    def test_both_interfaces_open_against_a_running_controller(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        assert driver.is_connected
        assert fake_rtde.receive.host == HOST
        assert fake_rtde.control.host == HOST

    def test_connecting_twice_is_a_no_op_success(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        assert driver.connect_eagerly() is None
        assert len(fake_rtde.controls) == 1, "a second connect must not open a second interface"

    def test_a_protective_stopped_controller_is_refused_before_control_opens(
        self, monkeypatch: pytest.MonkeyPatch, fake_rtde: FakeRTDE
    ) -> None:
        """The receive side is interrogated first, and its answer stops the dial.

        A controller in a protective stop accepts an RTDE *control* connection
        and performs no motion afterwards. Opening it anyway would leave the
        driver reporting ``connected`` for an arm that cannot move.
        """
        original = fake_rtde.make_receive

        def stopped(host: str, frequency: float | None = None) -> object:
            interface = original(host, frequency)
            interface.safety_mode = 3  # PROTECTIVE_STOP
            return interface

        monkeypatch.setattr("rtde_receive.RTDEReceiveInterface", stopped)
        driver = URDriver(tool_name="ur5e", port=HOST)

        reason = driver.connect_eagerly()

        assert reason is not None
        assert "PROTECTIVE_STOP" in reason
        assert not driver.is_connected
        assert fake_rtde.controls == [], "the control interface must not be opened for an arm that cannot move"

    def test_an_unpowered_controller_is_refused_by_robot_mode(
        self, monkeypatch: pytest.MonkeyPatch, fake_rtde: FakeRTDE
    ) -> None:
        original = fake_rtde.make_receive

        def powered_off(host: str, frequency: float | None = None) -> object:
            interface = original(host, frequency)
            interface.robot_mode = 3  # POWER_OFF
            return interface

        monkeypatch.setattr("rtde_receive.RTDEReceiveInterface", powered_off)

        reason = URDriver(tool_name="ur5e", port=HOST).connect_eagerly()

        assert reason is not None
        assert "POWER_OFF" in reason and "RUNNING" in reason

    def test_no_address_is_reported_rather_than_dialled(self, fake_rtde: FakeRTDE) -> None:
        reason = URDriver(tool_name="ur5e").connect_eagerly()
        assert reason is not None
        assert "port=" in reason
        assert fake_rtde.receives == []

    def test_a_missing_sdk_leaves_the_driver_usable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off-SDK the driver reports a reason; it does not raise through the tool surface."""
        import sys

        monkeypatch.setitem(sys.modules, "rtde_control", None)
        monkeypatch.setitem(sys.modules, "rtde_receive", None)
        driver = URDriver(tool_name="ur5e", port=HOST)

        reason = driver.connect_eagerly()

        assert reason is not None
        assert "ur_rtde" in reason
        assert not driver.is_connected
        assert driver.send_action({"elbow_joint": 1.4})["status"] == "error"


class TestCommanding:
    """``send_action`` -> ``servoJ``, and every gate in front of it."""

    def test_the_setpoint_reaches_servoj_in_wire_order(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde, control_frequency=50.0)
        target = MEASURED_Q[2] + 0.01

        envelope = driver.send_action({"elbow_joint": target})

        assert envelope["status"] == "success", text_of(envelope)
        sent, params = fake_rtde.control.servoj_calls[-1]
        assert sent == [MEASURED_Q[0], MEASURED_Q[1], target, MEASURED_Q[3], MEASURED_Q[4], MEASURED_Q[5]]
        assert params[2] == pytest.approx(1 / 50.0), "servoJ's time argument is the control period"
        assert params[3] == SERVOJ_LOOKAHEAD_TIME
        assert params[4] == SERVOJ_GAIN

    def test_the_envelope_names_the_joints_it_commanded(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        body = json_of(driver.send_action({"elbow_joint": MEASURED_Q[2] + 0.001}))
        assert set(body["joints"]) == set(JOINT_NAMES)
        assert body["robot"] == "ur5e"

    def test_a_setpoint_the_controller_declines_is_not_reported_as_success(self, fake_rtde: FakeRTDE) -> None:
        """``servoJ`` returning False means the arm is not tracking the command.

        The mode gate cannot cover this on its own: an arm can be running, in a
        safety mode that moves, and still decline a particular write.
        """
        driver = _connected(fake_rtde)
        fake_rtde.control.accepts = False

        envelope = driver.send_action({"elbow_joint": MEASURED_Q[2] + 0.001})

        assert envelope["status"] == "error"
        assert "declined" in text_of(envelope)

    def test_a_stop_that_lands_after_connecting_refuses_the_next_write(self, fake_rtde: FakeRTDE) -> None:
        """The mode is re-read per write, not cached from connect time."""
        driver = _connected(fake_rtde)
        fake_rtde.receive.safety_mode = 5  # SAFEGUARD_STOP

        envelope = driver.send_action({"elbow_joint": MEASURED_Q[2] + 0.001})

        assert envelope["status"] == "error"
        assert "SAFEGUARD_STOP" in text_of(envelope)
        assert fake_rtde.control.servoj_calls == [], "nothing may reach the wire behind a stop"

    def test_commanding_before_connecting_is_refused(self, fake_rtde: FakeRTDE) -> None:
        envelope = URDriver(tool_name="ur5e", port=HOST).send_action({"elbow_joint": 1.4})
        assert envelope["status"] == "error"
        assert "not connected" in text_of(envelope)

    def test_another_robots_name_is_refused_rather_than_applied_here(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        envelope = driver.send_action({"elbow_joint": 1.4}, robot_name="ur10e")
        assert envelope["status"] == "error"
        assert "ur10e" in text_of(envelope)
        assert fake_rtde.control.servoj_calls == []


class TestReadingState:
    """The three quantities the arm reports, in one round trip."""

    def test_state_carries_joints_tcp_pose_and_wrench(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        body = json_of(driver.state())
        assert body["joints"] == dict(zip(JOINT_NAMES, MEASURED_Q, strict=True))
        assert body["tcp_pose"] == list(MEASURED_TCP_POSE)
        assert body["wrench"] == list(MEASURED_WRENCH)
        assert body["robot_mode"] == "RUNNING"
        assert body["safety_mode"] == "NORMAL"

    def test_state_before_connecting_is_refused(self) -> None:
        envelope = URDriver(tool_name="ur5e", port=HOST).state()
        assert envelope["status"] == "error"
        assert "not connected" in text_of(envelope)

    def test_the_mesh_joint_read_reports_the_six_joints(self, fake_rtde: FakeRTDE) -> None:
        """``get_observation`` plus ``is_connected`` is what puts joints on the mesh."""
        driver = _connected(fake_rtde)
        assert driver.get_observation() == dict(zip(JOINT_NAMES, MEASURED_Q, strict=True))

    def test_an_arm_that_never_connected_publishes_no_joints(self) -> None:
        assert URDriver(tool_name="ur5e", port=HOST).get_observation() == {}

    def test_status_reports_the_controller_vocabulary(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        status = json_of(asyncio.run(driver.get_status()))
        assert status["connected"] is True
        assert status["robot_mode"] == "RUNNING"
        assert status["model"] == "ur5e"
        assert status["host"] == HOST

    def test_a_controller_reporting_the_wrong_axis_count_is_refused(self, fake_rtde: FakeRTDE) -> None:
        """A seven-axis answer would mis-index every joint after the extra one."""
        driver = _connected(fake_rtde)
        fake_rtde.receive.q = [0.0] * 7

        envelope = driver.send_action({"elbow_joint": 0.1})

        assert envelope["status"] == "error"
        assert "six-axis" in text_of(envelope)


class TestRollingOutAPolicy:
    """``run_policy`` streams setpoints and reports why it stopped."""

    def test_a_step_budget_commands_exactly_that_many_setpoints(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde, control_frequency=200.0)

        def hold(observation: dict[str, float]) -> dict[str, float]:
            return {"elbow_joint": observation["elbow_joint"] + 0.0005}

        assert driver.run_policy(hold, n_steps=3)["status"] == "success"
        _wait_for_exit(driver)

        snapshot = json_of(driver.get_task_status())
        assert snapshot["steps"] == 3
        assert snapshot["exit_reason"] == "n_steps"
        assert len(fake_rtde.control.servoj_calls) == 3

    def test_a_policy_the_arm_refuses_ends_the_rollout_with_that_reason(self, fake_rtde: FakeRTDE) -> None:
        """A refusal is the exit reason, not a step silently skipped."""
        driver = _connected(fake_rtde, control_frequency=200.0)

        def too_far(observation: dict[str, float]) -> dict[str, float]:
            return {"shoulder_pan_joint": observation["shoulder_pan_joint"] + 1.0}

        assert driver.run_policy(too_far, n_steps=5)["status"] == "success"
        _wait_for_exit(driver)

        snapshot = json_of(driver.get_task_status())
        assert snapshot["exit_reason"] == "refused"
        assert "rad/s ceiling" in snapshot["refusal"]
        assert snapshot["steps"] == 0

    def test_a_policy_returning_no_action_dict_ends_the_rollout(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde, control_frequency=200.0)

        policy: Any = lambda observation: None  # noqa: E731 - one of the shapes run_policy admits
        assert driver.run_policy(policy, n_steps=2)["status"] == "success"
        _wait_for_exit(driver)

        snapshot = json_of(driver.get_task_status())
        assert snapshot["exit_reason"] == "policy"
        assert "action dict" in snapshot["refusal"]

    def test_a_built_policys_get_actions_sync_is_the_step(self, fake_rtde: FakeRTDE) -> None:
        """A ``Policy`` is driven through its own sync entry point, with the instruction."""
        driver = _connected(fake_rtde, control_frequency=200.0)
        seen: list[str] = []

        class _Policy:
            def get_actions_sync(self, observation: dict[str, float], instruction: str) -> dict[str, float]:
                seen.append(instruction)
                return {"elbow_joint": observation["elbow_joint"]}

        policy: Any = _Policy()
        assert driver.run_policy(policy, instruction="hold still", n_steps=1)["status"] == "success"
        _wait_for_exit(driver)

        assert seen == ["hold still"]

    def test_an_object_that_is_no_kind_of_policy_is_refused(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        envelope = driver.run_policy(object())  # type: ignore[arg-type]
        assert envelope["status"] == "error"
        assert "get_actions_sync" in text_of(envelope)

    def test_a_rollout_needs_a_connected_arm(self, fake_rtde: FakeRTDE) -> None:
        envelope = URDriver(tool_name="ur5e", port=HOST).run_policy(lambda observation: {})
        assert envelope["status"] == "error"
        assert "not connected" in text_of(envelope)

    @pytest.mark.parametrize(
        ("kwargs", "quoted"),
        [({"duration": 0}, "duration"), ({"n_steps": 0}, "n_steps"), ({"n_steps": 1.5}, "n_steps")],
    )
    def test_an_unusable_budget_is_refused(self, fake_rtde: FakeRTDE, kwargs: dict[str, object], quoted: str) -> None:
        """A zero budget would exit instantly inside a success envelope."""
        driver = _connected(fake_rtde)
        envelope = driver.run_policy(lambda observation: {}, **kwargs)  # type: ignore[arg-type]
        assert envelope["status"] == "error"
        assert quoted in text_of(envelope)


class TestStopping:
    """The halt paths, and what they report."""

    def test_stop_task_decelerates_the_arm_and_reports_the_steps(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde, control_frequency=200.0)
        driver.run_policy(lambda observation: {"elbow_joint": observation["elbow_joint"]}, n_steps=2)
        _wait_for_exit(driver)

        body = json_of(driver.stop_task())

        assert body["stopped"] is True
        assert body["steps"] == 2
        assert fake_rtde.control.servo_stops == 1

    def test_stopping_a_disconnected_arm_is_refused(self) -> None:
        envelope = URDriver(tool_name="ur5e", port=HOST).stop_task()
        assert envelope["status"] == "error"
        assert "not connected" in text_of(envelope)

    def test_cleanup_stops_the_arm_and_releases_both_interfaces(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)

        driver.cleanup()

        assert fake_rtde.control.servo_stops == 1
        assert fake_rtde.control.disconnected
        assert fake_rtde.receive.disconnected
        assert not driver.is_connected

    def test_cleanup_is_idempotent(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde)
        driver.cleanup()
        driver.cleanup()
        assert fake_rtde.control.servo_stops == 1


class TestTheConstructorRefusesAnUnusableConfiguration:
    """Raised at construction: a rate the loop cannot pace on is not a connection."""

    @pytest.mark.parametrize("frequency", [0, -5.0, float("nan"), float("inf"), True])
    def test_a_control_frequency_off_the_domain_is_refused(self, frequency: object) -> None:
        with pytest.raises(ValueError, match="control_frequency"):
            URDriver(tool_name="ur5e", port=HOST, control_frequency=frequency)  # type: ignore[arg-type]

    def test_a_port_suffix_that_is_not_the_rtde_port_is_refused(self) -> None:
        """RTDE's port is fixed, so a different one is a caller mistake worth naming."""
        with pytest.raises(ValueError, match="30004"):
            URDriver(tool_name="ur5e", port=f"{HOST}:502")

    def test_the_rtde_port_suffix_is_accepted_and_stripped(self, fake_rtde: FakeRTDE) -> None:
        driver = URDriver(tool_name="ur5e", port=f"{HOST}:30004")
        assert driver.connect_eagerly() is None
        assert fake_rtde.receive.host == HOST


class TestTheStepGateIsAnchoredOnTheCommandedTrajectory:
    """The gate bounds commanded velocity, not tracking error.

    ``servoJ`` setpoints form a trajectory the controller interpolates toward, so
    the arm is always behind the setpoint it was last given. A gate that measured
    each step from the *measured* pose would charge the step for that lag and
    refuse a policy the arm can easily follow - measured in simulation at 50 Hz,
    288 of 420 setpoints from a trajectory whose increments were a quarter of the
    ceiling. The doubles here hold their measured pose still, which is that lag
    taken to its limit.
    """

    #: 50 Hz, so a UR5e joint may travel pi * 0.02 = 0.0628 rad per setpoint.
    CONTROL_HZ = 50.0
    INCREMENT = 0.02
    STEPS = 20

    def test_a_stream_of_reachable_increments_survives_a_lagging_arm(self, fake_rtde: FakeRTDE) -> None:
        driver = _connected(fake_rtde, control_frequency=self.CONTROL_HZ)

        for index in range(1, self.STEPS + 1):
            target = MEASURED_Q[2] + self.INCREMENT * index
            envelope = driver.send_action({"elbow_joint": target})
            assert envelope["status"] == "success", (index, text_of(envelope))

        assert len(fake_rtde.control.servoj_calls) == self.STEPS
        travelled = self.INCREMENT * self.STEPS
        assert fake_rtde.control.servoj_calls[-1][0][2] == pytest.approx(MEASURED_Q[2] + travelled)
        assert travelled > math.pi / self.CONTROL_HZ, "the total must exceed one step's ceiling to mean anything"

    def test_the_anchor_is_not_a_way_around_the_ceiling(self, fake_rtde: FakeRTDE) -> None:
        """Each individual step is still sized against the ceiling."""
        driver = _connected(fake_rtde, control_frequency=self.CONTROL_HZ)
        assert driver.send_action({"elbow_joint": MEASURED_Q[2] + self.INCREMENT})["status"] == "success"

        envelope = driver.send_action({"elbow_joint": MEASURED_Q[2] + self.INCREMENT + 0.3})

        assert envelope["status"] == "error"
        assert "rad/s ceiling" in text_of(envelope)

    def test_a_stop_re_anchors_the_gate_on_the_measured_pose(self, fake_rtde: FakeRTDE) -> None:
        """The arm may have been moved while the stream was down.

        Resuming from a stale setpoint would command a jump from a pose the arm
        no longer holds, so a halt drops the anchor and the next setpoint is
        sized from what the controller actually reports.
        """
        driver = _connected(fake_rtde, control_frequency=self.CONTROL_HZ)
        for index in range(1, 11):
            driver.send_action({"elbow_joint": MEASURED_Q[2] + self.INCREMENT * index})
        stale = json_of(driver.get_task_status())  # no rollout; the anchor is send_action's
        assert stale["running"] is False

        assert driver.stop_task()["status"] == "success"

        # 0.19 rad away from the stale setpoint, but 0.01 from where the arm is.
        envelope = driver.send_action({"elbow_joint": MEASURED_Q[2] + 0.01})

        assert envelope["status"] == "success", text_of(envelope)

    def _stream_ten_increments(self, driver: URDriver) -> float:
        """Advance the anchor 0.20 rad past the measured pose, and return it."""
        for index in range(1, 11):
            envelope = driver.send_action({"elbow_joint": MEASURED_Q[2] + self.INCREMENT * index})
            assert envelope["status"] == "success", (index, text_of(envelope))
        return MEASURED_Q[2] + self.INCREMENT * 10

    @pytest.mark.parametrize(
        ("attribute", "value", "named"),
        [
            ("safety_mode", 3, "PROTECTIVE_STOP"),
            ("safety_mode", 5, "SAFEGUARD_STOP"),
            ("robot_mode", 3, "POWER_OFF"),
        ],
    )
    def test_a_controller_initiated_halt_re_anchors_the_gate_on_the_measured_pose(
        self,
        fake_rtde: FakeRTDE,
        attribute: str,
        value: int,
        named: str,
    ) -> None:
        """The halt the driver did not order is the one the arm gets jogged after.

        A protective stop ends the stream at the mode gate rather than through
        any halt verb, and clearing it from the pendant is exactly when an
        operator jogs the arm. If the anchor outlived that halt, a setpoint one
        increment from the *stale* anchor would pass the speed gate and hand
        ``servoJ`` a 0.21 rad jump from the pose the arm actually holds.
        """
        driver = _connected(fake_rtde, control_frequency=self.CONTROL_HZ)
        anchor = self._stream_ten_increments(driver)

        setattr(fake_rtde.receive, attribute, value)
        halted = driver.send_action({"elbow_joint": anchor + self.INCREMENT})
        assert halted["status"] == "error"
        assert named in text_of(halted)

        # The operator clears the stop, having jogged the arm back to where the
        # controller reports it. One increment past the stale anchor is 0.21 rad
        # from there - 3.3x what a 50 Hz period allows.
        setattr(fake_rtde.receive, attribute, 1 if attribute == "safety_mode" else 7)
        wrote = len(fake_rtde.control.servoj_calls)

        envelope = driver.send_action({"elbow_joint": anchor + 0.01})

        assert envelope["status"] == "error", "a jump from the stale anchor must not be admitted"
        assert "rad/s ceiling" in text_of(envelope)
        assert len(fake_rtde.control.servoj_calls) == wrote, "nothing may reach the wire"
        # And the gate is now sized from measurement, not merely closed.
        assert driver.send_action({"elbow_joint": MEASURED_Q[2] + 0.01})["status"] == "success"

    def test_a_rollout_that_ends_on_its_own_budget_drops_the_anchor(self, fake_rtde: FakeRTDE) -> None:
        """A rollout leaves the arm at rest however it exited.

        ``stop_task`` is not the only way a stream ends - a rollout that spends
        its step budget ends one too, and nothing calls a halt verb on that path.
        The arm may be moved before the next setpoint either way.
        """
        driver = _connected(fake_rtde, control_frequency=self.CONTROL_HZ)
        steps = {"taken": 0}

        def advance(_observation: dict[str, float]) -> dict[str, float]:
            steps["taken"] += 1
            return {"elbow_joint": MEASURED_Q[2] + self.INCREMENT * steps["taken"]}

        assert driver.run_policy(advance, n_steps=10)["status"] == "success"
        _wait_for_exit(driver)
        snapshot = json_of(driver.get_task_status())
        assert snapshot["exit_reason"] == "n_steps", snapshot
        anchor = MEASURED_Q[2] + self.INCREMENT * 10
        assert fake_rtde.control.servoj_calls[-1][0][2] == pytest.approx(anchor)
        wrote = len(fake_rtde.control.servoj_calls)

        envelope = driver.send_action({"elbow_joint": anchor + 0.01})

        assert envelope["status"] == "error", "the rollout's last setpoint is not a live anchor"
        assert "rad/s ceiling" in text_of(envelope)
        assert len(fake_rtde.control.servoj_calls) == wrote, "nothing may reach the wire"

    def test_a_value_gate_refusal_leaves_the_anchor_standing(self, fake_rtde: FakeRTDE) -> None:
        """A refused setpoint is not a broken stream.

        The arm is still running and still tracking toward the anchor, so this
        refusal must not re-anchor on measurement - doing so would charge the
        next step for accumulated tracking lag, which is the refusal storm the
        commanded anchor exists to prevent. This is the boundary of the halt
        handling above, and the reason it keys on the mode gate rather than on
        any refusal.
        """
        driver = _connected(fake_rtde, control_frequency=self.CONTROL_HZ)
        anchor = self._stream_ten_increments(driver)

        too_far = driver.send_action({"elbow_joint": anchor + 0.3})
        assert too_far["status"] == "error"
        assert "rad/s ceiling" in text_of(too_far)

        # The stream resumes from the anchor, 0.20 rad ahead of the still pose
        # the doubles report - which a measured-pose anchor would refuse.
        envelope = driver.send_action({"elbow_joint": anchor + self.INCREMENT})

        assert envelope["status"] == "success", text_of(envelope)
        assert fake_rtde.control.servoj_calls[-1][0][2] == pytest.approx(anchor + self.INCREMENT)


def _wait_for_exit(driver: URDriver, timeout: float = 5.0) -> None:
    """Block until the rollout thread has finished, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not json_of(driver.get_task_status())["running"]:
            return
        time.sleep(0.01)
    raise AssertionError("the rollout did not finish within its budget")
