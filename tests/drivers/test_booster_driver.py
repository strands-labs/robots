"""The Booster T1 native driver: the balance rule, the write gate, the wire frame.

Everything here runs with no T1 attached. One double stands in for
``booster_robotics_sdk_python`` and it is faithful in the one respect the tests
depend on: it **records** the ``LowCmd`` it is handed rather than discarding it.
A double that dropped the frame could not tell "the driver zero-gained the legs"
from "the driver sent nothing", and that distinction is the entire subject of
:class:`TestTheFrameLeavesBalanceToTheRobot` - the property that keeps a 1.2 m
biped standing.

The one cell that needs the *real* SDK says so with ``importorskip``: the joint
table is graded against the vendor's own ``JointIndex`` enum, and a stub would
compare the driver's constant against a copy of itself.
"""

from __future__ import annotations

import asyncio
import math
import sys
import threading
from typing import Any

import pytest
from strands.types.tools import ToolUse

from strands_robots.drivers import get_native_driver_class
from strands_robots.drivers.base import HardwareDriver, missing_driver_members
from strands_robots.drivers.booster import (
    BOOSTER_JOINT_INDEX,
    CMD_TYPE_STATE_FIELD,
    FALL_STATE_NAMES,
    FALL_STATE_READY,
    POSITION_MODE,
    ROBOT_MODES,
    UPPER_BODY_KD,
    UPPER_BODY_KP,
    UPPER_BODY_SLOTS,
    BoosterDriver,
    build_frame,
    parse_low_state,
    resolve_targets,
)
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK
from strands_robots.utils import MAX_DDS_DOMAIN_ID

# The width a T1 reports. Every frame the driver builds is bounded by what the
# robot itself said, so the fixtures carry a full-width state rather than a
# constant the driver could have guessed.
_WIDTH = len(BOOSTER_JOINT_INDEX)


# --------------------------------------------------------------------------- #
# The SDK double.                                                             #
# --------------------------------------------------------------------------- #


class _FakeMotorCmd:
    """One motor slot, holding whatever the driver wrote into it."""

    def __init__(self) -> None:
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.mode = 0


class _FakeLowCmd:
    """``LowCmd``'s resize/index surface, as the compiled core exposes it."""

    def __init__(self) -> None:
        self.cmd_type: Any = None
        self._motors: list[_FakeMotorCmd] = []

    def resize_motor_cmd(self, size: int) -> None:
        self._motors = [_FakeMotorCmd() for _ in range(size)]

    def motor_cmd_at(self, index: int) -> _FakeMotorCmd:
        return self._motors[index]

    def motor_cmd_size(self) -> int:
        return len(self._motors)


class _FakeMotorState:
    def __init__(self, q: float) -> None:
        self.q = q
        self.dq = 0.0
        self.tau_est = 0.0
        self.temperature = 30.0


class _FakeLowState:
    """A ``LowState`` carrying a different value per convention.

    The two arrays differ so a test can tell which one the driver read - the
    point of :class:`TestTheHoldPositionComesFromTheDeclaredConvention`.
    """

    def __init__(self, parallel: list[float], serial: list[float]) -> None:
        self.motor_state_parallel = [_FakeMotorState(q) for q in parallel]
        self.motor_state_serial = [_FakeMotorState(q) for q in serial]
        self.imu_state = _FakeImu()


class _FakeImu:
    rpy = (0.01, 0.02, 0.03)
    gyro = (0.1, 0.2, 0.3)
    acc = (0.0, 0.0, 9.81)


class _FakeBattery:
    def __init__(self, soc: float) -> None:
        self.soc = soc
        self.voltage = 48.2
        self.current = -3.1


class _FakeFall:
    """``FallDownState`` carries an enum-ish member, so the double does too."""

    def __init__(self, code: int) -> None:
        self.fall_down_state = type("State", (), {"value": code})()
        self.is_recovery_available = True


class _FakeLocoClient:
    """Records every high-level call, and can be told to refuse one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.refuse: set[str] = set()

    def _record(self, name: str, *args: Any) -> None:
        if name in self.refuse:
            raise RuntimeError(f"code = 100, {name} timed out")
        self.calls.append((name, args))

    def Init(self) -> None:  # noqa: N802 - the SDK's own spelling
        self._record("Init")

    def InitWithName(self, robot_name: str) -> None:  # noqa: N802
        self._record("InitWithName", robot_name)

    def UpperBodyCustomControl(self, on: bool) -> None:  # noqa: N802
        self._record("UpperBodyCustomControl", on)

    def MoveCommand(self, vx: float, vy: float, vyaw: float) -> None:  # noqa: N802
        self._record("MoveCommand", vx, vy, vyaw)

    def RotateHead(self, pitch: float, yaw: float) -> None:  # noqa: N802
        self._record("RotateHead", pitch, yaw)

    def ChangeMode(self, mode: Any) -> None:  # noqa: N802
        self._record("ChangeMode", mode)

    def GetMode(self) -> Any:  # noqa: N802
        self._record("GetMode")
        return type("Response", (), {"mode": type("Mode", (), {"name": "kCustom"})()})()


class _FakeChannel:
    def __init__(self, handler: Any = None) -> None:
        self.handler = handler
        self.written: list[_FakeLowCmd] = []
        self.opened_as: str | None = None
        self.closed = False
        self.accept = True

    def InitChannel(self) -> None:  # noqa: N802
        self.opened_as = "default"

    def InitChannelWithName(self, robot_name: str) -> None:  # noqa: N802
        self.opened_as = robot_name

    def CloseChannel(self) -> None:  # noqa: N802
        self.closed = True

    def Write(self, msg: _FakeLowCmd) -> bool:  # noqa: N802
        self.written.append(msg)
        return self.accept


class _FakeLowCmdType:
    """``LowCmdType``'s two members, at the vendor's own values."""

    PARALLEL = 0
    SERIAL = 1


class _FakeRobotMode:
    """``RobotMode``'s members, at the vendor's own values."""

    kUnknown = -1
    kDamping = 0
    kPrepare = 1
    kWalking = 2
    kCustom = 3
    kSoccer = 4


class _FakeSdk:
    """The module surface the driver reaches for, and nothing more."""

    LowCmd = _FakeLowCmd
    LowCmdType = _FakeLowCmdType
    RobotMode = _FakeRobotMode

    def __init__(self) -> None:
        self.client = _FakeLocoClient()
        self.subscriber: _FakeChannel | None = None
        self.publisher = _FakeChannel()
        self.battery: _FakeChannel | None = None
        self.fall: _FakeChannel | None = None
        self.factory_init: tuple[int, str] | None = None
        sdk = self

        class _Factory:
            @staticmethod
            def Instance() -> Any:  # noqa: N802
                return _Factory()

            def Init(self, domain_id: int, ip: str) -> None:  # noqa: N802
                sdk.factory_init = (domain_id, ip)

        self.ChannelFactory = _Factory

    def B1LocoClient(self) -> _FakeLocoClient:  # noqa: N802
        return self.client

    def B1LowStateSubscriber(self, handler: Any) -> _FakeChannel:  # noqa: N802
        self.subscriber = _FakeChannel(handler)
        return self.subscriber

    def B1LowCmdPublisher(self) -> _FakeChannel:  # noqa: N802
        return self.publisher

    def B1BatteryStateSubscriber(self, handler: Any) -> _FakeChannel:  # noqa: N802
        self.battery = _FakeChannel(handler)
        return self.battery

    def B1FallDownStateSubscriber(self, handler: Any) -> _FakeChannel:  # noqa: N802
        self.fall = _FakeChannel(handler)
        return self.fall


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeSdk:
    """Install the SDK double under the name the driver imports."""
    fake = _FakeSdk()
    monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", fake)
    return fake


def _tool_use(action: str) -> ToolUse:
    """One agent invocation, shaped as the ``ToolUse`` TypedDict requires."""
    return {"toolUseId": "t1", "name": "booster_t1", "input": {"action": action}}


def _live_driver(sdk: _FakeSdk, *, cmd_type: str = "parallel", enable: bool = True) -> BoosterDriver:
    """A connected driver holding one observed frame, ready to write."""
    driver = BoosterDriver(cmd_type=cmd_type)
    assert driver.connect_eagerly() is None
    if enable:
        assert driver.enable_upper_body(True)["status"] == "success"
    parallel = [round(0.1 * slot, 3) for slot in range(_WIDTH)]
    serial = [round(-0.1 * slot, 3) for slot in range(_WIDTH)]
    assert sdk.subscriber is not None
    sdk.subscriber.handler(_FakeLowState(parallel, serial))
    return driver


# --------------------------------------------------------------------------- #
# The safety rule.                                                            #
# --------------------------------------------------------------------------- #


class TestTheFrameLeavesBalanceToTheRobot:
    """A frame may only put stiffness on the eight joints the T1 lends out.

    The onboard whole-body controller keeps the T1 standing. Gain on a leg slot
    fights it, and the publish succeeds either way - so this is the property
    with nothing downstream to catch it.
    """

    def test_every_slot_gets_the_gain_its_owner_implies(self) -> None:
        """Commanded, held, and zero-gained: one row per slot, no exceptions."""
        held = [round(0.1 * slot, 3) for slot in range(_WIDTH)]
        commanded = BOOSTER_JOINT_INDEX["left_shoulder_pitch"]
        frame = build_frame({commanded: {"q": -0.4, "kp": UPPER_BODY_KP, "kd": UPPER_BODY_KD}}, held)

        assert len(frame) == len(held)
        for slot, motor in enumerate(frame):
            if slot == commanded:
                expected = (-0.4, UPPER_BODY_KP, UPPER_BODY_KD)
            elif slot in UPPER_BODY_SLOTS:
                expected = (held[slot], UPPER_BODY_KP, UPPER_BODY_KD)
            else:
                expected = (0.0, 0.0, 0.0)
            assert (motor["q"], motor["kp"], motor["kd"]) == expected, (
                f"slot {slot} carries {motor}, expected q/kp/kd {expected}"
            )
            assert motor["mode"] == POSITION_MODE

    def test_no_joint_outside_the_upper_body_can_be_given_a_gain(self) -> None:
        """The refusal is in the resolver, so no caller reaches build_frame with one."""
        outside = sorted(name for name, slot in BOOSTER_JOINT_INDEX.items() if slot not in UPPER_BODY_SLOTS)
        assert outside, "the fixture is vacuous if every joint is upper body"
        for name in outside:
            reason = resolve_targets({name: 0.1})
            assert isinstance(reason, str) and name in reason
            assert ("rotate_head()" if name.startswith("head_") else "move()") in reason


# --------------------------------------------------------------------------- #
# The write gate and the action contract.                                     #
# --------------------------------------------------------------------------- #


class TestSendActionRefusesRatherThanGuesses:
    """Each refusal names the state that has to change, not just the failure."""

    def test_an_unenabled_driver_names_the_call_that_opens_the_gate(self, sdk: _FakeSdk) -> None:
        """A frame sent without UpperBodyCustomControl is ignored and reported as sent."""
        driver = _live_driver(sdk, enable=False)
        result = driver.send_action({"left_shoulder_pitch": 0.2})
        assert result["status"] == "error"
        assert "enable_upper_body()" in result["content"][0]["text"]
        assert not sdk.publisher.written

    def test_a_driver_with_no_observed_frame_will_not_invent_a_hold(self, sdk: _FakeSdk) -> None:
        """Frame width and hold positions both come from the robot's own report."""
        driver = BoosterDriver()
        assert driver.connect_eagerly() is None
        assert driver.enable_upper_body(True)["status"] == "success"
        result = driver.send_action({"left_shoulder_pitch": 0.2})
        assert result["status"] == "error"
        assert "LowState" in result["content"][0]["text"]
        assert not sdk.publisher.written

    def test_an_unconnected_driver_refuses_before_touching_the_sdk(self) -> None:
        """No SDK installed at all is the same answer: a named refusal."""
        result = BoosterDriver().send_action({"left_shoulder_pitch": 0.2})
        assert result["status"] == "error"
        assert "connect_eagerly()" in result["content"][0]["text"]

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            pytest.param({}, "non-empty", id="empty"),
            pytest.param({"left_elbow": 0.1}, "unknown joint", id="unknown-name"),
            pytest.param({"left_knee_pitch": 0.1}, "onboard whole-body", id="leg-slot"),
            pytest.param({"head_yaw": 0.1}, "rotate_head()", id="head-slot"),
            pytest.param({"left_shoulder_pitch": {"kp": 10.0}}, "no 'q'", id="mapping-without-q"),
            pytest.param({"left_shoulder_pitch": float("nan")}, "finite", id="nan-target"),
            pytest.param({"left_shoulder_pitch": {"q": 0.1, "kp": float("inf")}}, "finite", id="infinite-gain"),
            pytest.param({"left_shoulder_pitch": "0.1"}, "finite", id="string-target"),
        ],
    )
    def test_the_resolver_names_the_joint_and_the_reason(self, action: dict[str, Any], expected: str) -> None:
        """One table for the whole action contract; each row is one way to be wrong."""
        reason = resolve_targets(action)
        assert isinstance(reason, str), f"{action} should have been refused, got {reason}"
        assert expected in reason


class TestSendActionReachesTheWire:
    """What the driver actually publishes, read back off the recorded frame."""

    def test_the_published_frame_carries_the_target_and_the_hold(self, sdk: _FakeSdk) -> None:
        """One write, full width, caller's gains honoured, legs at zero gain."""
        driver = _live_driver(sdk)
        result = driver.send_action(
            {"left_shoulder_pitch": -0.4, "right_elbow_pitch": {"q": 0.7, "kp": 25.0, "kd": 1.5}}
        )
        assert result["status"] == "success", result
        payload = result["content"][0]["json"]
        assert payload["joints"] == ["left_shoulder_pitch", "right_elbow_pitch"]
        assert payload["frame_width"] == _WIDTH
        assert payload["cmd_type"] == "parallel"

        assert len(sdk.publisher.written) == 1
        cmd = sdk.publisher.written[0]
        assert cmd.cmd_type == _FakeLowCmdType.PARALLEL
        assert cmd.motor_cmd_size() == _WIDTH

        left = cmd.motor_cmd_at(BOOSTER_JOINT_INDEX["left_shoulder_pitch"])
        assert (left.q, left.kp, left.kd, left.mode) == (-0.4, UPPER_BODY_KP, UPPER_BODY_KD, POSITION_MODE)
        right = cmd.motor_cmd_at(BOOSTER_JOINT_INDEX["right_elbow_pitch"])
        assert (right.q, right.kp, right.kd) == (0.7, 25.0, 1.5)
        held = cmd.motor_cmd_at(BOOSTER_JOINT_INDEX["left_elbow_yaw"])
        assert (held.q, held.kp) == (0.5, UPPER_BODY_KP), "an uncommanded arm joint holds its observed q"
        knee = cmd.motor_cmd_at(BOOSTER_JOINT_INDEX["left_knee_pitch"])
        assert (knee.q, knee.kp, knee.kd) == (0.0, 0.0, 0.0), "a leg slot must carry no gain"

    def test_a_publisher_that_rejects_the_frame_is_reported(self, sdk: _FakeSdk) -> None:
        """``Write`` returning False is a refusal, not a success with a caveat."""
        driver = _live_driver(sdk)
        sdk.publisher.accept = False
        result = driver.send_action({"left_shoulder_pitch": 0.1})
        assert result["status"] == "error"
        assert "rejected" in result["content"][0]["text"]


class TestTheHoldPositionComesFromTheDeclaredConvention:
    """``cmd_type`` and the state array it holds from must be the same convention.

    The parallel and serial arrays are two spellings of the same motors. A frame
    declaring one while holding positions read from the other sends targets in
    the wrong frame of reference - and every value is plausible, so nothing
    downstream notices.
    """

    @pytest.mark.parametrize("cmd_type", sorted(CMD_TYPE_STATE_FIELD))
    def test_each_convention_holds_from_its_own_array(self, sdk: _FakeSdk, cmd_type: str) -> None:
        """Signs differ between the doubles' arrays, so the source is decidable."""
        driver = _live_driver(sdk, cmd_type=cmd_type)
        assert driver.send_action({"left_shoulder_pitch": 0.0})["status"] == "success"
        held = sdk.publisher.written[-1].motor_cmd_at(BOOSTER_JOINT_INDEX["left_elbow_yaw"])
        assert held.q == (0.5 if cmd_type == "parallel" else -0.5)

    def test_the_snapshot_reads_the_field_it_was_asked_for(self) -> None:
        """:func:`parse_low_state` is the one place the convention is applied."""
        state = _FakeLowState([1.0] * _WIDTH, [-1.0] * _WIDTH)
        assert parse_low_state(state, "motor_state_parallel")["joints"][0] == 1.0
        assert parse_low_state(state, "motor_state_serial")["joints"][0] == -1.0
        assert parse_low_state(state, "motor_state_parallel")["imu"]["rpy"] == [0.01, 0.02, 0.03]


# --------------------------------------------------------------------------- #
# Lifecycle, reads and the seam.                                              #
# --------------------------------------------------------------------------- #


class TestTheDriverSurvivesABadFrameAndABadRobot:
    """The SDK's own thread runs the callback, so it must never raise there."""

    def test_a_non_finite_joint_is_dropped_rather_than_cached(self, sdk: _FakeSdk) -> None:
        """A NaN reaching the cache becomes a NaN hold target one frame later."""
        driver = _live_driver(sdk)
        good = driver.read_state()["joints"]
        assert sdk.subscriber is not None
        sdk.subscriber.handler(_FakeLowState([math.nan] * _WIDTH, [math.nan] * _WIDTH))
        assert driver.read_state()["joints"] == good

    def test_an_unreadable_frame_leaves_the_subscription_alive(self, sdk: _FakeSdk) -> None:
        """A frame missing the array is logged, not raised into the SDK thread."""
        driver = _live_driver(sdk)
        assert sdk.subscriber is not None
        sdk.subscriber.handler(object())
        assert driver.read_state()["joints"], "the previous good frame is still cached"

    def test_a_partial_channel_set_is_released(self, sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch) -> None:
        """A publisher that will not open must not leave a live subscriber behind."""

        def refuse(self: _FakeChannel) -> None:
            raise RuntimeError("channel busy")

        monkeypatch.setattr(_FakeChannel, "InitChannel", refuse)
        driver = BoosterDriver()
        reason = driver.connect_eagerly()
        assert reason is not None and "did not open" in reason
        assert not driver.is_connected
        assert sdk.subscriber is not None and sdk.subscriber.closed

    def test_a_missing_sdk_is_a_named_reason_not_an_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off-hardware the driver stays constructible and every write refuses."""
        monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", None)
        reason = BoosterDriver().connect_eagerly()
        assert reason is not None and "pip install booster_robotics_sdk_python" in reason


class TestAFallenRobotIsNotWrittenTo:
    """A held arm posture on a robot leaving the floor obstructs its recovery."""

    @pytest.mark.parametrize(
        ("code", "state", "admitted"),
        [
            pytest.param(0, "IS_READY", True, id="ready"),
            pytest.param(1, "IS_FALLING", False, id="falling"),
            pytest.param(2, "HAS_FALLEN", False, id="fallen"),
            pytest.param(3, "IS_GETTING_UP", False, id="getting-up"),
        ],
    )
    def test_only_a_ready_robot_admits_a_write(self, sdk: _FakeSdk, code: int, state: str, admitted: bool) -> None:
        driver = _live_driver(sdk)
        assert sdk.fall is not None
        sdk.fall.handler(_FakeFall(code))
        result = driver.send_action({"left_shoulder_pitch": 0.1})
        assert (result["status"] == "success") is admitted, result
        if not admitted:
            assert state in result["content"][0]["text"]
        assert asyncio.run(driver.get_status())["content"][0]["json"]["fall_state"] == state

    def test_a_silent_fall_topic_does_not_refuse_every_frame(self, sdk: _FakeSdk) -> None:
        """The gate reads evidence of a fall, not the absence of a reading."""
        driver = _live_driver(sdk)
        assert asyncio.run(driver.get_status())["content"][0]["json"]["fall_state"] is None
        assert driver.send_action({"left_shoulder_pitch": 0.1})["status"] == "success"

    def test_an_unknown_fall_code_is_ignored_rather_than_trusted(self, sdk: _FakeSdk) -> None:
        """A code the driver cannot name must not silently become a gate value."""
        driver = _live_driver(sdk)
        assert sdk.fall is not None
        sdk.fall.handler(_FakeFall(99))
        assert asyncio.run(driver.get_status())["content"][0]["json"]["fall_state"] is None


class TestTheBatteryReadIsReportedAndNotGatedOn:
    """``battery_pct`` is the shared triple's third field; the scale is the SDK's."""

    def test_the_charge_read_reaches_the_shared_status_field(self, sdk: _FakeSdk) -> None:
        driver = _live_driver(sdk)
        assert sdk.battery is not None
        sdk.battery.handler(_FakeBattery(87.5))
        assert asyncio.run(driver.get_status())["content"][0]["json"]["battery_pct"] == 87.5

    def test_a_low_charge_does_not_refuse_a_write(self, sdk: _FakeSdk) -> None:
        """No floor: the SDK documents no scale, so a floor would be theatre."""
        driver = _live_driver(sdk)
        assert sdk.battery is not None
        sdk.battery.handler(_FakeBattery(0.02))
        assert driver.send_action({"left_shoulder_pitch": 0.1})["status"] == "success"


class TestStopReportsBothHalvesOfTheHalt:
    """A driver that stopped walking and kept the arms is not stopped."""

    def test_a_stop_halts_locomotion_and_hands_the_arms_back(self, sdk: _FakeSdk) -> None:
        driver = _live_driver(sdk)
        outcome = driver.stop_task()
        assert outcome["status"] == "success"
        assert outcome["content"][0]["json"] == {"locomotion_halted": True, "upper_body_released": True}
        assert ("MoveCommand", (0.0, 0.0, 0.0)) in sdk.client.calls
        assert ("UpperBodyCustomControl", (False,)) in sdk.client.calls
        assert driver.send_action({"left_shoulder_pitch": 0.1})["status"] == "error"

    def test_a_half_that_refuses_is_reported_rather_than_asserted(self, sdk: _FakeSdk) -> None:
        """The robot refusing the release must not read as a completed stop."""
        driver = _live_driver(sdk)
        sdk.client.refuse = {"UpperBodyCustomControl"}
        outcome = driver.stop_task()
        assert outcome["status"] == "error"
        assert outcome["content"][0]["json"] == {"locomotion_halted": True, "upper_body_released": False}


class TestTheReadSurface:
    """What the mesh and the agent see."""

    def test_the_observation_names_every_slot_the_robot_reported(self, sdk: _FakeSdk) -> None:
        observation = _live_driver(sdk).get_observation()
        assert set(observation) == set(BOOSTER_JOINT_INDEX)
        assert observation["left_elbow_yaw"] == 0.5

    def test_an_unconnected_driver_reads_empty_rather_than_raising(self) -> None:
        driver = BoosterDriver()
        assert driver.get_observation() == {}
        assert driver.read_state() == {}
        assert driver.read_mode() is None

    def test_the_status_reports_the_gate_and_the_frame_width(self, sdk: _FakeSdk) -> None:
        driver = _live_driver(sdk)
        payload = asyncio.run(driver.get_status())["content"][0]["json"]
        assert payload["connected"] is True
        assert payload["upper_body_enabled"] is True
        assert payload["frame_width"] == _WIDTH
        assert payload["mode"] == "kCustom"

    def test_the_agent_verbs_are_read_only_plus_a_stop(self, sdk: _FakeSdk) -> None:
        """A model must not be able to choose a motion verb on a 1.2 m biped."""
        driver = _live_driver(sdk)
        spec = driver.tool_spec
        assert spec["inputSchema"]["json"]["properties"]["action"]["enum"] == ["sensors", "status", "stop"]

        async def collect() -> list[Any]:
            return [event async for event in driver.stream(_tool_use("sensors"), {})]

        events = asyncio.run(collect())
        assert events[-1]["status"] == "success"
        assert events[-1]["content"][0]["json"]["joints"]


class TestTheSeamCanBuildIt:
    """The registration, and the surface the registration checks."""

    def test_the_driver_satisfies_the_hardware_contract(self) -> None:
        assert missing_driver_members(BoosterDriver) == ()
        assert isinstance(BoosterDriver(), HardwareDriver)

    def test_booster_t1_resolves_to_this_driver(self) -> None:
        assert get_native_driver_class("booster_t1") is BoosterDriver

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param({"cmd_type": "hybrid"}, "cmd_type must be one of", id="unknown-cmd-type"),
            pytest.param({"domain_id": -1}, "invalid domain_id", id="below-the-floor-domain"),
            pytest.param({"domain_id": True}, "invalid domain_id", id="bool-domain"),
            pytest.param({"domain_id": MAX_DDS_DOMAIN_ID + 1}, "invalid domain_id", id="above-the-rtps-ceiling-domain"),
        ],
    )
    def test_the_constructor_refuses_an_unusable_knob(self, kwargs: dict[str, Any], expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            BoosterDriver(**kwargs)

    @pytest.mark.parametrize("domain", [0, 1, 7, 101, MAX_DDS_DOMAIN_ID], ids=repr)
    def test_a_domain_that_names_one_is_stored_verbatim(self, domain: int) -> None:
        """The accepted end of the domain, including the ceiling itself.

        Two things are pinned here that a refusal test cannot reach. The bound
        is inclusive, so ``MAX_DDS_DOMAIN_ID`` is a domain and not the first
        value past one - a guard written with the wrong comparison would refuse
        it and every refusal cell would still pass. And the value is kept as it
        arrived: ``_domain_id`` is what reaches
        ``ChannelFactory.Init(domain_id, ip)``, so a coercion here is what opens
        channels on a domain the caller never named, silently and with nothing
        to report. Neither is observable from the refusal side.
        """
        assert BoosterDriver(domain_id=domain)._domain_id == domain

    def test_a_named_robot_opens_named_channels(self, sdk: _FakeSdk) -> None:
        """Two T1s on one network must not share a topic."""
        driver = BoosterDriver(robot_name="t1_left", port="10.0.0.2", domain_id=3)
        assert driver.connect_eagerly() is None
        assert sdk.factory_init == (3, "10.0.0.2")
        assert sdk.subscriber is not None and sdk.subscriber.opened_as == "t1_left"
        assert sdk.publisher.opened_as == "t1_left"
        assert ("InitWithName", ("t1_left",)) in sdk.client.calls


class TestTheJointTableMatchesTheVendorEnum:
    """The driver's own joint map, graded against the SDK that owns the numbers.

    Needs the real SDK: a stub would compare the driver's constant against a
    copy of itself. Skipped where the vendor wheel is not installed, which is
    why every other cell here runs on the double instead.

    The comparison is per slot rather than per name, because the vendor spells
    the side inconsistently: eighteen members lead with it (``kLeftHipPitch``)
    and the four ankle crank members trail with it (``kCrankUpLeft``). The
    driver normalises all twenty-three to a leading side, so the pin is that
    each slot names the same joint - same words, any order - and that the
    numbering is identical. A vendor renumbering, or a typo in the driver's
    table, still fails; the deliberate reordering does not.
    """

    @staticmethod
    def _words(name: str) -> tuple[str, ...]:
        """The lowercase words of a joint name, from either spelling."""
        camel = "".join(f" {c.lower()}" if c.isupper() else c for c in name.removeprefix("k"))
        return tuple(sorted(camel.replace("_", " ").split()))

    def test_every_slot_names_the_vendor_joint_at_that_number(self) -> None:
        sdk = pytest.importorskip("booster_robotics_sdk_python")
        vendor = {int(getattr(sdk.JointIndex, name)): name for name in dir(sdk.JointIndex) if name.startswith("k")}
        driver = {slot: name for name, slot in BOOSTER_JOINT_INDEX.items()}

        assert sorted(driver) == sorted(vendor), "the driver's slot numbers are not the vendor's"
        assert len(driver) == int(sdk.kJointCnt)
        mismatched = {
            slot: (driver[slot], vendor[slot])
            for slot in vendor
            if self._words(driver[slot]) != self._words(vendor[slot])
        }
        assert not mismatched, f"slot -> (driver name, vendor name) disagree on the joint: {mismatched}"

    def test_the_fall_state_codes_are_the_vendor_codes(self) -> None:
        """The gate compares against numbers the vendor enum owns."""
        sdk = pytest.importorskip("booster_robotics_sdk_python")
        vendor = {
            int(getattr(sdk.FallDownStateType, name)): name for name in dir(sdk.FallDownStateType) if name.isupper()
        }
        assert FALL_STATE_NAMES == vendor
        assert FALL_STATE_READY == vendor[0]

    def test_the_wire_literals_are_the_vendor_literals(self) -> None:
        """Position mode and the two cmd_type conventions, read off the SDK."""
        sdk = pytest.importorskip("booster_robotics_sdk_python")
        assert POSITION_MODE == 0x0A
        assert set(CMD_TYPE_STATE_FIELD) == {"parallel", "serial"}
        for name in CMD_TYPE_STATE_FIELD:
            assert hasattr(sdk.LowCmdType, name.upper())
        # Every mode this driver offers, not a sample of two: ROBOT_MODES is the
        # set change_mode admits, and its one consumer reads the member off this
        # enum. A subset assertion over two names cannot see a build that
        # renamed or dropped one of the other three. kUnknown is excluded on
        # purpose - see ROBOT_MODES - so the claim is one-directional.
        vendor_modes = {mode for mode in dir(sdk.RobotMode) if mode.startswith("k")}
        assert vendor_modes >= set(ROBOT_MODES), (
            f"the SDK declares no {sorted(set(ROBOT_MODES) - vendor_modes)}; "
            "ROBOT_MODES claims a vocabulary this build does not have"
        )
        assert "kUnknown" in vendor_modes and "kUnknown" not in ROBOT_MODES


# --------------------------------------------------------------------------- #
# Endpoint construction is serialised against the rest of the process.         #
# --------------------------------------------------------------------------- #


class TestEndpointsAreBuiltUnderTheSharedDdsLock:
    """The channel set is constructed under the lock every DDS user shares.

    Constructing one CycloneDDS endpoint while another is being constructed
    segfaults the bindings, and the loss is not catchable: the process dies,
    possibly while a 1.2 m biped is standing under its own controller. The
    subscriber set in MODULE ``strands_robots.tools.g1._dds_engine`` builds every
    subscriber under ``_DDS_INIT_LOCK``, so this driver has to take the *same*
    lock rather than one of its own - a private lock would exclude nothing.

    That is what these cells measure: the connect blocks while a competing
    holder owns the shared lock, and no endpoint is built until it is released.
    """

    @staticmethod
    def _connect_in_a_thread(driver: BoosterDriver) -> tuple[threading.Thread, threading.Event, list[str | None]]:
        done = threading.Event()
        result: list[str | None] = []

        def run() -> None:
            result.append(driver.connect_eagerly())
            done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, done, result

    def test_connect_waits_for_a_competing_holder_of_the_shared_lock(self, sdk: _FakeSdk) -> None:
        """Held elsewhere, the connect makes no progress and builds no endpoint."""
        driver = BoosterDriver()
        with _DDS_INIT_LOCK:
            thread, done, result = self._connect_in_a_thread(driver)
            assert not done.wait(0.5), "connect_eagerly did not wait for the shared DDS lock"
            assert sdk.factory_init is None, "the channel factory was initialised while another holder had the lock"
            assert sdk.subscriber is None, "a subscriber was constructed while another holder had the lock"
            assert not driver.is_connected

        assert done.wait(5.0), "connect_eagerly never completed after the lock was released"
        thread.join(timeout=5.0)
        assert result == [None]
        assert sdk.factory_init is not None
        assert sdk.subscriber is not None
        assert driver.is_connected

    def test_the_lock_is_released_for_the_next_endpoint_user(self, sdk: _FakeSdk) -> None:
        """A connect that owns the lock must not keep it: DDS is process-wide."""
        driver = BoosterDriver()
        assert driver.connect_eagerly() is None
        assert _DDS_INIT_LOCK.acquire(timeout=1.0), "connect_eagerly held the shared DDS lock past its own use"
        _DDS_INIT_LOCK.release()
