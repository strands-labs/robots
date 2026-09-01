"""The Go2 native driver gates, maps and publishes what the Go2 wire expects.

One suite for one driver, table-driven where the behaviour is a family of
refusals, because the thing worth pinning is a *behaviour* and not each constant
it happens to read.

The headline contract is
:func:`test_the_wire_slot_order_is_legid_not_the_description_declaration_order`.
``LowCmd_.motor_cmd`` is indexed by Unitree's ``LegID`` order (front-right,
front-left, rear-right, rear-left) while the Go2's own URDF/MJCF description
declares its joints front-left, front-right, rear-left, rear-right. Those two
orders are a transposition of each other, so a driver that zipped one onto the
other would command the mirror-image leg with a perfectly valid CRC - a
quadruped that walks sideways with nothing in any log to say why. That is why
:data:`~strands_robots.drivers.go2.GO2_JOINT_INDEX` is keyed by name and why the
mapping is pinned here rather than left to review.

No test here needs a Go2, a DDS bus or ``unitree_sdk2py``: the SDK is stubbed on
:mod:`sys.modules` for the frame-building lanes (the same production code path
hardware drives) and the publisher is a recorder with
:class:`~strands_robots.drivers.g1.DDSPublisher`'s acceptance contract.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class, list_native_drivers
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.go2 import (
    _LEVEL_FLAG_LOW,
    _LOWCMD_HEAD,
    _MOTOR_MODE_SERVO,
    _SDK_KD,
    _SDK_KP,
    GO2_JOINT_INDEX,
    Go2Driver,
    build_lowcmd_from_action,
    build_zero_torque_lowcmd,
    decode_mode_name,
)
from strands_robots.registry import get_driver, resolve_name

#: The order the Go2's official description declares its twelve leg joints, as
#: read from ``go2_mj_description``'s actuator list. Recorded as a literal rather
#: than loaded from MuJoCo so this suite needs neither the package nor a network
#: fetch; it is the *other* order, and the whole point of the pin below is that
#: it is not the wire's.
_DESCRIPTION_DECLARATION_ORDER: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


class _StubMotorCmd:
    """Stand-in for a ``unitree_go`` ``MotorCmd_`` slot, zeroed like the real one."""

    def __init__(self) -> None:
        self.mode: int = 0
        self.q: float = 0.0
        self.dq: float = 0.0
        self.tau: float = 0.0
        self.kp: float = 0.0
        self.kd: float = 0.0


class _StubLowCmd:
    """Stand-in for ``unitree_sdk2py.idl.default.unitree_go_msg_dds__LowCmd_()``.

    Twenty ``motor_cmd`` slots, matching the ``unitree_go`` IDL width the driver's
    own slot guard checks, and - deliberately - ``head`` and ``level_flag`` left
    at their zero defaults, because the point of the header contract is that the
    real constructor does not set them either.
    """

    def __init__(self) -> None:
        self.head: list[int] = [0, 0]
        self.level_flag: int = 0
        self.gpio: int = 0
        self.motor_cmd: list[_StubMotorCmd] = [_StubMotorCmd() for _ in range(20)]
        self.crc: int = 0


class _StubCRC:
    """Stand-in for ``unitree_sdk2py.utils.crc.CRC``.

    Firmware verifies the real CRC; what is graded here is that the builder calls
    ``.Crc(cmd)`` at all, after every other field is populated. ``42`` is a
    distinguishable non-zero.
    """

    def Crc(self, _cmd: Any) -> int:
        return 42


class _RecordingPublisher:
    """Records ``publish`` calls without touching a DDS bus.

    Same acceptance contract as
    :class:`~strands_robots.tools.g1._dds_engine.DDSPublisher`: ``publish``
    returns ``None`` on success and a reason string on failure. Every call lands
    in :attr:`writes` so a test can walk the wire capture.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, type, Any]] = []
        self.publish_should_return: str | None = None
        self.close_calls = 0

    def publish(self, topic: str, message_class: type, message: Any) -> str | None:
        if self.publish_should_return is not None:
            return self.publish_should_return
        self.writes.append((topic, message_class, message))
        return None

    def close(self) -> None:
        self.close_calls += 1


class _RecordingMotionSwitcher:
    """A motion-switcher double that reports a queue of modes and counts releases.

    ``CheckMode`` pops the next scripted reading, which is how the release loop's
    "release, then verify" shape is graded without a robot: a script of
    ``["ai", ""]`` is a mode that goes away after one ``ReleaseMode``.
    """

    def __init__(self, readings: list[Any]) -> None:
        self._readings = list(readings)
        self.release_calls = 0

    def CheckMode(self) -> Any:
        return self._readings.pop(0) if self._readings else (0, {"name": ""})

    def ReleaseMode(self) -> None:
        self.release_calls += 1


@pytest.fixture
def stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``unitree_sdk2py`` stub for the duration of one test.

    The Go2 driver imports ``unitree_sdk2py.idl.default``,
    ``unitree_sdk2py.utils.crc`` and ``unitree_sdk2py.idl.unitree_go.msg.dds_``
    inside function bodies. Registering each on :mod:`sys.modules` drives the same
    production lane hardware drives, on a box with no SDK.
    ``monkeypatch.setitem`` restores the previous entries - normally absent - on
    teardown.
    """
    names = {
        "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.default": types.ModuleType("unitree_sdk2py.idl.default"),
        "unitree_sdk2py.idl.unitree_go": types.ModuleType("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_"),
        "unitree_sdk2py.utils": types.ModuleType("unitree_sdk2py.utils"),
        "unitree_sdk2py.utils.crc": types.ModuleType("unitree_sdk2py.utils.crc"),
    }
    names["unitree_sdk2py.idl.default"].unitree_go_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    names["unitree_sdk2py.idl.unitree_go.msg.dds_"].LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    names["unitree_sdk2py.utils.crc"].CRC = _StubCRC  # type: ignore[attr-defined]
    for name, module in names.items():
        monkeypatch.setitem(sys.modules, name, module)


def _released_driver() -> tuple[Go2Driver, _RecordingPublisher]:
    """Return a driver whose gates admit, and the recorder attached to it.

    Sport mode released, battery well above the floor, publisher a recorder. Every
    write-path test starts here so the cells read as "given a gate-passing
    driver".
    """
    driver = Go2Driver(tool_name="go2", port="192.168.123.161")
    driver._connected = True
    driver._sport_mode_released = True
    driver._battery = {"pct": 88.0, "current": 1.0, "cycle": 3}
    pub = _RecordingPublisher()
    driver._pubs = pub  # type: ignore[assignment]
    return driver, pub


def _text(envelope: dict[str, Any]) -> str:
    """Return the refusal text from an error envelope."""
    return str(envelope["content"][0]["text"])


# --------------------------------------------------------------------------- #
# Load hygiene and the driver seam.                                           #
# --------------------------------------------------------------------------- #


def test_the_import_pulls_no_sdk_module() -> None:
    """The driver module is loadable on a host without ``unitree_sdk2py``.

    Every SDK touch lives inside a function body, so the module imports on CI and
    on a dev box before any bring-up. A module-level import would break every
    headless runner (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.drivers.go2")
    leaked = {name for name in set(sys.modules) - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.drivers.go2 imports pulled SDK submodules: {leaked}. The rule "
        "for this package is that the SDK loads only inside function bodies "
        "(refs strands-labs/robots#358)."
    )


def test_the_driver_is_the_registered_native_driver_for_the_go2() -> None:
    """``Robot("go2", mode="real")`` resolves to this driver, by canonical name or alias.

    Three facts in one cell because they are one behaviour - the robot is
    natively driven: the class satisfies the driver surface, the seam registered
    it for the canonical name, and the registry entry declares ``strands`` so the
    native driver is the default rather than something a caller must ask for.
    """
    assert missing_driver_members(Go2Driver) == ()
    assert get_native_driver_class("unitree_go2") is Go2Driver
    assert get_native_driver_class("go2") is Go2Driver, "the go2 alias must resolve to the same driver"
    assert list_native_drivers()["unitree_go2"] == "Go2Driver"
    assert get_driver(resolve_name("go2")) == "strands"


def test_the_subscription_plan_reads_the_quadruped_idl_package() -> None:
    """Every subscribed topic decodes through ``unitree_go``, not the G1's ``unitree_hg``.

    The IDL package is the difference between a Go2 driver and a G1 driver, so it
    is graded rather than reviewed: a ``unitree_hg`` ``LowState_`` on a Go2
    deserialises garbage or nothing at all.
    """
    plan = Go2Driver()._subscription_plan()
    topics = [topic for topic, _cls, _decoder in plan]
    assert topics == ["rt/lowstate", "rt/sportmodestate"]
    for _topic, (module_path, _class_name), decoder in plan:
        assert module_path == "unitree_sdk2py.idl.unitree_go.msg.dds_"
        assert callable(decoder)


# --------------------------------------------------------------------------- #
# The headline: wire order is not description order.                          #
# --------------------------------------------------------------------------- #


def test_the_wire_slot_order_is_legid_not_the_description_declaration_order() -> None:
    """``motor_cmd`` slots follow LegID (FR, FL, RR, RL), transposing the description.

    The Go2's description declares FL, FR, RL, RR; the wire indexes FR, FL, RR,
    RL. Both orders contain the same twelve names, which is exactly why the
    mistake is invisible: zipping the description's order onto ``motor_cmd``
    produces twelve valid commands sent to the mirror-image legs. This cell pins
    the wire order, pins that it differs from the description's, and pins that
    the two are a front/rear-pairwise transposition rather than an arbitrary
    permutation - so a future edit that "sorts" the table fails here.
    """
    by_slot = [name for name, _slot in sorted(GO2_JOINT_INDEX.items(), key=lambda kv: kv[1])]
    assert by_slot == [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    # Same twelve joints as the description, different order. Both halves matter:
    # a missing joint and a reordered joint are different defects.
    assert set(by_slot) == set(_DESCRIPTION_DECLARATION_ORDER)
    assert by_slot != list(_DESCRIPTION_DECLARATION_ORDER), (
        "if these ever agree, either the SDK changed its LegID order or this table "
        "was 'fixed' to match the description - check the SDK before believing it"
    )
    # The transposition is per pair: swapping each leg-pair of the description's
    # order yields the wire's order exactly.
    swapped: list[str] = []
    for pair_start in (0, 6):
        swapped += list(_DESCRIPTION_DECLARATION_ORDER[pair_start + 3 : pair_start + 6])
        swapped += list(_DESCRIPTION_DECLARATION_ORDER[pair_start : pair_start + 3])
    assert by_slot == swapped


def test_every_joint_takes_the_reference_gain_for_its_own_slot() -> None:
    """The gain tables are indexed by wire slot and cover every named joint.

    A gain table shorter than the joint table would raise IndexError on the
    highest slot only - the rear-left calf, i.e. after a rollout has already been
    commanding the other eleven joints.
    """
    assert len(_SDK_KP) == len(_SDK_KD) == len(GO2_JOINT_INDEX)
    for slot in GO2_JOINT_INDEX.values():
        assert _SDK_KP[slot] > 0.0
        assert _SDK_KD[slot] > 0.0


# --------------------------------------------------------------------------- #
# The write path.                                                             #
# --------------------------------------------------------------------------- #


def test_a_gated_write_reaches_the_wire_with_the_go_protocol_header(stub_unitree_sdk: None) -> None:
    """One ``send_action`` publishes one framed ``LowCmd_`` on ``rt/lowcmd``.

    The whole wire contract in one cell, because it is one frame: the protocol
    header and low-level flag the ``unitree_go`` constructor does *not* set, the
    enable byte on the commanded slot, the reference gains for that slot, the CRC
    stamped last, and - the part a reader most wants guaranteed - every
    uncommanded slot left disabled.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    result = driver.send_action({"FL_calf_joint": -1.5})
    assert result["status"] == "success", _text(result)

    payload = result["content"][0]["json"]
    assert payload["topic"] == "rt/lowcmd"
    assert payload["joints"] == ["FL_calf_joint"]
    assert payload["slots"] == [GO2_JOINT_INDEX["FL_calf_joint"]]

    assert len(pub.writes) == 1
    topic, _cls, cmd = pub.writes[0]
    assert topic == "rt/lowcmd"
    assert (cmd.head[0], cmd.head[1]) == _LOWCMD_HEAD
    assert cmd.level_flag == _LEVEL_FLAG_LOW
    assert cmd.crc == 42, "the CRC must be stamped after the frame is populated"

    slot = GO2_JOINT_INDEX["FL_calf_joint"]
    motor = cmd.motor_cmd[slot]
    assert motor.mode == _MOTOR_MODE_SERVO, "an unset mode byte commands nothing however valid the CRC"
    assert motor.q == pytest.approx(-1.5)
    assert motor.kp == pytest.approx(_SDK_KP[slot])
    assert motor.kd == pytest.approx(_SDK_KD[slot])
    for other in range(len(cmd.motor_cmd)):
        if other != slot:
            assert cmd.motor_cmd[other].mode == 0, f"slot {other} was not commanded and must stay disabled"


def test_per_joint_gains_override_the_reference_gains(stub_unitree_sdk: None) -> None:
    """A per-joint dict places ``q``, gains and feed-forward effort on its own slot."""
    del stub_unitree_sdk
    driver, pub = _released_driver()
    result = driver.send_action({"RR_thigh_joint": {"q": 0.8, "kp": 12.0, "kd": 0.5, "dq": 0.25, "tau": 1.5}})
    assert result["status"] == "success", _text(result)
    _topic, _cls, cmd = pub.writes[0]
    motor = cmd.motor_cmd[GO2_JOINT_INDEX["RR_thigh_joint"]]
    assert (motor.q, motor.kp, motor.kd, motor.dq, motor.tau) == pytest.approx((0.8, 12.0, 0.5, 0.25, 1.5))


@pytest.mark.parametrize(
    ("action", "expected_fragment"),
    [
        pytest.param(["FL_hip_joint"], "must be a dict", id="not-a-dict"),
        pytest.param({}, "nothing to command", id="empty"),
        pytest.param({"elbow": 0.0}, "unknown joint name", id="unknown-joint"),
        pytest.param({"FL_hip_joint": {"kp": 1.0}}, "missing required key 'q'", id="missing-q"),
        pytest.param({"FL_hip_joint": {"q": 0.0, "kx": 1.0}}, "unknown per-joint keys", id="unknown-inner-key"),
        pytest.param({"FL_hip_joint": "forward"}, "must be a finite number", id="not-a-number"),
        pytest.param({"FL_hip_joint": float("nan")}, "must be a finite number", id="nan-target"),
    ],
)
def test_an_unusable_action_is_refused_and_nothing_reaches_the_wire(
    stub_unitree_sdk: None, action: Any, expected_fragment: str
) -> None:
    """Every unusable action refuses whole; a partial frame never publishes.

    Silently dropping a joint the caller believed was commanded is the worst
    failure mode on a legged robot, so each of these refuses the entire action
    rather than the offending key - and the publisher stays untouched, which is
    the half that matters.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    result = driver.send_action(action)
    assert result["status"] == "error"
    assert expected_fragment in _text(result)
    assert pub.writes == [], "a refused action must not reach the wire at all"


def test_the_zero_torque_frame_keeps_the_servos_enabled_at_zero_gain(stub_unitree_sdk: None) -> None:
    """The soft-stop frame zeroes gains and effort but leaves the servos enabled.

    A *Disable* frame cuts the motors dead and drops the robot onto its knees; an
    enabled zero-gain frame lets it settle under its own weight. That difference
    is the whole reason this helper exists, so it is pinned rather than described.
    """
    del stub_unitree_sdk
    cmd, err = build_zero_torque_lowcmd()
    assert err is None, err
    for name, slot in GO2_JOINT_INDEX.items():
        motor = cmd.motor_cmd[slot]
        assert motor.mode == _MOTOR_MODE_SERVO, f"{name} must stay enabled in a soft stop"
        assert (motor.kp, motor.kd, motor.tau, motor.q, motor.dq) == pytest.approx((0.0,) * 5)
    assert cmd.crc == 42


def test_an_idl_whose_slot_count_disagrees_is_named_not_indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``motor_cmd`` array of the wrong width is refused by name.

    Without the guard the mismatch surfaces as an ``IndexError`` raised from
    inside the frame builder on whichever slot happens to overflow first - a
    traceback that names neither the SDK nor the expectation it broke.
    """

    class _ShortLowCmd(_StubLowCmd):
        def __init__(self) -> None:
            super().__init__()
            self.motor_cmd = self.motor_cmd[:12]

    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_go_msg_dds__LowCmd_ = _ShortLowCmd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", types.ModuleType("unitree_sdk2py"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl", types.ModuleType("unitree_sdk2py.idl"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.default", default)

    cmd, err = build_lowcmd_from_action({"FL_hip_joint": 0.0})
    assert cmd is None
    assert err is not None
    assert "12 slots, expected 20" in err


# --------------------------------------------------------------------------- #
# The gates.                                                                  #
# --------------------------------------------------------------------------- #


def test_an_unreleased_sport_mode_refuses_every_write_path(stub_unitree_sdk: None) -> None:
    """Until sport mode is released, no write path reaches the wire.

    All three write entry points share one gate, so all three are graded here: a
    driver that refused ``send_action`` but let ``run_policy`` spin up a 500 Hz
    thread would hand the legs to two controllers at once.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    driver._sport_mode_released = False
    driver._sport_mode_name = "ai"

    for label, envelope in (
        ("send_action", driver.send_action({"FL_hip_joint": 0.0})),
        ("run_policy", driver.run_policy(lambda _state: {"FL_hip_joint": 0.0})),
        ("start_task", driver.start_task("walk")),
    ):
        assert envelope["status"] == "error", label
        reason = _text(envelope)
        assert "sport mode is not released" in reason, label
        assert "release_sport_mode" in reason, f"{label} must name the call that opens the gate"
    assert pub.writes == []


def test_a_battery_under_the_floor_refuses_the_write(stub_unitree_sdk: None) -> None:
    """The battery floor refuses separately from the sport-mode gate.

    Two independent gates, named separately, so a caller can tell which one
    refused instead of reading one message that covers both.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    driver._battery = {"pct": 4.0}
    result = driver.send_action({"FL_hip_joint": 0.0})
    assert result["status"] == "error"
    reason = _text(result)
    assert "battery 4.0%" in reason
    assert "under floor" in reason
    assert pub.writes == []


def test_a_non_finite_battery_floor_is_refused_at_construction() -> None:
    """A ``nan`` floor is refused rather than stored.

    ``reading < nan`` is False for every reading, so a ``nan`` floor would be
    reported by ``get_status`` and enforced against nothing - a safety gate that
    looks configured and is not.
    """
    with pytest.raises(ValueError, match="battery_floor_pct"):
        Go2Driver(battery_floor_pct=float("nan"))


@pytest.mark.parametrize(
    ("reading", "expected_fragment"),
    [
        pytest.param("released", "must return a (status, result) pair", id="not-a-pair"),
        pytest.param((0, {"name": ""}, 1), "must return a (status, result) pair", id="wrong-length"),
        pytest.param((True, {"name": ""}), "must be an int response code", id="bool-status"),
        pytest.param((7, {"name": ""}), "CheckMode() failed: 7", id="error-code"),
        pytest.param((0, "ai"), "result must be a dict", id="result-not-a-dict"),
        pytest.param((0, {"form": 1}), "has no 'name' key", id="no-name-key"),
        pytest.param((0, {"name": 3}), "'name' must be a string", id="name-not-a-string"),
    ],
)
def test_an_undecodable_mode_reading_is_refused_rather_than_read_as_released(
    reading: Any, expected_fragment: str
) -> None:
    """A reading that cannot be decoded is never evidence that the robot is free.

    The failure mode this forbids is the dangerous default: treating an
    unparseable ``CheckMode`` response as "no mode active" would open the write
    gate on a robot the onboard controller is still driving.
    """
    mode_name, refusal = decode_mode_name(reading)
    assert mode_name is None
    assert refusal is not None
    assert expected_fragment in refusal


def test_release_sport_mode_polls_until_the_robot_reports_no_mode() -> None:
    """The release loops - ``ReleaseMode`` then re-read - because the release is async.

    A single ``ReleaseMode()`` call followed by an assumption of success is the
    bug this shape avoids: the robot reports ``"ai"`` until it has actually let
    go, so the driver re-reads and only then opens its gate.
    """
    switcher = _RecordingMotionSwitcher([(0, {"name": "ai"}), (0, {"name": "ai"}), (0, {"name": ""})])
    driver = Go2Driver(motion_switcher_client_factory=lambda _iface: switcher)
    assert driver._sport_mode_released is False

    result = driver.release_sport_mode()
    assert result["status"] == "success", _text(result)
    assert result["content"][0]["json"]["sport_mode_released"] is True
    assert result["content"][0]["json"]["released_mode"] == "ai"
    assert switcher.release_calls == 2, "one release per non-empty reading, then the confirming read"
    assert driver._sport_mode_released is True


def test_release_sport_mode_gives_up_by_name_when_the_mode_will_not_clear() -> None:
    """A mode that never clears refuses with the attempt count and stays shut.

    Reporting success on an unreleased robot would let the next ``send_action``
    publish into a fight with the onboard controller.
    """
    switcher = _RecordingMotionSwitcher([(0, {"name": "normal"})] * 6)
    driver = Go2Driver(motion_switcher_client_factory=lambda _iface: switcher)
    result = driver.release_sport_mode(attempts=2)
    assert result["status"] == "error"
    reason = _text(result)
    assert "'normal'" in reason
    assert "after 2 release attempts" in reason
    assert driver._sport_mode_released is False


# --------------------------------------------------------------------------- #
# The task path.                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        pytest.param({"duration": float("nan")}, "duration", id="nan-duration"),
        pytest.param({"duration": float("inf")}, "duration", id="inf-duration"),
        pytest.param({"duration": 0.0}, "duration", id="zero-duration"),
        pytest.param({"n_steps": 0}, "n_steps", id="zero-steps"),
        pytest.param({"n_steps": True}, "n_steps", id="bool-steps"),
        pytest.param({"n_steps": 2.5}, "n_steps", id="fractional-steps"),
    ],
)
def test_run_policy_refuses_an_unusable_budget_before_starting_a_thread(
    stub_unitree_sdk: None, kwargs: dict[str, Any], expected_fragment: str
) -> None:
    """A budget that cannot bound the loop is refused before the thread exists.

    ``nan`` poisons the deadline comparison so the loop would actuate with no
    budget at all; ``inf`` never expires; ``0`` and a ``bool`` cap a rollout that
    commanded nothing inside a success envelope.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    result = driver.run_policy(lambda _state: {"FL_hip_joint": 0.0}, **kwargs)
    assert result["status"] == "error"
    assert expected_fragment in _text(result)
    assert pub.writes == []
    assert driver.get_task_status()["content"][0]["json"]["running"] is False


@pytest.mark.parametrize(
    ("policy", "expected_fragment"),
    [
        pytest.param(None, "policy_object is required", id="none"),
        pytest.param(object(), "must be callable or expose a .step()", id="not-callable"),
    ],
)
def test_run_policy_refuses_a_policy_it_cannot_call(
    stub_unitree_sdk: None, policy: Any, expected_fragment: str
) -> None:
    """A policy the loop could not call is refused up front, not mid-rollout."""
    del stub_unitree_sdk
    driver, _pub = _released_driver()
    result = driver.run_policy(policy)
    assert result["status"] == "error"
    assert expected_fragment in _text(result)


def test_a_rollout_publishes_frames_then_soft_stops_and_reports_its_exit(stub_unitree_sdk: None) -> None:
    """A bounded rollout commands its steps, soft-stops, and stays pollable after.

    The behaviour worth pinning end to end: frames reach ``rt/lowcmd``, the loop
    ends itself on the step budget, the last frame is the zero-gain soft stop,
    and the terminal snapshot survives the thread so a caller who polls late
    still learns why it stopped instead of "no task has been started".
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    calls: list[dict[str, Any]] = []

    def policy(state: dict[str, Any]) -> dict[str, Any]:
        calls.append(state)
        return {"FL_thigh_joint": 0.3}

    started = driver.run_policy(policy, n_steps=3, duration=5.0)
    assert started["status"] == "success", _text(started)
    assert driver._loop is not None
    driver._loop._thread.join(timeout=5.0)  # type: ignore[union-attr]

    snapshot = driver.get_task_status()["content"][0]["json"]
    assert snapshot["running"] is False
    assert snapshot["exit_reason"] == "n_steps"
    assert snapshot["steps"] == 3
    assert calls, "the policy must be called with the driver's cached state"
    assert "imu" in calls[0] and "sport_mode_released" in calls[0]

    # Three commanded frames plus the zero-torque frame the loop publishes on the
    # way out - the soft stop is not optional, so it is counted.
    assert len(pub.writes) == 4
    assert {topic for topic, _cls, _cmd in pub.writes} == {"rt/lowcmd"}
    final = pub.writes[-1][2]
    for slot in GO2_JOINT_INDEX.values():
        assert final.motor_cmd[slot].kp == pytest.approx(0.0)
        assert final.motor_cmd[slot].mode == _MOTOR_MODE_SERVO


def test_a_rollout_ends_when_the_policy_returns_an_unusable_action(stub_unitree_sdk: None) -> None:
    """A policy returning something unmappable ends the loop with a named reason.

    Not a silently-skipped step: a policy emitting joint names this robot does
    not have is broken, and continuing to hold the last posture at 500 Hz while
    discarding its output would hide that indefinitely.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    result = driver.run_policy(lambda _state: {"nonexistent_joint": 0.0}, n_steps=5, duration=5.0)
    assert result["status"] == "success", _text(result)
    driver._loop._thread.join(timeout=5.0)  # type: ignore[union-attr]

    snapshot = driver.get_task_status()["content"][0]["json"]
    assert snapshot["exit_reason"] == "policy"
    assert "unknown joint name" in str(snapshot["exit_detail"])
    assert snapshot["steps"] == 0
    # Only the soft stop reached the wire; the unusable action did not.
    assert len(pub.writes) == 1


def test_stop_task_is_idempotent_when_no_rollout_is_running() -> None:
    """Stopping nothing is a success naming the state, not an error."""
    driver, _pub = _released_driver()
    result = driver.stop_task()
    assert result["status"] == "success"
    assert "no task is running" in str(result["content"][0]["text"])


def test_get_task_status_says_so_before_any_rollout() -> None:
    """A driver that never ran a task reports that, rather than an empty snapshot."""
    payload = Go2Driver().get_task_status()["content"][0]["json"]
    assert payload["running"] is False
    assert "no task has been started" in payload["reason"]


# --------------------------------------------------------------------------- #
# Telemetry.                                                                  #
# --------------------------------------------------------------------------- #


def test_lowstate_fills_the_caches_the_gate_and_the_mesh_read() -> None:
    """``rt/lowstate`` yields the IMU, the battery the gate reads, and named joints.

    The Go2 carries its state of charge inside ``LowState_``, where the G1
    publishes a separate topic - so this one callback feeds the battery floor as
    well as the mesh. Joints come back keyed by *name*, which is what makes the
    read path safe against the same LegID transposition the write path guards.
    """

    class _Imu:
        quaternion = [1.0, 0.0, 0.0, 0.0]
        gyroscope = [0.1, 0.2, 0.3]
        accelerometer = [0.0, 0.0, 9.81]
        rpy = [0.0, 0.05, 0.0]

    class _Bms:
        soc = 73
        current = -2.5
        cycle = 41

    class _Motor:
        def __init__(self, q: float) -> None:
            self.q = q
            self.dq = 0.0
            self.tau_est = 1.25
            self.temperature = 34

    class _LowState:
        imu_state = _Imu()
        bms_state = _Bms()
        # Indexed by wire slot, so slot 3 is the front-LEFT hip.
        motor_state = [_Motor(float(i)) for i in range(20)]

    driver = Go2Driver()
    driver._on_lowstate(_LowState())
    state = driver.state

    assert state["imu"]["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert state["battery"]["pct"] == pytest.approx(73.0)
    assert state["joints"]["FL_hip_joint"]["q"] == pytest.approx(3.0), (
        "slot 3 is the front-left hip on the LegID wire order"
    )
    assert state["joints"]["FR_hip_joint"]["q"] == pytest.approx(0.0)
    assert set(state["joints"]) == set(GO2_JOINT_INDEX)

    # The battery the gate reads is the one this callback wrote.
    driver._sport_mode_released = True
    assert driver._check_motion_gates("send_action") is None


def test_sportmode_state_is_cached_for_a_rollout_cross_check() -> None:
    """``rt/sportmodestate`` gives body height and velocity - what the robot did."""

    class _SportState:
        mode = 1
        gait_type = 2
        body_height = 0.32
        position = [0.5, 0.0, 0.32]
        velocity = [0.4, 0.0, 0.0]
        yaw_speed = 0.1
        foot_force = [12, 14, 13, 11]

    driver = Go2Driver()
    driver._on_sportmode(_SportState())
    sport = driver.state["sport"]
    assert sport["body_height"] == pytest.approx(0.32)
    assert sport["velocity"] == [0.4, 0.0, 0.0]
    assert sport["foot_force"] == [12, 14, 13, 11]


# -- regression pin: the default MotionSwitcherClient import path is the shared
#    helper, not a direct SDK import from a module that never contained the class


class TestMotionSwitcherClientImportPath:
    """Pin that ``_open_motion_switcher_client`` delegates to the shared helper.

    The Go2 sport-mode gate depends on ``MotionSwitcherClient``, which lives in
    ``unitree_sdk2py.comm.motion_switcher.motion_switcher_client`` -- NOT in
    ``unitree_sdk2py.go2.sport.sport_client`` (that module ships only
    ``SportClient``).  A wrong import path is swallowed by the ``except
    Exception`` and silently gates shut every write surface on real hardware.

    This test reads the source AST to verify the import target without needing
    the SDK installed, so it runs on every CI host.
    """

    def test_default_path_delegates_to_the_shared_helper(self) -> None:
        import ast
        import inspect
        import textwrap

        source = inspect.getsource(Go2Driver._open_motion_switcher_client)
        tree = ast.parse(textwrap.dedent(source))

        # Collect every ImportFrom node inside the method
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

        # The method must import from the shared helper module, not directly
        # from the SDK
        helper_imports = [node for node in imports if node.module and "g1._motion_switcher" in node.module]
        sdk_direct_imports = [node for node in imports if node.module and "unitree_sdk2py" in node.module]

        assert helper_imports, (
            "_open_motion_switcher_client must delegate to the shared "
            "strands_robots.tools.g1._motion_switcher helper (one-owner "
            "pattern for the SDK module path); found no such import"
        )
        assert not sdk_direct_imports, (
            "_open_motion_switcher_client must not import directly from "
            f"unitree_sdk2py; found: {[n.module for n in sdk_direct_imports]}. "
            "The correct import path lives in "
            "strands_robots.tools.g1._motion_switcher._SDK_MODULE"
        )
