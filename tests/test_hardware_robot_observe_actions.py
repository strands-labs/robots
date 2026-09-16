"""``Robot(mode="real")`` can be READ by an agent without running a policy on it.

Before these actions existed the real-hardware tool offered ``execute``,
``start``, ``status`` and ``stop`` - four verbs, three of them about a rollout.
Asked to "read the joint positions, do not move any joint", an agent's only
route was ``execute`` with a mock policy and a ten-second duration: a rollout
requested in order to look. The operator gate caught it, which is the gate
doing its job and the tool failing at its own.

These tests run on a fake lerobot robot whose bus records every register it is
asked for, so they grade the observable facts without a serial port:

* ``get_state`` opens the bus alone (never the robot's ``connect()``, whose
  ``configure()`` writes servo registers), reads positions raw and normalised,
  torque and voltage, and never writes.
* An UNCALIBRATED arm is readable: degrees come from the encoder centre and
  the text says so, naming ``lerobot-calibrate``.
* A reading the arm did not give is reported as unread, not as its reassuring
  opposite: a calibration flag whose own read raises is not "uncalibrated" (the
  degrees stay the arm's own), and a torque register no motor answered is not
  "off" (which reads as "safe to grab").
* ``render`` opens the named camera lazily and saves a PNG inside the render
  sandbox; an unknown camera or an outside path is refused with the remedy.
* ``get_status()`` on a never-connected arm is a verdict, not the degraded
  ``{"error": ...}`` dict (``is_calibrated`` used to be read off a closed bus).
* ``_connect_robot()`` refusing an uncalibrated arm closes what it opened, and
  refuses again on the second call instead of short-circuiting past the gate
  on ``is_connected`` - measured on a real SO-101 before the fix.
* A device an observe action opened is handed back before ``connect()`` - the
  camera ``render`` opened as much as the bus ``get_state`` opened. The fakes
  follow lerobot's ``@check_if_already_connected`` contract on every device's
  ``connect()``, because an open camera is what let the driver's own connect
  *succeed* with ``configure()`` skipped: a one-camera arm then reads
  ``is_connected`` and its servos never received an operating mode or a
  torque limit. A hand-back that fails refuses the connect instead.

One test is marked ``hardware`` and reads a real arm when ``STRANDS_HW_PORT``
names its port; it is skipped everywhere else.
"""

from __future__ import annotations

import asyncio
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from lerobot.utils.errors import DeviceAlreadyConnectedError
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolUse

from strands_robots import hardware_observe
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor


class _Mode(Enum):
    DEGREES = "degrees"
    RANGE_0_100 = "range_0_100"


class _Motor:
    def __init__(self, id_: int, mode: _Mode) -> None:
        self.id = id_
        self.model = "sts3215"
        self.norm_mode = mode


class _NotConnected(Exception):
    pass


class _FakeBusWithoutCalibrationNotion:
    """The surface of lerobot's ``MotorsBus`` an observe action touches, with a ledger.

    No ``is_calibrated``: lerobot's contract for that property is "should be
    always True if not applicable", and a driver over a plain servo bus need not
    model calibration at all. Separate class rather than a subclass override,
    because the point is the *absence* of the attribute.
    """

    def __init__(self, *, calibrated: bool, ticks: dict[str, int] | None = None) -> None:
        self.port = "/dev/fake-bus"
        self.motors = {
            "shoulder_pan": _Motor(1, _Mode.DEGREES),
            "elbow_flex": _Motor(3, _Mode.DEGREES),
            "gripper": _Motor(6, _Mode.RANGE_0_100),
        }
        self.ticks = ticks or {"shoulder_pan": 2048, "elbow_flex": 3072, "gripper": 2180}
        self.torque = {name: 0 for name in self.motors}
        self._calibrated = calibrated
        self.is_connected = False
        self.connects = 0
        self.disconnects: list[bool] = []
        self.writes: list[tuple[str, Any]] = []
        self.reads: list[tuple[str, str]] = []

    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("already connected")
        self.connects += 1
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnects.append(disable_torque)
        self.is_connected = False

    def sync_read(self, register: str, *, normalize: bool = True, num_retry: int = 0) -> dict[str, float]:
        if not self.is_connected:
            raise _NotConnected("FakeBus is not connected")
        self.reads.append((register, "sync"))
        if not normalize:
            return dict(self.ticks)
        if not self._calibrated:
            raise RuntimeError(f"{self} has no calibration registered.")
        # A stand-in calibration: degrees around 2048, gripper 0-100.
        out: dict[str, float] = {}
        for name, t in self.ticks.items():
            out[name] = (t - 2048) / 4096 * 360 + 1.0 if name != "gripper" else 42.0
        return out

    def read(self, register: str, motor: str, *, normalize: bool = True) -> int:
        if not self.is_connected:
            raise _NotConnected("FakeBus is not connected")
        self.reads.append((register, motor))
        if register == "Torque_Enable":
            return self.torque[motor]
        if register == "Present_Voltage":
            return 56
        raise KeyError(register)

    def write(self, *args: Any, **kwargs: Any) -> None:
        self.writes.append((args[0], args[1:]))

    def sync_write(self, *args: Any, **kwargs: Any) -> None:
        self.writes.append((args[0], args[1:]))

    def enable_torque(self, *a: Any, **k: Any) -> None:
        self.writes.append(("Torque_Enable", 1))

    def disable_torque(self, *a: Any, **k: Any) -> None:
        self.writes.append(("Torque_Enable", 0))


class FakeBus(_FakeBusWithoutCalibrationNotion):
    """The same bus, declaring ``is_calibrated`` the way a lerobot bus does."""

    @property
    def is_calibrated(self) -> bool:
        if not self.is_connected:
            raise _NotConnected("FakeBus is not connected. Run `.connect()` first.")
        return self._calibrated


class _BusWhoseCalibrationReadRaises(FakeBus):
    """A bus that declares ``is_calibrated`` and cannot answer it.

    ``error`` is what the read raises: an ``AttributeError`` from inside the
    property (a driver whose flag is ``self.bus.is_calibrated`` over a lazily
    built bus) or a serial failure of the homing-offset sweep the property runs
    on a live arm. Either way the read failed; the arm is not "uncalibrated".
    """

    def __init__(self, *, error: Exception, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.error = error

    @property
    def is_calibrated(self) -> bool:
        raise self.error


class _BusWithoutTorqueRegister(FakeBus):
    """A bus whose control table cannot answer ``Torque_Enable``.

    ``answers`` names the motors that do reply, so one cell grades "no motor
    answered" and another "some did".
    """

    def __init__(self, *, answers: tuple[str, ...] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.answers = answers

    def read(self, register: str, motor: str, *, normalize: bool = True) -> int:
        if register == "Torque_Enable" and motor not in self.answers:
            raise KeyError(register)
        return super().read(register, motor, normalize=normalize)


class FakeCamera:
    def __init__(self, shape: tuple[int, int, int] = (480, 640, 3)) -> None:
        self.shape = shape
        self.is_connected = False
        self.connects = 0
        self.reads = 0

    def connect(self) -> None:
        # lerobot's every camera ``connect()`` is ``@check_if_already_connected``,
        # and the robot's connect loop has no handler of its own for it.
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")
        self.connects += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def read(self) -> np.ndarray:
        self.reads += 1
        frame = np.zeros(self.shape, dtype=np.uint8)
        frame[..., 0] = 200
        return frame


class _CamConfig:
    def __init__(self) -> None:
        self.index_or_path = 0
        self.width = 640
        self.height = 480
        self.fps = 30


class _RobotConfig:
    def __init__(self, cameras: dict[str, Any]) -> None:
        self.cameras = cameras


class FakeLeRobot:
    """A lerobot-shaped arm: bus + cameras, ``connect()`` runs configure (a WRITE)."""

    name = "so_follower"
    robot_type = "so_follower"

    def __init__(
        self,
        *,
        calibrated: bool = True,
        cameras: dict[str, FakeCamera] | None = None,
        bus: Any = None,
    ) -> None:
        self.bus: Any = bus if bus is not None else FakeBus(calibrated=calibrated)
        self.cameras = cameras if cameras is not None else {}
        self.config = _RobotConfig({name: _CamConfig() for name in self.cameras})
        self.configure_calls = 0

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(c.is_connected for c in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def configure(self) -> None:
        self.configure_calls += 1
        self.bus.writes.append(("Operating_Mode", 0))

    def disconnect(self) -> None:
        self.bus.disconnect(disable_torque=True)
        for cam in self.cameras.values():
            cam.disconnect()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.bus.sync_write("Goal_Position", action)
        return action

    def __str__(self) -> str:
        return "fake SOFollower"


class _CountingCalibration(FakeLeRobot):
    """Records every evaluation of ``is_calibrated``: on a real arm each is a bus sweep."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calibration_reads = 0

    @property
    def is_calibrated(self) -> bool:
        self.calibration_reads += 1
        return self.bus.is_calibrated


class _NoCalibrationNotion:
    """A driver with no ``is_calibrated`` at all: lerobot allows that, so must the gate.

    Not a ``FakeLeRobot`` subclass, because the point is the *absence* of the
    attribute and an inherited property cannot be removed.
    """

    def __init__(self, *, calibrated: bool = True, cameras: dict[str, FakeCamera] | None = None) -> None:
        self.bus = FakeBus(calibrated=calibrated)
        self.cameras = cameras if cameras is not None else {}
        self.config = _RobotConfig({name: _CamConfig() for name in self.cameras})

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        self.bus.disconnect(disable_torque=True)
        for cam in self.cameras.values():
            cam.disconnect()

    def __str__(self) -> str:
        return "fake driver without calibration"


class _RaisingCalibration(FakeLeRobot):
    """A driver whose ``is_calibrated`` raises ``AttributeError`` from inside itself."""

    @property
    def is_calibrated(self) -> bool:
        return bool(self.bus.no_calibration_here)  # type: ignore[attr-defined]


def _make_hw(robot: FakeLeRobot) -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = 30.0
    hw.action_sleep_time = 1.0 / 30.0
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = robot
    return hw


def _call(hw: HwRobot, **tool_input: Any) -> dict[str, Any]:
    tool_use = cast(ToolUse, {"toolUseId": "tu-observe", "input": tool_input})

    async def _run() -> list:
        return [ev async for ev in hw.stream(tool_use, {})]

    events = asyncio.run(_run())
    assert isinstance(events[-1], ToolResultEvent)
    return dict(events[-1].tool_result)


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "renders"
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(root))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    return root


class TestToolSpecIsHonest:
    def test_observe_actions_are_in_the_enum_and_the_description_leads_with_them(self) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True))
        spec = hw.tool_spec
        enum = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert set(hardware_observe.OBSERVE_ACTIONS) <= set(enum)
        assert enum.index("get_state") < enum.index("execute")
        assert spec["inputSchema"]["json"]["properties"]["action"]["default"] == "get_state"
        # The description tells the agent how to look without moving, names the
        # port, and says which sim verbs do NOT exist here.
        assert "get_state" in spec["description"].split("Motion")[0]
        assert "/dev/fake-bus" in spec["description"]
        assert "No set_joint_positions/move_to" in spec["description"]

    def test_unknown_action_lists_every_real_action(self) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True))
        result = _call(hw, action="wave")
        assert result["status"] == "error"
        for name in ("get_state", "list_cameras", "render", "execute", "status", "stop"):
            assert name in _text(result)


class TestGetStateIsAReadAndOnlyARead:
    @pytest.mark.parametrize("action", ["get_state", "get_robot_state"])
    def test_calibrated_arm_reports_degrees_from_the_calibration(self, action: str) -> None:
        robot = FakeLeRobot(calibrated=True)
        hw = _make_hw(robot)

        result = _call(hw, action=action)

        assert result["status"] == "success", _text(result)
        state = _json(result)
        assert state["calibrated"] is True
        assert state["opened_bus"] is True
        assert state["joints"]["elbow_flex"]["ticks"] == 3072
        assert state["joints"]["elbow_flex"]["degrees_source"] == "calibration"
        assert state["joints"]["elbow_flex"]["degrees"] == pytest.approx(91.0, abs=0.01)
        assert state["joints"]["gripper"]["normalized"] == 42.0
        assert state["joints"]["gripper"]["normalized_unit"] == "range_0_100"
        assert state["joints"]["shoulder_pan"]["torque_enabled"] is False
        assert state["joints"]["shoulder_pan"]["voltage_v"] == 5.6
        assert state["torque_enabled_any"] is False
        text = _text(result)
        assert "torque OFF on all joints" in text
        assert "calibrated" in text and "NOT calibrated" not in text
        assert "elbow_flex (id 3): +91.0°  [3072 ticks]  torque off  5.6 V" in text

    def test_the_bus_is_opened_alone_and_nothing_is_written(self) -> None:
        robot = FakeLeRobot(calibrated=True, cameras={"front": FakeCamera()})
        hw = _make_hw(robot)

        _call(hw, action="get_state")

        assert robot.bus.connects == 1
        assert robot.bus.is_connected is True
        assert robot.configure_calls == 0, "the robot's connect() runs configure(), which writes servo registers"
        assert robot.cameras["front"].connects == 0, "a joint read does not open cameras"
        assert robot.bus.writes == []
        registers = {r for r, _ in robot.bus.reads}
        assert registers == {"Present_Position", "Torque_Enable", "Present_Voltage"}

    def test_second_read_reuses_the_open_bus(self) -> None:
        robot = FakeLeRobot(calibrated=True)
        hw = _make_hw(robot)
        _call(hw, action="get_state")
        second = _call(hw, action="get_state")
        assert robot.bus.connects == 1
        assert _json(second)["opened_bus"] is False

    def test_uncalibrated_arm_is_still_readable_and_says_so(self) -> None:
        robot = FakeLeRobot(calibrated=False)
        robot.bus.ticks["shoulder_pan"] = 1978
        hw = _make_hw(robot)

        result = _call(hw, action="get_state")

        assert result["status"] == "success", _text(result)
        state = _json(result)
        assert state["calibrated"] is False
        joint = state["joints"]["shoulder_pan"]
        assert joint["degrees_source"] == "encoder_estimate"
        assert joint["degrees"] == pytest.approx(hardware_observe.ticks_to_degrees(1978), abs=0.01)
        assert joint["degrees"] == pytest.approx(-6.15, abs=0.01)
        text = _text(result)
        assert "NOT calibrated" in text
        assert "lerobot-calibrate" in text
        assert "2048 ticks = 0°" in text
        assert robot.bus.writes == []

    def test_torque_state_is_reported_per_joint(self) -> None:
        robot = FakeLeRobot(calibrated=True)
        robot.bus.torque["gripper"] = 1
        hw = _make_hw(robot)
        result = _call(hw, action="get_state")
        assert _json(result)["torque_enabled_any"] is True
        assert "torque ON on gripper; OFF on the rest" in _text(result)

    @pytest.mark.parametrize(
        "error",
        [
            AttributeError("'FakeBus' object has no attribute '_calibration_cache'"),
            ConnectionError("No status packet received from shoulder_pan"),
        ],
        ids=["raised_inside_the_property", "the_servo_sweep_failed"],
    )
    def test_a_calibration_flag_that_cannot_be_read_is_unread_not_uncalibrated(self, error: Exception) -> None:
        """A failed calibration read used to demote the degrees to an estimate.

        The flag was read with ``getattr(bus, "is_calibrated", False)``, which
        answers ``False`` for an ``AttributeError`` from *inside* the property -
        what a driver whose flag is ``self.bus.is_calibrated`` over a lazily
        built bus raises - and propagates any other failure of the servo sweep
        the property runs, killing a read that already had the positions. Read
        as "not calibrated", the arm's own normalised reading was never asked
        for: the degrees silently became encoder estimates, and the text sent the
        operator to ``lerobot-calibrate`` for a fault that is not calibration.
        """
        robot = FakeLeRobot(bus=_BusWhoseCalibrationReadRaises(calibrated=True, error=error))

        result = _call(_make_hw(robot), action="get_state")

        state = _json(result)
        assert state["calibrated"] is None, "unread, which is neither True nor False"
        assert state["calibration_error"] == str(error)
        # The arm can still normalise, so the degrees are ITS degrees, not ticks arithmetic.
        elbow = state["joints"]["elbow_flex"]
        assert elbow["degrees_source"] == "calibration"
        assert elbow["degrees"] == 91.0
        assert elbow["degrees"] != pytest.approx(hardware_observe.ticks_to_degrees(3072))
        text = _text(result)
        assert "calibration state UNREAD" in text
        assert "lerobot-calibrate" not in text, "the calibration is not the fault to report"

    def test_a_bus_with_no_notion_of_calibration_is_not_reported_uncalibrated(self) -> None:
        """lerobot's contract: ``is_calibrated`` "should be always True if not applicable".

        A driver that does not model calibration was reported NOT calibrated,
        with text promising a rollout would refuse until ``lerobot-calibrate``
        ran - while the connect gate lets exactly that driver through.
        """
        robot = FakeLeRobot(bus=_FakeBusWithoutCalibrationNotion(calibrated=True))

        result = _call(_make_hw(robot), action="get_state")

        state = _json(result)
        assert state["calibrated"] is True
        assert "calibration_error" not in state
        assert "NOT calibrated" not in _text(result)

    def test_an_unread_torque_register_is_not_reported_as_torque_off(self) -> None:
        """ "the arm can be moved by hand" is a claim about a live arm.

        ``any()`` over registers no motor answered used to make it for free.
        """
        robot = FakeLeRobot(bus=_BusWithoutTorqueRegister(calibrated=True))

        result = _call(_make_hw(robot), action="get_state")

        state = _json(result)
        assert state["torque_enabled_any"] is None
        assert all(j["torque_enabled"] is None for j in state["joints"].values())
        text = _text(result)
        assert "torque state UNREAD on every joint" in text
        assert "moved by hand" not in text

    def test_a_partly_unread_torque_register_names_the_unread_joints(self) -> None:
        bus = _BusWithoutTorqueRegister(calibrated=True, answers=("gripper",))
        bus.torque["gripper"] = 1

        result = _call(_make_hw(FakeLeRobot(bus=bus)), action="get_state")

        assert _json(result)["torque_enabled_any"] is True
        assert "torque ON on gripper; UNREAD on shoulder_pan, elbow_flex" in _text(result)

    def test_a_bus_that_cannot_open_becomes_an_error_naming_the_port(self) -> None:
        robot = FakeLeRobot(calibrated=True)

        def _refuse(handshake: bool = True) -> None:
            raise ConnectionError("Could not connect on port '/dev/fake-bus'. Try running `lerobot-find-port`")

        robot.bus.connect = _refuse  # type: ignore[method-assign]
        hw = _make_hw(robot)

        result = _call(hw, action="get_state")

        assert result["status"] == "error"
        text = _text(result)
        assert "/dev/fake-bus" in text
        assert "lsof" in text and "lerobot-find-port" in text


class TestCameras:
    def test_list_cameras_opens_nothing(self) -> None:
        robot = FakeLeRobot(calibrated=True, cameras={"front": FakeCamera(), "wrist": FakeCamera()})
        hw = _make_hw(robot)

        result = _call(hw, action="list_cameras")

        assert result["status"] == "success"
        cams = _json(result)["cameras"]
        assert [c["name"] for c in cams] == ["front", "wrist"]
        assert cams[0] == {
            "name": "front",
            "type": "FakeCamera",
            "connected": False,
            "index_or_path": "0",
            "width": 640,
            "height": 480,
            "fps": 30,
        }
        assert robot.bus.connects == 0
        assert all(c.connects == 0 for c in robot.cameras.values())
        assert "front: FakeCamera source 0 640x480@30fps - closed (opens on first render)" in _text(result)

    def test_list_cameras_with_none_tells_how_to_add_one(self) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True))
        result = _call(hw, action="list_cameras")
        assert result["status"] == "success"
        assert "cameras=" in _text(result) and "index_or_path" in _text(result)

    def test_render_opens_the_camera_lazily_and_saves_a_png_in_the_sandbox(self, sandbox: Path) -> None:
        cam = FakeCamera()
        robot = FakeLeRobot(calibrated=True, cameras={"front": cam})
        hw = _make_hw(robot)

        result = _call(hw, action="render", camera_name="front")

        assert result["status"] == "success", _text(result)
        frame = _json(result)
        path = Path(frame["path"])
        assert path.is_relative_to(sandbox)
        assert path.suffix == ".png" and path.name.startswith("arm-front-")
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert (frame["width"], frame["height"], frame["channels"]) == (640, 480, 3)
        assert frame["opened_camera"] is True
        assert cam.connects == 1 and cam.reads == 1
        assert robot.bus.connects == 0, "a frame grab does not touch the motor bus"
        assert "640x480" in _text(result) and str(path) in _text(result)
        # the model is HANDED the frame - a path alone is a frame it describes from expectation
        images = [block["image"] for block in result["content"] if "image" in block]
        assert len(images) == 1 and images[0]["format"] == "png"
        assert images[0]["source"]["bytes"] == path.read_bytes()
        assert "png" not in frame, "the json block is metadata; the bytes travel in the image block"

        again = _call(hw, action="render", camera_name="front", output_path="second.png")
        assert _json(again)["opened_camera"] is False
        assert Path(_json(again)["path"]) == sandbox / "second.png"

    def test_render_with_one_camera_needs_no_name(self, sandbox: Path) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True, cameras={"only": FakeCamera()}))
        result = _call(hw, action="render")
        assert result["status"] == "success", _text(result)
        assert _json(result)["camera"] == "only"

    def test_render_with_two_cameras_and_no_name_is_refused_naming_them(self, sandbox: Path) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True, cameras={"a": FakeCamera(), "b": FakeCamera()}))
        result = _call(hw, action="render")
        assert result["status"] == "error"
        assert "camera_name is required" in _text(result) and "['a', 'b']" in _text(result)

    def test_render_unknown_camera_lists_the_real_ones(self, sandbox: Path) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True, cameras={"front": FakeCamera()}))
        result = _call(hw, action="render", camera_name="top")
        assert result["status"] == "error"
        assert "Camera 'top' not found. Available: ['front']" in _text(result)

    def test_render_outside_the_sandbox_is_refused_with_the_env_var(self, sandbox: Path, tmp_path: Path) -> None:
        cam = FakeCamera()
        hw = _make_hw(FakeLeRobot(calibrated=True, cameras={"front": cam}))
        outside = tmp_path / "elsewhere" / "frame.png"
        result = _call(hw, action="render", camera_name="front", output_path=str(outside))
        assert result["status"] == "error"
        assert "STRANDS_ROBOTS_RENDER_ALLOW_ABS" in _text(result)
        assert not outside.exists()

    def test_render_with_no_cameras_configured_says_how_to_add_one(self, sandbox: Path) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True))
        result = _call(hw, action="render")
        assert result["status"] == "error"
        assert "cameras=" in _text(result)


class TestStatusOnAnIdleArm:
    def test_get_status_before_any_connect_is_a_verdict_not_an_error(self) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=True))
        status = asyncio.run(hw.get_status())
        assert "error" not in status
        assert status["is_connected"] is False
        assert status["is_calibrated"] is None, "unknown until the bus is open"
        assert status["task_status"] == "idle"

    def test_get_status_after_a_read_knows_the_calibration(self) -> None:
        hw = _make_hw(FakeLeRobot(calibrated=False))
        _call(hw, action="get_state")
        status = asyncio.run(hw.get_status())
        assert status["is_connected"] is True
        assert status["is_calibrated"] is False


class TestCalibrationGateHoldsOnEveryCall:
    def test_refusal_closes_what_it_opened_and_refuses_again(self) -> None:
        robot = FakeLeRobot(calibrated=False, cameras={"front": FakeCamera()})
        hw = _make_hw(robot)

        first = asyncio.run(hw._connect_robot())
        assert first[0] is False and "not calibrated" in first[1]
        assert robot.is_connected is False, "a refused connect must not hold the port"
        assert robot.cameras["front"].is_connected is False

        second = asyncio.run(hw._connect_robot())
        assert second[0] is False, "used to short-circuit on is_connected and return (True, '')"
        assert "not calibrated" in second[1]

    def test_a_bus_opened_by_get_state_is_handed_back_to_connect(self) -> None:
        robot = FakeLeRobot(calibrated=True, cameras={"front": FakeCamera()})
        hw = _make_hw(robot)
        _call(hw, action="get_state")
        assert robot.bus.is_connected and not robot.is_connected

        ok, err = asyncio.run(hw._connect_robot())

        assert (ok, err) == (True, "")
        assert robot.is_connected is True
        assert robot.configure_calls == 1, "the driver ran its full open sequence"
        # The hand-back closed without a torque write: nothing had been energised.
        assert robot.bus.disconnects[0] is False

    def test_a_camera_opened_by_render_is_handed_back_to_connect(self, sandbox: Path) -> None:
        """An open camera used to make connect() SUCCEED with configure() skipped.

        lerobot's ``connect()`` opens the bus, then each camera, then runs
        ``configure()``. A camera already open raises in that loop - after the
        bus is up, before ``configure()`` - and the handler read it as "already
        connected". With one camera ``is_connected`` then reads True, so the
        rollout drove servos that never received ``Operating_Mode`` or the
        gripper's torque limits. Measured on this fake before the fix:
        ``(True, "")`` with ``configure_calls == 0``.
        """
        cam = FakeCamera()
        robot = FakeLeRobot(calibrated=True, cameras={"front": cam})
        hw = _make_hw(robot)
        _call(hw, action="render", camera_name="front")
        assert cam.is_connected and not robot.bus.is_connected

        ok, err = asyncio.run(hw._connect_robot())

        assert (ok, err) == (True, "")
        assert robot.is_connected is True
        assert robot.configure_calls == 1, "the driver ran its full open sequence, configure() included"
        assert cam.connects == 2, "handed back, then reopened by the driver"

    def test_a_bus_opened_by_get_state_on_a_camera_less_arm_is_handed_back_too(self) -> None:
        """With no cameras, the open bus alone IS ``is_connected``.

        A hand-back gated on ``not is_connected`` never fires here, so connect()
        is skipped outright and a rollout follows on an arm whose ``configure()``
        never ran - the same servos-never-configured state as the camera case,
        reached without a camera. Measured before the fix: ``(True, "")`` with
        ``configure_calls == 0``.
        """
        robot = FakeLeRobot(calibrated=True)
        hw = _make_hw(robot)
        _call(hw, action="get_state")
        assert robot.is_connected, "no cameras: an open bus reads as a connected robot"

        ok, err = asyncio.run(hw._connect_robot())

        assert (ok, err) == (True, "")
        assert robot.configure_calls == 1, "the driver's connect() ran, configure() included"
        assert robot.bus.disconnects == [False] and robot.bus.connects == 2

    def test_a_robot_the_caller_connected_is_left_as_they_left_it(self) -> None:
        """The ledger records what the tool borrowed, not what the caller opened."""
        robot = FakeLeRobot(calibrated=True, cameras={"front": FakeCamera()})
        robot.connect()
        hw = _make_hw(robot)

        assert asyncio.run(hw._connect_robot()) == (True, "")
        assert robot.configure_calls == 1, "already connected by the caller; not torn down and redone"
        assert robot.bus.disconnects == []

    def test_a_teleop_write_hands_back_the_bus_get_state_opened(self) -> None:
        """The teleop loop's lazy connect is the second path a write reaches the arm by."""
        robot = FakeLeRobot(calibrated=True)
        hw = _make_hw(robot)
        _call(hw, action="get_state")

        result = hw.send_action({"shoulder_pan.pos": 1.0})

        assert result["status"] == "success", result
        assert robot.configure_calls == 1, "a write must not reach servos configure() never set up"
        assert robot.bus.disconnects == [False]

    def test_a_camera_opened_by_render_on_a_two_camera_arm_is_handed_back_too(self, sandbox: Path) -> None:
        """With two cameras the same state failed the connect outright; now it connects."""
        robot = FakeLeRobot(calibrated=True, cameras={"front": FakeCamera(), "wrist": FakeCamera()})
        hw = _make_hw(robot)
        _call(hw, action="render", camera_name="wrist")

        ok, err = asyncio.run(hw._connect_robot())

        assert (ok, err) == (True, "")
        assert robot.configure_calls == 1

    def test_a_device_that_cannot_be_handed_back_refuses_the_connect(self, sandbox: Path) -> None:
        """A failed hand-back must not fall through to the driver's "already connected".

        If the close raised and connect() went ahead, the driver would raise
        ``DeviceAlreadyConnectedError`` in its camera loop and the handler would
        wave it through - the exact skip-configure state the hand-back exists to
        prevent. So the refusal is the connect's, and nothing was configured.
        """

        class _StuckCamera(FakeCamera):
            def disconnect(self) -> None:
                raise OSError("v4l2 device is busy")

        cam = _StuckCamera()
        robot = FakeLeRobot(calibrated=True, cameras={"front": cam})
        hw = _make_hw(robot)
        _call(hw, action="render", camera_name="front")

        ok, err = asyncio.run(hw._connect_robot())

        assert ok is False and "device is busy" in err, err
        assert robot.configure_calls == 0, "a connect that was refused must not have configured anything"
        assert robot.bus.is_connected is False, "the refusal rolled back the bus the driver had opened"

    def test_the_bus_is_handed_back_under_the_lock_an_observe_read_holds(self) -> None:
        """A read in flight finishes before the hand-back closes the port under it."""
        from strands_robots.bus_access import bus_lock

        robot = FakeLeRobot(calibrated=True)
        hw = _make_hw(robot)
        _call(hw, action="get_state")
        assert robot.bus.is_connected

        outcome: list[tuple[bool, str]] = []
        lock = bus_lock(robot)
        lock.acquire()  # a probe mid-read
        try:
            worker = threading.Thread(target=lambda: outcome.append(asyncio.run(hw._connect_robot())))
            worker.start()
            worker.join(timeout=0.3)
            assert worker.is_alive(), "the hand-back closed the port under a read that held the lock"
            assert robot.bus.disconnects == []
        finally:
            lock.release()
        worker.join(timeout=5)

        assert outcome == [(True, "")]
        assert robot.bus.disconnects == [False]
        assert robot.configure_calls == 1

    def test_an_uncalibrated_arm_read_first_is_still_refused_by_connect(self) -> None:
        robot = FakeLeRobot(calibrated=False)
        hw = _make_hw(robot)
        _call(hw, action="get_state")
        ok, err = asyncio.run(hw._connect_robot())
        assert ok is False and "not calibrated" in err
        assert robot.bus.is_connected is False

    def test_the_gate_reads_the_calibration_once_per_connect(self) -> None:
        """``is_calibrated`` is a bus sweep, so asking twice doubles it for nothing."""
        robot = _CountingCalibration(calibrated=True, cameras={"front": FakeCamera()})
        hw = _make_hw(robot)

        assert asyncio.run(hw._connect_robot()) == (True, "")

        assert robot.calibration_reads == 1, "hasattr on the instance evaluated the property too"

    def test_the_gate_is_not_skipped_when_is_calibrated_raises(self) -> None:
        """A property that raises must refuse the arm, not be read as "no such property".

        ``hasattr`` swallows an ``AttributeError`` raised *inside* the property -
        which is what a driver whose ``is_calibrated`` is ``self.bus.is_calibrated``
        over a lazily built bus raises - and the whole gate was then skipped:
        ``(True, "")`` for an arm whose calibration was never checked, cameras open
        and ``configure()`` already run.
        """
        robot = _RaisingCalibration(calibrated=False, cameras={"front": FakeCamera()})
        hw = _make_hw(robot)

        ok, err = asyncio.run(hw._connect_robot())

        assert ok is False, "an unanswerable calibration check must not permit motion"
        assert "no_calibration_here" in err, err
        assert robot.is_connected is False, "the refused arm must not keep the port"

    def test_a_driver_with_no_calibration_notion_is_still_allowed(self) -> None:
        """Absent is not unreadable: lerobot's contract is "always True if not applicable"."""
        robot = _NoCalibrationNotion(cameras={"front": FakeCamera()})
        hw = _make_hw(cast(Any, robot))

        assert asyncio.run(hw._connect_robot()) == (True, "")

    def test_calibration_declared_on_the_instance_still_gates(self) -> None:
        """A driver that carries the flag as a plain attribute, not a property, is checked."""
        robot = _NoCalibrationNotion(cameras={"front": FakeCamera()})
        robot.is_calibrated = False  # type: ignore[attr-defined]  # a plain flag, not a property
        hw = _make_hw(cast(Any, robot))

        ok, err = asyncio.run(hw._connect_robot())

        assert ok is False and "not calibrated" in err


class TestTicks:
    @pytest.mark.parametrize(
        ("ticks", "degrees"),
        [(2048, 0.0), (0, -180.0), (4096, 180.0), (3072, 90.0), (1978, -6.15234375)],
    )
    def test_encoder_estimate(self, ticks: int, degrees: float) -> None:
        assert hardware_observe.ticks_to_degrees(ticks) == pytest.approx(degrees)


@pytest.mark.hardware
def test_real_arm_get_state_reads_six_joints_torque_off() -> None:
    """Read the real arm named by STRANDS_HW_PORT. Never writes; the bus is closed after."""
    port = os.environ.get("STRANDS_HW_PORT")
    if not port:
        pytest.skip("set STRANDS_HW_PORT=/dev/... to read a real arm")
    from strands_robots import Robot

    arm = Robot("so101", mode="real", port=port)
    try:
        result = _call(arm, action="get_state")
        assert result["status"] == "success", _text(result)
        state = _json(result)
        assert state["port"] == port
        assert len(state["joints"]) == 6
        assert all("ticks" in j and "torque_enabled" in j for j in state["joints"].values())
        assert state["torque_enabled_any"] is False, "the lane's arm must be readable by hand"
        assert asyncio.run(arm.get_status())["is_connected"] is True
    finally:
        arm.cleanup()
    assert arm.robot.bus.is_connected is False
