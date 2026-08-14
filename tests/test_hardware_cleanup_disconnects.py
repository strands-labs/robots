# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``cleanup()`` must close the devices, not just the resources the library owns.

``Robot.cleanup()`` tears down everything *software* holds -- it latches
``_shutdown_event``, stops a teleop loop, stops a ``RUNNING`` task, shuts the
task executor, stops the mesh client and the ROS bridge. It never called
``self.robot.disconnect()``, so the only resources that are a physical device --
the motors bus serial port, plus one ``/dev/video*`` node and one read thread
per camera -- were the only ones it left open. Measured with a driver double
recording every call, on a one-camera arm:

    after                          bus open   camera open   disconnect calls
    connect + a healthy rollout      yes          yes              0
    cleanup()                        yes          yes              0
    a second cleanup()               yes          yes              0

That is not untidiness. A serial port is exclusive on Linux and macOS, so a
second process -- or a re-constructed ``Robot`` in this one -- cannot open the
same ``/dev/tty*``, which makes the documented recovery for a wedged arm (tear
down, reconnect) unavailable without exiting the process. ``cleanup()`` is also
what ``__del__`` calls, so a script that simply ends left the arm energised at
its last commanded position instead of going through the driver's disconnect,
where torque disable and gripper release live. And it was unrecoverable
afterwards: the executor is shut down and ``_shutdown_event`` is set, so no
library entry point remained that would reach a disconnect, while lerobot's
``Robot.disconnect()`` is ``@check_if_not_connected`` and so cannot be called by
hand in a half-open state.

What these tests pin:

    - the round trip -- connect, rollout, ``cleanup()`` -- closes the bus and
      every camera, each disconnected exactly once, and the port is free for
      the next holder;
    - ``cleanup()`` stays idempotent: a second call disconnects nothing again
      rather than raising ``DeviceNotConnectedError`` from the driver;
    - the driver's own ``disconnect()`` is what runs while it can, because that
      is where torque disable lives;
    - a camera whose close raises does not keep the bus or the remaining
      cameras open, and is warned rather than propagated -- lerobot's own
      ``disconnect()`` is a single unguarded loop, so the first close that
      raises abandons every device after it;
    - a half-open robot (``is_connected`` False with the port still open) still
      gets that port closed, which is the state no other entry point can reach;
    - a driver exposing no ``disconnect`` at all is not an error;
    - the devices close *last*, after the mesh and the ROS bridge, and after the
      executor has drained -- ``send_action`` re-opens the robot lazily, so a
      port closed while any command source is still live gets re-opened behind
      the teardown and then stays open for the life of the process.

No serial port and no camera device is opened: the lerobot driver, its bus and
its cameras are in-memory doubles that mirror lerobot's connect ordering, its
``is_connected`` composition and its decorator contracts. Port exclusivity is
modelled explicitly so the consequence, not just the call, is asserted.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor

#: Upper bound on any wait, so a broken contract fails instead of hanging.
DEADLINE = 10.0


class _Port:
    """An exclusive device node: at most one holder at a time.

    Modelled because it is the reason the leak matters. Asserting only that
    ``disconnect()`` was called would pass on a driver that records the call
    and keeps the node.
    """

    def __init__(self) -> None:
        self.held_by: str | None = None

    def open(self, holder: str) -> None:
        if self.held_by is not None:
            raise OSError(f"[Errno 16] Device or resource busy: /dev/ttyACM0 held by {self.held_by}")
        self.held_by = holder

    def close(self) -> None:
        self.held_by = None


class _Bus:
    """Motors-bus double holding an exclusive port and recording every close."""

    def __init__(self, port: _Port, *, holder: str = "arm", log: list[str] | None = None) -> None:
        self.port = port
        self.holder = holder
        self.log = log if log is not None else []
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []

    def connect(self) -> None:
        self.connect_calls += 1
        self.port.open(self.holder)
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        self.log.append("bus.disconnect")
        self.port.close()
        self.is_connected = False


class _Camera:
    """Mirrors ``OpenCVCamera``'s connect / disconnect / ``is_connected``."""

    def __init__(self, name: str, *, close_raises: bool = False, log: list[str] | None = None) -> None:
        self.name = name
        self.log = log if log is not None else []
        self._close_raises = close_raises
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self, warmup: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        self.connect_calls += 1
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.log.append(f"camera.{self.name}.disconnect")
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        if self._close_raises:
            raise OSError(f"{self.name} close failed")
        self.is_connected = False


class _Driver:
    """Mirrors lerobot ``SOFollower``: composed ``is_connected``, gated methods.

    ``disconnect()`` reproduces lerobot's single unguarded loop -- the bus
    first, then each camera in turn, with nothing catching a close that raises
    -- because that shape is what the fallback has to cover.
    """

    def __init__(self, cameras: dict[str, _Camera], bus: _Bus, log: list[str] | None = None) -> None:
        self.name = self.robot_type = "fake_arm"
        self.bus = bus
        self.cameras = cameras
        self.log = log if log is not None else []
        self.is_calibrated = True
        self.disconnect_calls = 0
        self.commands: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": cameras})()

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        if self.is_connected:  # @check_if_already_connected
            raise DeviceAlreadyConnectedError("fake_arm is already connected.")
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.log.append("robot.disconnect")
        if not self.is_connected:  # @check_if_not_connected
            raise DeviceNotConnectedError("fake_arm is not connected. Run `.connect()` first.")
        self.bus.disconnect(True)
        for cam in self.cameras.values():
            cam.disconnect()

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.commands.append(action)


class _BusOnlyDriver:
    """A driver exposing a bus but no ``disconnect`` of its own."""

    def __init__(self, bus: _Bus) -> None:
        self.name = self.robot_type = "bus_only_arm"
        self.bus = bus
        self.cameras: dict[str, _Camera] = {}
        self.is_calibrated = True

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        self.bus.connect()


class _Mesh:
    """Mesh component double: any object exposing ``stop()``."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def stop(self) -> None:
        self.log.append("mesh.stop")


class _Bridge:
    """ROS 2 bridge double, torn down by ``_shutdown_ros_bridge()``."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def shutdown(self) -> None:
        self.log.append("ros_bridge.shutdown")


def _make_robot(driver: Any) -> HwRobot:
    """Construct a ``Robot`` around ``driver``, bypassing hardware init."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = 1000.0
    hw.action_sleep_time = 0.001
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw.mesh = None
    hw.peer_id = None
    hw.robot = driver
    return hw


def _arm(port: _Port | None = None, log: list[str] | None = None, **camera_kwargs: Any) -> _Driver:
    """A one-camera arm on its own port."""
    shared: list[str] = log if log is not None else []
    return _Driver(
        {"wrist": _Camera("wrist_cam", log=shared, **camera_kwargs)},
        _Bus(port or _Port(), log=shared),
        log=shared,
    )


class TestCleanupClosesTheDevices:
    """The round trip: connect, drive, ``cleanup()``, nothing left open."""

    def test_cleanup_closes_the_bus_and_every_camera(self) -> None:
        """Every device the bring-up opened is shut, each exactly once.

        The bus close carries ``disable_torque=True``: the driver's own
        ``disconnect()`` ran, which is where a follower disables torque and
        releases the gripper. Closing the port underneath it would skip both
        and leave the arm energised at its last commanded position.
        """
        driver = _arm()
        hw = _make_robot(driver)
        ok, _ = asyncio.run(hw._connect_robot())
        assert ok is True
        hw.send_action({"j0.pos": 1.0})

        hw.cleanup()

        assert driver.disconnect_calls == 1
        assert driver.bus.is_connected is False
        assert driver.bus.disconnect_calls == [True]
        assert driver.cameras["wrist"].is_connected is False
        assert driver.cameras["wrist"].disconnect_calls == 1

    def test_the_port_is_free_for_the_next_holder(self) -> None:
        """The documented recovery for a wedged arm works without exiting.

        A serial port is exclusive, so this is the user-visible consequence:
        while ``cleanup()`` held the node, a re-constructed ``Robot`` on the
        same port raised ``EBUSY`` and the only recovery was a new process.
        """
        port = _Port()
        first = _make_robot(_arm(port))
        asyncio.run(first._connect_robot())
        first.cleanup()

        second = _make_robot(_arm(port))
        ok, err = asyncio.run(second._connect_robot())

        assert ok is True, err
        assert port.held_by == "arm"
        second.cleanup()

    def test_a_second_cleanup_does_not_disconnect_again(self) -> None:
        """``cleanup()`` is called by ``__del__`` as well as by hand, so a
        repeat must be a no-op rather than a ``DeviceNotConnectedError`` from
        the driver's gated ``disconnect()``."""
        driver = _arm()
        hw = _make_robot(driver)
        asyncio.run(hw._connect_robot())
        hw.cleanup()

        hw.cleanup()

        assert driver.disconnect_calls == 1
        assert driver.bus.disconnect_calls == [True]
        assert driver.cameras["wrist"].disconnect_calls == 1

    def test_cleanup_on_a_robot_that_never_connected_touches_nothing(self) -> None:
        """Nothing is open, so nothing is closed -- and the gated driver
        ``disconnect()`` is not called just to have it raise."""
        driver = _arm()
        hw = _make_robot(driver)

        hw.cleanup()

        assert driver.disconnect_calls == 0
        assert driver.bus.disconnect_calls == []
        assert driver.cameras["wrist"].disconnect_calls == 0


class TestOneStuckDeviceCannotKeepTheRestOpen:
    """lerobot's ``disconnect()`` abandons every device after a close that
    raises. The teardown has to survive that, not inherit it."""

    def test_a_camera_that_will_not_close_leaves_nothing_else_open(self) -> None:
        """The driver's loop raises on the camera, so the cameras behind it are
        never reached. The fallback closes each remaining device
        independently, and the port -- the exclusive resource -- ends free."""
        port = _Port()
        log: list[str] = []
        driver = _Driver(
            {
                "wrist": _Camera("wrist_cam", close_raises=True, log=log),
                "top": _Camera("top_cam", log=log),
            },
            _Bus(port, log=log),
            log=log,
        )
        hw = _make_robot(driver)
        asyncio.run(hw._connect_robot())

        hw.cleanup()

        assert driver.cameras["wrist"].is_connected is True  # genuinely stuck
        assert driver.cameras["top"].is_connected is False  # closed by the fallback
        assert driver.cameras["top"].disconnect_calls == 1
        assert port.held_by is None

    def test_the_stuck_camera_is_warned_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        """``cleanup()`` is a teardown and is called from ``__del__``: it
        reports the failure and finishes the rest, and the report names the
        disconnect rather than being swallowed at debug level."""
        driver = _arm(close_raises=True)
        hw = _make_robot(driver)
        asyncio.run(hw._connect_robot())

        with caplog.at_level(logging.WARNING, logger="strands_robots.hardware_robot"):
            hw.cleanup()

        assert any("robot.disconnect() raised during cleanup" in r.getMessage() for r in caplog.records)

    def test_a_half_open_robot_still_gets_its_port_closed(self) -> None:
        """The state no other entry point can recover.

        With the port open and a camera shut, ``is_connected`` is False, so
        lerobot's ``@check_if_not_connected`` ``disconnect()`` refuses. The
        fallback closes what is actually open -- and without a torque write,
        which on a bus in this state would raise before ``closePort`` and
        leave the node held.
        """
        port = _Port()
        driver = _arm(port)
        hw = _make_robot(driver)
        asyncio.run(hw._connect_robot())
        driver.cameras["wrist"].disconnect()  # half-open: port up, camera down
        assert driver.is_connected is False

        hw.cleanup()

        assert driver.disconnect_calls == 0  # not called only to be refused
        assert driver.bus.disconnect_calls == [False]
        assert port.held_by is None

    def test_a_driver_without_a_disconnect_is_not_an_error(self) -> None:
        """``self.robot`` is any lerobot-shaped object, including one whose
        teardown is its bus. The port still ends closed."""
        port = _Port()
        driver = _BusOnlyDriver(_Bus(port))
        hw = _make_robot(driver)
        asyncio.run(hw._connect_robot())

        hw.cleanup()

        assert driver.bus.is_connected is False
        assert port.held_by is None


class TestTheDevicesCloseLast:
    """A port closed while any command source is still live gets re-opened."""

    def test_the_devices_close_after_the_mesh_and_the_ros_bridge(self) -> None:
        """``send_action`` re-opens the robot lazily on a command that finds it
        disconnected, and both the mesh dispatch and the ROS bridge command
        thread reach it. Closing the devices before those are down would let a
        late command re-open the port behind the teardown -- where nothing
        remains that would close it again."""
        log: list[str] = []
        driver = _arm(log=log)
        hw = _make_robot(driver)
        hw.mesh = _Mesh(log)
        hw._ros_bridge = _Bridge(log)
        asyncio.run(hw._connect_robot())

        hw.cleanup()

        assert log.index("mesh.stop") < log.index("robot.disconnect")
        assert log.index("ros_bridge.shutdown") < log.index("robot.disconnect")

    def test_a_command_still_draining_finds_an_open_port(self) -> None:
        """The disconnect runs after ``_executor.shutdown(wait=True)`` returns,
        so a rollout still draining commands a live bus instead of a closed
        one -- and its command does not lazily re-open what was just shut."""
        driver = _arm()
        hw = _make_robot(driver)
        asyncio.run(hw._connect_robot())

        def _late_command() -> None:
            time.sleep(0.05)  # still in flight when cleanup() starts
            hw.send_action({"j0.pos": 2.0})

        future = hw._executor.submit(_late_command)

        hw.cleanup()

        future.result(timeout=DEADLINE)
        assert driver.commands == [{"j0.pos": 2.0}]
        assert driver.bus.connect_calls == 1  # never re-opened
        assert driver.bus.is_connected is False
