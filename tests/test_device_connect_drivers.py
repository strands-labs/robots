"""Unit tests for Device Connect DeviceDriver adapters.

Tests RobotDeviceDriver, SimulationDeviceDriver, ReachyMiniDriver,
init_device_connect(), and the updated robot_mesh tool.

All external dependencies (Zenoh, LeRobot, device_connect_edge, strands) are mocked.
"""

import asyncio
import json
import sys
import unittest
from dataclasses import dataclass
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock heavy dependencies before importing ──────────────────────

# Mock device_connect_edge
mock_device_connect_edge = MagicMock()
mock_drivers = MagicMock()


class _FakeDeviceDriver:
    """Minimal stub so our drivers can subclass it."""

    device_type = None

    def __init__(self):
        self._transport = None

    def set_device(self, device):
        pass

    @property
    def transport(self):
        return self._transport


# Make @rpc, @emit, @periodic, @on pass-through decorators
def _passthrough_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]

    def wrapper(func):
        # Tag the function so tests can verify decorator usage
        for k, v in kwargs.items():
            setattr(func, f"_{k}", v)
        return func

    return wrapper


mock_drivers.DeviceDriver = _FakeDeviceDriver
mock_drivers.rpc = _passthrough_decorator
mock_drivers.emit = _passthrough_decorator
mock_drivers.periodic = _passthrough_decorator
mock_drivers.on = _passthrough_decorator

mock_types = MagicMock()


@dataclass
class FakeDeviceIdentity:
    device_type: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    description: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    arch: str | None = None
    commissioning_comment: str | None = None


@dataclass
class FakeDeviceStatus:
    availability: str = "idle"
    busy_score: float = 0.0
    location: str | None = None
    battery: int | None = None
    online: bool = True
    error_state: str | None = None


mock_types.DeviceIdentity = FakeDeviceIdentity
mock_types.DeviceStatus = FakeDeviceStatus

# Save originals so we can restore after this module's tests run
_saved_modules = {}
_mock_keys = (
    "device_connect_edge",
    "device_connect_edge.drivers",
    "device_connect_edge.types",
    "device_connect_edge.device",
)
# Also track strands_robots.device_connect submodules that will be imported
# with the mocked base class — they need to be purged so later tests re-import
# with the real base class.
_strands_dc_keys = [k for k in sys.modules if k.startswith("strands_robots.device_connect")]
for _key in list(_mock_keys) + _strands_dc_keys:
    _saved_modules[_key] = sys.modules.get(_key)

sys.modules["device_connect_edge"] = mock_device_connect_edge
sys.modules["device_connect_edge.drivers"] = mock_drivers
sys.modules["device_connect_edge.types"] = mock_types
sys.modules["device_connect_edge.device"] = MagicMock()

# Mock DeviceRuntime
mock_device_runtime = MagicMock()
mock_device_connect_edge.DeviceRuntime = mock_device_runtime

# Now import our modules (intentionally after the sys.modules mocking above)
from strands_robots.device_connect.robot_driver import RobotDeviceDriver  # noqa: E402
from strands_robots.device_connect.sim_driver import SimulationDeviceDriver  # noqa: E402


def teardown_module():
    """Restore real device_connect_edge modules so other test files are not affected.

    Also purge cached strands_robots.device_connect submodules that were imported
    with the mock base class, so later test files get fresh imports with the real base.
    """
    # Restore device_connect_edge modules
    for key, original in _saved_modules.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original
    # Purge ALL strands_robots.device_connect submodules — they were imported
    # with the mock DeviceDriver base class and must be re-imported fresh.
    for key in list(sys.modules):
        if key.startswith("strands_robots.device_connect"):
            sys.modules.pop(key, None)


# ── Task state mocks ──────────────────────────────────────────────


class FakeTaskStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class FakeTaskState:
    status: FakeTaskStatus = FakeTaskStatus.IDLE
    instruction: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    step_count: int = 0
    error_message: str = ""


def _make_mock_robot(tool_name="so100", task_status="idle"):
    robot = MagicMock()
    robot.tool_name_str = tool_name
    robot._task_state = FakeTaskState(
        status=FakeTaskStatus(task_status),
        instruction="pick up the cube" if task_status == "running" else "",
        step_count=42 if task_status == "running" else 0,
    )
    robot.start_task.return_value = {"status": "success", "content": [{"text": "Task started"}]}
    robot.stop_task.return_value = {"status": "success", "content": [{"text": "Task stopped"}]}
    robot.get_task_status.return_value = {"status": "success", "content": [{"text": "Status info"}]}
    robot.get_features.return_value = {
        "status": "success",
        "content": [{"json": {"observation_features": {"joint1": "float"}, "action_features": {"joint1": "float"}}}],
    }
    # Mock inner lerobot robot
    robot.robot = MagicMock()
    robot.robot.get_observation.return_value = {"joint1": 0.5, "joint2": -1.2}
    return robot


def _make_mock_sim(tool_name="so100_sim"):
    sim = MagicMock()
    sim.tool_name_str = tool_name

    # SimWorld-like structure
    robot_data = MagicMock()
    robot_data.policy_running = False
    robot_data.policy_steps = 0
    robot_data.policy_instruction = ""

    world = MagicMock()
    world.robots = {"so100": robot_data}
    world.sim_time = 0.0
    world.step_count = 0
    sim._world = world

    sim.start_policy.return_value = {"status": "success", "content": [{"text": "Policy started"}]}
    sim.get_state.return_value = {"status": "success", "content": [{"text": "State info"}]}
    sim.get_features.return_value = {"status": "success", "content": [{"json": {"features": {}}}]}
    sim.step.return_value = {"status": "success", "content": [{"text": "Stepped"}]}
    sim.reset.return_value = {"status": "success", "content": [{"text": "Reset"}]}
    return sim


# ── TestRobotDeviceDriver ─────────────────────────────────────────


class TestRobotDeviceDriver(unittest.TestCase):
    def test_identity(self):
        robot = _make_mock_robot(tool_name="so100")
        driver = RobotDeviceDriver(robot)
        identity = driver.identity
        self.assertEqual(identity.device_type, "strands_robot")
        self.assertEqual(identity.manufacturer, "strands-robots")
        self.assertEqual(identity.model, "so100")

    def test_status_idle(self):
        robot = _make_mock_robot(task_status="idle")
        driver = RobotDeviceDriver(robot)
        status = driver.status
        self.assertEqual(status.availability, "idle")
        self.assertEqual(status.busy_score, 0.0)

    def test_status_busy(self):
        robot = _make_mock_robot(task_status="running")
        driver = RobotDeviceDriver(robot)
        status = driver.status
        self.assertEqual(status.availability, "busy")
        self.assertEqual(status.busy_score, 1.0)

    def test_execute_rpc(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.execute("pick up cube", "groot", 30.0, 0))
        robot.start_task.assert_called_once_with("pick up cube", "groot", None, "localhost", 30.0)
        self.assertEqual(result["status"], "success")

    def test_execute_rpc_with_port(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        asyncio.run(driver.execute("wave", "groot", 10.0, 50051))
        robot.start_task.assert_called_once_with("wave", "groot", 50051, "localhost", 10.0)

    def test_stop_rpc(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.stop())
        robot.stop_task.assert_called_once()
        self.assertEqual(result["status"], "success")

    def test_get_status_rpc(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getStatus())
        robot.get_task_status.assert_called_once()
        self.assertEqual(result["status"], "success")

    def test_get_features_rpc(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getFeatures())
        robot.get_features.assert_called_once()
        self.assertEqual(result["status"], "success")

    def test_get_state_rpc(self):
        robot = _make_mock_robot(task_status="running")
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        self.assertEqual(result["task_status"], "running")
        self.assertEqual(result["instruction"], "pick up the cube")
        self.assertEqual(result["step_count"], 42)
        # Joints should be read from inner robot
        self.assertIn("joints", result)
        self.assertAlmostEqual(result["joints"]["joint1"], 0.5)

    def test_connect_disconnect_noop(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        asyncio.run(driver.connect())
        asyncio.run(driver.disconnect())
        # Should not raise

    def test_emergency_stop_handler(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        asyncio.run(driver.onEmergencyStop("other-robot", "emergencyStop", {"reason": "test"}))
        robot.stop_task.assert_called_once()


# ── TestSimulationDeviceDriver ────────────────────────────────────


class TestSimulationDeviceDriver(unittest.TestCase):
    def test_identity(self):
        sim = _make_mock_sim(tool_name="mujoco_sim")
        driver = SimulationDeviceDriver(sim)
        identity = driver.identity
        self.assertEqual(identity.device_type, "strands_sim")
        self.assertEqual(identity.manufacturer, "strands-robots")
        self.assertEqual(identity.model, "mujoco_sim")

    def test_status_idle(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        status = driver.status
        self.assertEqual(status.availability, "idle")

    def test_status_busy(self):
        sim = _make_mock_sim()
        sim._world.robots["so100"].policy_running = True
        driver = SimulationDeviceDriver(sim)
        status = driver.status
        self.assertEqual(status.availability, "busy")

    def test_identity_sim_type(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        self.assertEqual(driver.device_type, "strands_sim")

    def test_execute_rpc(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.execute("pick up cube", "mock", 10.0))
        sim.start_policy.assert_called_once_with(
            robot_name="so100",
            policy_provider="mock",
            instruction="pick up cube",
            duration=10.0,
        )
        self.assertEqual(result["status"], "success")

    def test_execute_with_robot_name(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.execute("wave", "mock", 5.0, robot_name="arm2"))
        sim.start_policy.assert_called_once_with(
            robot_name="arm2",
            policy_provider="mock",
            instruction="wave",
            duration=5.0,
        )

    def test_stop_rpc(self):
        sim = _make_mock_sim()
        sim._world.robots["so100"].policy_running = True
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.stop())
        self.assertEqual(result["status"], "success")
        self.assertFalse(sim._world.robots["so100"].policy_running)

    def test_get_status_rpc(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.getStatus())
        sim.get_state.assert_called_once()

    def test_get_features_rpc(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.getFeatures())
        sim.get_features.assert_called_once()

    def test_step_rpc(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.step(10))
        sim.step.assert_called_once_with(10)

    def test_reset_rpc(self):
        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.reset())
        sim.reset.assert_called_once()

    def test_emergency_stop_handler(self):
        sim = _make_mock_sim()
        sim._world.robots["so100"].policy_running = True
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.onEmergencyStop("other-device", "emergencyStop", {"reason": "test"}))
        self.assertFalse(sim._world.robots["so100"].policy_running)


# ── TestReachyMiniDriver ─────────────────────────────────────────


class TestReachyMiniDriver(unittest.TestCase):
    def setUp(self):
        # Mock reachy_transport module but keep real ZenohLink/WebSocketLink
        from strands_robots.device_connect.reachy_transport import WebSocketLink, ZenohLink

        self.mock_transport_mod = MagicMock()
        self.mock_transport_mod.api.return_value = {"status": "ok"}
        self.mock_transport_mod.rpy_to_pose.side_effect = lambda *args, **kwargs: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        self.mock_transport_mod.identity_pose.return_value = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.mock_transport_mod.ZenohLink = ZenohLink
        self.mock_transport_mod.WebSocketLink = WebSocketLink

        self.transport_patcher = patch.dict(
            sys.modules,
            {
                "strands_robots.device_connect.reachy_transport": self.mock_transport_mod,
            },
        )
        self.transport_patcher.start()

        # Re-import to pick up mocks
        if "strands_robots.device_connect.reachy_mini_driver" in sys.modules:
            del sys.modules["strands_robots.device_connect.reachy_mini_driver"]
        from strands_robots.device_connect.reachy_mini_driver import ReachyMiniDriver

        self.ReachyMiniDriver = ReachyMiniDriver

    def tearDown(self):
        self.transport_patcher.stop()

    def _make_driver(self, **kwargs):
        """Create a driver with a mocked Device Connect transport and ZenohLink-like _hw."""
        driver = self.ReachyMiniDriver(**kwargs)
        mock_transport = AsyncMock()
        mock_transport.publish = AsyncMock()
        mock_transport.subscribe = AsyncMock()
        driver._transport = mock_transport

        # Create a HW link that delegates to mock_transport (like ZenohLink does)
        prefix = driver._prefix

        class _MockZenohLink:
            async def send_cmd(self, cmd):
                await mock_transport.publish(f"{prefix}/command", json.dumps(cmd).encode())

            async def start(self, on_joints, on_imu):
                await mock_transport.subscribe(f"{prefix}/joint_positions", on_joints)
                await mock_transport.subscribe(f"{prefix}/imu_data", on_imu)

            async def stop(self):
                pass

        driver._hw = _MockZenohLink()
        return driver

    def test_identity(self):
        driver = self.ReachyMiniDriver(host="192.168.1.50")
        identity = driver.identity
        self.assertEqual(identity.device_type, "reachy_mini")
        self.assertEqual(identity.manufacturer, "Pollen Robotics")
        self.assertIn("192.168.1.50", identity.model)

    def test_look_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.look(pitch=15, yaw=30))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["pitch"], 15)
        self.assertEqual(result["yaw"], 30)
        # Verify transport.publish was called with the command topic
        driver._transport.publish.assert_awaited()
        topic = driver._transport.publish.call_args[0][0]
        self.assertEqual(topic, "reachy_mini/command")

    def test_antennas_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.antennas(left=45, right=-30))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["left"], 45)
        self.assertEqual(result["right"], -30)
        driver._transport.publish.assert_awaited()

    def test_get_joints_rpc(self):
        driver = self._make_driver()
        # Pre-populate cached joint data
        driver._latest_joints = {
            "head_joint_positions": [0.1, 0.2, 0.3],
            "antennas_joint_positions": [0.5, -0.5],
        }
        result = asyncio.run(driver.getJoints())
        self.assertEqual(result["status"], "success")
        self.assertIn("head", result)
        self.assertIn("antennas", result)

    def test_get_joints_no_data(self):
        driver = self._make_driver()
        result = asyncio.run(driver.getJoints())
        self.assertEqual(result["status"], "error")

    def test_get_imu_rpc(self):
        driver = self._make_driver()
        driver._latest_imu = {
            "accelerometer": [0.1, 0.2, 9.8],
            "gyroscope": [0.0, 0.0, 0.0],
            "quaternion": [1, 0, 0, 0],
            "temperature": 35.2,
        }
        result = asyncio.run(driver.getImu())
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(result["temperature"], 35.2)

    def test_get_imu_no_data(self):
        driver = self._make_driver()
        result = asyncio.run(driver.getImu())
        self.assertEqual(result["status"], "error")

    def test_enable_motors_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.enableMotors())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["enabled"], "all")
        driver._transport.publish.assert_awaited()

    def test_disable_motors_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.disableMotors(motor_ids="head_pitch,head_yaw"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["disabled"], "head_pitch,head_yaw")
        driver._transport.publish.assert_awaited()

    def test_play_move_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.playMove("happy", library="emotions"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["move"], "happy")

    def test_nod_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.nod())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["expression"], "nod")
        # nod sends multiple publish calls (head_pose animation)
        self.assertGreater(driver._transport.publish.await_count, 1)

    def test_shake_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.shake())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["expression"], "shake")

    def test_happy_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.happy())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["expression"], "happy")

    def test_wake_up_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.wakeUp())
        self.assertEqual(result["status"], "success")

    def test_sleep_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.sleep())
        self.assertEqual(result["status"], "success")

    def test_stop_motion_rpc(self):
        driver = self._make_driver()
        result = asyncio.run(driver.stopMotion())
        self.assertEqual(result["status"], "success")

    def test_daemon_status_rpc(self):
        self.mock_transport_mod.api.return_value = {"state": "ready", "version": "1.0"}
        driver = self._make_driver()
        result = asyncio.run(driver.getDaemonStatus())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["state"], "ready")

    @patch("strands_robots.device_connect.reachy_mini_driver.api")
    def test_connect_subscribes(self, mock_api):
        # Simulate wireless variant (wireless_version=True)
        mock_api.return_value = {"wireless_version": True}
        driver = self.ReachyMiniDriver()
        mock_transport = AsyncMock()
        mock_transport.publish = AsyncMock()
        mock_transport.subscribe = AsyncMock()
        driver._transport = mock_transport
        # connect() creates ZenohLink and subscribes via transport
        asyncio.run(driver.connect())
        self.assertEqual(mock_transport.subscribe.await_count, 2)
        topics = [call[0][0] for call in mock_transport.subscribe.call_args_list]
        self.assertIn("reachy_mini/joint_positions", topics)
        self.assertIn("reachy_mini/imu_data", topics)

    def test_disconnect(self):
        driver = self._make_driver()
        asyncio.run(driver.disconnect())

    def test_emergency_stop_handler(self):
        driver = self._make_driver()
        asyncio.run(driver.onEmergencyStop("other-device", "emergencyStop", {"reason": "test"}))
        # stopMotion calls REST API, disableMotors calls transport.publish
        driver._transport.publish.assert_awaited()

    def test_command_payload_format(self):
        """Verify that transport.publish receives correct JSON payload."""
        driver = self._make_driver()
        asyncio.run(driver.enableMotors())
        _, payload_bytes = driver._transport.publish.call_args[0]
        payload = json.loads(payload_bytes.decode())
        self.assertTrue(payload["torque"])
        self.assertIsNone(payload["ids"])


# ── TestInitDeviceConnect ─────────────────────────────────────────


class TestInitDeviceConnect(unittest.TestCase):
    @patch("strands_robots.device_connect.DeviceRuntime")
    def test_creates_robot_driver(self, MockRuntime):
        from strands_robots.device_connect import init_device_connect

        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock()
        mock_runtime.set_heartbeat_provider = MagicMock()
        MockRuntime.return_value = mock_runtime

        robot = _make_mock_robot()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(init_device_connect(robot, peer_id="test-1", peer_type="robot"))
        loop.close()

        # Verify DeviceRuntime was created with a RobotDeviceDriver
        call_kwargs = MockRuntime.call_args
        self.assertIsNotNone(call_kwargs)
        driver = call_kwargs.kwargs.get("driver") or call_kwargs[1].get("driver")
        self.assertEqual(type(driver).__name__, "RobotDeviceDriver")
        self.assertEqual(driver._robot, robot)

    @patch("strands_robots.device_connect.DeviceRuntime")
    def test_creates_sim_driver(self, MockRuntime):
        from strands_robots.device_connect import init_device_connect

        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock()
        mock_runtime.set_heartbeat_provider = MagicMock()
        MockRuntime.return_value = mock_runtime

        sim = _make_mock_sim()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(init_device_connect(sim, peer_id="test-sim", peer_type="sim"))
        loop.close()

        call_kwargs = MockRuntime.call_args
        driver = call_kwargs.kwargs.get("driver") or call_kwargs[1].get("driver")
        self.assertEqual(type(driver).__name__, "SimulationDeviceDriver")

    @patch("strands_robots.device_connect.DeviceRuntime")
    def test_generates_device_id(self, MockRuntime):
        from strands_robots.device_connect import init_device_connect

        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock()
        mock_runtime.set_heartbeat_provider = MagicMock()
        MockRuntime.return_value = mock_runtime

        robot = _make_mock_robot(tool_name="so100")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(init_device_connect(robot))
        loop.close()

        call_kwargs = MockRuntime.call_args
        device_id = call_kwargs.kwargs.get("device_id") or call_kwargs[1].get("device_id")
        self.assertTrue(device_id.startswith("so100-"))

    @patch("strands_robots.device_connect.DeviceRuntime")
    def test_explicit_device_id(self, MockRuntime):
        from strands_robots.device_connect import init_device_connect

        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock()
        mock_runtime.set_heartbeat_provider = MagicMock()
        MockRuntime.return_value = mock_runtime

        robot = _make_mock_robot()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(init_device_connect(robot, peer_id="my-robot-42"))
        loop.close()

        call_kwargs = MockRuntime.call_args
        device_id = call_kwargs.kwargs.get("device_id") or call_kwargs[1].get("device_id")
        self.assertEqual(device_id, "my-robot-42")

    @patch("strands_robots.device_connect.DeviceRuntime")
    def test_sets_heartbeat_provider(self, MockRuntime):
        from strands_robots.device_connect import init_device_connect

        mock_runtime = MagicMock()
        mock_runtime.run = AsyncMock()
        mock_runtime.set_heartbeat_provider = MagicMock()
        MockRuntime.return_value = mock_runtime

        robot = _make_mock_robot()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(init_device_connect(robot, peer_id="test-hb"))
        loop.close()

        mock_runtime.set_heartbeat_provider.assert_called_once()


# ── TestEmergencyStop (cross-driver) ──────────────────────────────


class TestEmergencyStop(unittest.TestCase):
    def test_robot_reacts_to_emergency_stop(self):
        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        asyncio.run(driver.onEmergencyStop("reachy-1", "emergencyStop", {"reason": "button pressed"}))
        robot.stop_task.assert_called_once()

    def test_sim_reacts_to_emergency_stop(self):
        sim = _make_mock_sim()
        sim._world.robots["so100"].policy_running = True
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.onEmergencyStop("barista-001", "emergencyStop", {"reason": "agent-initiated"}))
        self.assertFalse(sim._world.robots["so100"].policy_running)


# ── TestRobotMeshTool (Device Connect backend) ───────────────────


class TestRobotMeshToolDeviceConnect(unittest.TestCase):
    def setUp(self):
        # Mock device_connect_agent_tools.connection
        self.mock_conn = MagicMock()
        self.mock_conn.list_devices.return_value = [
            {
                "device_id": "so100-lab-1",
                "device_type": "strands_robot",
                "status": {"availability": "idle"},
                "functions": [{"name": "execute"}, {"name": "stop"}],
                "events": [],
            },
            {
                "device_id": "reachy-mini-1",
                "device_type": "reachy_mini",
                "status": {"availability": "idle"},
                "functions": [{"name": "look"}, {"name": "nod"}],
                "events": [],
            },
        ]
        self.mock_conn.invoke.return_value = {
            "jsonrpc": "2.0",
            "id": "test",
            "result": {"status": "accepted"},
        }

        # Mock the device_connect_agent_tools modules before importing
        mock_aft = MagicMock()
        mock_aft_conn = MagicMock()
        mock_aft_conn.get_connection.return_value = self.mock_conn
        self._saved_modules = {}
        for mod in [
            "device_connect_agent_tools",
            "device_connect_agent_tools.connection",
            "device_connect_agent_tools.tools",
            "device_connect_agent_tools.agent",
            "device_connect_agent_tools.adapters",
            "device_connect_agent_tools.adapters.strands",
        ]:
            self._saved_modules[mod] = sys.modules.get(mod)
            sys.modules[mod] = mock_aft if mod == "device_connect_agent_tools" else mock_aft_conn

        # NOTE: robot_mesh is intentionally NOT deleted from sys.modules.
        # _device_connect_dispatch imports get_connection lazily at call time, so
        # the mocked device_connect_agent_tools.connection installed above is
        # picked up without a re-import. Deleting + re-importing robot_mesh would
        # create a second module object and break sibling test files
        # (test_robot_mesh_tool / _security / deep_mesh) that hold a reference to
        # the original module — their _resolve_mesh patches would miss.

    def tearDown(self):
        for mod, saved in self._saved_modules.items():
            if saved is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = saved

    def test_peers_action(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch("peers", "", "", "", "mock", 0, 30.0, 30.0)
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("so100-lab-1", text)
        self.assertIn("reachy-mini-1", text)

    def test_tell_action(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch(
            "tell",
            "so100-lab-1",
            "pick up cube",
            "",
            "groot",
            0,
            30.0,
            30.0,
        )
        self.assertEqual(result["status"], "success")
        self.mock_conn.invoke.assert_called_once()
        call_args = self.mock_conn.invoke.call_args
        self.assertEqual(call_args[0][0], "so100-lab-1")
        self.assertEqual(call_args[0][1], "execute")

    def test_stop_action(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch("stop", "so100-lab-1", "", "", "mock", 0, 30.0, 30.0)
        self.assertEqual(result["status"], "success")
        self.mock_conn.invoke.assert_called_once_with("so100-lab-1", "stop", {}, timeout=5.0)

    def test_emergency_stop(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch("emergency_stop", "", "", "", "mock", 0, 30.0, 30.0)
        self.assertEqual(result["status"], "success")
        self.assertIn("2", result["content"][0]["text"])  # 2 devices stopped

    def test_missing_target(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch("tell", "", "do something", "", "mock", 0, 30.0, 30.0)
        self.assertEqual(result["status"], "error")
        self.assertIn("target", result["content"][0]["text"])

    def test_missing_instruction(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch("tell", "so100-lab-1", "", "", "mock", 0, 30.0, 30.0)
        self.assertEqual(result["status"], "error")
        self.assertIn("instruction", result["content"][0]["text"])

    def test_status_action(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        result = _device_connect_dispatch("status", "", "", "", "mock", 0, 30.0, 30.0)
        self.assertEqual(result["status"], "success")
        self.assertIn("2 device(s)", result["content"][0]["text"])


# ── TestReachyTransport ───────────────────────────────────────────


class TestReachyTransport(unittest.TestCase):
    """Test the extracted transport helpers."""

    def test_rpy_to_pose_identity(self):
        from strands_robots.device_connect.reachy_transport import rpy_to_pose

        pose = rpy_to_pose(0, 0, 0)
        # Should be close to identity rotation
        self.assertAlmostEqual(pose[0][0], 1.0, places=5)
        self.assertAlmostEqual(pose[1][1], 1.0, places=5)
        self.assertAlmostEqual(pose[2][2], 1.0, places=5)
        self.assertAlmostEqual(pose[3][3], 1.0, places=5)

    def test_rpy_to_pose_translation(self):
        from strands_robots.device_connect.reachy_transport import rpy_to_pose

        pose = rpy_to_pose(0, 0, 0, x_mm=100, y_mm=200, z_mm=300)
        self.assertAlmostEqual(pose[0][3], 0.1, places=5)  # 100mm = 0.1m
        self.assertAlmostEqual(pose[1][3], 0.2, places=5)
        self.assertAlmostEqual(pose[2][3], 0.3, places=5)

    def test_identity_pose(self):
        from strands_robots.device_connect.reachy_transport import identity_pose

        pose = identity_pose()
        self.assertEqual(pose, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def test_resolve_host_ip(self):
        from strands_robots.device_connect.reachy_transport import resolve_host

        # IP should pass through unchanged
        result = resolve_host("192.168.1.1")
        self.assertEqual(result, "192.168.1.1")


if __name__ == "__main__":
    unittest.main()
