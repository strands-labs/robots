"""Parametrized Device Connect tests across all 38 registered robots.

Validates that RobotDeviceDriver and SimulationDeviceDriver work correctly
with every robot's specific configuration (joint counts, observation shapes,
identity, status, RPC delegation). Also tests multi-robot simulation scenarios,
edge cases, and robot_mesh dispatch with diverse device types.

All external dependencies (Zenoh, LeRobot, device_connect_edge, strands) are mocked.
No Docker, GPU, or hardware required.
"""

import asyncio
import importlib.util
import json
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

# ── Mock heavy dependencies before importing ──────────────────────

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


def _passthrough_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]

    def wrapper(func):
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

_saved_modules = {}
_mock_keys = (
    "device_connect_edge",
    "device_connect_edge.drivers",
    "device_connect_edge.types",
    "device_connect_edge.device",
)
_strands_dc_keys = [k for k in sys.modules if k.startswith("strands_robots.device_connect")]
for _key in list(_mock_keys) + _strands_dc_keys:
    _saved_modules[_key] = sys.modules.get(_key)

# The robot_mesh dispatch tests below patch
# ``device_connect_agent_tools.connection.get_connection``, which forces an
# import of the real ``device_connect_agent_tools`` package (__init__ -> agent
# -> connection -> ``device_connect_edge.messaging``). Import it now, while the
# real ``device_connect_edge`` is still in sys.modules, so the module is cached
# before the mock is installed below -- otherwise those tests only pass when a
# sibling test imports it first (a hidden import-order dependency).
try:
    import device_connect_agent_tools.connection  # noqa: E402, F401
except Exception:
    pass

sys.modules["device_connect_edge"] = mock_device_connect_edge
sys.modules["device_connect_edge.drivers"] = mock_drivers
sys.modules["device_connect_edge.types"] = mock_types
sys.modules["device_connect_edge.device"] = MagicMock()

mock_device_runtime = MagicMock()
mock_device_connect_edge.DeviceRuntime = mock_device_runtime

from strands_robots.device_connect.robot_driver import RobotDeviceDriver  # noqa: E402
from strands_robots.device_connect.sim_driver import SimulationDeviceDriver  # noqa: E402


def teardown_module():
    """Restore real device_connect_edge modules."""
    for key, original in _saved_modules.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original
    for key in list(sys.modules):
        if key.startswith("strands_robots.device_connect"):
            sys.modules.pop(key, None)


# ── Load robot registry ──────────────────────────────────────────

_REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "registry" / "robots.json"
_REGISTRY = json.loads(_REGISTRY_PATH.read_text())["robots"]

ALL_ROBOTS = [(name, info) for name, info in _REGISTRY.items()]
SIM_ROBOTS = [(name, info) for name, info in ALL_ROBOTS if "asset" in info]
REAL_ONLY_ROBOTS = [(name, info) for name, info in ALL_ROBOTS if "asset" not in info]


# ── Task state mocks ─────────────────────────────────────────────


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


# ── Observation helper ───────────────────────────────────────────


class _FakeArray:
    """Mimics a numpy array with a .shape attribute."""

    def __init__(self, shape):
        self.shape = shape


def _generate_observation(joint_count, include_arrays=True):
    """Generate a realistic observation dict for a robot with N joints."""
    obs = {}
    for i in range(joint_count):
        obs[f"joint_{i}"] = float(i) * 0.1
    if include_arrays:
        obs["image"] = _FakeArray((480, 640, 3))
        obs["depth"] = _FakeArray((480, 640))
    return obs


# ── Mock factories ───────────────────────────────────────────────


def _get_joint_count(info):
    """Get joint count from registry info, defaulting to 6 for real-only robots."""
    return info.get("joints", 6)


def _make_mock_robot(name, info, task_status="idle"):
    """Create a mock robot matching the registry entry's configuration."""
    joint_count = _get_joint_count(info)
    robot = MagicMock()
    robot.tool_name_str = name
    robot._task_state = FakeTaskState(
        status=FakeTaskStatus(task_status),
        instruction="pick up the cube" if task_status == "running" else "",
        step_count=42 if task_status == "running" else 0,
    )
    robot.start_task.return_value = {"status": "success", "content": [{"text": "Task started"}]}
    robot.stop_task.return_value = {"status": "success", "content": [{"text": "Task stopped"}]}
    robot.get_task_status.return_value = {"status": "success", "content": [{"text": "Status info"}]}

    features = {f"joint_{i}": "float" for i in range(joint_count)}
    robot.get_features.return_value = {
        "status": "success",
        "content": [{"json": {"observation_features": features, "action_features": features}}],
    }

    robot.robot = MagicMock()
    robot.robot.get_observation.return_value = _generate_observation(joint_count)
    return robot


def _make_mock_sim(name, info, robots_in_world=None):
    """Create a mock simulation matching the registry entry's configuration."""
    sim = MagicMock()
    sim.tool_name_str = f"{name}_sim"

    world = MagicMock()
    if robots_in_world is None:
        robot_data = MagicMock()
        robot_data.policy_running = False
        robot_data.policy_steps = 0
        robot_data.policy_instruction = ""
        world.robots = {name: robot_data}
    else:
        world.robots = robots_in_world
    world.sim_time = 0.0
    world.step_count = 0
    sim._world = world

    sim.start_policy.return_value = {"status": "success", "content": [{"text": "Policy started"}]}
    sim.get_state.return_value = {"status": "success", "content": [{"text": "State info"}]}
    sim.get_features.return_value = {"status": "success", "content": [{"json": {"features": {}}}]}
    sim.step.return_value = {"status": "success", "content": [{"text": "Stepped"}]}
    sim.reset.return_value = {"status": "success", "content": [{"text": "Reset"}]}
    return sim


# ── TestRobotDriverAllRobots ─────────────────────────────────────


class TestRobotDriverAllRobots:
    """Parametrized tests for RobotDeviceDriver across all 38 registered robots."""

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_identity(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        assert driver.identity.device_type == "strands_robot"
        assert driver.identity.model == robot_name
        assert driver.identity.manufacturer == "strands-robots"

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_status_idle(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info, task_status="idle")
        driver = RobotDeviceDriver(robot)
        assert driver.status.availability == "idle"
        assert driver.status.busy_score == 0.0

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_status_busy(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info, task_status="running")
        driver = RobotDeviceDriver(robot)
        assert driver.status.availability == "busy"
        assert driver.status.busy_score == 1.0

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_execute_delegates(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.execute("pick up cube", "groot", 30.0, 0))
        robot.start_task.assert_called_once_with("pick up cube", "groot", None, "localhost", 30.0)
        assert result["status"] == "success"

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_get_state_joint_count(self, robot_name, robot_info):
        joint_count = _get_joint_count(robot_info)
        robot = _make_mock_robot(robot_name, robot_info, task_status="running")
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert "joints" in result
        assert len(result["joints"]) == joint_count

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_get_state_filters_arrays(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        if "joints" in result:
            for key, value in result["joints"].items():
                assert isinstance(value, float), f"Non-float value for {key}: {type(value)}"
                assert not key.startswith("image") and not key.startswith("depth")

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_get_state_task_info(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info, task_status="running")
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert result["task_status"] == "running"
        assert result["instruction"] == "pick up the cube"
        assert result["step_count"] == 42

    @pytest.mark.parametrize("robot_name,robot_info", ALL_ROBOTS, ids=[r[0] for r in ALL_ROBOTS])
    def test_get_features_delegates(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getFeatures())
        robot.get_features.assert_called_once()
        assert result["status"] == "success"


# ── TestSimDriverAllRobots ───────────────────────────────────────


class TestSimDriverAllRobots:
    """Parametrized tests for SimulationDeviceDriver across all 32 sim-capable robots."""

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_identity(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        driver = SimulationDeviceDriver(sim)
        assert driver.identity.device_type == "strands_sim"
        assert driver.identity.model == f"{robot_name}_sim"
        assert driver.identity.manufacturer == "strands-robots"

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_status_idle(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        driver = SimulationDeviceDriver(sim)
        assert driver.status.availability == "idle"

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_status_busy(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        driver = SimulationDeviceDriver(sim)
        sim._world.robots[robot_name].policy_running = True
        assert driver.status.availability == "busy"

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_execute_auto_detects_robot(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.execute("pick up cube", "mock", 30.0, ""))
        sim.start_policy.assert_called_once_with(
            robot_name=robot_name, policy_provider="mock", instruction="pick up cube", duration=30.0
        )
        assert result["status"] == "success"

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_stop_sets_policy_running_false(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        sim._world.robots[robot_name].policy_running = True
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.stop())
        assert sim._world.robots[robot_name].policy_running is False
        assert result["status"] == "success"

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_step_delegates(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.step(10))
        sim.step.assert_called_once_with(10)
        assert result["status"] == "success"

    @pytest.mark.parametrize("robot_name,robot_info", SIM_ROBOTS, ids=[r[0] for r in SIM_ROBOTS])
    def test_reset_delegates(self, robot_name, robot_info):
        sim = _make_mock_sim(robot_name, robot_info)
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.reset())
        sim.reset.assert_called_once()
        assert result["status"] == "success"


# ── TestRealOnlyRobots ───────────────────────────────────────────


class TestRealOnlyRobots:
    """Tests for real-only robots (no sim asset): lekiwi, reachy2, hope_jr, earthrover, omx, bi_openarm."""

    @pytest.mark.parametrize("robot_name,robot_info", REAL_ONLY_ROBOTS, ids=[r[0] for r in REAL_ONLY_ROBOTS])
    def test_driver_creation(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        assert driver is not None

    @pytest.mark.parametrize("robot_name,robot_info", REAL_ONLY_ROBOTS, ids=[r[0] for r in REAL_ONLY_ROBOTS])
    def test_identity_no_asset(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        assert driver.identity.model == robot_name
        assert driver.identity.device_type == "strands_robot"

    @pytest.mark.parametrize("robot_name,robot_info", REAL_ONLY_ROBOTS, ids=[r[0] for r in REAL_ONLY_ROBOTS])
    def test_execute_delegates(self, robot_name, robot_info):
        robot = _make_mock_robot(robot_name, robot_info)
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.execute("move forward", "mock", 10.0, 0))
        robot.start_task.assert_called_once()
        assert result["status"] == "success"


# ── TestMultiRobotSimulation ─────────────────────────────────────


class TestMultiRobotSimulation:
    """Tests for multi-robot simulation scenarios with diverse joint counts."""

    def _make_robot_data(self, running=False):
        robot_data = MagicMock()
        robot_data.policy_running = running
        robot_data.policy_steps = 0
        robot_data.policy_instruction = ""
        return robot_data

    def test_mixed_joint_counts(self):
        """so100 (13 joints) + unitree_g1 (46 joints) in one world."""
        robots_in_world = {
            "so100": self._make_robot_data(),
            "unitree_g1": self._make_robot_data(),
        }
        sim = _make_mock_sim("mixed", _REGISTRY["so100"], robots_in_world=robots_in_world)
        driver = SimulationDeviceDriver(sim)
        # Execute auto-detects first robot
        asyncio.run(driver.execute("test", "mock", 10.0, ""))
        sim.start_policy.assert_called_once()
        call_kwargs = sim.start_policy.call_args
        assert call_kwargs[1]["robot_name"] in ("so100", "unitree_g1")

    def test_stop_all_policies(self):
        """Stop sets policy_running=False on all robots in the world."""
        robots_in_world = {
            "so100": self._make_robot_data(running=True),
            "panda": self._make_robot_data(running=True),
            "unitree_go2": self._make_robot_data(running=True),
        }
        sim = _make_mock_sim("fleet", _REGISTRY["so100"], robots_in_world=robots_in_world)
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.stop())
        for name, robot_data in robots_in_world.items():
            assert robot_data.policy_running is False, f"{name} still running"

    def test_execute_with_explicit_robot_name(self):
        """Target a specific robot in a multi-robot sim."""
        robots_in_world = {
            "so100": self._make_robot_data(),
            "unitree_g1": self._make_robot_data(),
        }
        sim = _make_mock_sim("multi", _REGISTRY["so100"], robots_in_world=robots_in_world)
        driver = SimulationDeviceDriver(sim)
        asyncio.run(driver.execute("walk forward", "mock", 30.0, "unitree_g1"))
        sim.start_policy.assert_called_once_with(
            robot_name="unitree_g1", policy_provider="mock", instruction="walk forward", duration=30.0
        )

    def test_execute_empty_world(self):
        """Returns error when no robots in simulation."""
        sim = _make_mock_sim("empty", _REGISTRY["so100"], robots_in_world={})
        driver = SimulationDeviceDriver(sim)
        result = asyncio.run(driver.execute("test", "mock", 10.0, ""))
        assert result["status"] == "error"


# ── TestEdgeCases ────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for driver robustness."""

    def test_observation_all_arrays(self):
        """Observation with only array values → joints dict is empty."""
        robot = _make_mock_robot("so100", _REGISTRY["so100"])
        robot.robot.get_observation.return_value = {
            "camera_front": _FakeArray((480, 640, 3)),
            "camera_wrist": _FakeArray((480, 640, 3)),
            "depth": _FakeArray((480, 640)),
        }
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert result.get("joints", {}) == {}

    def test_observation_empty(self):
        """Empty observation → no joints key or empty joints."""
        robot = _make_mock_robot("so100", _REGISTRY["so100"])
        robot.robot.get_observation.return_value = {}
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert result.get("joints", {}) == {}

    def test_observation_raises(self):
        """get_observation() throws → getState still returns task info."""
        robot = _make_mock_robot("so100", _REGISTRY["so100"], task_status="running")
        robot.robot.get_observation.side_effect = RuntimeError("hardware disconnected")
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert result["task_status"] == "running"
        assert result["instruction"] == "pick up the cube"
        assert "joints" not in result

    def test_no_inner_robot(self):
        """robot.robot is None → getState skips observation."""
        robot = _make_mock_robot("so100", _REGISTRY["so100"], task_status="running")
        robot.robot = None
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert result["task_status"] == "running"
        assert "joints" not in result

    def test_no_task_state(self):
        """_task_state is None → status is idle, getState has no task info."""
        robot = _make_mock_robot("so100", _REGISTRY["so100"])
        robot._task_state = None
        driver = RobotDeviceDriver(robot)
        assert driver.status.availability == "idle"
        result = asyncio.run(driver.getState())
        assert "task_status" not in result

    def test_float_conversion_failure(self):
        """Non-numeric scalar in observation → graceful handling via exception catch."""
        robot = _make_mock_robot("so100", _REGISTRY["so100"])
        robot.robot.get_observation.return_value = {
            "joint_0": 0.5,
            "metadata": "not_a_number",
        }
        driver = RobotDeviceDriver(robot)
        # float("not_a_number") raises ValueError; the driver wraps get_observation
        # in a try/except, so it either filters it out or catches the error
        result = asyncio.run(driver.getState())
        # Either joints has only joint_0, or the whole observation was skipped
        if "joints" in result:
            assert "metadata" not in result["joints"] or isinstance(result["joints"].get("metadata"), float)

    def test_max_joint_robot(self):
        """unitree_g1 (46 joints) — all joints appear in getState."""
        info = _REGISTRY["unitree_g1"]
        robot = _make_mock_robot("unitree_g1", info, task_status="running")
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert len(result["joints"]) == 46

    def test_min_joint_robot(self):
        """koch (7 joints) — correct joint count."""
        info = _REGISTRY["koch"]
        robot = _make_mock_robot("koch", info, task_status="running")
        driver = RobotDeviceDriver(robot)
        result = asyncio.run(driver.getState())
        assert len(result["joints"]) == 7


# ── TestRobotMeshDispatchAllTypes ────────────────────────────────


# robot_mesh imports `strands` (@tool, ToolContext) and device_connect_agent_tools.
# Prefer the REAL packages when installed (both are hard deps here) so we never
# leave a stub in sys.modules that leaks into sibling test modules. Only fall
# back to stubs when a package genuinely is not importable.
def _passthrough_tool(*args, **kwargs):
    """Stub for strands @tool / @tool(context=True): return the function unchanged."""
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


if importlib.util.find_spec("strands") is None:
    _m = MagicMock()
    _m.tool = _passthrough_tool
    _types_tools = MagicMock()
    _types_tools.ToolContext = object
    sys.modules["strands"] = _m
    sys.modules["strands.types"] = MagicMock()
    sys.modules["strands.types.tools"] = _types_tools

if importlib.util.find_spec("device_connect_agent_tools") is None:
    sys.modules.setdefault("device_connect_agent_tools", MagicMock())
    sys.modules.setdefault("device_connect_agent_tools.connection", MagicMock())


class _FakeConnection:
    """Fake connection with all methods the dispatch uses."""

    def __init__(self, devices=None):
        self.zone = "default"
        self._devices = devices or []
        self._invoke_results = {}
        self._inbox = {}
        self._sync_subs = {}

    def list_devices(self, device_type=None):
        if device_type:
            return [d for d in self._devices if d.get("device_type") == device_type]
        return list(self._devices)

    def invoke(self, device_id, function, params=None, timeout=30.0):
        key = (device_id, function)
        if key in self._invoke_results:
            return self._invoke_results[key]
        return {"result": {"status": "ok"}}

    def broadcast(self, function, params=None, timeout=5.0):
        results = []
        for d in self._devices:
            try:
                r = self.invoke(d["device_id"], function, params, timeout=timeout)
                results.append({"device_id": d["device_id"], "result": r})
            except Exception as e:
                results.append({"device_id": d["device_id"], "error": str(e)})
        return results

    def subscribe_buffered(self, subject, name=None):
        name = name or subject
        self._inbox[name] = []
        self._sync_subs[name] = True
        return name

    def get_inbox(self, name=None):
        if name is not None:
            return {name: list(self._inbox.get(name, []))}
        return {k: list(v) for k, v in self._inbox.items()}


# Build a diverse fleet of sample devices from the registry
_CATEGORY_REPRESENTATIVES = {
    "arm": ("so100", "strands_robot"),
    "bimanual": ("aloha", "strands_robot"),
    "hand": ("shadow_hand", "strands_robot"),
    "humanoid": ("unitree_g1", "strands_sim"),
    "expressive": ("reachy_mini", "strands_robot"),
    "mobile": ("unitree_go2", "strands_sim"),
    "mobile_manip": ("google_robot", "strands_sim"),
}

DIVERSE_DEVICES = []
for category, (robot_name, device_type) in _CATEGORY_REPRESENTATIVES.items():
    DIVERSE_DEVICES.append(
        {
            "device_id": f"{robot_name}-{category}-1",
            "device_type": device_type,
            "status": {"availability": "idle"},
            "functions": [{"name": "execute"}, {"name": "stop"}, {"name": "getStatus"}],
            "events": ["taskStarted", "taskComplete"] if device_type == "strands_robot" else ["stateUpdate"],
        }
    )


class TestRobotMeshDispatchAllTypes:
    """Tests robot_mesh dispatch with a diverse fleet spanning all robot categories."""

    def _get_dispatch(self):
        from strands_robots.tools.robot_mesh import _device_connect_dispatch

        return _device_connect_dispatch

    def _call(self, dispatch, conn, action, **kwargs):
        defaults = dict(
            target="",
            instruction="",
            command="",
            policy_provider="mock",
            policy_port=0,
            duration=30.0,
            timeout=5.0,
        )
        defaults.update(kwargs)
        with patch("device_connect_agent_tools.connection.get_connection", return_value=conn):
            return dispatch(
                action,
                **{
                    k: defaults[k]
                    for k in [
                        "target",
                        "instruction",
                        "command",
                        "policy_provider",
                        "policy_port",
                        "duration",
                        "timeout",
                    ]
                },
            )

    def test_peers_lists_all_categories(self):
        conn = _FakeConnection(devices=DIVERSE_DEVICES)
        dispatch = self._get_dispatch()
        result = self._call(dispatch, conn, "peers")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert f"{len(DIVERSE_DEVICES)} device(s)" in text
        for device in DIVERSE_DEVICES:
            assert device["device_id"] in text

    def test_tell_arm_robot(self):
        conn = _FakeConnection(devices=DIVERSE_DEVICES)
        dispatch = self._get_dispatch()
        result = self._call(dispatch, conn, "tell", target="so100-arm-1", instruction="pick up cube")
        assert result["status"] == "success"
        assert "so100-arm-1" in result["content"][0]["text"]

    def test_tell_humanoid_sim(self):
        conn = _FakeConnection(devices=DIVERSE_DEVICES)
        dispatch = self._get_dispatch()
        result = self._call(dispatch, conn, "tell", target="unitree_g1-humanoid-1", instruction="walk forward")
        assert result["status"] == "success"
        assert "unitree_g1-humanoid-1" in result["content"][0]["text"]

    def test_tell_mobile_robot(self):
        conn = _FakeConnection(devices=DIVERSE_DEVICES)
        dispatch = self._get_dispatch()
        result = self._call(dispatch, conn, "tell", target="unitree_go2-mobile-1", instruction="navigate to door")
        assert result["status"] == "success"
        assert "unitree_go2-mobile-1" in result["content"][0]["text"]

    def test_emergency_stop_all_types(self):
        conn = _FakeConnection(devices=DIVERSE_DEVICES)
        dispatch = self._get_dispatch()
        result = self._call(dispatch, conn, "emergency_stop")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "E-STOP" in text
        assert f"{len(DIVERSE_DEVICES)}/{len(DIVERSE_DEVICES)}" in text

    def test_status_mixed_fleet(self):
        conn = _FakeConnection(devices=DIVERSE_DEVICES)
        dispatch = self._get_dispatch()
        result = self._call(dispatch, conn, "status")
        assert result["status"] == "success"
        assert f"{len(DIVERSE_DEVICES)} device(s)" in result["content"][0]["text"]
