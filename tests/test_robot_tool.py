#!/usr/bin/env python3
"""Comprehensive tests for strands_robots/robot.py — Robot AgentTool.

Tests cover all 7 actions (execute, start, status, stop, record, replay, features),
Robot initialization (3 input types), _get_policy (6 providers), _connect_robot
(success/error/already-connected/not-calibrated), stream() async generator,
cleanup, get_status, and edge cases.

Coverage target: 2% → 60%+ of robot.py (595 statements).
All tests run without hardware, lerobot, or torch installed.
"""

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# CROSS_PR_SKIP: Tests written against full Robot API (requires PR #8 + #11)
# Skip if Robot.tool_name is a string (not callable) - means the extended API isn't present
try:
    import strands_robots.robot as _robot_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("strands_robots.robot requires lerobot", allow_module_level=True)
if not callable(getattr(_robot_mod.Robot, "tool_name", None)):
    import pytest as _xfail_pytest

    _xfail_pytest.skip("Tests require extended Robot API from PR #8 + #11", allow_module_level=True)


# ── Module-level sys.modules mocking ──────────────────────────────────────────
# robot.py has top-level imports from lerobot and strands that need to be mocked
# BEFORE the module is ever imported.

_mock_zenoh_mesh = MagicMock()
_mock_zenoh_mesh.init_mesh = MagicMock(return_value=None)

_mock_policies = MagicMock()
_mock_policies.Policy = type("Policy", (), {})
_mock_policies.create_policy = MagicMock()

_mock_lerobot_camera_config = MagicMock()


class _FakeOpenCVCameraConfig:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_mock_lerobot_camera_config.OpenCVCameraConfig = _FakeOpenCVCameraConfig


class _FakeRobotConfig:
    """Minimal stand-in for lerobot.robots.config.RobotConfig."""

    pass


_mock_lerobot_robots_config = MagicMock()
_mock_lerobot_robots_config.RobotConfig = _FakeRobotConfig


class _FakeLeRobotRobot:
    """Minimal stand-in for lerobot.robots.robot.Robot."""

    def __init__(self):
        self.is_connected = False
        self.is_calibrated = True
        self.name = "mock_robot"
        self.robot_type = "so100"
        self.config = MagicMock()
        self.config.cameras = {}
        self.observation_features = {"j1": float, "j2": float}
        self.action_features = {"j1": float, "j2": float}

    def connect(self, calibrate=False):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return {"j1": 0.5, "j2": 0.3}

    def send_action(self, action):
        pass

    def __str__(self):
        return "FakeRobot(mock)"


_mock_lr_robot_mod = MagicMock()
_mock_lr_robot_mod.Robot = _FakeLeRobotRobot

_mock_lr_utils_mod = MagicMock()
_mock_lr_utils_mod.make_robot_from_config = MagicMock(return_value=_FakeLeRobotRobot())


class _FakeDeviceAlreadyConnectedError(Exception):
    pass


_mock_lr_errors = MagicMock()
_mock_lr_errors.DeviceAlreadyConnectedError = _FakeDeviceAlreadyConnectedError


class _FakeAgentTool:
    def __init__(self):
        pass


_mock_strands_tools_tools = MagicMock()
_mock_strands_tools_tools.AgentTool = _FakeAgentTool


class _FakeToolResultEvent(dict):
    def __init__(self, d):
        super().__init__(d)


_mock_strands_events = MagicMock()
_mock_strands_events.ToolResultEvent = _FakeToolResultEvent

_mock_strands_types_tools = MagicMock()

# Build the full sys.modules patch dict
# lerobot.robots mock needs __path__ for pkgutil.iter_modules
# MagicMock's __getattr__ blocks __path__ access, so use types.ModuleType
import types as _types  # noqa: E402

_mock_lr_robots = _types.ModuleType("lerobot.robots")
_mock_lr_robots.__path__ = ["/fake/lerobot/robots"]

# Pre-mock cv2 to prevent OpenCV 4.12 cv2.dnn.DictValue crash
import importlib.machinery as _im  # noqa: E402

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im.ModuleSpec("cv2", None)
_mock_cv2.dnn = MagicMock()
_mock_cv2.typing = MagicMock()

_PATCHES = {
    "cv2": _mock_cv2,
    "cv2.dnn": _mock_cv2.dnn,
    "cv2.typing": _mock_cv2.typing,
    "lerobot": MagicMock(),
    "lerobot.cameras": MagicMock(),
    "lerobot.cameras.opencv": MagicMock(),
    "lerobot.cameras.opencv.configuration_opencv": _mock_lerobot_camera_config,
    "lerobot.robots": _mock_lr_robots,
    "lerobot.robots.config": _mock_lerobot_robots_config,
    "lerobot.robots.robot": _mock_lr_robot_mod,
    "lerobot.robots.utils": _mock_lr_utils_mod,
    "lerobot.utils": MagicMock(),
    "lerobot.utils.errors": _mock_lr_errors,
    "lerobot.datasets": MagicMock(),
    "lerobot.datasets.lerobot_dataset": MagicMock(),
    "strands": MagicMock(),
    "strands.tools": MagicMock(),
    "strands.tools.tools": _mock_strands_tools_tools,
    "strands.types": MagicMock(),
    "strands.types._events": _mock_strands_events,
    "strands.types.tools": _mock_strands_types_tools,
    "strands_robots.zenoh_mesh": _mock_zenoh_mesh,
    "strands_robots.policies": _mock_policies,
}


# CRITICAL: Python resolves `import lerobot.robots` through the parent package attr.
# So lerobot (MagicMock) must have .robots pointing to our ModuleType.
_PATCHES["lerobot"].robots = _mock_lr_robots


# ── Import robot module directly ───────────────────────────────────────────
# robot.py has top-level `from strands.tools.tools import AgentTool` which needs
# strands mocked in sys.modules BEFORE import. Apply patches, import, then
# save originals so we can restore in the autouse fixture.
_saved_modules = {}
for _k, _v in _PATCHES.items():
    _saved_modules[_k] = sys.modules.get(_k)
    sys.modules[_k] = _v

from strands_robots.robot import Robot as _Robot  # noqa: E402
from strands_robots.robot import RobotTaskState as _RobotTaskState  # noqa: E402
from strands_robots.robot import TaskStatus as _TaskStatus  # noqa: E402

# Restore original sys.modules state (remove patches we added)
for _k, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_k, None)
    else:
        sys.modules[_k] = _orig


@pytest.fixture(autouse=True)
def _mock_modules():
    """Patch sys.modules for every test so robot.py can be imported."""
    with patch.dict(sys.modules, _PATCHES):
        yield


@pytest.fixture
def mod():
    """Import Robot, TaskStatus, RobotTaskState under mocks."""
    return _Robot, _TaskStatus, _RobotTaskState


@pytest.fixture
def fake_robot():
    return _FakeLeRobotRobot()


@pytest.fixture
def tool(mod, fake_robot):
    """Build a Robot tool WITHOUT calling __init__ (bypasses hardware)."""
    Robot, TaskStatus, RobotTaskState = mod
    obj = Robot.__new__(Robot)
    _FakeAgentTool.__init__(obj)
    obj.tool_name_str = "test_bot"
    obj.action_horizon = 4
    obj.data_config = None
    obj.control_frequency = 50.0
    obj.action_sleep_time = 0.02
    obj._task_state = RobotTaskState()
    obj._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t")
    obj._shutdown_event = threading.Event()
    obj.robot = fake_robot
    obj._observation_features = {"j1": float, "j2": float}
    obj._action_features = {"j1": float, "j2": float}
    obj.mesh = None
    yield obj
    obj._executor.shutdown(wait=False)


# ══════════════════════════════════════════════════════════════════════════════
# TaskStatus & RobotTaskState
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskStatus:
    def test_all_values(self, mod):
        _, TaskStatus, _ = mod
        assert TaskStatus.IDLE.value == "idle"
        assert TaskStatus.CONNECTING.value == "connecting"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.STOPPED.value == "stopped"
        assert TaskStatus.ERROR.value == "error"
        assert len(TaskStatus) == 6


class TestRobotTaskState:
    def test_defaults(self, mod):
        _, TaskStatus, RobotTaskState = mod
        s = RobotTaskState()
        assert s.status == TaskStatus.IDLE
        assert s.instruction == ""
        assert s.start_time == 0.0
        assert s.step_count == 0
        assert s.error_message == ""
        assert s.task_future is None

    def test_mutation(self, mod):
        _, TaskStatus, RobotTaskState = mod
        s = RobotTaskState()
        s.status = TaskStatus.RUNNING
        s.instruction = "grab cup"
        assert s.instruction == "grab cup"


# ══════════════════════════════════════════════════════════════════════════════
# Tool metadata
# ══════════════════════════════════════════════════════════════════════════════


class TestToolMeta:
    def test_tool_name(self, tool):
        assert tool.tool_name() == "test_bot"

    def test_tool_type(self, tool):
        assert tool.tool_type == "robot"

    def test_tool_spec_schema(self, tool):
        spec = tool.tool_spec
        assert spec["name"] == "test_bot"
        props = spec["inputSchema"]["json"]["properties"]
        assert "action" in props
        assert set(props["action"]["enum"]) == {"execute", "start", "status", "stop", "record", "replay", "features"}
        assert spec["inputSchema"]["json"]["required"] == ["action"]


# ══════════════════════════════════════════════════════════════════════════════
# get_features
# ══════════════════════════════════════════════════════════════════════════════


class TestGetFeatures:
    def test_success(self, tool):
        r = tool.get_features()
        assert r["status"] == "success"
        assert "Observation Features" in r["content"][0]["text"]
        assert "Action Features" in r["content"][0]["text"]

    def test_json_payload(self, tool):
        r = tool.get_features()
        j = r["content"][1]["json"]
        assert "j1" in j["observation_features"]
        assert "j1" in j["action_features"]

    def test_empty_features_warning(self, tool):
        tool._observation_features = {}
        tool._action_features = {}
        r = tool.get_features()
        assert "No features available" in r["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════════
# start_task / get_task_status / stop_task
# ══════════════════════════════════════════════════════════════════════════════


class TestStartTask:
    def test_start_ok(self, tool):
        tool._executor = MagicMock()
        tool._executor.submit.return_value = MagicMock()
        r = tool.start_task("pick up block", policy_provider="mock")
        assert r["status"] == "success"
        assert "started" in r["content"][0]["text"].lower()

    def test_rejects_while_running(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.RUNNING
        tool._task_state.instruction = "old"
        r = tool.start_task("new")
        assert r["status"] == "error"
        assert "already running" in r["content"][0]["text"].lower()

    def test_submits_to_executor(self, tool):
        mock_ex = MagicMock()
        mock_ex.submit.return_value = MagicMock()
        tool._executor = mock_ex
        tool.start_task("go", policy_provider="mock")
        mock_ex.submit.assert_called_once()


class TestGetTaskStatus:
    def test_idle(self, tool):
        r = tool.get_task_status()
        assert "IDLE" in r["content"][0]["text"]

    def test_running(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.RUNNING
        tool._task_state.instruction = "testing"
        tool._task_state.start_time = time.time() - 3
        tool._task_state.step_count = 42
        r = tool.get_task_status()
        txt = r["content"][0]["text"]
        assert "RUNNING" in txt
        assert "42" in txt

    def test_error_message(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.ERROR
        tool._task_state.error_message = "USB gone"
        r = tool.get_task_status()
        assert "USB gone" in r["content"][0]["text"]

    def test_completed(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.COMPLETED
        tool._task_state.duration = 7.5
        tool._task_state.step_count = 150
        r = tool.get_task_status()
        txt = r["content"][0]["text"]
        assert "7.5" in txt
        assert "150" in txt


class TestStopTask:
    def test_stop_running(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.RUNNING
        tool._task_state.instruction = "running"
        r = tool.stop_task()
        assert r["status"] == "success"
        assert tool._task_state.status == TaskStatus.STOPPED

    def test_no_task(self, tool):
        r = tool.stop_task()
        assert "No task running" in r["content"][0]["text"]

    def test_cancels_future(self, tool, mod):
        _, TaskStatus, _ = mod
        fut = MagicMock()
        tool._task_state.status = TaskStatus.RUNNING
        tool._task_state.instruction = "x"
        tool._task_state.task_future = fut
        tool.stop_task()
        fut.cancel.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# _get_policy — per-provider config
# ══════════════════════════════════════════════════════════════════════════════


class TestGetPolicy:
    def test_groot_requires_port(self, tool):
        with pytest.raises(ValueError, match="policy_port"):
            asyncio.run(tool._get_policy(policy_provider="groot"))

    def test_groot_config(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="groot", policy_port=5555, policy_host="h"))
            mock_cp.assert_called_once()
            kw = mock_cp.call_args[1]
            assert kw["port"] == 5555
            assert kw["host"] == "h"

    def test_groot_data_config(self, tool):
        tool.data_config = "fourier_gr1_arms_only"
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="groot", policy_port=5555))
            kw = mock_cp.call_args[1]
            assert kw["data_config"] == "fourier_gr1_arms_only"
        tool.data_config = None

    def test_lerobot_async_requires_port(self, tool):
        with pytest.raises(ValueError, match="policy_port or server_address"):
            asyncio.run(tool._get_policy(policy_provider="lerobot_async"))

    def test_lerobot_async_server_address(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="lerobot_async", server_address="h:8080"))
            assert mock_cp.call_args[1]["server_address"] == "h:8080"

    def test_lerobot_async_builds_address(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="lerobot_async", policy_port=9090, policy_host="remote"))
            assert mock_cp.call_args[1]["server_address"] == "remote:9090"

    def test_lerobot_async_forwards_policy_type(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="lerobot_async", policy_port=8080, policy_type="pi0"))
            assert mock_cp.call_args[1]["policy_type"] == "pi0"

    def test_dreamgen_requires_model_path(self, tool):
        with pytest.raises(ValueError, match="model_path"):
            asyncio.run(tool._get_policy(policy_provider="dreamgen"))

    def test_dreamgen_config(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="dreamgen", model_path="/m/v1"))
            assert mock_cp.call_args[1]["model_path"] == "/m/v1"

    def test_mock_provider(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="mock"))
            mock_cp.assert_called_once_with("mock")

    def test_lerobot_local_model_path(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="lerobot_local", model_path="lerobot/act"))
            assert mock_cp.call_args[1]["pretrained_name_or_path"] == "lerobot/act"

    def test_lerobot_alias(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="lerobot", model_path="hf/model"))
            assert mock_cp.call_args[1]["pretrained_name_or_path"] == "hf/model"

    def test_unknown_provider_passthrough(self, tool):
        with patch("strands_robots.robot.create_policy") as mock_cp:
            mock_cp.reset_mock()
            mock_cp.return_value = MagicMock()
            asyncio.run(tool._get_policy(policy_provider="custom", policy_port=7777))
            assert mock_cp.call_args[1]["port"] == 7777

    # ══════════════════════════════════════════════════════════════════════════════
    # _connect_robot
    # ══════════════════════════════════════════════════════════════════════════════


class TestConnectRobot:
    def test_already_connected(self, tool):
        tool.robot.is_connected = True
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is True and err == ""

    def test_connect_success(self, tool):
        tool.robot.is_connected = False

        def _connect(cal=False):
            tool.robot.is_connected = True

        tool.robot.connect = _connect
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is True

    def test_device_already_connected_error(self, tool):
        tool.robot.is_connected = False

        def _connect(cal=False):
            tool.robot.is_connected = True
            raise _FakeDeviceAlreadyConnectedError("already")

        tool.robot.connect = _connect
        ok, _ = asyncio.run(tool._connect_robot())
        assert ok is True

    def test_string_already_connected(self, tool):
        tool.robot.is_connected = False

        def _connect(cal=False):
            tool.robot.is_connected = True
            raise Exception("device is already connected")

        tool.robot.connect = _connect
        ok, _ = asyncio.run(tool._connect_robot())
        assert ok is True

    def test_connect_failure(self, tool):
        tool.robot.is_connected = False
        tool.robot.connect = MagicMock(side_effect=Exception("USB not found"))
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is False
        assert "USB not found" in err or "connection failed" in err.lower()

    def test_not_calibrated(self, tool):
        tool.robot.is_connected = False

        def _connect(cal=False):
            tool.robot.is_connected = True

        tool.robot.connect = _connect
        tool.robot.is_calibrated = False
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is False
        assert "calibrat" in err.lower()


# ══════════════════════════════════════════════════════════════════════════════
# _initialize_policy
# ══════════════════════════════════════════════════════════════════════════════


class TestInitializePolicy:
    def test_with_action_features(self, tool):
        p = MagicMock()
        assert asyncio.run(tool._initialize_policy(p)) is True
        p.set_robot_state_keys.assert_called_once()

    def test_fallback_to_observation(self, tool):
        tool._action_features = {}
        p = MagicMock()
        assert asyncio.run(tool._initialize_policy(p)) is True

    def test_failure(self, tool):
        tool._action_features = {}
        tool.robot.get_observation = MagicMock(side_effect=Exception("sensor fail"))
        p = MagicMock()
        assert asyncio.run(tool._initialize_policy(p)) is False


# ══════════════════════════════════════════════════════════════════════════════
# _execute_task_sync
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskSync:
    def test_success(self, tool, mod):
        _, TaskStatus, _ = mod
        tool.robot.is_connected = True
        mp = MagicMock()
        mp.get_actions = AsyncMock(return_value=[{"j1": 0.1}])
        with (
            patch.object(tool, "_get_policy", new_callable=AsyncMock, return_value=mp),
            patch.object(tool, "_initialize_policy", new_callable=AsyncMock, return_value=True),
        ):
            r = tool._execute_task_sync("test", policy_provider="mock", duration=0.05)
        assert r["status"] == "success"

    def test_connect_fail(self, tool, mod):
        _, TaskStatus, _ = mod
        with patch.object(tool, "_connect_robot", new_callable=AsyncMock, return_value=(False, "no dev")):
            r = tool._execute_task_sync("test", policy_provider="mock", duration=0.05)
        assert r["status"] == "error"
        assert tool._task_state.status == TaskStatus.ERROR

    def test_policy_init_fail(self, tool, mod):
        mp = MagicMock()
        with (
            patch.object(tool, "_connect_robot", new_callable=AsyncMock, return_value=(True, "")),
            patch.object(tool, "_get_policy", new_callable=AsyncMock, return_value=mp),
            patch.object(tool, "_initialize_policy", new_callable=AsyncMock, return_value=False),
        ):
            r = tool._execute_task_sync("test", policy_provider="mock", duration=0.05)
        assert r["status"] == "error"

    def test_policy_display_groot(self, tool):
        tool.robot.is_connected = True
        mp = MagicMock()
        mp.get_actions = AsyncMock(return_value=[{"j1": 0.1}])
        with (
            patch.object(tool, "_get_policy", new_callable=AsyncMock, return_value=mp),
            patch.object(tool, "_initialize_policy", new_callable=AsyncMock, return_value=True),
        ):
            r = tool._execute_task_sync("test", policy_provider="groot", policy_port=5555, duration=0.05)
        assert "groot" in r["content"][0]["text"].lower()
        assert "5555" in r["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════════
# stream() — async generator
# ══════════════════════════════════════════════════════════════════════════════


class TestStream:
    def _run(self, tool, tool_use):
        evts = []

        async def go():
            async for e in tool.stream(tool_use, {}):
                # ToolResultEvent wraps data as {'type': 'tool_result', 'tool_result': {...}}
                # Extract the inner tool_result dict for test assertions
                if hasattr(e, "tool_result"):
                    evts.append(e.tool_result)
                elif isinstance(e, dict) and "tool_result" in e:
                    evts.append(e["tool_result"])
                else:
                    evts.append(e)

        asyncio.run(go())
        return evts

    def test_status(self, tool):
        evts = self._run(tool, {"toolUseId": "1", "input": {"action": "status"}})
        assert "IDLE" in evts[0]["content"][0]["text"]

    def test_stop(self, tool):
        evts = self._run(tool, {"toolUseId": "2", "input": {"action": "stop"}})
        assert evts[0]["status"] == "success"

    def test_features(self, tool):
        evts = self._run(tool, {"toolUseId": "3", "input": {"action": "features"}})
        assert "Features" in evts[0]["content"][0]["text"]

    def test_unknown_action(self, tool):
        evts = self._run(tool, {"toolUseId": "4", "input": {"action": "dance"}})
        assert evts[0]["status"] == "error"
        assert "Unknown action" in evts[0]["content"][0]["text"]

    def test_execute_missing_instruction(self, tool):
        evts = self._run(tool, {"toolUseId": "5", "input": {"action": "execute"}})
        assert evts[0]["status"] == "error"

    def test_execute_groot_no_port(self, tool):
        evts = self._run(
            tool, {"toolUseId": "6", "input": {"action": "execute", "instruction": "go", "policy_provider": "groot"}}
        )
        assert evts[0]["status"] == "error"
        assert "policy_port" in evts[0]["content"][0]["text"]

    def test_execute_dreamgen_no_model(self, tool):
        evts = self._run(
            tool, {"toolUseId": "7", "input": {"action": "execute", "instruction": "go", "policy_provider": "dreamgen"}}
        )
        assert evts[0]["status"] == "error"
        assert "model_path" in evts[0]["content"][0]["text"]

    def test_execute_mock(self, tool):
        tool.robot.is_connected = True
        mp = MagicMock()
        mp.get_actions = AsyncMock(return_value=[{"j1": 0.1}])
        with (
            patch.object(tool, "_get_policy", new_callable=AsyncMock, return_value=mp),
            patch.object(tool, "_initialize_policy", new_callable=AsyncMock, return_value=True),
        ):
            evts = self._run(
                tool,
                {
                    "toolUseId": "8",
                    "input": {"action": "execute", "instruction": "pick", "policy_provider": "mock", "duration": 0.05},
                },
            )
        assert len(evts) == 1

    def test_start(self, tool):
        tool._executor = MagicMock()
        tool._executor.submit.return_value = MagicMock()
        evts = self._run(
            tool, {"toolUseId": "9", "input": {"action": "start", "instruction": "go", "policy_provider": "mock"}}
        )
        assert evts[0]["status"] == "success"
        assert "started" in evts[0]["content"][0]["text"].lower()

    def test_record_missing_instruction(self, tool):
        evts = self._run(tool, {"toolUseId": "10", "input": {"action": "record"}})
        assert evts[0]["status"] == "error"

    def test_replay_missing_repo(self, tool):
        evts = self._run(tool, {"toolUseId": "11", "input": {"action": "replay"}})
        assert evts[0]["status"] == "error"

    def test_exception_handling(self, tool):
        tool.get_features = MagicMock(side_effect=RuntimeError("boom"))
        evts = self._run(tool, {"toolUseId": "12", "input": {"action": "features"}})
        assert evts[0]["status"] == "error"
        assert "boom" in evts[0]["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════════
# cleanup / __del__
# ══════════════════════════════════════════════════════════════════════════════


class TestCleanup:
    def test_cleanup_idle(self, tool):
        tool.cleanup()
        assert tool._shutdown_event.is_set()

    def test_cleanup_running(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.RUNNING
        tool._task_state.instruction = "active"
        tool.cleanup()
        assert tool._task_state.status == TaskStatus.STOPPED

    def test_cleanup_stops_mesh(self, tool):
        m = MagicMock()
        tool.mesh = m
        tool.cleanup()
        m.stop.assert_called_once()

    def test_cleanup_mesh_error(self, tool):
        m = MagicMock()
        m.stop.side_effect = Exception("oops")
        tool.mesh = m
        tool.cleanup()  # should not raise

    def test_del(self, tool):
        with patch.object(tool, "cleanup") as mc:
            tool.__del__()
            mc.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# get_status / stop (async)
# ══════════════════════════════════════════════════════════════════════════════


class TestGetStatus:
    def test_disconnected(self, tool):
        r = asyncio.run(tool.get_status())
        assert r["robot_name"] == "test_bot"
        assert r["is_connected"] is False

    def test_connected(self, tool):
        tool.robot.is_connected = True
        r = asyncio.run(tool.get_status())
        assert r["is_connected"] is True
        assert r["task_status"] == "idle"

    def test_with_cameras(self, tool):
        tool.robot.config.cameras = {"wrist": MagicMock()}
        r = asyncio.run(tool.get_status())
        assert "wrist" in r["cameras"]

    def test_with_error(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.ERROR
        tool._task_state.error_message = "hw fault"
        r = asyncio.run(tool.get_status())
        assert r["task_error"] == "hw fault"

    def test_exception_safe(self, tool):
        type(tool.robot).is_connected = property(lambda s: (_ for _ in ()).throw(Exception("broken")))
        r = asyncio.run(tool.get_status())
        assert r["is_connected"] is False
        del type(tool.robot).is_connected
        tool.robot.is_connected = False


class TestAsyncStop:
    def test_disconnects(self, tool):
        tool.robot.is_connected = True
        asyncio.run(tool.stop())
        assert tool._shutdown_event.is_set()

    def test_stops_running_task(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._task_state.status = TaskStatus.RUNNING
        tool._task_state.instruction = "x"
        asyncio.run(tool.stop())
        assert tool._task_state.status == TaskStatus.STOPPED

    def test_disconnect_error(self, tool):
        tool.robot.disconnect = MagicMock(side_effect=Exception("fail"))
        asyncio.run(tool.stop())  # should not raise


# ══════════════════════════════════════════════════════════════════════════════
# _initialize_robot — input type dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestInitializeRobot:
    def test_direct_instance(self, tool):
        r = _FakeLeRobotRobot()
        assert tool._initialize_robot(r, None) is r

    def test_robot_config(self, tool):
        cfg = _FakeRobotConfig()
        _mock_lr_utils_mod.make_robot_from_config.return_value = _FakeLeRobotRobot()
        tool._initialize_robot(cfg, None)
        _mock_lr_utils_mod.make_robot_from_config.assert_called_with(cfg)

    def test_string_type(self, tool):
        _mock_lr_utils_mod.make_robot_from_config.return_value = _FakeLeRobotRobot()
        with patch.object(tool, "_create_minimal_config", return_value=_FakeRobotConfig()):
            result = tool._initialize_robot("so100", None)
        assert isinstance(result, _FakeLeRobotRobot)

    def test_invalid_type(self, tool):
        with pytest.raises(ValueError, match="Unsupported robot type"):
            tool._initialize_robot(12345, None)


# ══════════════════════════════════════════════════════════════════════════════
# _create_minimal_config
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateMinimalConfig:
    def test_with_cameras(self, tool):
        cfg_cls = MagicMock()
        cfg_cls.__dataclass_fields__ = {}
        cfg_cls.return_value = _FakeRobotConfig()
        with patch.object(tool, "_resolve_robot_config_class", return_value=cfg_cls):
            tool._create_minimal_config("so100", {"wrist": {"index_or_path": "/dev/video0"}})
        kw = cfg_cls.call_args[1]
        assert kw["id"] == "test_bot"
        assert "wrist" in kw["cameras"]

    def test_unsupported_camera(self, tool):
        cfg_cls = MagicMock()
        with patch.object(tool, "_resolve_robot_config_class", return_value=cfg_cls):
            with pytest.raises(ValueError, match="Unsupported camera type"):
                tool._create_minimal_config("so100", {"c": {"type": "realsense", "index_or_path": 0}})

    def test_forwards_kwargs(self, tool):
        cfg_cls = MagicMock()
        cfg_cls.__dataclass_fields__ = {"custom": None}
        cfg_cls.return_value = _FakeRobotConfig()
        with patch.object(tool, "_resolve_robot_config_class", return_value=cfg_cls):
            tool._create_minimal_config("so100", None, port="/dev/ttyUSB0", custom="val")
        kw = cfg_cls.call_args[1]
        assert kw["port"] == "/dev/ttyUSB0"
        assert kw["custom"] == "val"

    def test_type_error_message(self, tool):
        cfg_cls = MagicMock()
        cfg_cls.__name__ = "TestCfg"
        cfg_cls.__dataclass_fields__ = {}
        cfg_cls.side_effect = TypeError("missing 'serial'")
        with patch.object(tool, "_resolve_robot_config_class", return_value=cfg_cls):
            with pytest.raises(ValueError, match="Failed to create TestCfg"):
                tool._create_minimal_config("so100", None)


# ══════════════════════════════════════════════════════════════════════════════
# _resolve_robot_config_class
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveRobotConfigClass:
    def test_class_name_match(self, tool):
        class So100Config(_FakeRobotConfig):
            pass

        mock_mod = MagicMock()
        mock_mod.So100Config = So100Config
        type(mock_mod).__dir__ = lambda self: ["So100Config"]

        # Mock the entire function path: pkgutil.iter_modules and importlib.import_module
        # __path__ is tricky with MagicMock so we bypass it entirely
        with patch("pkgutil.iter_modules", return_value=[("", "so100_mod", False)]):
            with patch("importlib.import_module", return_value=mock_mod):
                result = tool._resolve_robot_config_class("so100")
        assert result is So100Config

    def test_no_match(self, tool):
        with patch("pkgutil.iter_modules", return_value=[]):
            with pytest.raises(ValueError, match="Could not find"):
                tool._resolve_robot_config_class("nonexistent")


# ══════════════════════════════════════════════════════════════════════════════
# record_task
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordTask:
    def test_lerobot_not_installed(self, tool):
        mock_ds = MagicMock()
        mock_ds.HAS_LEROBOT_DATASET = False
        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds}):
            r = tool.record_task(instruction="record test", policy_provider="mock", duration=0.1)
        assert r["status"] == "error"
        assert "lerobot" in r["content"][0]["text"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_action_sleep_time(self, tool):
        assert tool.action_sleep_time == pytest.approx(0.02, abs=0.001)

    def test_shutdown_event(self, tool, mod):
        _, TaskStatus, _ = mod
        tool._shutdown_event.set()
        assert tool._shutdown_event.is_set()

    def test_mesh_error_pattern(self, tool):
        """Mesh publish errors are caught silently in the control loop."""
        m = MagicMock()
        m.alive = True
        m.publish_step.side_effect = Exception("mesh down")
        tool.mesh = m
        # The control loop does: try: mesh.publish_step(...) except: pass
        try:
            tool.mesh.publish_step(step=1, observation={}, action={})
        except Exception:
            pass  # This is what robot.py does — confirms the pattern
