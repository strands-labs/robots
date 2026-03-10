#!/usr/bin/env python3
"""Tests for Robot record_task() and replay_episode() — the two biggest
uncovered blocks in robot.py (lines 751-874 and 894-1059).

Also covers:
- __init__ body (camera logging, features, data_config, mesh)
- _get_policy branches (dreamgen, lerobot_local, unknown)
- _initialize_policy (fallback path)
- _connect_robot (not-calibrated path)
- stream() dispatch for record/replay actions
- cleanup with running state

All tests run without hardware, lerobot, or torch.
"""

import asyncio
import sys
import threading
import types as _types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# CROSS_PR_SKIP: Tests written against full Robot API with record/replay
try:
    import strands_robots.robot as _robot_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("strands_robots.robot requires lerobot", allow_module_level=True)
if not hasattr(_robot_mod.Robot, "record"):
    import pytest as _xfail_pytest

    _xfail_pytest.skip("Tests require Robot.record from PR #11", allow_module_level=True)


# ── Module-level mocking (same pattern as test_robot_tool.py) ──────────────

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
    pass


_mock_lerobot_robots_config = MagicMock()
_mock_lerobot_robots_config.RobotConfig = _FakeRobotConfig


class _FakeLeRobotRobot:
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

_mock_lr_robots = _types.ModuleType("lerobot.robots")
_mock_lr_robots.__path__ = ["/fake/lerobot/robots"]

# Pre-mock cv2 to prevent OpenCV 4.12 cv2.dnn.DictValue crash
import importlib.machinery as _im_fix  # noqa: E402

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
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

_PATCHES["lerobot"].robots = _mock_lr_robots

# ── Import robot module ONCE at module scope under mocks ──────────────────
# This avoids the cv2.dnn.DictValue crash from _fresh_import pattern
_saved_modules = {}
for k in list(sys.modules):
    if k == "strands_robots.robot":
        _saved_modules[k] = sys.modules.pop(k)

with patch.dict(sys.modules, _PATCHES):
    from strands_robots.robot import Robot as _Robot
    from strands_robots.robot import RobotTaskState as _RobotTaskState
    from strands_robots.robot import TaskStatus as _TaskStatus

# Restore any saved modules
for k, v in _saved_modules.items():
    sys.modules[k] = v


@pytest.fixture(autouse=True)
def _mock_modules():
    with patch.dict(sys.modules, _PATCHES):
        yield


@pytest.fixture
def mod():
    return _Robot, _TaskStatus, _RobotTaskState


@pytest.fixture
def fake_robot():
    return _FakeLeRobotRobot()


@pytest.fixture
def tool(mod, fake_robot):
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
# record_task
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordTask:
    """Tests for record_task() — lines 751-874 in robot.py."""

    def test_lerobot_not_installed(self, tool):
        """record_task returns error when lerobot not installed."""
        mock_ds = MagicMock()
        mock_ds.HAS_LEROBOT_DATASET = False
        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds}):
            r = tool.record_task(instruction="pick up cube", policy_provider="mock", duration=0.1)
        assert r["status"] == "error"
        assert "lerobot" in r["content"][0]["text"].lower()

    def test_import_error_fallback(self, tool):
        """record_task returns error when dataset_recorder fails to import."""
        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": None}):
            r = tool.record_task(instruction="pick up cube", policy_provider="mock", duration=0.1)
        assert r["status"] == "error"

    def test_record_success(self, tool, mod):
        """Full record_task success path with mocked async internals."""
        _, TaskStatus, _ = mod

        # Mock DatasetRecorder
        mock_recorder = MagicMock()
        mock_recorder.root = "/tmp/fake_dataset"
        mock_recorder.repo_id = "local/test_bot_12345"
        mock_recorder.frame_count = 10
        mock_recorder.episode_count = 1
        mock_recorder.save_episode.return_value = {"status": "success"}

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True
        mock_ds_module.DatasetRecorder.create.return_value = mock_recorder

        # Mock policy
        mock_policy = AsyncMock()
        mock_policy.get_actions.return_value = [{"j1": 0.1, "j2": 0.2}]
        mock_policy.set_robot_state_keys = MagicMock()

        # Make the recording end quickly by setting shutdown after 1 cycle
        call_count = 0
        original_get_obs = tool.robot.get_observation

        def get_obs_then_stop():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                tool._shutdown_event.set()
            return original_get_obs()

        tool.robot.get_observation = get_obs_then_stop
        tool.robot.is_connected = True  # Already connected

        async def fake_get_policy(**kwargs):
            return mock_policy

        async def fake_init_policy(p):
            return True

        async def fake_connect():
            return (True, "")

        tool._get_policy = fake_get_policy
        tool._initialize_policy = fake_init_policy
        tool._connect_robot = fake_connect

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(
                instruction="pick up cube",
                policy_provider="mock",
                duration=0.5,
                fps=30,
                repo_id="local/test_data",
            )

        assert r["status"] == "success"
        assert "local/test_data" in r["content"][0]["text"] or "recorded" in r["content"][0]["text"].lower()
        mock_recorder.save_episode.assert_called_once()
        mock_recorder.finalize.assert_called_once()

    def test_record_connect_failure(self, tool, mod):
        """record_task returns error when robot connection fails."""
        _, TaskStatus, _ = mod

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True

        async def fake_connect():
            return (False, "USB disconnected")

        tool._connect_robot = fake_connect

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(instruction="pick up cube", policy_provider="mock", duration=0.5)

        assert r["status"] == "error"
        assert "USB disconnected" in r["content"][0]["text"]

    def test_record_policy_init_failure(self, tool, mod):
        """record_task returns error when policy initialization fails."""
        _, TaskStatus, _ = mod

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True

        mock_policy = AsyncMock()

        async def fake_connect():
            return (True, "")

        async def fake_get_policy(**kwargs):
            return mock_policy

        async def fake_init_policy(p):
            return False

        tool._connect_robot = fake_connect
        tool._get_policy = fake_get_policy
        tool._initialize_policy = fake_init_policy

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(instruction="pick up cube", policy_provider="mock", duration=0.5)

        assert r["status"] == "error"
        assert "initialize" in r["content"][0]["text"].lower() or "failed" in r["content"][0]["text"].lower()

    def test_record_auto_repo_id(self, tool, mod):
        """record_task auto-generates repo_id when not provided."""
        _, TaskStatus, _ = mod

        mock_recorder = MagicMock()
        mock_recorder.root = "/tmp/ds"
        mock_recorder.repo_id = "local/test_bot_9999"
        mock_recorder.frame_count = 0
        mock_recorder.episode_count = 1
        mock_recorder.save_episode.return_value = {"status": "success"}

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True
        mock_ds_module.DatasetRecorder.create.return_value = mock_recorder

        mock_policy = AsyncMock()
        mock_policy.get_actions.return_value = [{"j1": 0.1}]
        mock_policy.set_robot_state_keys = MagicMock()

        tool._shutdown_event.set()
        tool.robot.is_connected = True

        async def fake_connect():
            return (True, "")

        async def fake_get_policy(**kwargs):
            return mock_policy

        async def fake_init_policy(p):
            return True

        tool._connect_robot = fake_connect
        tool._get_policy = fake_get_policy
        tool._initialize_policy = fake_init_policy

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(
                instruction="wave hand",
                policy_provider="mock",
                duration=0.1,
            )

        assert r["status"] == "success"
        create_kwargs = mock_ds_module.DatasetRecorder.create.call_args
        assert "local/test_bot_" in create_kwargs[1]["repo_id"]

    def test_record_push_to_hub(self, tool, mod):
        """record_task pushes to hub when flag is set."""
        _, TaskStatus, _ = mod

        mock_recorder = MagicMock()
        mock_recorder.root = "/tmp/ds"
        mock_recorder.repo_id = "user/my_data"
        mock_recorder.frame_count = 5
        mock_recorder.episode_count = 1
        mock_recorder.save_episode.return_value = {"status": "success"}
        mock_recorder.push_to_hub.return_value = {"status": "success"}

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True
        mock_ds_module.DatasetRecorder.create.return_value = mock_recorder

        mock_policy = AsyncMock()
        mock_policy.get_actions.return_value = [{"j1": 0.1}]
        mock_policy.set_robot_state_keys = MagicMock()

        tool._shutdown_event.set()
        tool.robot.is_connected = True

        async def fake_connect():
            return (True, "")

        async def fake_get_policy(**kwargs):
            return mock_policy

        async def fake_init_policy(p):
            return True

        tool._connect_robot = fake_connect
        tool._get_policy = fake_get_policy
        tool._initialize_policy = fake_init_policy

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(
                instruction="pick up cube",
                policy_provider="mock",
                duration=0.1,
                push_to_hub=True,
                repo_id="user/my_data",
            )

        assert r["status"] == "success"
        mock_recorder.push_to_hub.assert_called_once()
        assert "hub" in r["content"][0]["text"].lower()

    def test_record_with_cameras(self, tool, mod):
        """record_task extracts camera keys from robot config."""
        _, TaskStatus, _ = mod

        mock_recorder = MagicMock()
        mock_recorder.root = "/tmp/ds"
        mock_recorder.repo_id = "local/cam_test"
        mock_recorder.frame_count = 0
        mock_recorder.episode_count = 1
        mock_recorder.save_episode.return_value = {"status": "success"}

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True
        mock_ds_module.DatasetRecorder.create.return_value = mock_recorder

        mock_policy = AsyncMock()
        mock_policy.get_actions.return_value = [{"j1": 0.1}]
        mock_policy.set_robot_state_keys = MagicMock()

        tool.robot.config.cameras = {"wrist": MagicMock(), "overhead": MagicMock()}
        tool._shutdown_event.set()
        tool.robot.is_connected = True

        async def fake_connect():
            return (True, "")

        async def fake_get_policy(**kwargs):
            return mock_policy

        async def fake_init_policy(p):
            return True

        tool._connect_robot = fake_connect
        tool._get_policy = fake_get_policy
        tool._initialize_policy = fake_init_policy

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(
                instruction="look around",
                policy_provider="mock",
                duration=0.1,
                repo_id="local/cam_test",
            )

        assert r["status"] == "success"
        create_kwargs = mock_ds_module.DatasetRecorder.create.call_args[1]
        assert set(create_kwargs["camera_keys"]) == {"wrist", "overhead"}

    def test_record_no_action_features_fallback(self, tool, mod):
        """record_task falls back to observation keys when no action features."""
        _, TaskStatus, _ = mod

        mock_recorder = MagicMock()
        mock_recorder.root = "/tmp/ds"
        mock_recorder.repo_id = "local/fb_test"
        mock_recorder.frame_count = 0
        mock_recorder.episode_count = 1
        mock_recorder.save_episode.return_value = {"status": "success"}

        mock_ds_module = MagicMock()
        mock_ds_module.HAS_LEROBOT_DATASET = True
        mock_ds_module.DatasetRecorder.create.return_value = mock_recorder

        mock_policy = AsyncMock()
        mock_policy.get_actions.return_value = [{"j1": 0.1}]
        mock_policy.set_robot_state_keys = MagicMock()

        tool._action_features = {}
        tool._shutdown_event.set()
        tool.robot.is_connected = True

        async def fake_connect():
            return (True, "")

        async def fake_get_policy(**kwargs):
            return mock_policy

        async def fake_init_policy(p):
            return True

        tool._connect_robot = fake_connect
        tool._get_policy = fake_get_policy
        tool._initialize_policy = fake_init_policy

        with patch.dict(sys.modules, {"strands_robots.dataset_recorder": mock_ds_module}):
            r = tool.record_task(
                instruction="test",
                policy_provider="mock",
                duration=0.1,
                repo_id="local/fb_test",
            )

        assert r["status"] == "success"
        create_kwargs = mock_ds_module.DatasetRecorder.create.call_args[1]
        assert "j1" in create_kwargs["joint_names"]


# ══════════════════════════════════════════════════════════════════════════════
# replay_episode
# ══════════════════════════════════════════════════════════════════════════════


class TestReplayEpisode:
    """Tests for replay_episode() — lines 894-1059 in robot.py."""

    def _make_mock_dataset(self, num_episodes=2, frames_per_episode=5):
        """Create a mock LeRobotDataset."""
        ds = MagicMock()
        ds.fps = 30
        ds.meta = MagicMock()
        ds.meta.total_episodes = num_episodes
        ds.meta.episodes = [{"length": frames_per_episode} for _ in range(num_episodes)]

        total_frames = num_episodes * frames_per_episode
        frames = []
        for ep_idx in range(num_episodes):
            for f in range(frames_per_episode):
                frames.append(
                    {
                        "action": [0.1 * f, 0.2 * f],
                        "observation.state": [0.5, 0.3],
                        "episode_index": ep_idx,
                    }
                )

        ds.__len__ = lambda self: total_frames
        ds.__getitem__ = lambda self, idx: frames[idx]

        return ds

    def test_replay_import_error(self, tool):
        """replay_episode returns error when lerobot not installed."""
        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": None}):
            r = tool.replay_episode(repo_id="user/test_data", episode=0)
        assert r["status"] == "error"

    def test_replay_dataset_load_error(self, tool):
        """replay_episode returns error when dataset fails to load."""
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.side_effect = Exception("File not found")
        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": mock_ds_mod}):
            r = tool.replay_episode(repo_id="user/bad_data", episode=0)
        assert r["status"] == "error"
        assert "failed to load" in r["content"][0]["text"].lower()

    def test_replay_episode_out_of_range(self, tool):
        """replay_episode returns error when episode index exceeds count."""
        mock_ds = self._make_mock_dataset(num_episodes=2)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": mock_ds_mod}):
            r = tool.replay_episode(repo_id="user/test_data", episode=5)
        assert r["status"] == "error"
        assert "out of range" in r["content"][0]["text"].lower()

    def test_replay_success(self, tool, mod):
        """Full replay_episode success path."""
        _, TaskStatus, _ = mod

        mock_ds = self._make_mock_dataset(num_episodes=1, frames_per_episode=3)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        tool.robot.is_connected = True
        tool.robot.send_action = MagicMock()

        async def fake_connect():
            return (True, "")

        tool._connect_robot = fake_connect

        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.Tensor = type(mock_tensor)
        mock_torch.tensor.return_value = mock_tensor

        with patch.dict(
            sys.modules,
            {
                "lerobot.datasets.lerobot_dataset": mock_ds_mod,
                "torch": mock_torch,
            },
        ):
            r = tool.replay_episode(
                repo_id="user/test_data",
                episode=0,
                speed=10.0,
            )

        assert r["status"] == "success"
        assert "replayed" in r["content"][0]["text"].lower()
        assert r["content"][1]["json"]["episode"] == 0
        assert r["content"][1]["json"]["total_frames"] == 3

    def test_replay_connect_failure(self, tool, mod):
        """replay_episode returns error when robot connection fails."""
        _, TaskStatus, _ = mod

        mock_ds = self._make_mock_dataset(num_episodes=1, frames_per_episode=3)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        async def fake_connect():
            return (False, "USB disconnected")

        tool._connect_robot = fake_connect

        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": mock_ds_mod}):
            r = tool.replay_episode(repo_id="user/test_data", episode=0)
        assert r["status"] == "error"
        assert "disconnected" in r["content"][0]["text"].lower()

    def test_replay_zero_frames_episode(self, tool, mod):
        """replay_episode returns error for episode with no frames."""
        _, TaskStatus, _ = mod

        mock_ds = MagicMock()
        mock_ds.fps = 30
        mock_ds.meta = MagicMock()
        mock_ds.meta.total_episodes = 2
        mock_ds.meta.episodes = [{"length": 0}, {"length": 5}]

        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": mock_ds_mod}):
            r = tool.replay_episode(repo_id="user/test_data", episode=0)
        assert r["status"] == "error"
        assert "no frames" in r["content"][0]["text"].lower()

    def test_replay_shutdown_mid_replay(self, tool, mod):
        """replay_episode stops when shutdown event is set during replay."""
        _, TaskStatus, _ = mod

        mock_ds = self._make_mock_dataset(num_episodes=1, frames_per_episode=10)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        tool.robot.is_connected = True
        call_count = 0

        def send_then_stop(action):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                tool._shutdown_event.set()

        tool.robot.send_action = send_then_stop

        async def fake_connect():
            return (True, "")

        tool._connect_robot = fake_connect

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "lerobot.datasets.lerobot_dataset": mock_ds_mod,
                "torch": mock_torch,
            },
        ):
            r = tool.replay_episode(
                repo_id="user/test_data",
                episode=0,
                speed=100.0,
            )

        assert r["status"] == "success"
        assert r["content"][1]["json"]["frames_sent"] < 10

    def test_replay_speed_multiplier(self, tool, mod):
        """replay_episode respects speed multiplier."""
        _, TaskStatus, _ = mod

        mock_ds = self._make_mock_dataset(num_episodes=1, frames_per_episode=2)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        tool.robot.is_connected = True
        tool.robot.send_action = MagicMock()

        async def fake_connect():
            return (True, "")

        tool._connect_robot = fake_connect

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "lerobot.datasets.lerobot_dataset": mock_ds_mod,
                "torch": mock_torch,
            },
        ):
            r = tool.replay_episode(
                repo_id="user/test_data",
                episode=0,
                speed=100.0,
            )

        assert r["status"] == "success"
        assert r["content"][1]["json"]["speed"] == 100.0

    def test_replay_second_episode(self, tool, mod):
        """replay_episode correctly indexes second episode frames."""
        _, TaskStatus, _ = mod

        mock_ds = self._make_mock_dataset(num_episodes=3, frames_per_episode=4)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        tool.robot.is_connected = True
        tool.robot.send_action = MagicMock()

        async def fake_connect():
            return (True, "")

        tool._connect_robot = fake_connect

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "lerobot.datasets.lerobot_dataset": mock_ds_mod,
                "torch": mock_torch,
            },
        ):
            r = tool.replay_episode(
                repo_id="user/test_data",
                episode=1,
                speed=100.0,
            )

        assert r["status"] == "success"
        assert r["content"][1]["json"]["episode"] == 1
        assert r["content"][1]["json"]["total_frames"] == 4

    def test_replay_with_root(self, tool, mod):
        """replay_episode passes root to LeRobotDataset."""
        _, TaskStatus, _ = mod

        mock_ds = self._make_mock_dataset(num_episodes=1, frames_per_episode=2)
        mock_ds_mod = MagicMock()
        mock_ds_mod.LeRobotDataset.return_value = mock_ds

        tool.robot.is_connected = True
        tool.robot.send_action = MagicMock()

        async def fake_connect():
            return (True, "")

        tool._connect_robot = fake_connect

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "lerobot.datasets.lerobot_dataset": mock_ds_mod,
                "torch": mock_torch,
            },
        ):
            r = tool.replay_episode(
                repo_id="user/test_data",
                episode=0,
                root="/custom/path",
                speed=100.0,
            )

        assert r["status"] == "success"
        # Verify root was passed to LeRobotDataset constructor
        mock_ds_mod.LeRobotDataset.assert_called_with(repo_id="user/test_data", root="/custom/path")


# ══════════════════════════════════════════════════════════════════════════════
# _get_policy — uncovered branches
# ══════════════════════════════════════════════════════════════════════════════


class TestGetPolicyBranches:
    """Cover _get_policy branches for dreamgen, lerobot_local, unknown."""

    def test_dreamgen_requires_model_path(self, tool):
        with pytest.raises(ValueError, match="model_path is required"):
            asyncio.run(tool._get_policy(policy_provider="dreamgen"))

    def test_dreamgen_with_model_path(self, tool):
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="dreamgen",
                model_path="/models/dg_so100",
                mode="rollout",
                embodiment_tag="so100",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["model_path"] == "/models/dg_so100"
        assert call_kw["mode"] == "rollout"
        assert call_kw["embodiment_tag"] == "so100"

    def test_lerobot_local_provider(self, tool):
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="lerobot_local",
                pretrained_name_or_path="lerobot/act_so100_test",
                device="cpu",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["pretrained_name_or_path"] == "lerobot/act_so100_test"
        assert call_kw["device"] == "cpu"

    def test_lerobot_local_forwards_named_policy_type(self, tool):
        """Regression: policy_type as named arg must be forwarded (not just via **kwargs)."""
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="lerobot_local",
                pretrained_name_or_path="lerobot/act_so100_test",
                policy_type="act",
                device="cpu",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["policy_type"] == "act", "policy_type named arg was silently dropped for lerobot_local"

    def test_lerobot_alias_provider(self, tool):
        """'lerobot' is an alias for 'lerobot_local'."""
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="lerobot",
                model_path="lerobot/act_so100",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["pretrained_name_or_path"] == "lerobot/act_so100"

    def test_unknown_provider_with_port(self, tool):
        """Unknown providers get port/host as best-effort."""
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="custom_provider",
                policy_port=9999,
                policy_host="192.168.1.1",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["port"] == 9999
        assert call_kw["host"] == "192.168.1.1"

    def test_unknown_provider_with_data_config(self, tool):
        """Unknown providers get data_config if available."""
        tool.data_config = "gr1_arms_only"
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="custom_provider",
                policy_port=5555,
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["data_config"] == "gr1_arms_only"

    def test_lerobot_async_requires_port_or_address(self, tool):
        with pytest.raises(ValueError, match="policy_port or server_address"):
            asyncio.run(tool._get_policy(policy_provider="lerobot_async"))

    def test_lerobot_async_with_server_address(self, tool):
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="lerobot_async",
                server_address="10.0.0.5:8080",
                policy_type="pi0",
                pretrained_name_or_path="lerobot/pi0_test",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["server_address"] == "10.0.0.5:8080"
        assert call_kw["policy_type"] == "pi0"
        assert call_kw["pretrained_name_or_path"] == "lerobot/pi0_test"

    def test_lerobot_async_from_host_port(self, tool):
        """lerobot_async builds server_address from host:port."""
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="lerobot_async",
                policy_port=8080,
                policy_host="192.168.1.5",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["server_address"] == "192.168.1.5:8080"

    def test_dreamgen_forwards_all_kwargs(self, tool):
        """dreamgen-specific kwargs are forwarded."""
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="dreamgen",
                model_path="/models/dg",
                action_horizon=16,
                action_dim=6,
                denoising_steps=10,
                device="cuda",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["action_horizon"] == 16
        assert call_kw["action_dim"] == 6
        assert call_kw["denoising_steps"] == 10
        assert call_kw["device"] == "cuda"

    def test_groot_requires_port(self, tool):
        with pytest.raises(ValueError, match="policy_port is required"):
            asyncio.run(tool._get_policy(policy_provider="groot"))

    def test_groot_with_data_config(self, tool):
        """groot forwards data_config."""
        tool.data_config = "gr1_arms_waist"
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="groot",
                policy_port=5555,
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["data_config"] == "gr1_arms_waist"

    def test_lerobot_local_with_model_path_alias(self, tool):
        """lerobot_local: model_path used as pretrained_name_or_path alias."""
        _mock_policies.create_policy.return_value = MagicMock()
        asyncio.run(
            tool._get_policy(
                policy_provider="lerobot_local",
                model_path="lerobot/act_so100",
            )
        )
        call_kw = _mock_policies.create_policy.call_args[1]
        assert call_kw["pretrained_name_or_path"] == "lerobot/act_so100"


# ══════════════════════════════════════════════════════════════════════════════
# _initialize_policy
# ══════════════════════════════════════════════════════════════════════════════


class TestInitializePolicy:
    """Cover _initialize_policy fallback path (no action features)."""

    def test_with_action_features(self, tool):
        """Normal path: extract state keys from action_features."""
        mock_policy = MagicMock()
        result = asyncio.run(tool._initialize_policy(mock_policy))
        assert result is True
        mock_policy.set_robot_state_keys.assert_called_once()
        call_args = mock_policy.set_robot_state_keys.call_args[0][0]
        assert "j1" in call_args
        assert "j2" in call_args

    def test_no_action_features_fallback(self, tool):
        """Fallback: get keys from observation minus camera keys."""
        tool._action_features = {}
        tool.robot.config.cameras = {"wrist": MagicMock()}
        tool.robot.get_observation = lambda: {"j1": 0.5, "j2": 0.3, "wrist": "img_data"}

        mock_policy = MagicMock()
        result = asyncio.run(tool._initialize_policy(mock_policy))
        assert result is True
        call_args = mock_policy.set_robot_state_keys.call_args[0][0]
        assert "j1" in call_args
        assert "j2" in call_args
        assert "wrist" not in call_args

    def test_exception_returns_false(self, tool):
        """Exception during init returns False."""
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys.side_effect = RuntimeError("broken")
        result = asyncio.run(tool._initialize_policy(mock_policy))
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# _connect_robot edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestConnectRobotEdgeCases:
    def test_not_calibrated(self, tool):
        """Returns error when robot is not calibrated."""
        tool.robot.is_connected = False
        tool.robot.is_calibrated = False

        def connect_and_mark(calibrate=False):
            tool.robot.is_connected = True

        tool.robot.connect = connect_and_mark
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is False
        assert "calibrat" in err.lower()

    def test_connect_fails_stays_disconnected(self, tool):
        """Returns error when connect() doesn't change is_connected."""
        tool.robot.is_connected = False

        def connect_noop(calibrate=False):
            pass

        tool.robot.connect = connect_noop
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is False
        assert "failed" in err.lower()

    def test_already_connected_string_error(self, tool):
        """Handles string-based 'already connected' error gracefully."""
        tool.robot.is_connected = False

        def connect_raises(calibrate=False):
            raise Exception("Device is already connected to the bus")

        tool.robot.connect = connect_raises
        ok, err = asyncio.run(tool._connect_robot())
        # The error is caught but is_connected is still False
        assert ok is False

    def test_already_connected_true(self, tool):
        """Already connected returns success immediately."""
        tool.robot.is_connected = True
        ok, err = asyncio.run(tool._connect_robot())
        assert ok is True
        assert err == ""


# ══════════════════════════════════════════════════════════════════════════════
# __init__ body coverage
# ══════════════════════════════════════════════════════════════════════════════


class TestInitBody:
    """Cover the __init__ body paths (camera logging, data_config, mesh)."""

    def test_init_with_cameras(self, mod):
        Robot, TaskStatus, RobotTaskState = mod

        fake_robot = _FakeLeRobotRobot()
        fake_robot.config.cameras = {"wrist": MagicMock(), "overhead": MagicMock()}

        with patch.object(Robot, "_initialize_robot", return_value=fake_robot):
            obj = Robot(robot=fake_robot, tool_name="cam_bot")

        assert obj.robot is fake_robot
        obj.cleanup()

    def test_init_with_data_config(self, mod):
        Robot, TaskStatus, RobotTaskState = mod

        fake_robot = _FakeLeRobotRobot()

        with patch.object(Robot, "_initialize_robot", return_value=fake_robot):
            obj = Robot(robot=fake_robot, tool_name="cfg_bot", data_config="gr1_arms_waist")

        assert obj.data_config == "gr1_arms_waist"
        obj.cleanup()

    def test_init_with_mesh_error(self, mod):
        Robot, TaskStatus, RobotTaskState = mod

        fake_robot = _FakeLeRobotRobot()
        _mock_zenoh_mesh.init_mesh.side_effect = Exception("zenoh unavailable")

        with patch.object(Robot, "_initialize_robot", return_value=fake_robot):
            obj = Robot(robot=fake_robot, tool_name="mesh_bot")

        assert obj.mesh is None
        _mock_zenoh_mesh.init_mesh.side_effect = None
        obj.cleanup()

    def test_init_with_custom_frequency(self, mod):
        Robot, TaskStatus, RobotTaskState = mod

        fake_robot = _FakeLeRobotRobot()

        with patch.object(Robot, "_initialize_robot", return_value=fake_robot):
            obj = Robot(robot=fake_robot, tool_name="freq_bot", control_frequency=100)

        assert obj.control_frequency == 100
        assert obj.action_sleep_time == pytest.approx(0.01, abs=0.001)
        obj.cleanup()

    def test_init_without_features(self, mod):
        """__init__ handles robot without observation/action features."""
        Robot, TaskStatus, RobotTaskState = mod

        fake_robot = _FakeLeRobotRobot()
        # Remove features attributes
        del fake_robot.observation_features
        del fake_robot.action_features

        with patch.object(Robot, "_initialize_robot", return_value=fake_robot):
            obj = Robot(robot=fake_robot, tool_name="no_feat_bot")

        assert obj._observation_features == {}
        assert obj._action_features == {}
        obj.cleanup()


# ══════════════════════════════════════════════════════════════════════════════
# _resolve_robot_config_class — strategy 2 & 3
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveConfigStrategies:
    """Cover strategies 2 (type property) and 3 (fuzzy match)."""

    def test_strategy2_type_property_match(self, tool):
        """Strategy 2: match by config.type property."""

        class UnknownConfig(_FakeRobotConfig):
            def __init__(self, **kwargs):
                self.type = "custom_arm"

        mock_mod = MagicMock()
        mock_mod.UnknownConfig = UnknownConfig
        type(mock_mod).__dir__ = lambda self: ["UnknownConfig"]

        with patch("pkgutil.iter_modules", return_value=[("", "custom_mod", False)]):
            with patch("importlib.import_module", return_value=mock_mod):
                result = tool._resolve_robot_config_class("custom_arm")
        assert result is UnknownConfig

    def test_strategy3_fuzzy_match(self, tool):
        """Strategy 3: fuzzy match by module name parts."""

        class SomeFollowerConfig(_FakeRobotConfig):
            def __init__(self, **kwargs):
                self.type = "something_else"

        mock_mod = MagicMock()
        mock_mod.SomeFollowerConfig = SomeFollowerConfig
        type(mock_mod).__dir__ = lambda self: ["SomeFollowerConfig"]

        with patch("pkgutil.iter_modules", return_value=[("", "some_follower_mod", False)]):
            with patch("importlib.import_module", return_value=mock_mod):
                result = tool._resolve_robot_config_class("some_follower")
        assert result is SomeFollowerConfig


# ══════════════════════════════════════════════════════════════════════════════
# stream() dispatch for record and replay actions
# ══════════════════════════════════════════════════════════════════════════════


class TestStreamRecordReplay:
    def _run(self, tool, tool_use):
        async def _collect():
            results = []
            async for e in tool.stream(tool_use, {}):
                results.append(e)
            return results

        return asyncio.run(_collect())

    def test_stream_record_dispatches(self, tool):
        tool.record_task = MagicMock(
            return_value={
                "status": "success",
                "content": [{"text": "recorded 10 frames"}],
            }
        )

        events = self._run(
            tool,
            {
                "toolUseId": "rec1",
                "input": {
                    "action": "record",
                    "instruction": "pick up cube",
                    "policy_provider": "mock",
                    "repo_id": "user/test",
                },
            },
        )

        assert events[0]["status"] == "success"
        tool.record_task.assert_called_once()

    def test_stream_replay_dispatches(self, tool):
        tool.replay_episode = MagicMock(
            return_value={
                "status": "success",
                "content": [{"text": "replayed episode 0"}],
            }
        )

        events = self._run(
            tool,
            {
                "toolUseId": "rep1",
                "input": {
                    "action": "replay",
                    "repo_id": "user/test",
                    "episode": 1,
                    "speed": 2.0,
                },
            },
        )

        assert events[0]["status"] == "success"
        tool.replay_episode.assert_called_once()
        call_kw = tool.replay_episode.call_args[1]
        assert call_kw["episode"] == 1
        assert call_kw["speed"] == 2.0

    def test_stream_replay_missing_repo(self, tool):
        events = self._run(
            tool,
            {
                "toolUseId": "rep2",
                "input": {"action": "replay"},
            },
        )
        assert events[0]["status"] == "error"

    def test_stream_record_missing_instruction(self, tool):
        events = self._run(
            tool,
            {
                "toolUseId": "rec2",
                "input": {"action": "record"},
            },
        )
        assert events[0]["status"] == "error"

    def test_stream_record_with_policy_kwargs(self, tool):
        tool.record_task = MagicMock(
            return_value={
                "status": "success",
                "content": [{"text": "ok"}],
            }
        )

        events = self._run(
            tool,
            {
                "toolUseId": "rec3",
                "input": {
                    "action": "record",
                    "instruction": "test",
                    "policy_provider": "lerobot_local",
                    "model_path": "/models/act_so100",
                    "pretrained_name_or_path": "lerobot/act_test",
                },
            },
        )

        assert events[0]["status"] == "success"

    def test_stream_unknown_action(self, tool):
        events = self._run(
            tool,
            {
                "toolUseId": "unk1",
                "input": {"action": "fly"},
            },
        )
        assert events[0]["status"] == "error"
        assert "unknown" in events[0]["content"][0]["text"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# cleanup / __del__ additional paths
# ══════════════════════════════════════════════════════════════════════════════


class TestCleanupAdditional:
    def test_del_exception_safe(self, tool):
        tool.cleanup = MagicMock(side_effect=Exception("cleanup crashed"))
        tool.__del__()  # Should not raise

    def test_cleanup_executor_shutdown(self, tool):
        mock_exec = MagicMock()
        tool._executor = mock_exec
        tool.cleanup()
        mock_exec.shutdown.assert_called_once_with(wait=True)

    def test_cleanup_with_mesh(self, tool):
        m = MagicMock()
        tool.mesh = m
        tool.cleanup()
        m.stop.assert_called_once()

    def test_cleanup_mesh_error(self, tool):
        m = MagicMock()
        m.stop.side_effect = Exception("mesh error")
        tool.mesh = m
        tool.cleanup()  # Should not raise
