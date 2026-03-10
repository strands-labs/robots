"""Tests for strands_robots/leisaac.py — LeIsaac × LeRobot EnvHub integration.

All LeRobot, IsaacSim, and GPU dependencies are mocked. CPU-only.
Covers:
- list_tasks() and format_task_table()
- RolloutResult dataclass
- LeIsaacEnv initialization, load, reset, step, render, close
- LeIsaacEnv.rollout() — policy evaluation with mocked env
- LeIsaacEnv.record_video() — video recording with mocked env
- _obs_to_dict() and _dict_to_action() conversion utilities
- LeIsaacEnv.get_joint_names() edge cases
- leisaac_env AgentTool (all actions)
- create_leisaac_env convenience function
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Mock strands if not installed ───
try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types as _types

    _mock_strands = _types.ModuleType("strands")
    _mock_strands.tool = lambda f: f
    _mock_strands.tools = _types.ModuleType("strands.tools")
    _mock_strands.tools.decorator = _types.ModuleType("strands.tools.decorator")
    _mock_strands.tools.decorator.tool = lambda f: f
    sys.modules["strands"] = _mock_strands
    sys.modules["strands.tools"] = _mock_strands.tools
    sys.modules["strands.tools.decorator"] = _mock_strands.tools.decorator
    HAS_STRANDS = False

# ─── Pre-mock cv2 if needed (prevents OpenCV 4.12 crash) ───
_cv2_needs_mock = "cv2" not in sys.modules
if _cv2_needs_mock:
    import importlib.machinery as _im_fix

    _mock_cv2 = MagicMock()
    _mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
    _mock_cv2.dnn = MagicMock()
    _mock_cv2.dnn.DictValue = MagicMock()
    sys.modules["cv2"] = _mock_cv2

# ─── Import module under test ───
for _key in list(sys.modules.keys()):
    if "strands_robots.leisaac" in _key:
        del sys.modules[_key]

from strands_robots.leisaac import (  # noqa: E402
    LEISAAC_TASKS,
    LeIsaacEnv,
    RolloutResult,
    create_leisaac_env,
    format_task_table,
    list_tasks,
)

# Try to import the tool (may be None if strands not available)
try:
    from strands_robots.leisaac import leisaac_env
except ImportError:
    leisaac_env = None

# Restore cv2 if we mocked it
if _cv2_needs_mock and "cv2" in sys.modules and sys.modules["cv2"] is _mock_cv2:
    del sys.modules["cv2"]


# ═════════════════════════════════════════════════════════════════════════════
# list_tasks / format_task_table Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestListTasks:
    def test_returns_list(self):
        tasks = list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == len(LEISAAC_TASKS)

    def test_task_structure(self):
        tasks = list_tasks()
        for task in tasks:
            assert "name" in task
            assert "env_id" in task
            assert "description" in task
            assert "robot" in task
            assert "category" in task
            assert "instruction" in task

    def test_known_tasks_present(self):
        tasks = list_tasks()
        names = [t["name"] for t in tasks]
        assert "so101_pick_orange" in names
        assert "so101_lift_cube" in names


class TestFormatTaskTable:
    def test_returns_string(self):
        table = format_task_table()
        assert isinstance(table, str)

    def test_contains_header(self):
        table = format_task_table()
        assert "LeIsaac" in table
        assert "Task" in table

    def test_contains_all_tasks(self):
        table = format_task_table()
        for name in LEISAAC_TASKS:
            assert name in table

    def test_contains_envhub_source(self):
        table = format_task_table()
        assert "LightwheelAI" in table


# ═════════════════════════════════════════════════════════════════════════════
# RolloutResult Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRolloutResult:
    def test_defaults(self):
        r = RolloutResult()
        assert r.n_episodes == 0
        assert r.n_successes == 0
        assert r.success_rate == 0.0
        assert r.avg_steps == 0.0
        assert r.avg_reward == 0.0
        assert r.total_time == 0.0
        assert r.episodes == []

    def test_to_dict(self):
        r = RolloutResult(
            n_episodes=10,
            n_successes=7,
            success_rate=0.7,
            avg_steps=150.3456,
            avg_reward=12.3456789,
            total_time=45.6789,
        )
        d = r.to_dict()
        assert d["n_episodes"] == 10
        assert d["n_successes"] == 7
        assert d["success_rate"] == 0.7
        assert d["avg_steps"] == 150.3
        assert d["avg_reward"] == 12.3457
        assert d["total_time"] == 45.7

    def test_to_dict_rounding(self):
        r = RolloutResult(
            success_rate=0.33333333,
            avg_steps=99.9999,
            avg_reward=0.00001234,
            total_time=0.00001,
        )
        d = r.to_dict()
        assert d["success_rate"] == 0.3333
        assert d["avg_steps"] == 100.0
        assert d["avg_reward"] == 0.0
        assert d["total_time"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# LeIsaacEnv Initialization Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLeIsaacEnvInit:
    def test_known_task(self):
        env = LeIsaacEnv("so101_pick_orange")
        assert env.task_name == "so101_pick_orange"
        assert env.task_info == LEISAAC_TASKS["so101_pick_orange"]
        assert env._loaded is False
        assert env._env is None

    def test_custom_task(self):
        env = LeIsaacEnv("custom/repo:envs/my_task.py")
        assert env.task_name == "custom/repo:envs/my_task.py"
        assert env.task_info["robot"] == "unknown"
        assert env.env_script == "custom/repo:envs/my_task.py"

    def test_default_params(self):
        env = LeIsaacEnv()
        assert env.task_name == "so101_pick_orange"
        assert env.n_envs == 1
        assert env.render_mode == "rgb_array"

    def test_custom_params(self):
        env = LeIsaacEnv("so101_lift_cube", n_envs=4, render_mode="human")
        assert env.n_envs == 4
        assert env.render_mode == "human"

    def test_repr(self):
        env = LeIsaacEnv("so101_pick_orange")
        r = repr(env)
        assert "so101_pick_orange" in r
        assert "loaded=False" in r

    def test_repr_loaded(self):
        env = LeIsaacEnv("so101_pick_orange")
        env._loaded = True
        r = repr(env)
        assert "loaded=True" in r


# ═════════════════════════════════════════════════════════════════════════════
# LeIsaacEnv.load Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLeIsaacEnvLoad:
    def test_load_success(self):
        mock_inner_env = MagicMock()
        mock_sync_vec = MagicMock()
        mock_sync_vec.envs = [MagicMock(unwrapped=mock_inner_env)]

        mock_make_env = MagicMock(return_value={"suite": (mock_sync_vec,)})

        mock_factory = ModuleType("lerobot.envs.factory")
        mock_factory.make_env = mock_make_env

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.envs": MagicMock(),
                "lerobot.envs.factory": mock_factory,
            },
        ):
            env = LeIsaacEnv("so101_pick_orange")
            result = env.load()

        assert result is True
        assert env._loaded is True
        assert env._raw_env is mock_inner_env

    def test_load_with_initialize(self):
        mock_inner_env = MagicMock()
        mock_inner_env.initialize = MagicMock()
        mock_sync_vec = MagicMock()
        mock_sync_vec.envs = [MagicMock(unwrapped=mock_inner_env)]

        mock_make_env = MagicMock(return_value={"suite": (mock_sync_vec,)})

        mock_factory = ModuleType("lerobot.envs.factory")
        mock_factory.make_env = mock_make_env

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.envs": MagicMock(),
                "lerobot.envs.factory": mock_factory,
            },
        ):
            env = LeIsaacEnv("bi_so101_fold_cloth")
            result = env.load()

        assert result is True
        mock_inner_env.initialize.assert_called_once()

    def test_load_import_error(self):
        env = LeIsaacEnv("so101_pick_orange")
        result = env.load()
        assert result is False
        assert env._loaded is False

    def test_load_runtime_error(self):
        mock_factory = ModuleType("lerobot.envs.factory")
        mock_factory.make_env = MagicMock(side_effect=RuntimeError("GPU not available"))

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.envs": MagicMock(),
                "lerobot.envs.factory": mock_factory,
            },
        ):
            env = LeIsaacEnv("so101_pick_orange")
            result = env.load()

        assert result is False


# ═════════════════════════════════════════════════════════════════════════════
# LeIsaacEnv core methods (reset, step, render, close)
# ═════════════════════════════════════════════════════════════════════════════


class TestLeIsaacEnvMethods:
    def _make_loaded_env(self):
        """Create a LeIsaacEnv with a mocked inner environment."""
        env = LeIsaacEnv("so101_pick_orange")
        mock_raw = MagicMock()
        mock_raw.action_space = MagicMock()
        mock_raw.action_space.shape = (6,)
        mock_raw.reset.return_value = (np.zeros(6), {})
        mock_raw.step.return_value = (np.zeros(6), 1.0, False, False, {})
        mock_raw.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_raw.joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]
        env._raw_env = mock_raw
        env._loaded = True
        return env

    def test_reset_loaded(self):
        env = self._make_loaded_env()
        env.reset()
        env._raw_env.reset.assert_called_once()

    def test_reset_auto_loads(self):
        mock_inner = MagicMock()
        mock_inner.reset.return_value = (np.zeros(6), {})
        mock_sync_vec = MagicMock()
        mock_sync_vec.envs = [MagicMock(unwrapped=mock_inner)]

        mock_factory = ModuleType("lerobot.envs.factory")
        mock_factory.make_env = MagicMock(return_value={"suite": (mock_sync_vec,)})

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.envs": MagicMock(),
                "lerobot.envs.factory": mock_factory,
            },
        ):
            env = LeIsaacEnv("so101_pick_orange")
            env.reset()
            assert env._loaded is True

    def test_reset_load_failure_raises(self):
        env = LeIsaacEnv("so101_pick_orange")
        with pytest.raises(RuntimeError, match="Failed to load"):
            env.reset()

    def test_step(self):
        env = self._make_loaded_env()
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        env.step(action)
        env._raw_env.step.assert_called_once()

    def test_render(self):
        env = self._make_loaded_env()
        frame = env.render()
        assert frame is not None
        env._raw_env.render.assert_called_once()

    def test_render_no_render_method(self):
        env = LeIsaacEnv("so101_pick_orange")
        env._raw_env = MagicMock(spec=[])
        result = env.render()
        assert result is None

    def test_close(self):
        env = self._make_loaded_env()
        env.close()
        env._raw_env.close.assert_called_once()

    def test_close_no_env(self):
        env = LeIsaacEnv("so101_pick_orange")
        env.close()  # Should not raise


# ═════════════════════════════════════════════════════════════════════════════
# get_joint_names Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGetJointNames:
    def test_from_joint_names_attr(self):
        env = LeIsaacEnv("so101_pick_orange")
        env._raw_env = MagicMock()
        env._raw_env.joint_names = ["j0", "j1", "j2"]
        assert env.get_joint_names() == ["j0", "j1", "j2"]

    def test_from_robot_joint_names_attr(self):
        env = LeIsaacEnv("so101_pick_orange")
        mock = MagicMock(spec=["robot_joint_names", "action_space"])
        mock.robot_joint_names = ["rj0", "rj1"]
        env._raw_env = mock
        assert env.get_joint_names() == ["rj0", "rj1"]

    def test_from_action_names_attr(self):
        env = LeIsaacEnv("so101_pick_orange")
        mock = MagicMock(spec=["action_names", "action_space"])
        mock.action_names = ["a0", "a1", "a2", "a3"]
        env._raw_env = mock
        assert env.get_joint_names() == ["a0", "a1", "a2", "a3"]

    def test_fallback_from_action_space(self):
        env = LeIsaacEnv("so101_pick_orange")
        mock = MagicMock(spec=["action_space"])
        mock.action_space = MagicMock()
        mock.action_space.shape = (4,)
        env._raw_env = mock
        assert env.get_joint_names() == [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
        ]

    def test_no_raw_env(self):
        env = LeIsaacEnv("so101_pick_orange")
        assert env.get_joint_names() == []

    def test_no_relevant_attrs(self):
        env = LeIsaacEnv("so101_pick_orange")
        mock = MagicMock(spec=[])
        env._raw_env = mock
        assert env.get_joint_names() == []


# ═════════════════════════════════════════════════════════════════════════════
# _obs_to_dict / _dict_to_action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestObsConversion:
    def _make_env(self, joint_names):
        env = LeIsaacEnv("so101_pick_orange")
        env._raw_env = MagicMock()
        env._raw_env.joint_names = joint_names
        return env

    def test_obs_dict_with_state(self):
        env = self._make_env(["j0", "j1"])
        result = env._obs_to_dict({"state": np.array([0.5, 0.7])})
        assert result["j0"] == pytest.approx(0.5)
        assert result["j1"] == pytest.approx(0.7)

    def test_obs_dict_with_image(self):
        env = self._make_env(["j0"])
        result = env._obs_to_dict({"cam": np.zeros((480, 640, 3), dtype=np.uint8)})
        assert "cam" in result
        assert result["cam"].shape == (480, 640, 3)

    def test_obs_dict_with_tensor(self):
        env = self._make_env(["j0", "j1"])
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = np.array([0.3, 0.4])
        result = env._obs_to_dict({"state": mock_tensor})
        assert result["j0"] == pytest.approx(0.3)
        assert result["j1"] == pytest.approx(0.4)

    def test_obs_ndarray(self):
        env = self._make_env(["j0", "j1", "j2"])
        result = env._obs_to_dict(np.array([0.1, 0.2, 0.3]))
        assert result["j0"] == pytest.approx(0.1)
        assert result["j1"] == pytest.approx(0.2)
        assert result["j2"] == pytest.approx(0.3)

    def test_obs_other_type(self):
        env = self._make_env(["j0"])
        assert env._obs_to_dict("raw_string") == {"observation": "raw_string"}

    def test_obs_dict_non_array_passthrough(self):
        env = self._make_env(["j0"])
        result = env._obs_to_dict({"flag": True, "count": 42})
        assert result["flag"] is True
        assert result["count"] == 42

    def test_dict_to_action(self):
        env = self._make_env(["j0", "j1", "j2"])
        arr = env._dict_to_action({"j0": 0.1, "j1": 0.5, "j2": 0.9})
        np.testing.assert_allclose(arr, [0.1, 0.5, 0.9])

    def test_dict_to_action_missing_keys(self):
        env = self._make_env(["j0", "j1", "j2"])
        arr = env._dict_to_action({"j0": 0.5})
        np.testing.assert_allclose(arr, [0.5, 0.0, 0.0])

    def test_dict_to_action_empty(self):
        env = self._make_env(["j0", "j1"])
        np.testing.assert_allclose(env._dict_to_action({}), [0.0, 0.0])


# ═════════════════════════════════════════════════════════════════════════════
# LeIsaacEnv.rollout Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRollout:
    def _make_rollout_env(self, terminated_at=None, truncated_at=None):
        env = LeIsaacEnv("so101_pick_orange")
        mock_raw = MagicMock()
        mock_raw.joint_names = ["j0", "j1"]
        mock_raw.action_space = MagicMock()
        mock_raw.action_space.shape = (2,)

        call_count = [0]

        def step_fn(action):
            call_count[0] += 1
            done = terminated_at is not None and call_count[0] >= terminated_at
            trunc = truncated_at is not None and call_count[0] >= truncated_at
            info = {"is_success": True} if done else {}
            return np.zeros(2), 1.0, done, trunc, info

        mock_raw.reset.return_value = (np.zeros(2), {})
        mock_raw.step.side_effect = step_fn
        env._raw_env = mock_raw
        env._loaded = True
        return env

    def test_rollout_basic(self):
        env = self._make_rollout_env(terminated_at=3)
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()

        async def fake_get_actions(obs, instr, **kw):
            return [{"j0": 0.1, "j1": 0.2}]

        mock_policy.get_actions = fake_get_actions
        result = env.rollout(mock_policy, instruction="test", n_episodes=1, max_steps=10)

        assert result.n_episodes == 1
        assert result.n_successes == 1
        assert result.success_rate == 1.0
        assert result.avg_steps == 3.0
        mock_policy.set_robot_state_keys.assert_called_once_with(["j0", "j1"])

    def test_rollout_truncated(self):
        env = self._make_rollout_env(truncated_at=2)
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()

        async def fake_get_actions(obs, instr, **kw):
            return [{"j0": 0.0, "j1": 0.0}]

        mock_policy.get_actions = fake_get_actions
        result = env.rollout(mock_policy, n_episodes=1, max_steps=100)

        assert result.n_successes == 0
        assert result.avg_steps == 2.0

    def test_rollout_empty_actions(self):
        env = self._make_rollout_env(truncated_at=1)
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()

        async def fake_get_actions(obs, instr, **kw):
            return []

        mock_policy.get_actions = fake_get_actions
        result = env.rollout(mock_policy, n_episodes=1, max_steps=5)
        assert result.n_episodes == 1

    def test_rollout_default_instruction(self):
        env = self._make_rollout_env(terminated_at=1)
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()
        captured = []

        async def fake_get_actions(obs, instr, **kw):
            captured.append(instr)
            return [{"j0": 0.0, "j1": 0.0}]

        mock_policy.get_actions = fake_get_actions
        env.rollout(mock_policy, instruction="", n_episodes=1, max_steps=5)
        assert captured[0] == LEISAAC_TASKS["so101_pick_orange"]["default_instruction"]

    def test_rollout_multiple_episodes(self):
        env = self._make_rollout_env(terminated_at=2)
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()

        async def fake_get_actions(obs, instr, **kw):
            return [{"j0": 0.1, "j1": 0.2}]

        mock_policy.get_actions = fake_get_actions
        result = env.rollout(mock_policy, n_episodes=3, max_steps=10)

        assert result.n_episodes == 3
        assert len(result.episodes) == 3
        assert result.total_time > 0


# ═════════════════════════════════════════════════════════════════════════════
# LeIsaacEnv.record_video Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRecordVideo:
    def _make_record_env(self, steps_to_terminate=3):
        env = LeIsaacEnv("so101_pick_orange")
        mock_raw = MagicMock()
        mock_raw.joint_names = ["j0"]
        mock_raw.action_space = MagicMock()
        mock_raw.action_space.shape = (1,)
        mock_raw.reset.return_value = (np.zeros(1), {})
        mock_raw.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        step_count = [0]

        def step_fn(action):
            step_count[0] += 1
            done = step_count[0] >= steps_to_terminate
            return np.zeros(1), 0.5, done, False, {"is_success": done}

        mock_raw.step.side_effect = step_fn
        env._raw_env = mock_raw
        env._loaded = True
        return env

    def test_record_imageio_fallback(self, tmp_path):
        """Test recording via imageio fallback (strands_robots.video import fails)."""
        env = self._make_record_env(steps_to_terminate=3)
        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()

        async def fake_get_actions(obs, instr, **kw):
            return [{"j0": 0.1}]

        mock_policy.get_actions = fake_get_actions
        output = str(tmp_path / "test.mp4")

        # Force video import to fail → imageio fallback
        mock_imageio = MagicMock()
        mock_writer = MagicMock()
        mock_imageio.get_writer.return_value = mock_writer

        # Temporarily make strands_robots.video import raise ImportError
        saved = sys.modules.get("strands_robots.video")
        sys.modules["strands_robots.video"] = None  # causes ImportError on `from X import Y`

        try:
            with patch.dict("sys.modules", {"imageio": mock_imageio}):
                info = env.record_video(mock_policy, output, max_steps=10)
        finally:
            if saved is not None:
                sys.modules["strands_robots.video"] = saved
            else:
                sys.modules.pop("strands_robots.video", None)

        assert info["frames"] >= 1
        assert info["steps"] >= 1
        assert info["success"] is True
        mock_imageio.get_writer.assert_called_once()
        mock_writer.close.assert_called_once()

    def test_record_no_frames(self, tmp_path):
        env = LeIsaacEnv("so101_pick_orange")
        mock_raw = MagicMock()
        mock_raw.joint_names = ["j0"]
        mock_raw.action_space = MagicMock()
        mock_raw.action_space.shape = (1,)
        mock_raw.reset.return_value = (np.zeros(1), {})
        mock_raw.render.return_value = None
        mock_raw.step.return_value = (np.zeros(1), 0.0, True, False, {})
        env._raw_env = mock_raw
        env._loaded = True

        mock_policy = MagicMock()
        mock_policy.set_robot_state_keys = MagicMock()

        async def fake_get_actions(obs, instr, **kw):
            return [{"j0": 0.0}]

        mock_policy.get_actions = fake_get_actions
        info = env.record_video(mock_policy, str(tmp_path / "empty.mp4"), max_steps=5)
        assert info["frames"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# leisaac_env AgentTool Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLeisaacEnvTool:
    @pytest.fixture(autouse=True)
    def check_tool(self):
        if leisaac_env is None:
            pytest.skip("leisaac_env tool not available (strands not installed)")

    def test_list_action(self):
        result = leisaac_env(action="list")
        assert result["status"] == "success"
        assert "so101_pick_orange" in result["content"][0]["text"]

    def test_info_known_task(self):
        result = leisaac_env(action="info", task="so101_pick_orange")
        assert result["status"] == "success"
        assert "Pick" in result["content"][0]["text"]

    def test_info_unknown_task(self):
        result = leisaac_env(action="info", task="nonexistent_task")
        assert result["status"] == "success"
        assert "Unknown task" in result["content"][0]["text"]

    def test_load_fails_gracefully(self):
        result = leisaac_env(action="load", task="so101_pick_orange")
        assert result["status"] == "error"
        assert "Failed" in result["content"][0]["text"]

    def test_unknown_action(self):
        result = leisaac_env(action="invalid_action")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_rollout_load_fails(self):
        result = leisaac_env(action="rollout", task="so101_pick_orange")
        assert result["status"] == "error"

    def test_record_load_fails(self):
        result = leisaac_env(action="record", task="so101_pick_orange")
        assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# create_leisaac_env Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCreateLeisaacEnv:
    def test_creates_env(self):
        env = create_leisaac_env("so101_lift_cube", auto_load=False)
        assert isinstance(env, LeIsaacEnv)
        assert env.task_name == "so101_lift_cube"
        assert env._loaded is False

    def test_auto_load_false(self):
        env = create_leisaac_env(auto_load=False)
        assert env._loaded is False

    def test_auto_load_true_fails_gracefully(self):
        env = create_leisaac_env(auto_load=True)
        assert isinstance(env, LeIsaacEnv)
        assert env._loaded is False

    def test_custom_n_envs(self):
        env = create_leisaac_env(n_envs=8, auto_load=False)
        assert env.n_envs == 8


# ═════════════════════════════════════════════════════════════════════════════
# Module Exports Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_all_exports(self):
        from strands_robots import leisaac

        assert "LeIsaacEnv" in leisaac.__all__
        assert "create_leisaac_env" in leisaac.__all__
        assert "list_tasks" in leisaac.__all__
        assert "RolloutResult" in leisaac.__all__
        assert "LEISAAC_TASKS" in leisaac.__all__

    def test_leisaac_tasks_dict(self):
        assert isinstance(LEISAAC_TASKS, dict)
        assert len(LEISAAC_TASKS) >= 4
        for name, info in LEISAAC_TASKS.items():
            assert "env_script" in info
            assert "env_id" in info
            assert "description" in info
