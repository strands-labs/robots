#!/usr/bin/env python3
"""Tests for strands_robots/tools/isaac_sim.py — Isaac Sim GPU-Accelerated Simulation Tool.

All Isaac Sim / IsaacLab / GPU dependencies are mocked. CPU-only CI safe.
Tests cover the tool's action routing, parameter parsing, error handling,
robot/task registries, and integration with the mocked IsaacSimBackend.

Uses `isaac_sim._tool_func(...)` to call the raw function directly.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock strands if not installed so tests can run without the full SDK
try:
    import strands

    # Verify it's the real strands, not a mock from another test file
    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types
    from unittest.mock import MagicMock

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f  # @tool decorator becomes identity
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

_requires_strands = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def call():
    """Return a callable that invokes the isaac_sim tool's raw function."""
    from strands_robots.tools.isaac_sim import isaac_sim

    return getattr(isaac_sim, "_tool_func", isaac_sim)


@pytest.fixture(autouse=True)
def clean_backend():
    """Reset global backend before and after each test."""
    import importlib

    mod = importlib.import_module("strands_robots.tools.isaac_sim")
    mod._backend = None
    mod._backend_config = None
    mod._isaac_env = None
    yield
    mod._backend = None
    mod._backend_config = None
    mod._isaac_env = None


def _make_mock_backend():
    """Create a mock IsaacSimBackend with common method stubs."""
    backend = MagicMock()
    backend.config = MagicMock()
    backend.config.num_envs = 1
    backend.config.device = "cuda:0"
    backend.create_world.return_value = {
        "status": "success",
        "content": [{"text": "🌍 Isaac Sim world created"}],
    }
    backend.add_robot.return_value = {
        "status": "success",
        "content": [{"text": "🤖 Robot added"}],
    }
    backend.step.return_value = {
        "status": "success",
        "observations": {"joint_pos": [0.1, 0.2, 0.3]},
    }
    backend.get_observation.return_value = {
        "joint_pos": [0.1, 0.2, 0.3],
        "joint_vel": [0.0, 0.0, 0.0],
    }
    backend.render.return_value = {
        "status": "success",
        "content": [{"text": "🎥 Rendered image"}],
    }
    backend.reset.return_value = {
        "status": "success",
        "content": [{"text": "🔄 Reset complete"}],
    }
    backend.run_policy.return_value = {
        "status": "success",
        "content": [{"text": "✅ Policy executed"}],
    }
    backend.destroy.return_value = {
        "status": "success",
        "content": [{"text": "💥 Backend destroyed"}],
    }
    backend._robot = MagicMock()
    return backend


# ─────────────────────────────────────────────────────────────────────
# Import & Smoke Tests
# ─────────────────────────────────────────────────────────────────────


class TestIsaacSimImport:

    def test_clean_import(self):
        from strands_robots.tools.isaac_sim import isaac_sim

        assert isaac_sim is not None

    @_requires_strands
    def test_tool_attributes(self):
        from strands_robots.tools.isaac_sim import isaac_sim

        assert hasattr(isaac_sim, "_tool_func")
        assert hasattr(isaac_sim, "tool_name")
        assert isaac_sim.tool_name == "isaac_sim"

    def test_robot_registry_exists(self):
        from strands_robots.tools.isaac_sim import _ISAAC_ROBOTS

        assert isinstance(_ISAAC_ROBOTS, dict)
        assert len(_ISAAC_ROBOTS) == 17  # 17 built-in robots

    def test_task_registry_exists(self):
        from strands_robots.tools.isaac_sim import _ISAAC_TASKS

        assert isinstance(_ISAAC_TASKS, dict)
        assert len(_ISAAC_TASKS) > 0

    def test_backend_globals_exist(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.isaac_sim")
        assert hasattr(mod, "_backend")
        assert hasattr(mod, "_backend_config")
        assert hasattr(mod, "_isaac_env")


# ─────────────────────────────────────────────────────────────────────
# Robot Registry Integrity
# ─────────────────────────────────────────────────────────────────────


class TestRobotRegistry:

    def test_all_robots_have_required_keys(self):
        from strands_robots.tools.isaac_sim import _ISAAC_ROBOTS

        for name, info in _ISAAC_ROBOTS.items():
            assert "type" in info, f"Robot '{name}' missing 'type'"
            assert "source" in info, f"Robot '{name}' missing 'source'"
            assert "joints" in info, f"Robot '{name}' missing 'joints'"
            assert isinstance(info["joints"], int)
            assert info["joints"] > 0

    def test_expected_robot_types(self):
        from strands_robots.tools.isaac_sim import _ISAAC_ROBOTS

        types = set(info["type"] for info in _ISAAC_ROBOTS.values())
        assert "quadruped" in types
        assert "humanoid" in types
        assert "manipulator" in types
        assert "hand" in types
        assert "classic" in types

    def test_expected_robots_present(self):
        from strands_robots.tools.isaac_sim import _ISAAC_ROBOTS

        expected = ["unitree_go2", "anymal_c", "panda", "ur5e", "shadow_hand", "cartpole"]
        for name in expected:
            assert name in _ISAAC_ROBOTS, f"Expected robot '{name}' not in registry"


# ─────────────────────────────────────────────────────────────────────
# Task Registry Integrity
# ─────────────────────────────────────────────────────────────────────


class TestTaskRegistry:

    def test_tasks_are_strings(self):
        from strands_robots.tools.isaac_sim import _ISAAC_TASKS

        for key, val in _ISAAC_TASKS.items():
            assert isinstance(key, str)
            assert isinstance(val, str)
            assert val.startswith("Isaac-")

    def test_expected_tasks(self):
        from strands_robots.tools.isaac_sim import _ISAAC_TASKS

        expected = ["cartpole", "anymal_c_flat", "humanoid", "franka_cabinet"]
        for task in expected:
            assert task in _ISAAC_TASKS, f"Expected task '{task}' not in registry"


# ─────────────────────────────────────────────────────────────────────
# Unknown / Invalid Action
# ─────────────────────────────────────────────────────────────────────


class TestUnknownAction:

    def test_unknown_action(self, call):
        result = call(action="bogus")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        assert "bogus" in result["content"][0]["text"]

    def test_unknown_action_lists_valid(self, call):
        result = call(action="fake_action")
        text = result["content"][0]["text"]
        for valid in ["create_world", "add_robot", "step", "destroy", "benchmark", "list_robots", "list_tasks"]:
            assert valid in text


# ─────────────────────────────────────────────────────────────────────
# create_world Action
# ─────────────────────────────────────────────────────────────────────


class TestCreateWorld:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_create_world_basic(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="create_world", num_envs=4096, device="cuda:0")
        assert result["status"] == "success"
        mock_destroy.assert_called_once()
        mock_backend.create_world.assert_called_once()

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_create_world_with_gravity(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="create_world", gravity="[0,0,-3.7]")
        mock_backend.create_world.assert_called_once_with(gravity=[0, 0, -3.7])


# ─────────────────────────────────────────────────────────────────────
# add_robot Action
# ─────────────────────────────────────────────────────────────────────


class TestAddRobot:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_robot_by_type(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="add_robot", robot_type="unitree_go2", position="[0,0,0.5]")
        assert result["status"] == "success"
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["data_config"] == "unitree_go2"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_robot_by_usd(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot", name="my_robot", usd_path="/tmp/robot.usd")
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["usd_path"] == "/tmp/robot.usd"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_robot_default_name(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot")
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["name"] == "robot"


# ─────────────────────────────────────────────────────────────────────
# add_object Action
# ─────────────────────────────────────────────────────────────────────


class TestAddObject:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_object_basic(self, mock_get, call):
        mock_backend = _make_mock_backend()
        del mock_backend.add_object  # Simulate backend without add_object
        mock_get.return_value = mock_backend

        result = call(
            action="add_object", object_type="box", name="cube", position="[0,0,1]", object_size="[0.2,0.2,0.2]"
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "cube" in text or "box" in text

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_object_with_backend_support(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.add_object.return_value = {"status": "success", "content": [{"text": "Box spawned"}]}
        mock_get.return_value = mock_backend

        result = call(action="add_object", object_type="sphere")
        assert result["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_object_defaults(self, mock_get, call):
        mock_backend = _make_mock_backend()
        del mock_backend.add_object
        mock_get.return_value = mock_backend

        result = call(action="add_object")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "box" in text.lower()  # Default type


# ─────────────────────────────────────────────────────────────────────
# step Action
# ─────────────────────────────────────────────────────────────────────


class TestStep:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_step_single(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="step", steps=1)
        assert result["status"] == "success"
        assert mock_backend.step.call_count == 1

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_step_multiple(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="step", steps=50)
        assert result["status"] == "success"
        assert mock_backend.step.call_count == 50

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_step_with_observations(self, mock_get, call):
        import numpy as np

        mock_backend = _make_mock_backend()
        mock_backend.step.return_value = {
            "status": "success",
            "observations": {"joint_pos": np.array([0.1, 0.2, 0.3])},
        }
        mock_get.return_value = mock_backend

        result = call(action="step", steps=5)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "joint_pos" in text

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_step_no_observations(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.step.return_value = {"status": "success"}
        mock_get.return_value = mock_backend

        result = call(action="step", steps=1)
        assert result["status"] == "success"


# ─────────────────────────────────────────────────────────────────────
# observe Action
# ─────────────────────────────────────────────────────────────────────


class TestObserve:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_observe_success(self, mock_get, call):
        import numpy as np

        mock_backend = _make_mock_backend()
        mock_backend.get_observation.return_value = {
            "joint_pos": np.array([0.1, 0.2]),
            "joint_vel": np.array([0.0, 0.0]),
        }
        mock_get.return_value = mock_backend

        result = call(action="observe")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Observations" in text
        assert "joint_pos" in text

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_observe_empty(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.get_observation.return_value = {}
        mock_get.return_value = mock_backend

        result = call(action="observe")
        assert result["status"] == "success"
        assert "No observations" in result["content"][0]["text"]

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_observe_with_robot_name(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="observe", name="my_robot")
        mock_backend.get_observation.assert_called_once_with(robot_name="my_robot")


# ─────────────────────────────────────────────────────────────────────
# render Action
# ─────────────────────────────────────────────────────────────────────


class TestRender:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_render(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="render", camera_name="front_cam", width=1280, height=720)
        mock_backend.render.assert_called_once_with(camera_name="front_cam", width=1280, height=720)


# ─────────────────────────────────────────────────────────────────────
# reset Action
# ─────────────────────────────────────────────────────────────────────


class TestReset:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_reset_with_method(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="reset")
        mock_backend.reset.assert_called_once()

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_reset_fallback_recreate(self, mock_destroy, mock_get, call):
        """If backend has no reset method, should destroy and recreate."""
        import importlib

        mod = importlib.import_module("strands_robots.tools.isaac_sim")

        mock_backend = _make_mock_backend()
        del mock_backend.reset  # Remove reset method
        mod._backend_config = {"num_envs": 1, "device": "cuda:0"}

        # First call returns backend without reset, second call returns new backend
        new_backend = _make_mock_backend()
        mock_get.side_effect = [mock_backend, new_backend]

        result = call(action="reset")
        assert result["status"] == "success"
        assert "reset" in result["content"][0]["text"].lower()


# ─────────────────────────────────────────────────────────────────────
# run_policy Action
# ─────────────────────────────────────────────────────────────────────


class TestRunPolicy:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_run_policy(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="run_policy", name="go2", policy_provider="mock", instruction="walk", duration=5.0)
        mock_backend.run_policy.assert_called_once_with(
            robot_name="go2",
            policy_provider="mock",
            instruction="walk",
            duration=5.0,
        )

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_run_policy_default_name(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="run_policy")
        call_args = mock_backend.run_policy.call_args
        assert call_args[1]["robot_name"] == "robot"


# ─────────────────────────────────────────────────────────────────────
# set_joint_pos Action
# ─────────────────────────────────────────────────────────────────────


class TestSetJointPos:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_set_joint_pos_with_method(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.set_joint_positions.return_value = {"status": "success", "content": [{"text": "Joints set"}]}
        mock_get.return_value = mock_backend

        result = call(action="set_joint_pos", joint_positions="[0.1, 0.2, 0.3]")
        assert result["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_set_joint_pos_no_positions(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend
        result = call(action="set_joint_pos")
        assert result["status"] == "error"
        assert "joint_positions required" in result["content"][0]["text"]

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_set_joint_pos_no_robot(self, mock_get, call):
        mock_backend = _make_mock_backend()
        del mock_backend.set_joint_positions
        mock_backend._robot = None
        mock_get.return_value = mock_backend

        result = call(action="set_joint_pos", joint_positions="[0.1]")
        assert result["status"] == "error"
        assert "No robot" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# get_contacts Action
# ─────────────────────────────────────────────────────────────────────


class TestGetContacts:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_get_contacts_with_method(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.get_contact_forces.return_value = {"status": "success", "content": [{"text": "Contact forces"}]}
        mock_get.return_value = mock_backend

        result = call(action="get_contacts")
        assert result["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_get_contacts_fallback(self, mock_get, call):
        mock_backend = _make_mock_backend()
        del mock_backend.get_contact_forces
        mock_get.return_value = mock_backend

        result = call(action="get_contacts")
        assert result["status"] == "success"
        assert "requires Isaac Sim" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# save_state / load_state Actions
# ─────────────────────────────────────────────────────────────────────


class TestSaveLoadState:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_save_state_success(self, mock_get, call, tmp_path):
        import importlib

        import numpy as np

        mod = importlib.import_module("strands_robots.tools.isaac_sim")
        mock_backend = _make_mock_backend()
        mock_backend.get_observation.return_value = {
            "joint_pos": np.array([0.1, 0.2, 0.3]),
        }
        mod._backend = mock_backend
        mod._backend_config = {"num_envs": 1}

        outpath = str(tmp_path / "state.json")
        result = call(action="save_state", output_path=outpath)
        assert result["status"] == "success"
        assert os.path.exists(outpath)

        # Verify JSON content
        with open(outpath) as f:
            data = json.load(f)
        assert "observations" in data
        assert "config" in data

    def test_save_state_no_backend(self, call):
        result = call(action="save_state")
        assert result["status"] == "error"
        assert "No active simulation" in result["content"][0]["text"]

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_load_state_success(self, mock_destroy, mock_get, call, tmp_path):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        # Create state file
        state_path = str(tmp_path / "state.json")
        state = {
            "observations": {"joint_pos": [0.1, 0.2]},
            "config": {"num_envs": 4, "device": "cuda:0"},
        }
        with open(state_path, "w") as f:
            json.dump(state, f)

        result = call(action="load_state", input_path=state_path)
        assert result["status"] == "success"
        assert "loaded" in result["content"][0]["text"].lower()

    def test_load_state_missing_file(self, call):
        result = call(action="load_state", input_path="/nonexistent/state.json")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"].lower()


# ─────────────────────────────────────────────────────────────────────
# destroy Action
# ─────────────────────────────────────────────────────────────────────


class TestDestroy:

    def test_destroy_no_backend(self, call):
        result = call(action="destroy")
        assert result["status"] == "success"
        assert "No active backend" in result["content"][0]["text"]

    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_destroy_active_backend(self, mock_destroy, call):
        import importlib

        mod = importlib.import_module("strands_robots.tools.isaac_sim")
        mock_backend = _make_mock_backend()
        mod._backend = mock_backend

        result = call(action="destroy")
        assert result["status"] == "success"
        mock_backend.destroy.assert_called_once()
        mock_destroy.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# list_robots Action
# ─────────────────────────────────────────────────────────────────────


class TestListRobots:

    def test_list_robots(self, call):
        result = call(action="list_robots")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Isaac Sim Available Robots" in text
        assert "17 robots" in text
        assert "unitree_go2" in text
        assert "panda" in text
        assert "Quadruped" in text or "quadruped" in text.lower()

    def test_list_robots_all_types_shown(self, call):
        result = call(action="list_robots")
        text = result["content"][0]["text"]
        for robot_type in ["Quadruped", "Humanoid", "Manipulator", "Hand", "Classic"]:
            assert robot_type in text, f"Robot type '{robot_type}' not in list output"


# ─────────────────────────────────────────────────────────────────────
# list_tasks Action
# ─────────────────────────────────────────────────────────────────────


class TestListTasks:

    def test_list_tasks(self, call):
        result = call(action="list_tasks")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Isaac Lab Tasks" in text
        assert "cartpole" in text
        assert "Isaac-CartPole-v0" in text

    def test_list_tasks_has_usage_hint(self, call):
        result = call(action="list_tasks")
        text = result["content"][0]["text"]
        assert "create_env" in text


# ─────────────────────────────────────────────────────────────────────
# create_env Action
# ─────────────────────────────────────────────────────────────────────


class TestCreateEnv:

    @patch("strands_robots.isaac.isaac_lab_env.create_isaac_env", create=True)
    def test_create_env(self, mock_create, call):
        mock_env = MagicMock()
        mock_create.return_value = mock_env

        with patch.dict(
            "sys.modules",
            {
                "strands_robots.isaac.isaac_lab_env": MagicMock(create_isaac_env=mock_create),
            },
        ):
            result = call(action="create_env", task="cartpole", num_envs=1024)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "cartpole" in text
        assert "1024" in text

    @patch(
        "strands_robots.isaac.isaac_lab_env.create_isaac_env",
        side_effect=ImportError("Isaac Lab not installed"),
        create=True,
    )
    def test_create_env_no_isaac_lab(self, mock_create, call):
        result = call(action="create_env", task="cartpole")
        assert result["status"] == "error"
        assert "Isaac Lab" in result["content"][0]["text"] or "Error" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# train Action
# ─────────────────────────────────────────────────────────────────────


class TestTrain:

    def test_train_action(self, call):
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"status": "success", "content": [{"text": "Training complete"}]}
        mock_config = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "strands_robots.isaac.isaac_lab_trainer": MagicMock(
                    IsaacLabTrainer=MagicMock(return_value=mock_trainer),
                    IsaacLabTrainerConfig=mock_config,
                ),
            },
        ):
            result = call(action="train", task="cartpole", rl_framework="rsl_rl", max_iterations=100)
        assert result["status"] == "success"


# ─────────────────────────────────────────────────────────────────────
# export_policy Action
# ─────────────────────────────────────────────────────────────────────


class TestExportPolicy:

    def test_export_no_input(self, call):
        result = call(action="export_policy")
        assert result["status"] == "error"
        assert "input_path required" in result["content"][0]["text"]

    def test_export_with_input(self, call):
        result = call(action="export_policy", input_path="/tmp/model.pt")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "/tmp/model.pt" in text
        assert "/tmp/model.onnx" in text

    def test_export_custom_output(self, call):
        result = call(action="export_policy", input_path="/tmp/model.pt", output_path="/tmp/custom.onnx")
        text = result["content"][0]["text"]
        assert "/tmp/custom.onnx" in text


# ─────────────────────────────────────────────────────────────────────
# benchmark Action
# ─────────────────────────────────────────────────────────────────────


class TestBenchmark:

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_benchmark(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.create_world.return_value = {"status": "success"}
        mock_get.return_value = mock_backend

        result = call(action="benchmark", benchmark_envs=512, benchmark_steps=50)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Benchmark" in text
        assert "512" in text
        assert "env-steps/s" in text

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_benchmark_create_world_error(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.create_world.return_value = {"status": "error", "content": [{"text": "GPU not available"}]}
        mock_get.return_value = mock_backend

        result = call(action="benchmark")
        assert result["status"] == "error"


# ─────────────────────────────────────────────────────────────────────
# convert_asset Action
# ─────────────────────────────────────────────────────────────────────


class TestConvertAsset:

    def test_convert_no_input(self, call):
        result = call(action="convert_asset")
        assert result["status"] == "error"
        assert "input_path required" in result["content"][0]["text"]

    def test_convert_unknown_format(self, call):
        result = call(action="convert_asset", input_path="/tmp/model.abc")
        assert result["status"] == "error"
        assert "Unknown format" in result["content"][0]["text"]

    def test_convert_mjcf_to_usd(self, call):
        mock_convert = MagicMock(return_value={"status": "success", "content": [{"text": "Converted"}]})
        with patch.dict(
            "sys.modules",
            {
                "strands_robots.isaac.asset_converter": MagicMock(
                    convert_mjcf_to_usd=mock_convert,
                    convert_usd_to_mjcf=MagicMock(),
                ),
            },
        ):
            result = call(action="convert_asset", input_path="/tmp/robot.xml", output_path="/tmp/robot.usd")
        assert result["status"] == "success"

    def test_convert_usd_to_mjcf(self, call):
        mock_convert = MagicMock(return_value={"status": "success", "content": [{"text": "Converted"}]})
        with patch.dict(
            "sys.modules",
            {
                "strands_robots.isaac.asset_converter": MagicMock(
                    convert_mjcf_to_usd=MagicMock(),
                    convert_usd_to_mjcf=mock_convert,
                ),
            },
        ):
            result = call(action="convert_asset", input_path="/tmp/robot.usd")
        assert result["status"] == "success"


# ─────────────────────────────────────────────────────────────────────
# list_extensions Action
# ─────────────────────────────────────────────────────────────────────


class TestListExtensions:

    def test_list_extensions(self, call):
        result = call(action="list_extensions")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Isaac Sim Extensions" in text
        assert "Core" in text
        assert "Sensors" in text
        assert "isaacsim.core.api" in text


# ─────────────────────────────────────────────────────────────────────
# Exception Handling
# ─────────────────────────────────────────────────────────────────────


class TestExceptionHandling:

    @patch("strands_robots.tools.isaac_sim._get_backend", side_effect=ImportError("Isaac Sim not installed"))
    def test_import_error_caught(self, mock_get, call):
        result = call(action="step")
        assert result["status"] == "error"
        assert "Isaac Sim" in result["content"][0]["text"]

    @patch("strands_robots.tools.isaac_sim._get_backend", side_effect=RuntimeError("CUDA error"))
    def test_runtime_error_caught(self, mock_get, call):
        result = call(action="create_world")
        assert result["status"] == "error"
        assert "CUDA" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# Backend Management
# ─────────────────────────────────────────────────────────────────────


class TestBackendManagement:

    def test_destroy_backend_clears_globals(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.isaac_sim")
        mock_backend = MagicMock()
        mock_env = MagicMock()
        mod._backend = mock_backend
        mod._backend_config = {"num_envs": 1}
        mod._isaac_env = mock_env

        mod._destroy_backend()
        assert mod._backend is None
        assert mod._backend_config is None
        assert mod._isaac_env is None
        mock_backend.destroy.assert_called_once()
        mock_env.close.assert_called_once()

    def test_destroy_backend_noop_when_none(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.isaac_sim")
        mod._backend = None
        mod._backend_config = None
        mod._isaac_env = None

        mod._destroy_backend()  # Should not crash

    def test_get_backend_reuses_existing(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.isaac_sim")
        existing = MagicMock()
        mod._backend = existing

        result = mod._get_backend()
        assert result is existing
