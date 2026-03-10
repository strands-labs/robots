#!/usr/bin/env python3
"""Tests for strands_robots/tools/newton_sim.py — Newton GPU-Accelerated Physics Simulation Tool.

All Newton/Warp/GPU dependencies are mocked. CPU-only CI safe.
Tests cover the tool's action routing, parameter parsing, error handling,
and integration with the mocked NewtonBackend.

Uses `newton_sim._tool_func(...)` to call the raw function directly,
bypassing the `@tool` decorator's `DecoratedFunctionTool` wrapper.
"""

import json
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
    """Return a callable that invokes the newton_sim tool's raw function."""
    from strands_robots.tools.newton_sim import newton_sim

    return getattr(newton_sim, "_tool_func", newton_sim)


@pytest.fixture(autouse=True)
def clean_backend():
    """Reset global backend before and after each test."""
    import importlib

    mod = importlib.import_module("strands_robots.tools.newton_sim")
    mod._backend = None
    mod._backend_config = None
    yield
    mod._backend = None
    mod._backend_config = None


def _make_mock_backend():
    """Create a mock NewtonBackend with common method stubs."""
    backend = MagicMock()
    backend.create_world.return_value = {"success": True, "message": "World created"}
    backend.add_robot.return_value = {
        "success": True,
        "robot_info": {
            "name": "test_robot",
            "format": "urdf",
            "model_path": "/tmp/robot.urdf",
            "num_joints": 12,
            "num_bodies": 13,
            "position": (0, 0, 0),
        },
    }
    backend.add_cloth.return_value = {"success": True, "message": "Cloth added"}
    backend.add_cable.return_value = {"success": True, "message": "Cable added"}
    backend.add_particles.return_value = {"success": True, "message": "Particles added"}
    backend.replicate.return_value = {
        "success": True,
        "env_info": {
            "num_envs": 4096,
            "bodies_total": 53248,
            "joints_total": 49152,
            "solver": "mujoco",
            "device": "cuda:0",
        },
    }
    backend.step.return_value = {
        "success": True,
        "sim_time": 0.016,
        "step_count": 1,
    }
    backend.get_observation.return_value = {
        "success": True,
        "sim_time": 0.032,
        "observations": {
            "robot": {
                "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            },
        },
    }
    backend.reset.return_value = {"success": True, "message": "Reset complete"}
    backend.get_state.return_value = {
        "config": {"solver": "mujoco", "device": "cuda:0", "num_envs": 1},
        "step_count": 100,
        "sim_time": 1.6,
        "robots": {"robot": {}},
        "cloths": {},
        "sensors": [],
        "joints_per_world": 12,
        "bodies_per_world": 13,
    }
    backend.run_policy.return_value = {
        "success": True,
        "steps_executed": 600,
        "wall_time": 10.0,
        "realtime_factor": 1.0,
        "errors": [],
    }
    backend.run_diffsim.return_value = {
        "success": True,
        "iterations": 100,
        "final_loss": 0.001234,
        "optimize_param": "initial_velocity",
    }
    backend.add_sensor.return_value = {"success": True, "message": "Sensor 'contact_0' added"}
    backend.read_sensor.return_value = {"success": True, "data": [0.0, 1.5, 3.2]}
    backend.solve_ik.return_value = {"success": True, "iterations": 15, "error": 0.0012}
    backend.enable_dual_solver.return_value = {"success": True, "message": "Dual solver enabled"}
    backend.destroy.return_value = {"message": "Backend destroyed"}
    return backend


# ─────────────────────────────────────────────────────────────────────
# Import & Smoke Tests
# ─────────────────────────────────────────────────────────────────────


class TestNewtonSimImport:
    """Validate imports and tool structure."""

    def test_clean_import(self):
        from strands_robots.tools.newton_sim import newton_sim

        assert newton_sim is not None

    @_requires_strands
    def test_tool_attributes(self):
        from strands_robots.tools.newton_sim import newton_sim

        assert hasattr(newton_sim, "_tool_func")
        assert hasattr(newton_sim, "tool_name")
        assert newton_sim.tool_name == "newton_sim"

    def test_backend_globals_exist(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.newton_sim")
        assert hasattr(mod, "_backend")
        assert hasattr(mod, "_backend_config")
        assert hasattr(mod, "_get_backend")
        assert hasattr(mod, "_destroy_backend")


# ─────────────────────────────────────────────────────────────────────
# Unknown / Invalid Action
# ─────────────────────────────────────────────────────────────────────


class TestUnknownAction:

    def test_unknown_action_returns_error(self, call):
        result = call(action="bogus_action")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        assert "bogus_action" in result["content"][0]["text"]

    def test_unknown_action_lists_valid_actions(self, call):
        result = call(action="definitely_fake")
        text = result["content"][0]["text"]
        for valid in ["create_world", "add_robot", "step", "destroy", "benchmark"]:
            assert valid in text


# ─────────────────────────────────────────────────────────────────────
# create_world Action
# ─────────────────────────────────────────────────────────────────────


class TestCreateWorld:

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_create_world_basic(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="create_world", solver="mujoco", device="cuda:0")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Newton world created" in text
        assert "mujoco" in text
        mock_destroy.assert_called_once()

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_create_world_differentiable(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="create_world", enable_differentiable=True)
        text = result["content"][0]["text"]
        assert "Differentiable" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_create_world_no_ground(self, mock_destroy, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="create_world", ground_plane=False)
        assert result["status"] == "success"
        mock_backend.create_world.assert_called_once_with(ground_plane=False)

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_create_world_not_differentiable(self, mock_destroy, mock_get, call):
        """Non-differentiable mode should NOT show the differentiable line."""
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="create_world", enable_differentiable=False)
        text = result["content"][0]["text"]
        assert "Differentiable: ON" not in text


# ─────────────────────────────────────────────────────────────────────
# add_robot Action
# ─────────────────────────────────────────────────────────────────────


class TestAddRobot:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot_success(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="add_robot", name="quadruped", urdf_path="/tmp/robot.urdf", position="1,2,3")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "test_robot" in text
        assert "Joints: 12" in text
        assert "Bodies: 13" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.add_robot.return_value = {"success": False, "message": "URDF not found"}
        mock_get.return_value = mock_backend

        result = call(action="add_robot", name="bad_robot")
        assert result["status"] == "error"
        assert "URDF not found" in result["content"][0]["text"]

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot_default_name(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot", urdf_path="/tmp/test.urdf")
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["name"] == "robot"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot_with_usd(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot", name="usd_robot", usd_path="/tmp/robot.usd", scale=0.5)
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["usd_path"] == "/tmp/robot.usd"
        assert call_args[1]["scale"] == 0.5

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot_with_data_config(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot", name="bot", data_config='{"embodiment": "go2"}')
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["data_config"] == {"embodiment": "go2"}


# ─────────────────────────────────────────────────────────────────────
# add_cloth / add_cable / add_particles
# ─────────────────────────────────────────────────────────────────────


class TestSoftBodies:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_cloth(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="add_cloth", name="cloth_0", position="0,1,0", density=0.05)
        assert result["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_cloth_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.add_cloth.return_value = {"success": False, "message": "Solver not supported"}
        mock_get.return_value = mock_backend

        result = call(action="add_cloth")
        assert result["status"] == "error"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_cable(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="add_cable", name="cable_0", position="0,0,1", data_config='{"end": [1, 0, 1]}')
        assert result["status"] == "success"
        mock_backend.add_cable.assert_called_once()

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_cable_default_end(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_cable")
        call_args = mock_backend.add_cable.call_args
        assert call_args[1]["name"] == "cable"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_particles(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="add_particles", name="sand", density=0.1)
        assert result["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_particles_with_positions(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        result = call(action="add_particles", data_config=json.dumps({"positions": positions}))
        assert result["status"] == "success"


# ─────────────────────────────────────────────────────────────────────
# replicate Action
# ─────────────────────────────────────────────────────────────────────


class TestReplicate:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_replicate_success(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="replicate", num_envs=4096)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "4096" in text
        assert "environments" in text.lower()

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_replicate_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.replicate.return_value = {"success": False, "message": "Out of GPU memory"}
        mock_get.return_value = mock_backend

        result = call(action="replicate", num_envs=100000)
        assert result["status"] == "error"
        assert "GPU memory" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# step Action
# ─────────────────────────────────────────────────────────────────────


class TestStep:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_step_single(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="step", num_steps=1)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "1 steps" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_step_multiple(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="step", num_steps=100)
        assert result["status"] == "success"
        assert mock_backend.step.call_count == 100
        text = result["content"][0]["text"]
        assert "100 steps" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_step_failure_mid_loop(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.step.side_effect = [
            {"success": True, "sim_time": 0.016, "step_count": 1},
            {"success": True, "sim_time": 0.032, "step_count": 2},
            {"success": False, "error": "NaN detected"},
        ]
        mock_get.return_value = mock_backend

        result = call(action="step", num_steps=10)
        assert result["status"] == "error"
        assert "NaN" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# observe Action
# ─────────────────────────────────────────────────────────────────────


class TestObserve:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_observe_success(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="observe")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Observations" in text
        assert "robot" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_observe_with_robot_name(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="observe", robot_name="quadruped")
        mock_backend.get_observation.assert_called_once_with("quadruped")

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_observe_no_data(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.get_observation.return_value = {
            "success": False,
            "observations": {},
        }
        mock_get.return_value = mock_backend

        result = call(action="observe")
        assert "No observations" in result["content"][0]["text"]

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_observe_truncates_long_arrays(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.get_observation.return_value = {
            "success": True,
            "sim_time": 0.1,
            "observations": {
                "robot": {
                    "joint_positions": list(range(20)),
                },
            },
        }
        mock_get.return_value = mock_backend

        result = call(action="observe")
        text = result["content"][0]["text"]
        assert "..." in text


# ─────────────────────────────────────────────────────────────────────
# reset Action
# ─────────────────────────────────────────────────────────────────────


class TestReset:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_reset_all(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="reset")
        assert result["status"] == "success"
        mock_backend.reset.assert_called_once_with(env_ids=None)

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_reset_specific_envs(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="reset", env_ids="0,5,10")
        mock_backend.reset.assert_called_once_with(env_ids=[0, 5, 10])

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_reset_empty_env_ids(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="reset", env_ids="")
        mock_backend.reset.assert_called_once_with(env_ids=None)


# ─────────────────────────────────────────────────────────────────────
# get_state Action
# ─────────────────────────────────────────────────────────────────────


class TestGetState:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_get_state(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="get_state")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Newton State" in text
        assert "mujoco" in text
        assert "Steps: 100" in text


# ─────────────────────────────────────────────────────────────────────
# run_policy Action
# ─────────────────────────────────────────────────────────────────────


class TestRunPolicy:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_run_policy_success(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(
            action="run_policy",
            robot_name="quadruped",
            policy_provider="mock",
            instruction="walk forward",
            duration=5.0,
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "mock" in text
        assert "quadruped" in text
        assert "Steps: 600" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_run_policy_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.run_policy.return_value = {
            "success": False,
            "steps_executed": 10,
            "wall_time": 0.5,
            "realtime_factor": 0.5,
            "errors": ["Policy crashed"],
        }
        mock_get.return_value = mock_backend

        result = call(action="run_policy", robot_name="robot")
        assert result["status"] == "error"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_run_policy_uses_name_fallback(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="run_policy", name="my_bot", robot_name="")
        call_args = mock_backend.run_policy.call_args
        assert call_args[1]["robot_name"] == "my_bot"


# ─────────────────────────────────────────────────────────────────────
# run_diffsim Action
# ─────────────────────────────────────────────────────────────────────


class TestRunDiffsim:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_diffsim_success(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="run_diffsim", lr=0.01, iterations=50, optimize_param="initial_velocity")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "DiffSim" in text
        assert "initial_velocity" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_diffsim_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.run_diffsim.return_value = {
            "success": False,
            "message": "Gradient exploded",
        }
        mock_get.return_value = mock_backend

        result = call(action="run_diffsim")
        assert result["status"] == "error"
        assert "Gradient" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# Sensor Actions
# ─────────────────────────────────────────────────────────────────────


class TestSensors:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_sensor(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="add_sensor", name="foot_contact", sensor_type="contact")
        assert result["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_read_sensor(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="read_sensor", name="foot_contact")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "foot_contact" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_read_sensor_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.read_sensor.return_value = {"success": False, "message": "Sensor not found"}
        mock_get.return_value = mock_backend

        result = call(action="read_sensor", name="nonexistent")
        assert result["status"] == "error"


# ─────────────────────────────────────────────────────────────────────
# solve_ik Action
# ─────────────────────────────────────────────────────────────────────


class TestSolveIK:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_solve_ik_success(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="solve_ik", robot_name="panda", position="0.5,0.0,0.3")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "IK converged" in text
        assert "15" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_solve_ik_failure(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_backend.solve_ik.return_value = {"success": False, "message": "Target unreachable"}
        mock_get.return_value = mock_backend

        result = call(action="solve_ik", position="100,100,100")
        assert result["status"] == "error"
        assert "unreachable" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# enable_dual_solver Action
# ─────────────────────────────────────────────────────────────────────


class TestDualSolver:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_dual_solver_default_vbd(self, mock_get, call):
        """With no data_config, cloth solver defaults to 'vbd'."""
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="enable_dual_solver", solver="featherstone")
        assert result["status"] == "success"
        mock_backend.enable_dual_solver.assert_called_once_with(rigid_solver="featherstone", cloth_solver="vbd")

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_dual_solver_with_json_dict_defaults_to_vbd(self, mock_get, call):
        """When data_config is a JSON dict (starts with '{'), cloth solver defaults to 'vbd'."""
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        result = call(action="enable_dual_solver", solver="mujoco", data_config='{"some_key": "value"}')
        assert result["status"] == "success"
        mock_backend.enable_dual_solver.assert_called_once_with(rigid_solver="mujoco", cloth_solver="vbd")

    def test_dual_solver_plain_string_data_config_fails_json(self, call):
        """Non-JSON data_config like 'xpbd' hits json.loads error before reaching
        the enable_dual_solver branch. This is a known code limitation."""
        result = call(action="enable_dual_solver", data_config="xpbd")
        # The outer try/except catches JSONDecodeError
        assert result["status"] == "error"
        assert "Error" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# destroy Action
# ─────────────────────────────────────────────────────────────────────


class TestDestroy:

    def test_destroy_no_backend(self, call):
        result = call(action="destroy")
        assert result["status"] == "success"
        assert "No active backend" in result["content"][0]["text"]

    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_destroy_active_backend(self, mock_destroy, call):
        import importlib

        mod = importlib.import_module("strands_robots.tools.newton_sim")
        mock_backend = _make_mock_backend()
        mod._backend = mock_backend

        result = call(action="destroy")
        assert result["status"] == "success"
        mock_backend.destroy.assert_called_once()
        mock_destroy.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# list_assets Action
# ─────────────────────────────────────────────────────────────────────


class TestListAssets:

    def test_list_assets_newton_not_installed(self, call):
        """list_assets gracefully handles missing newton package."""
        import sys

        # Ensure newton.examples is not importable
        with patch.dict(sys.modules, {"newton": None, "newton.examples": None}):
            result = call(action="list_assets")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "not installed" in text.lower()

    def test_list_assets_with_newton(self, call):
        """list_assets shows assets when newton is available."""
        import sys

        mock_ne = MagicMock()
        mock_ne.get_asset_directory.return_value = "/tmp/newton_assets"

        with patch.dict(sys.modules, {"newton": MagicMock(), "newton.examples": mock_ne}):
            with patch("os.listdir", return_value=["quadruped.urdf", "humanoid.xml", "box.usd"]):
                result = call(action="list_assets")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Newton Bundled Assets" in text


# ─────────────────────────────────────────────────────────────────────
# benchmark Action
# ─────────────────────────────────────────────────────────────────────


class TestBenchmark:

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_benchmark_with_newton_examples(self, mock_destroy, mock_get, call):
        """Benchmark with newton.examples available (add_robot succeeds)."""
        import sys

        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        mock_ne = MagicMock()
        mock_ne.get_asset.return_value = "/tmp/quadruped.urdf"

        with patch.dict(sys.modules, {"newton": MagicMock(), "newton.examples": mock_ne}):
            result = call(action="benchmark", solver="mujoco", benchmark_envs=1024, benchmark_steps=50)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Benchmark" in text
        assert "mujoco" in text
        assert "1024" in text

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_benchmark_fallback_warp(self, mock_destroy, mock_get, call):
        """Benchmark when newton.examples fails but warp is available."""
        import sys

        mock_backend = _make_mock_backend()
        mock_backend._lazy_init = MagicMock()
        mock_backend._builder = MagicMock()
        mock_backend._model = None
        mock_get.return_value = mock_backend

        mock_wp = MagicMock()
        mock_wp.transform.return_value = (0, 0, 0.5)
        mock_wp.quat_identity.return_value = (0, 0, 0, 1)

        # newton.examples fails, warp succeeds
        with patch.dict(sys.modules, {"newton": None, "newton.examples": None, "warp": mock_wp}):
            result = call(action="benchmark", solver="mujoco", benchmark_envs=512, benchmark_steps=20)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Benchmark" in text


# ─────────────────────────────────────────────────────────────────────
# Exception Handling
# ─────────────────────────────────────────────────────────────────────


class TestExceptionHandling:

    @patch("strands_robots.tools.newton_sim._get_backend", side_effect=ImportError("newton not installed"))
    def test_import_error_caught(self, mock_get, call):
        result = call(action="step")
        assert result["status"] == "error"
        assert "newton not installed" in result["content"][0]["text"]

    @patch("strands_robots.tools.newton_sim._get_backend", side_effect=RuntimeError("CUDA OOM"))
    def test_runtime_error_caught(self, mock_get, call):
        result = call(action="step")
        assert result["status"] == "error"
        assert "CUDA OOM" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# Position Parsing
# ─────────────────────────────────────────────────────────────────────


class TestPositionParsing:

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_position_parsing_default(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot", position="")
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["position"] == (0, 0, 0)

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_position_parsing_custom(self, mock_get, call):
        mock_backend = _make_mock_backend()
        mock_get.return_value = mock_backend

        call(action="add_robot", position="1.5,2.5,3.5")
        call_args = mock_backend.add_robot.call_args
        assert call_args[1]["position"] == (1.5, 2.5, 3.5)

    def test_position_parsing_invalid(self, call):
        """Invalid position string should be caught by outer try/except."""
        result = call(action="add_robot", position="not,a,number")
        assert result["status"] == "error"


# ─────────────────────────────────────────────────────────────────────
# _get_backend / _destroy_backend
# ─────────────────────────────────────────────────────────────────────


class TestBackendManagement:

    def test_destroy_backend_clears_globals(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.newton_sim")
        mock_backend = MagicMock()
        mod._backend = mock_backend
        mod._backend_config = {"solver": "mujoco"}

        mod._destroy_backend()
        assert mod._backend is None
        assert mod._backend_config is None
        mock_backend.destroy.assert_called_once()

    def test_destroy_backend_noop_when_none(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.newton_sim")
        mod._backend = None
        mod._backend_config = None

        mod._destroy_backend()  # Should not crash
        assert mod._backend is None

    @patch("strands_robots.newton.NewtonBackend")
    @patch("strands_robots.newton.NewtonConfig")
    def test_get_backend_creates_new(self, mock_config_cls, mock_backend_cls):
        import importlib

        mod = importlib.import_module("strands_robots.tools.newton_sim")
        mod._backend = None

        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config
        mock_backend = MagicMock()
        mock_backend_cls.return_value = mock_backend

        result = mod._get_backend({"solver": "featherstone"})
        assert result == mock_backend
        assert mod._backend == mock_backend

    def test_get_backend_reuses_existing(self):
        import importlib

        mod = importlib.import_module("strands_robots.tools.newton_sim")
        existing = MagicMock()
        mod._backend = existing

        result = mod._get_backend()
        assert result is existing
