#!/usr/bin/env python3
"""Tests for Sample 01: Hello Robot and Sample 02: Policy Playground.

These tests verify the sample scripts work correctly using mock deps.
All heavy dependencies (mujoco, PIL, imageio) are mocked so tests run on CPU CI.
"""

# ──────────────────────────────────────────────────────────────────────
# Pre-mock heavy deps at module level to prevent import-time crashes
# ──────────────────────────────────────────────────────────────────────
# Mock cv2 (OpenCV 4.12 dnn.DictValue crash)
import importlib.machinery as _im_fix
import sys
from unittest.mock import MagicMock

import pytest

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
_mock_cv2_dnn = MagicMock()
_mock_cv2.dnn = _mock_cv2_dnn
sys.modules.setdefault("cv2", _mock_cv2)
sys.modules.setdefault("cv2.dnn", _mock_cv2_dnn)


class TestSample01HelloRobot:
    """Test hello_robot.py script logic."""

    def test_robot_factory_creates_simulation(self):
        """Robot('so100') returns a Simulation instance."""
        from strands_robots import Robot

        try:
            sim = Robot("so100")
            # Should be a Simulation (MuJoCo backend)
            from strands_robots.simulation import Simulation

            assert isinstance(sim, Simulation)
            sim.destroy()
        except ImportError:
            pytest.skip("mujoco not installed")

    def test_list_robots_returns_data(self):
        """list_robots() returns a list of robot dicts."""
        from strands_robots import list_robots

        robots = list_robots()
        assert isinstance(robots, list)
        assert len(robots) > 0
        # Check structure
        for robot in robots:
            assert "name" in robot
            assert "description" in robot

    def test_list_robots_includes_so100(self):
        """SO-100 is in the robot registry."""
        from strands_robots import list_robots

        robots = list_robots()
        names = [r["name"] for r in robots]
        assert "so100" in names

    def test_list_robots_filter_sim(self):
        """list_robots(mode='sim') filters to sim-capable robots."""
        from strands_robots import list_robots

        sim_robots = list_robots(mode="sim")
        for r in sim_robots:
            assert r["has_sim"] is True

    def test_mock_policy_creation(self):
        """create_policy('mock') returns a MockPolicy."""
        from strands_robots import MockPolicy, create_policy

        policy = create_policy("mock")
        assert isinstance(policy, MockPolicy)
        assert policy.provider_name == "mock"

    def test_mock_policy_generates_actions(self):
        """MockPolicy.get_actions returns sinusoidal actions."""
        import asyncio

        from strands_robots import create_policy

        policy = create_policy("mock")
        policy.set_robot_state_keys(["joint_0", "joint_1", "joint_2"])

        observation = {"observation.state": [0.0, 0.0, 0.0]}
        actions = asyncio.run(policy.get_actions(observation, "test"))

        assert isinstance(actions, list)
        assert len(actions) == 8  # default action horizon
        for action in actions:
            assert "joint_0" in action
            assert "joint_1" in action
            assert "joint_2" in action
            # Values should be bounded sinusoidal
            for key in action:
                assert -1.0 <= action[key] <= 1.0


class TestSample02PolicyPlayground:
    """Test policy_playground.py script logic."""

    def test_list_providers_returns_strings(self):
        """list_providers() returns a list of provider name strings."""
        from strands_robots import list_providers

        providers = list_providers()
        assert isinstance(providers, list)
        assert all(isinstance(p, str) for p in providers)
        # Mock should always be available
        assert "mock" in providers

    def test_list_providers_count(self):
        """At least 10 providers are registered."""
        from strands_robots import list_providers

        providers = list_providers()
        assert len(providers) >= 10

    def test_create_policy_mock(self):
        """create_policy('mock') works and returns MockPolicy."""
        from strands_robots import create_policy

        policy = create_policy("mock")
        assert policy.provider_name == "mock"

    def test_register_custom_policy(self):
        """register_policy() allows custom policies."""
        from strands_robots import list_providers, register_policy
        from strands_robots.policies import Policy

        class TestPolicy(Policy):
            @property
            def provider_name(self) -> str:
                return "test_custom"

            def set_robot_state_keys(self, keys):
                self.keys = keys

            async def get_actions(self, obs, instruction, **kwargs):
                return [{"joint_0": 0.0}]

        register_policy("test_custom", lambda: TestPolicy)
        providers = list_providers()
        assert "test_custom" in providers

    def test_mock_policy_auto_generates_keys(self):
        """MockPolicy auto-generates joint keys from observation dimension."""
        import asyncio

        from strands_robots import create_policy

        policy = create_policy("mock")
        # Don't call set_robot_state_keys — let it auto-detect
        observation = {"observation.state": [0.1, 0.2, 0.3, 0.4]}
        actions = asyncio.run(policy.get_actions(observation, "auto test"))

        assert len(actions) == 8
        # Should have auto-generated 4 keys
        first_action = actions[0]
        assert len(first_action) == 4

    def test_policy_abc_interface(self):
        """Policy ABC requires provider_name, set_robot_state_keys, get_actions."""
        from strands_robots.policies import Policy

        # Verify abstract methods
        assert hasattr(Policy, "provider_name")
        assert hasattr(Policy, "set_robot_state_keys")
        assert hasattr(Policy, "get_actions")


class TestSample01TryAllRobots:
    """Test the robot registry for all robots used in try_all_robots.py."""

    @pytest.mark.parametrize(
        "robot_name",
        [
            "so100",
            "panda",
            "aloha",
            "unitree_g1",
            "unitree_go2",
            "reachy_mini",
        ],
    )
    def test_robot_in_registry(self, robot_name):
        """Each robot used in try_all_robots.py exists in the registry."""
        from strands_robots import list_robots

        robots = list_robots(mode="sim")
        names = [r["name"] for r in robots]
        assert robot_name in names, f"{robot_name} not found in sim robot registry"

    def test_robot_aliases(self):
        """Common aliases resolve correctly."""
        from strands_robots.factory import _resolve_name

        assert _resolve_name("franka") == "panda"
        assert _resolve_name("g1") == "unitree_g1"
        assert _resolve_name("go2") == "unitree_go2"
        assert _resolve_name("h1") == "unitree_h1"
        assert _resolve_name("a1") == "unitree_a1"


class TestSample02AnatomyOfAPolicy:
    """Test the custom policy pattern from anatomy_of_a_policy.py."""

    def test_custom_policy_subclass(self):
        """Custom Policy subclass works correctly."""
        import asyncio

        from strands_robots.policies import Policy

        class CenterPolicy(Policy):
            def __init__(self):
                self.keys = []

            @property
            def provider_name(self):
                return "center"

            def set_robot_state_keys(self, keys):
                self.keys = keys

            async def get_actions(self, obs, instruction, **kwargs):
                return [{k: 0.0 for k in self.keys}]

        policy = CenterPolicy()
        policy.set_robot_state_keys(["j0", "j1", "j2"])

        actions = asyncio.run(policy.get_actions({}, "go to center"))
        assert len(actions) == 1
        assert actions[0] == {"j0": 0.0, "j1": 0.0, "j2": 0.0}

    def test_register_and_create(self):
        """Registered policies can be created via create_policy()."""
        from strands_robots import create_policy, register_policy
        from strands_robots.policies import Policy

        class DemoPolicy(Policy):
            @property
            def provider_name(self):
                return "demo_test"

            def set_robot_state_keys(self, keys):
                pass

            async def get_actions(self, obs, instruction, **kwargs):
                return [{}]

        register_policy("demo_test", lambda: DemoPolicy)
        policy = create_policy("demo_test")
        assert policy.provider_name == "demo_test"
