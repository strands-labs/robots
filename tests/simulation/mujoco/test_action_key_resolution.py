"""Regression tests for action key resolution (bugs #2 and #3 from hot-path audit).

Bug #2: When send_action drops ALL action keys (unresolved), run_policy must NOT
         return status='success'. This catches any policy that produces keys that
         don't match the robot's actuators.

Bug #3: The error/warning message when action keys are unresolved should enumerate
         the actual valid actuator names, not hardcoded examples that don't exist
         on the loaded robot.
"""

import logging
from typing import Any

import pytest

from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import Simulation


class _StubbornPolicy(Policy):
    """A policy that ignores set_robot_state_keys and always emits wrong keys.

    Simulates a misconfigured external policy whose output keys don't match
    the robot's actuators. This is the failure mode Bug #2 is about.
    """

    @property
    def provider_name(self) -> str:
        return "stubborn_test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        # Intentionally ignore the correct keys.
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        # Always emit generic keys that don't match so100 actuators.
        return [{"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3, "joint_3": 0.4, "joint_4": 0.5, "joint_5": 0.6}]


@pytest.fixture
def sim():
    s = Simulation()
    s.create_world()
    s.add_robot("so100")
    return s


class TestActionKeyResolution:
    """Tests that unresolved action keys produce actionable diagnostics."""

    def test_send_action_invalid_keys_returns_error(self, sim):
        """send_action with keys that don't match any actuator returns error status."""
        action = {"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3}
        result = sim.send_action(action)
        assert result["status"] == "error"
        # All three keys should be unresolved
        json_block = next((c for c in result["content"] if "json" in c), None)
        assert json_block is not None
        assert set(json_block["json"]["unresolved_keys"]) == {"joint_0", "joint_1", "joint_2"}

    def test_send_action_error_shows_valid_keys(self, sim):
        """The error message should list actual actuator names, not hardcoded examples."""
        action = {"nonexistent_joint": 1.0}
        result = sim.send_action(action)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        # Must NOT contain the old hardcoded examples
        assert "shoulder_pan" not in text
        assert "elbow_flex" not in text
        # Must contain actual so100 actuator names
        assert "Rotation" in text
        assert "Pitch" in text
        assert "Jaw" in text

    def test_send_action_valid_keys_returns_success(self, sim):
        """send_action with correct actuator names returns success."""
        action = {"Rotation": 0.5, "Pitch": 0.3}
        result = sim.send_action(action)
        assert result["status"] == "success"

    def test_warn_unresolved_includes_valid_names(self, sim, caplog):
        """The warning log includes valid actuator names for the robot."""
        with caplog.at_level(logging.WARNING):
            sim.send_action({"bogus_key": 1.0})
        # Find the warning about the unresolved key
        warn_msgs = [r.message for r in caplog.records if "bogus_key" in r.message]
        assert len(warn_msgs) >= 1
        msg = warn_msgs[0]
        # Should list the real actuator names
        assert "Rotation" in msg
        assert "Jaw" in msg
        # Should NOT suggest hardcoded names that don't exist on so100
        assert "shoulder_pan" not in msg
        assert "elbow_flex" not in msg


class TestPolicyRunnerActionErrors:
    """Tests that run_policy propagates action-key failures to the final status."""

    def test_stubborn_policy_wrong_keys_reports_error(self, sim):
        """A policy that ignores set_robot_state_keys must trigger error status."""
        policy = _StubbornPolicy()
        result = sim.run_policy(
            policy_object=policy,
            duration=0.1,
            control_frequency=10.0,
        )
        # Pre-fix this was "success" (false). Post-fix it must be "error".
        assert result["status"] == "error"
        assert "unresolved" in result["content"][0]["text"].lower()

    def test_mock_policy_via_provider_succeeds(self, sim):
        """MockPolicy via the provider path (keys auto-set) returns success."""
        result = sim.run_policy(
            policy_provider="mock",
            duration=0.1,
            control_frequency=10.0,
        )
        assert result["status"] == "success"

    def test_mock_policy_via_policy_object_succeeds(self, sim):
        """MockPolicy passed as policy_object (keys auto-set by base) returns success."""
        from strands_robots.policies.mock import MockPolicy

        policy = MockPolicy()
        result = sim.run_policy(
            policy_object=policy,
            duration=0.1,
            control_frequency=10.0,
        )
        # run_policy calls set_robot_state_keys before the loop, so MockPolicy
        # gets the correct keys. This must succeed.
        assert result["status"] == "success"
