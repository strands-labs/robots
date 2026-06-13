"""Regression test: _resolve_single_robot returns structured error, not ValueError.

Before this fix, calling get_robot_state(), run_policy(), or start_policy()
with robot_name=None in a multi-robot scene raised a raw ValueError instead
of returning a structured {"status": "error", ...} dict. This broke the
agent-tool contract where every method returns a dict — agents caught errors
by inspecting result["status"], not by catching exceptions.

The describe() note also incorrectly stated that get_robot_state required an
explicit robot_name even in single-robot scenes. In reality it resolves
automatically via _resolve_single_robot, just like get_observation.
"""

import pytest

from strands_robots.simulation import create_simulation


@pytest.fixture
def multi_robot_sim():
    sim = create_simulation()
    sim.create_world()
    sim.add_robot("arm1", data_config="so100")
    sim.add_robot("arm2", data_config="so100", position=[0.5, 0, 0])
    yield sim
    sim.cleanup()


@pytest.fixture
def single_robot_sim():
    sim = create_simulation()
    sim.create_world()
    sim.add_robot("arm1", data_config="so100")
    yield sim
    sim.cleanup()


class TestGetRobotStateStructuredError:
    """get_robot_state must return error dict, not raise ValueError."""

    def test_multi_robot_none_returns_error_dict(self, multi_robot_sim):
        result = multi_robot_sim.get_robot_state(robot_name=None)
        assert result["status"] == "error"
        assert "Multiple robots" in result["content"][0]["text"]
        assert "arm1" in result["content"][0]["text"]
        assert "arm2" in result["content"][0]["text"]

    def test_single_robot_none_resolves(self, single_robot_sim):
        result = single_robot_sim.get_robot_state(robot_name=None)
        assert result["status"] == "success"


class TestRunPolicyStructuredError:
    """run_policy must return error dict, not raise ValueError."""

    def test_multi_robot_none_returns_error_dict(self, multi_robot_sim):
        result = multi_robot_sim.run_policy(robot_name=None)
        assert result["status"] == "error"
        assert "Multiple robots" in result["content"][0]["text"]

    def test_single_robot_none_resolves(self, single_robot_sim):
        result = single_robot_sim.run_policy(robot_name=None, policy_provider="mock", duration=0.02, fast_mode=True)
        assert result["status"] == "success"


class TestStartPolicyStructuredError:
    """start_policy must return error dict, not raise ValueError."""

    def test_multi_robot_none_returns_error_dict(self, multi_robot_sim):
        result = multi_robot_sim.start_policy(robot_name=None)
        assert result["status"] == "error"
        assert "Multiple robots" in result["content"][0]["text"]

    def test_single_robot_none_resolves(self, single_robot_sim):
        result = single_robot_sim.start_policy(robot_name=None, policy_provider="mock", duration=0.02, fast_mode=True)
        assert result["status"] == "success"
        # Wait for background thread to finish
        import time

        time.sleep(0.5)


class TestDescribeNoteAccuracy:
    """describe() note must accurately reflect auto-resolution behavior."""

    def test_note_mentions_get_robot_state(self, single_robot_sim):
        desc = single_robot_sim.describe()
        note = desc["note"]
        assert "get_robot_state" in note
        # The old note said "pass the robot name explicitly" for get_robot_state.
        # That's wrong - it resolves automatically.
        assert (
            "pass the robot name explicitly" not in note or "get_robot_state" not in note.split("pass")[0]
            if "pass" in note
            else True
        )

    def test_note_mentions_multiple_methods(self, single_robot_sim):
        desc = single_robot_sim.describe()
        note = desc["note"]
        # All methods that use _resolve_single_robot should be mentioned
        assert "get_observation" in note
        assert "get_robot_state" in note
        assert "run_policy" in note
