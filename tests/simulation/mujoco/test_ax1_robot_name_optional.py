"""AX-1 regression test: robot_name optional across get_robot_state-family methods.

This test MUST FAIL before the fix and PASS after it.

The inconsistency: get_observation(robot_name=None) works (infers single robot),
but get_robot_state(robot_name) requires the arg. An agent (or human) who learns
one signature cannot transfer to the other.

Fix: a shared _resolve_single_robot(robot_name) helper that resolves None when
exactly one robot exists, and raises ValueError listing candidates when ambiguous.

Applied to: get_robot_state, run_policy, start_policy.
Already correct: get_observation, send_action (default to None).
"""

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def single_robot_sim():
    """Create a sim with exactly ONE robot (so100)."""
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    sim = MuJoCoSimEngine(tool_name="test_ax1_single")
    sim.create_world()
    result = sim.add_robot("alice", data_config="so100")
    assert result["status"] == "success", f"add_robot failed: {result}"
    yield sim
    sim.cleanup()


@pytest.fixture
def two_robot_sim():
    """Create a sim with TWO robots (both so100, different names)."""
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    sim = MuJoCoSimEngine(tool_name="test_ax1_multi")
    sim.create_world()
    r1 = sim.add_robot("alice", data_config="so100")
    assert r1["status"] == "success", f"add_robot alice failed: {r1}"
    r2 = sim.add_robot("bob", data_config="so100", position=[1.0, 0.0, 0.0])
    assert r2["status"] == "success", f"add_robot bob failed: {r2}"
    yield sim
    sim.cleanup()


class TestGetRobotStateOptional:
    """get_robot_state() should work without robot_name when exactly one robot."""

    def test_single_robot_no_arg(self, single_robot_sim):
        """Core AX-1: single-robot sim -> get_robot_state() (no arg) succeeds."""
        result = single_robot_sim.get_robot_state()
        assert result["status"] == "success", f"Expected success, got: {result}"
        # Should contain joint state
        assert any("json" in c for c in result["content"]), "Expected JSON state payload"

    def test_single_robot_explicit_arg(self, single_robot_sim):
        """Explicit robot_name still works (backwards compat)."""
        result = single_robot_sim.get_robot_state(robot_name="alice")
        assert result["status"] == "success"

    def test_two_robots_no_arg_raises(self, two_robot_sim):
        """Two-robot sim -> get_robot_state() raises ValueError listing both names."""
        with pytest.raises(ValueError, match=".*alice.*bob.*|.*bob.*alice.*"):
            two_robot_sim.get_robot_state()

    def test_two_robots_explicit_arg_works(self, two_robot_sim):
        """Two-robot sim -> get_robot_state(robot_name='alice') works fine."""
        result = two_robot_sim.get_robot_state(robot_name="alice")
        assert result["status"] == "success"
        result2 = two_robot_sim.get_robot_state(robot_name="bob")
        assert result2["status"] == "success"


class TestRunPolicyOptional:
    """run_policy() should infer robot_name when exactly one robot."""

    def test_single_robot_no_arg(self, single_robot_sim):
        """Single robot -> run_policy() (no robot_name) uses that robot."""
        result = single_robot_sim.run_policy(policy_provider="mock", duration=0.1, control_frequency=10.0)
        assert result["status"] == "success", f"Expected success, got: {result}"

    def test_two_robots_no_arg_raises(self, two_robot_sim):
        """Two robots -> run_policy() raises ValueError listing both names."""
        with pytest.raises(ValueError, match=".*alice.*bob.*|.*bob.*alice.*"):
            two_robot_sim.run_policy(policy_provider="mock", duration=0.1)


class TestStartPolicyOptional:
    """start_policy() should infer robot_name when exactly one robot."""

    def test_single_robot_no_arg(self, single_robot_sim):
        """Single robot -> start_policy() (no robot_name) uses that robot."""
        result = single_robot_sim.start_policy(policy_provider="mock", duration=0.1, control_frequency=10.0)
        assert result["status"] == "success", f"Expected success, got: {result}"
        # Wait for async policy to finish
        import time

        time.sleep(0.5)

    def test_two_robots_no_arg_raises(self, two_robot_sim):
        """Two robots -> start_policy() raises ValueError listing both names."""
        with pytest.raises(ValueError, match=".*alice.*bob.*|.*bob.*alice.*"):
            two_robot_sim.start_policy(policy_provider="mock", duration=0.1)


class TestRobotFactoryEntryPoint:
    """AX-4 verification: Robot("so100", mode="sim") -> get_robot_state() works."""

    def test_robot_factory_get_state_no_arg(self):
        """Robot('so100') factory -> get_robot_state() just works."""
        from strands_robots.robot import Robot

        sim = Robot("so100", mode="sim", mesh=False)
        try:
            result = sim.get_robot_state()
            assert result["status"] == "success", f"Expected success, got: {result}"
        finally:
            sim.cleanup()


class TestResolveSingleRobotHelper:
    """Unit tests for the _resolve_single_robot helper."""

    def test_explicit_name_passthrough(self, single_robot_sim):
        """Explicit name is returned unchanged."""
        resolved = single_robot_sim._resolve_single_robot("alice")
        assert resolved == "alice"

    def test_none_single_robot(self, single_robot_sim):
        """None + one robot -> that robot's name."""
        resolved = single_robot_sim._resolve_single_robot(None)
        assert resolved == "alice"

    def test_none_multiple_robots(self, two_robot_sim):
        """None + multiple robots -> ValueError with both names."""
        with pytest.raises(ValueError) as exc_info:
            two_robot_sim._resolve_single_robot(None)
        msg = str(exc_info.value)
        assert "alice" in msg
        assert "bob" in msg

    def test_none_no_robots(self):
        """None + no robots -> ValueError."""
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        sim = MuJoCoSimEngine(tool_name="test_ax1_empty")
        sim.create_world()
        try:
            with pytest.raises(ValueError, match="[Nn]o robot"):
                sim._resolve_single_robot(None)
        finally:
            sim.cleanup()
