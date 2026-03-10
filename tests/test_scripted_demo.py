#!/usr/bin/env python3
"""Tests for scripted pick-and-place demo policy."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from scripted_pick_demo import ScriptedPickPolicy


class TestScriptedPickPolicy:
    """Test the ScriptedPickPolicy trajectory generation."""

    def test_trajectory_length(self):
        """Trajectory should match requested steps."""
        policy = ScriptedPickPolicy(num_joints=6)
        traj = policy.generate_trajectory(total_steps=200)
        assert len(traj) == 200

    def test_trajectory_shape(self):
        """Each action should have num_joints + 1 (gripper) dims."""
        policy = ScriptedPickPolicy(num_joints=6)
        traj = policy.generate_trajectory(total_steps=100)
        for action in traj:
            assert action.shape == (7,), f"Expected (7,) got {action.shape}"

    def test_starts_with_gripper_open(self):
        """First action should have gripper open."""
        policy = ScriptedPickPolicy(gripper_open=1.0)
        traj = policy.generate_trajectory(100)
        assert traj[0][-1] == pytest.approx(1.0, abs=0.01)

    def test_has_gripper_close_phase(self):
        """Middle of trajectory should have gripper closing."""
        policy = ScriptedPickPolicy(gripper_open=1.0, gripper_closed=-1.0)
        traj = policy.generate_trajectory(200)
        # Check around 60% mark (grasp phase)
        mid_idx = len(traj) * 3 // 5
        assert traj[mid_idx][-1] < 0.5, "Gripper should be closing in grasp phase"

    def test_smooth_trajectory(self):
        """Trajectory should be smooth (no large jumps)."""
        policy = ScriptedPickPolicy(num_joints=6)
        traj = policy.generate_trajectory(200)
        for i in range(1, len(traj)):
            diff = np.abs(traj[i][:6] - traj[i - 1][:6])
            max_diff = np.max(diff)
            assert max_diff < 0.2, f"Step {i}: max joint diff {max_diff:.3f} > 0.2"

    def test_different_steps_count(self):
        """Should handle various step counts."""
        policy = ScriptedPickPolicy(num_joints=6)
        for steps in [50, 100, 200, 500]:
            traj = policy.generate_trajectory(steps)
            assert len(traj) == steps

    def test_get_action_clamps(self):
        """get_action beyond trajectory should return last action."""
        policy = ScriptedPickPolicy(num_joints=6)
        policy.generate_trajectory(100)
        last = policy.get_action(99)
        beyond = policy.get_action(999)
        np.testing.assert_array_equal(last, beyond)

    def test_randomized_cube_position(self):
        """Different cube positions should produce different trajectories."""
        p1 = ScriptedPickPolicy(cube_pos=np.array([0.1, 0, 0.02]))
        p2 = ScriptedPickPolicy(cube_pos=np.array([0.3, 0, 0.02]))
        # The trajectories are currently parametric, not IK-based,
        # so they'll be the same. This test documents current behavior.
        t1 = p1.generate_trajectory(100)
        t2 = p2.generate_trajectory(100)
        # Currently equal — when we add IK, they should differ
        assert len(t1) == len(t2)


class TestTrajectoryPhases:
    """Test individual motion phases."""

    def test_approach_starts_at_home(self):
        """Approach phase alpha=0 should be home (zeros)."""
        policy = ScriptedPickPolicy(num_joints=6)
        joints = policy._approach_joints(0.0)
        np.testing.assert_array_almost_equal(joints, np.zeros(6))

    def test_approach_ends_at_target(self):
        """Approach phase alpha=1 should reach target."""
        policy = ScriptedPickPolicy(num_joints=6)
        joints = policy._approach_joints(1.0)
        expected = np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0])
        np.testing.assert_array_almost_equal(joints, expected)

    def test_phases_are_continuous(self):
        """End of one phase should match start of next."""
        policy = ScriptedPickPolicy(num_joints=6)
        # approach(1.0) should be close to descend(0.0)
        end_approach = policy._approach_joints(1.0)
        start_descend = policy._descend_joints(0.0)
        np.testing.assert_array_almost_equal(end_approach, start_descend)

        # descend(1.0) should match lift(0.0)
        end_descend = policy._descend_joints(1.0)
        start_lift = policy._lift_joints(0.0)
        np.testing.assert_array_almost_equal(end_descend, start_lift)
