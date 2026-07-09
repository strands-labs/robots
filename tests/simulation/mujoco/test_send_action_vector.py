"""Regression tests: send_action accepts an ordered action vector.

A policy's ``get_actions`` naturally emits an ordered action *vector* (a list /
tuple / 1-D numpy array), not a ``{joint: value}`` mapping. Before the fix,
passing such a vector to ``send_action`` crashed deep in the actuator/joint
name-lookup loop with ``AttributeError: 'list' object has no attribute 'items'``
- a cryptic failure far from the call site. ``replay_episode`` binds a recorded
action vector positionally to ``robot_action_keys`` (the robot's actuator keys,
the ordering the dataset recorder writes the action column in), so ``send_action``
is made consistent: a vector is zipped against the robot's actuator order, a
mapping is applied unchanged, and an ill-typed / wrong-length action returns an
actionable error instead of crashing or being silently dropped. For a robot whose
actuators mirror its joints (e.g. so101) the two orderings coincide.
"""

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation()
    s.create_world()
    s.add_robot("so101")
    yield s
    s.cleanup()


class TestSendActionVector:
    def test_list_vector_applies_positionally(self, sim):
        """A list action vector binds positionally to robot_joint_names."""
        joints = sim.robot_joint_names("so101")
        vector = [0.3, 0.2, 0.1, 0.0, 0.0, 0.0]
        assert len(vector) == len(joints)

        result = sim.send_action(vector, robot_name="so101", n_substeps=10)

        assert result["status"] == "success", result
        # All joints resolved -> no unresolved-keys json block.
        assert not any(isinstance(b, dict) and b.get("json", {}).get("unresolved_keys") for b in result["content"])

    def test_numpy_vector_applies(self, sim):
        """A 1-D numpy array is accepted just like a list."""
        result = sim.send_action(np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0]), robot_name="so101", n_substeps=5)
        assert result["status"] == "success", result

    def test_vector_actually_moves_the_arm(self, sim):
        """A non-trivial vector target drives the joints away from rest."""
        before = sim.get_observation(robot_name="so101", skip_images=True)
        sim.send_action([0.6, 0.5, 0.4, 0.0, 0.0, 0.0], robot_name="so101", n_substeps=60)
        after = sim.get_observation(robot_name="so101", skip_images=True)
        joints = sim.robot_joint_names("so101")
        moved = sum(abs(float(after[j]) - float(before[j])) for j in joints if j in before and j in after)
        assert moved > 1e-3, f"arm did not move under a vector action (delta={moved})"

    def test_dict_action_still_works(self, sim):
        """The original mapping contract is unchanged (backward compatible)."""
        result = sim.send_action({"1": 0.0, "2": 0.0}, robot_name="so101", n_substeps=2)
        assert result["status"] == "success", result

    def test_wrong_length_vector_is_actionable_error(self, sim):
        """A length mismatch reports the joint count + names, not a crash."""
        result = sim.send_action([0.1, 0.2, 0.3], robot_name="so101")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "length 3" in text
        assert "action-key count 6" in text
        # The valid actuator order is surfaced so the caller can self-correct.
        assert "1" in text and "6" in text

    def test_scalar_action_is_actionable_error(self, sim):
        """A scalar (no length) is rejected with a clear message, not a crash."""
        result = sim.send_action(5.0, robot_name="so101")
        assert result["status"] == "error"
        assert "mapping" in result["content"][0]["text"]

    def test_string_action_is_rejected(self, sim):
        """A str is iterable but never a valid action; reject it explicitly."""
        result = sim.send_action("oops", robot_name="so101")
        assert result["status"] == "error"
        assert "mapping" in result["content"][0]["text"]


class TestSendActionScalarValues:
    """A dict action value must be a scalar the actuator loop can ``float()``.

    ``_apply_action_by_name`` applies each value as ``float(value)`` with no
    guard, so a mapping carrying a non-scalar value (a list / tuple /
    multi-element array - exactly what a policy emitting a vector-valued key
    like ``base_velocity: [vx, vy, omega]`` produces) raised an unhandled
    ``TypeError`` out of ``send_action`` and crashed the caller mid-rollout,
    after partially writing ``data.ctrl`` for the earlier keys. ``send_action``
    returns a structured error for every other malformed input (bad vector
    length, non-numeric vector entry, scalar, string, unresolved keys) - a
    non-scalar dict value is now rejected the same way, atomically.
    """

    def test_list_value_is_actionable_error_not_a_crash(self, sim):
        result = sim.send_action({"1": [0.1, 0.2, 0.3]}, robot_name="so101")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "scalar" in text and "'1'" in text and "list" in text

    def test_multielement_array_value_is_actionable_error(self, sim):
        result = sim.send_action({"1": np.array([0.1, 0.2])}, robot_name="so101")
        assert result["status"] == "error"
        assert "scalar" in result["content"][0]["text"]

    def test_bad_value_rejects_the_whole_action_atomically(self, sim):
        """A good key alongside a bad key applies neither (no partial ctrl write)."""
        model = sim.mj_model
        act_id = model.actuator("so101/1").id  # the good key's actuator
        before_ctrl = float(sim.mj_data.ctrl[act_id])
        result = sim.send_action({"1": 0.5, "2": [0.1, 0.2]}, robot_name="so101")
        assert result["status"] == "error"
        # The whole action is rejected before any ctrl write: the good key's
        # actuator command is untouched.
        assert float(sim.mj_data.ctrl[act_id]) == before_ctrl

    def test_scalar_like_values_still_accepted(self, sim):
        """np.float64, a python float and a numeric string all coerce cleanly."""
        result = sim.send_action(
            {"1": np.float64(0.1), "2": 0.2, "3": "0.05"},
            robot_name="so101",
            n_substeps=2,
        )
        assert result["status"] == "success", result
