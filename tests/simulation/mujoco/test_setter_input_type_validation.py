"""Type-coercion guards for MuJoCo sim config mutators.

These pin the contract that ``set_gravity``, ``set_timestep`` and
``add_camera`` reject malformed *types* (a non-numeric entry inside an
otherwise correctly-shaped vector, a non-numeric scalar, or a non-sized
argument) with a structured ``{"status": "error"}`` dict rather than
propagating a ``TypeError`` / ``ValueError`` out of the call.

Existing suites already cover numeric-but-invalid input (wrong length,
NaN, Inf, non-positive). The branches exercised here are the ``float(...)``
and ``len(...)`` coercion-failure paths, which fire only for genuinely
non-numeric / non-sized arguments an agent can still supply.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty compiled world."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


class TestSetterInputTypeValidation:
    def test_set_gravity_non_numeric_entry_errors(self, sim_with_world):
        """A correctly-shaped vector with a non-numeric entry is rejected via
        the float() coercion path, not by raising ValueError to the caller."""
        res = sim_with_world.set_gravity(["x", 0.0, 0.0])
        assert res["status"] == "error"
        assert "numbers" in res["content"][0]["text"]

    def test_set_timestep_non_numeric_string_errors(self, sim_with_world):
        """A non-numeric string timestep is rejected via float() coercion."""
        res = sim_with_world.set_timestep("fast")
        assert res["status"] == "error"
        assert "positive number" in res["content"][0]["text"]

    def test_set_timestep_none_errors(self, sim_with_world):
        """``None`` (TypeError under float()) is rejected, not raised."""
        res = sim_with_world.set_timestep(None)
        assert res["status"] == "error"
        assert "positive number" in res["content"][0]["text"]

    def test_add_camera_non_sized_position_errors(self, sim_with_world):
        """A non-sized position (no ``len()``) is rejected via the TypeError
        branch of the shape check rather than raising to the caller."""
        res = sim_with_world.add_camera(name="cam", position=5)
        assert res["status"] == "error"
        assert "list of 3 numbers" in res["content"][0]["text"]
