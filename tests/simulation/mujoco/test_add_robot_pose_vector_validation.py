"""Regression tests: ``add_robot`` rejects a non-finite / malformed base pose.

``add_robot`` bakes the caller-supplied ``position`` (3) / ``orientation``
(4-element wxyz quaternion) straight into the injected robot's frame pos/quat.
It shared the numeric-vector failure classes already guarded on ``add_object`` /
``move_object`` / ``add_camera`` but did not validate the pose itself, so three
failures slipped through:

* A ``nan`` / ``inf`` ``position`` / ``orientation`` was written verbatim into
  the base transform and propagated across the whole physics state by
  ``mj_forward`` while ``add_robot`` still reported ``status="success"`` -- a
  silent corruption (``data.xpos`` went non-finite).
* A wrong-length vector produced a generic "Failed to inject robot" with no hint
  that the length was the problem.
* A non-numeric element raised a bare MuJoCo ``add_frame(): incompatible
  function arguments`` ``TypeError`` that escaped the structured
  ``{"status": "error"}`` tool-result contract.

These pin the fix: an invalid pose is rejected up front with a structured,
actionable error and leaves the simulation state finite / the robot
unregistered, while a valid pose (including NumPy scalar components) still adds
the robot.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_add_robot_pose_validation_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"position": [float("nan"), 0.0, 0.0]}, "finite"),
        ({"position": [0.0, float("inf"), 0.0]}, "finite"),
        ({"orientation": [1.0, 0.0, float("inf"), 0.0]}, "finite"),
        ({"orientation": [float("nan"), 0.0, 0.0, 0.0]}, "finite"),
    ],
)
def test_add_robot_rejects_nonfinite_pose(sim, kwargs, expect):
    """A nan/inf pose is rejected, the robot is not added, and xpos stays finite.

    Before the guard, a nan/inf position/orientation returned ``success`` and
    poisoned ``data.xpos`` through ``mj_forward``.
    """
    result = sim.add_robot(data_config="so101", **kwargs)
    assert result["status"] == "error", result
    assert expect in result["content"][0]["text"]
    assert "so101" not in sim._world.robots
    assert np.all(np.isfinite(sim._world._data.xpos))
    assert np.all(np.isfinite(sim._world._data.qpos))


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"position": ["a", "b", "c"]}, "must be numbers"),
        ({"orientation": ["a", "b", "c", "d"]}, "must be numbers"),
        ({"position": [[1], [2], [3]]}, "must be numbers"),
    ],
)
def test_add_robot_rejects_nonnumeric_pose(sim, kwargs, expect):
    """A non-numeric element returns a structured error, not a bare TypeError."""
    result = sim.add_robot(data_config="so101", **kwargs)
    assert result["status"] == "error", result
    assert expect in result["content"][0]["text"]
    assert "so101" not in sim._world.robots


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": [1.0, 2.0]},
        {"position": [1.0, 2.0, 3.0, 4.0]},
        {"orientation": [1.0, 0.0, 0.0]},
        {"orientation": [1.0, 0.0, 0.0, 0.0, 0.0]},
    ],
)
def test_add_robot_rejects_wrong_length_pose(sim, kwargs):
    """A wrong-length position (!=3) / orientation (!=4) is a structured error."""
    result = sim.add_robot(data_config="so101", **kwargs)
    assert result["status"] == "error", result
    assert "element vector" in result["content"][0]["text"]
    assert "so101" not in sim._world.robots


def test_add_robot_accepts_valid_pose_with_numpy_scalars(sim):
    """A valid pose still adds the robot, and NumPy scalar components are accepted."""
    result = sim.add_robot(
        data_config="so101",
        position=[np.float64(0.1), np.float32(0.2), 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )
    assert result["status"] == "success", result
    assert "so101" in sim._world.robots
    assert np.all(np.isfinite(sim._world._data.xpos))
    assert np.all(np.isfinite(sim._world._data.qpos))
