"""Regression tests: ``move_object`` rejects a non-finite / malformed pose.

``move_object`` moves a dynamic object by writing its ``[x, y, z, qw..qz]``
slice straight into ``data.qpos`` (then a forward pass), and a static object by
editing the spec body pose and recompiling. Neither path validated the caller's
``position`` / ``orientation`` before mutating state, so two failure classes
slipped through:

* A ``nan`` / ``inf`` component was written verbatim into ``data.qpos``;
  ``mj_forward`` then propagated the non-finite value across the whole physics
  state while ``move_object`` still reported ``status="success"`` - a silent
  corruption.
* A non-numeric (``["a", "b", "c"]``) or wrong-length (``[1.0, 2.0]``) vector
  raised a bare ``ValueError`` from inside the numpy assignment / broadcast,
  escaping the structured ``{"status": "error"}`` tool-result contract.

These pin the fix: an invalid pose is rejected up front with a structured error
and leaves the simulation state finite, while a valid pose (including NumPy
scalar components) still moves the object.
"""

import math

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_move_object_pose_validation_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    s.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.5], is_static=False)
    s.add_object("wall", shape="box", size=[0.1, 0.1, 0.1], position=[1.0, 0.0, 0.5], is_static=True)
    yield s
    s.cleanup()


@pytest.mark.parametrize(
    "position",
    [
        [float("nan"), 0.0, 0.5],
        [0.0, float("inf"), 0.5],
        [0.0, 0.0, -float("inf")],
    ],
)
def test_move_object_rejects_nonfinite_position(sim, position):
    """A nan/inf position is rejected and never poisons qpos.

    Before the guard this returned ``success`` and left ``data.qpos`` holding
    the non-finite value.
    """
    result = sim.move_object("cube", position=position)
    assert result["status"] == "error", result
    assert "finite" in result["content"][0]["text"]
    assert np.all(np.isfinite(sim._world._data.qpos))


def test_move_object_rejects_nonfinite_orientation(sim):
    """A nan/inf quaternion component is rejected and qpos stays finite."""
    result = sim.move_object("cube", orientation=[float("inf"), 0.0, 0.0, 0.0])
    assert result["status"] == "error", result
    assert "finite" in result["content"][0]["text"]
    assert np.all(np.isfinite(sim._world._data.qpos))


def test_move_object_rejects_nonnumeric_position(sim):
    """A non-numeric vector returns a structured error, not a bare ValueError."""
    result = sim.move_object("cube", position=["a", "b", "c"])
    assert result["status"] == "error", result
    assert "must be numbers" in result["content"][0]["text"]


@pytest.mark.parametrize(
    ("position", "orientation"),
    [
        ([1.0, 2.0], None),
        ([1.0, 2.0, 3.0, 4.0], None),
        (None, [1.0, 0.0, 0.0]),
        (None, [1.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_move_object_rejects_wrong_length(sim, position, orientation):
    """A wrong-length position (!=3) or orientation (!=4) is a structured error."""
    result = sim.move_object("cube", position=position, orientation=orientation)
    assert result["status"] == "error", result
    assert "element vector" in result["content"][0]["text"]


def test_move_object_rejects_nonfinite_position_static_object(sim):
    """The static-object reposition path is guarded too (no spec mutation)."""
    result = sim.move_object("wall", position=[1.0, float("nan"), 0.5])
    assert result["status"] == "error", result
    assert "finite" in result["content"][0]["text"]
    assert np.all(np.isfinite(sim._world._data.qpos))
    # State untouched: the wall is still at its spawn position.
    assert sim._world.objects["wall"].position == [1.0, 0.0, 0.5]


def test_move_object_accepts_valid_pose_with_numpy_scalars(sim):
    """A valid pose still succeeds, and NumPy scalar components are accepted."""
    quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    result = sim.move_object(
        "cube",
        position=[np.float64(0.2), np.float32(0.0), 0.6],
        orientation=quat,
    )
    assert result["status"] == "success", result
    assert np.all(np.isfinite(sim._world._data.qpos))
    bid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "cube")
    assert [float(x) for x in sim._world._data.xpos[bid]] == pytest.approx([0.2, 0.0, 0.6], abs=1e-6)
