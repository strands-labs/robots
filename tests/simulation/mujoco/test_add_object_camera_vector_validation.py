"""Regression tests: ``add_object`` / ``add_camera`` reject non-finite / malformed vectors.

Both are scene-construction methods that bake caller-supplied numeric vectors
into the compiled MJCF. Neither validated those vectors' contents before doing
so, letting the two failure classes the numeric-input campaign targets slip
through:

* ``add_object`` wrote a ``nan`` / ``inf`` ``position`` / ``orientation``
  verbatim into the object's freejoint ``qpos``; ``mj_forward`` then propagated
  the non-finite value across the whole physics state while ``add_object`` still
  reported ``status="success"``. A ``nan`` ``size`` aborted the recompile with a
  cryptic "spec recompile refused", and a non-numeric ``color`` / ``size``
  raised a bare ``TypeError`` from inside MuJoCo's ``add_geom`` / the
  ``size <= 0`` comparison - escaping the structured ``{"status": "error"}``
  tool-result contract. A bare scalar ``color`` / ``size`` (a non-iterable
  passed where a vector is expected) is the same class of malformed input and
  is likewise rejected up front rather than raising a bare ``TypeError`` from
  ``add_geom``.
* ``add_camera`` baked a ``nan`` / ``inf`` ``position`` / ``target`` into the
  camera's ``xyaxes`` (``fwd /= flen`` divides by ``nan`` -> a silently
  degenerate camera that renders garbage while reporting ``success``), and a
  non-numeric element raised a bare ``TypeError`` from the degenerate-orientation
  ``abs(pos[i] - tgt[i])`` comparison.

These pin the fix: an invalid vector is rejected up front with a structured
error and leaves the simulation state finite / the entity unregistered, while a
valid vector (including NumPy scalar components) still adds the object/camera.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_add_vector_validation_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


# --------------------------------------------------------------------------- #
# add_object                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"position": [float("nan"), 0.0, 0.03]}, "finite"),
        ({"position": [0.0, float("inf"), 0.03]}, "finite"),
        ({"orientation": [1.0, 0.0, float("inf"), 0.0]}, "finite"),
        ({"color": [float("nan"), 0.0, 0.0, 1.0]}, "finite"),
        ({"size": [float("nan"), 0.05, 0.05]}, "finite"),
    ],
)
def test_add_object_rejects_nonfinite(sim, kwargs, expect):
    """A nan/inf vector is rejected, the object is not added, and qpos stays finite.

    Before the guard, nan/inf position/orientation returned ``success`` and
    poisoned ``data.qpos``; a nan size aborted with a cryptic recompile error.
    """
    result = sim.add_object("bad", shape="box", **kwargs)
    assert result["status"] == "error", result
    assert expect in result["content"][0]["text"]
    assert "bad" not in sim._world.objects
    assert np.all(np.isfinite(sim._world._data.qpos))


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"position": ["a", "b", "c"]}, "must be numbers"),
        ({"color": ["r", "g", "b", "a"]}, "must be numbers"),
        ({"size": ["a", 0.05, 0.05]}, "must be numbers"),
        ({"position": [[1], [2], [3]]}, "must be numbers"),
    ],
)
def test_add_object_rejects_nonnumeric(sim, kwargs, expect):
    """A non-numeric element returns a structured error, not a bare TypeError."""
    result = sim.add_object("bad", shape="box", **kwargs)
    assert result["status"] == "error", result
    assert expect in result["content"][0]["text"]
    assert "bad" not in sim._world.objects


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"color": 5}, "must be a sequence of numbers"),
        ({"color": np.float64(1.0)}, "must be a sequence of numbers"),
        ({"size": 0.05}, "must be a list/tuple of numbers"),
        ({"size": np.float32(0.05)}, "must be a list/tuple of numbers"),
    ],
)
def test_add_object_rejects_scalar_color_size(sim, kwargs, expect):
    """A bare scalar ``color`` / ``size`` is rejected, not raised as a TypeError.

    A caller passing a single number instead of a vector is a non-iterable that
    would otherwise reach ``iter(vec)`` / MuJoCo's ``add_geom`` and raise a bare
    ``TypeError``, escaping the ``{"status": "error"}`` tool-result contract.
    The guard turns it into a structured error naming the offending parameter.
    ``size`` is content-validated without a fixed length (its count is
    shape-dependent and checked against the shape afterwards); ``color`` shares
    the rgba coercion with ``set_geom_properties``, so it reports that helper's
    wording.
    """
    result = sim.add_object("bad", shape="box", **kwargs)
    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert expect in text
    assert next(iter(kwargs)) in text
    assert "bad" not in sim._world.objects


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": [1.0, 2.0]},
        {"position": [1.0, 2.0, 3.0, 4.0]},
        {"orientation": [1.0, 0.0, 0.0]},
        {"orientation": [1.0, 0.0, 0.0, 0.0, 0.0]},
    ],
)
def test_add_object_rejects_wrong_length_pose(sim, kwargs):
    """A wrong-length position (!=3) / orientation (!=4) is a structured error."""
    result = sim.add_object("bad", shape="box", **kwargs)
    assert result["status"] == "error", result
    assert "element vector" in result["content"][0]["text"]
    assert "bad" not in sim._world.objects


def test_add_object_accepts_valid_vectors_with_numpy_scalars(sim):
    """A valid object still adds, and NumPy scalar components are accepted."""
    result = sim.add_object(
        "cube",
        shape="box",
        position=[np.float64(0.2), np.float32(0.0), 0.03],
        orientation=[1.0, 0.0, 0.0, 0.0],
        color=[np.float32(1.0), 0.0, 0.0, 1.0],
        size=[np.float64(0.05), 0.05, 0.05],
    )
    assert result["status"] == "success", result
    assert "cube" in sim._world.objects
    assert np.all(np.isfinite(sim._world._data.qpos))


# --------------------------------------------------------------------------- #
# add_camera                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("position", "target"),
    [
        ([float("nan"), 0.0, 0.3], [0.2, 0.0, 0.05]),
        ([0.5, 0.0, 0.3], [float("inf"), 0.0, 0.05]),
    ],
)
def test_add_camera_rejects_nonfinite(sim, position, target):
    """A nan/inf position/target is rejected and the camera is not registered.

    Before the guard this returned ``success`` and baked a degenerate camera
    (nan xyaxes) that rendered garbage.
    """
    result = sim.add_camera("bad", position=position, target=target)
    assert result["status"] == "error", result
    assert "finite" in result["content"][0]["text"]
    assert "bad" not in sim._world.cameras


@pytest.mark.parametrize(
    ("position", "target"),
    [
        (["a", "b", "c"], [0.2, 0.0, 0.05]),
        ([[1], [2], [3]], [0.2, 0.0, 0.05]),
    ],
)
def test_add_camera_rejects_nonnumeric(sim, position, target):
    """A non-numeric element returns a structured error, not a bare TypeError."""
    result = sim.add_camera("bad", position=position, target=target)
    assert result["status"] == "error", result
    assert "must be numbers" in result["content"][0]["text"]
    assert "bad" not in sim._world.cameras


def test_add_camera_rejects_wrong_length(sim):
    """A non-3 position/target is a structured error."""
    result = sim.add_camera("bad", position=[0.5, 0.3], target=[0.2, 0.0, 0.05])
    assert result["status"] == "error", result
    assert "element vector" in result["content"][0]["text"]
    assert "bad" not in sim._world.cameras


def test_add_camera_accepts_valid_vectors_with_numpy_scalars(sim):
    """A valid camera still registers, and NumPy scalar components are accepted."""
    result = sim.add_camera(
        "front",
        position=[np.float32(0.55), np.float64(0.0), 0.35],
        target=[0.2, 0.0, 0.05],
    )
    assert result["status"] == "success", result
    assert "front" in sim._world.cameras
