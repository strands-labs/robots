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
        assert "list/tuple of 3 numbers" in res["content"][0]["text"]


class TestSetGravityNumpyScalar:
    """A scalar z-only gravity may arrive as a NumPy real scalar.

    ``set_gravity`` accepts a bare number as z-only gravity. A value produced by
    NumPy math (``np.float32``, ``np.int64``, ``np.degrees(...)`` etc.) is a real
    scalar but is not an instance of Python ``int`` / ``float`` (only
    ``np.float64`` subclasses ``float``), so the old ``(int, float)`` guard
    skipped the scalar branch and the value fell through to ``len(gravity)`` --
    raising ``TypeError`` internally and surfacing a misleading
    "must be a 3-element list of numbers (... has no len())" error. These pin
    that any ``numbers.Real`` scalar is treated like a plain float.
    """

    def test_numpy_float32_scalar_accepted(self, sim_with_world):
        import numpy as np

        res = sim_with_world.set_gravity(np.float32(-9.81))
        assert res["status"] == "success"
        gravity = sim_with_world._world._model.opt.gravity
        assert list(gravity[:2]) == [0.0, 0.0]
        assert gravity[2] == pytest.approx(-9.81, abs=1e-4)

    def test_numpy_int64_scalar_accepted(self, sim_with_world):
        import numpy as np

        res = sim_with_world.set_gravity(np.int64(-3))
        assert res["status"] == "success"
        assert list(sim_with_world._world._model.opt.gravity) == [0.0, 0.0, -3.0]

    def test_numpy_array_still_takes_vector_path(self, sim_with_world):
        """A 3-element NumPy array is not a scalar and must set x/y/z directly."""
        import numpy as np

        res = sim_with_world.set_gravity(np.array([0.0, 0.0, -3.7]))
        assert res["status"] == "success"
        assert list(sim_with_world._world._model.opt.gravity) == [0.0, 0.0, -3.7]

    def test_numpy_bool_scalar_still_rejected(self, sim_with_world):
        """``np.bool_`` is not ``numbers.Real`` and has no ``len()`` -- it stays
        refused with a structured error rather than becoming a 1.0 z-gravity."""
        import numpy as np

        res = sim_with_world.set_gravity(np.bool_(True))
        assert res["status"] == "error"
        assert "numbers" in res["content"][0]["text"]


class TestSetGeomPropertiesRejectsInvalid:
    """``set_geom_properties`` must reject physically invalid numeric inputs.

    Unlike ``set_gravity`` / ``set_body_properties`` / ``set_timestep``, the
    geom mutator historically wrote ``color`` / ``friction`` / ``size`` straight
    into the MuJoCo model with no validation. A ``nan`` / ``inf`` -- or a
    negative friction / non-positive size -- landed directly in ``geom_rgba``,
    ``geom_friction`` or ``geom_size`` (making ``geom_rbound`` inf, silently
    breaking broadphase collision) while the call still returned
    ``status="success"``. These pin that such inputs are rejected with a
    structured error and leave the model untouched, and that valid values
    (including NumPy scalars) still apply.
    """

    @pytest.fixture
    def sim_with_box(self):
        sim = Simulation()
        sim.create_world()
        sim.add_object("box", shape="box", position=[0.3, 0.0, 0.1], size=[0.05, 0.05, 0.05])
        yield sim
        sim.destroy()

    def _geom_id(self, sim):
        import mujoco

        return sim._resolve_mj_name(mujoco.mjtObj.mjOBJ_GEOM, "box_geom")

    def test_nan_friction_rejected_and_model_untouched(self, sim_with_box):
        gid = self._geom_id(sim_with_box)
        before = sim_with_box._world._model.geom_friction[gid].copy()
        res = sim_with_box.set_geom_properties(geom_name="box", friction=[float("nan"), 0.005, 0.0001])
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]
        assert (sim_with_box._world._model.geom_friction[gid] == before).all()

    def test_inf_size_rejected_and_rbound_stays_finite(self, sim_with_box):
        import numpy as np

        gid = self._geom_id(sim_with_box)
        res = sim_with_box.set_geom_properties(geom_name="box", size=[float("inf"), 0.05, 0.05])
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]
        assert bool(np.isfinite(sim_with_box._world._model.geom_rbound[gid]))

    def test_non_positive_size_rejected(self, sim_with_box):
        for bad in ([-0.05, 0.05, 0.05], [0.0, 0.05, 0.05]):
            res = sim_with_box.set_geom_properties(geom_name="box", size=bad)
            assert res["status"] == "error"
            assert "> 0" in res["content"][0]["text"]

    def test_negative_friction_rejected(self, sim_with_box):
        res = sim_with_box.set_geom_properties(geom_name="box", friction=[-1.0, 0.005, 0.0001])
        assert res["status"] == "error"
        assert ">= 0" in res["content"][0]["text"]

    def test_nan_color_rejected(self, sim_with_box):
        res = sim_with_box.set_geom_properties(geom_name="box", color=[float("nan"), 0.0, 0.0, 1.0])
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_non_numeric_entry_rejected_not_raised(self, sim_with_box):
        res = sim_with_box.set_geom_properties(geom_name="box", friction=["a", "b", "c"])
        assert res["status"] == "error"
        assert "numbers" in res["content"][0]["text"]

    def test_valid_numpy_scalar_values_still_apply(self, sim_with_box):
        import numpy as np

        gid = self._geom_id(sim_with_box)
        res = sim_with_box.set_geom_properties(
            geom_name="box", friction=[np.float32(0.8), 0.01, 0.001], size=[0.06, 0.06, 0.06]
        )
        assert res["status"] == "success"
        assert abs(float(sim_with_box._world._model.geom_size[gid][0]) - 0.06) < 1e-6
        assert abs(float(sim_with_box._world._model.geom_friction[gid][0]) - 0.8) < 1e-4

    @pytest.mark.parametrize(
        ("kwargs", "param"),
        [
            ({"color": 0.5}, "color"),
            ({"friction": 1.0}, "friction"),
            ({"size": 0.05}, "size"),
        ],
    )
    def test_bare_scalar_vector_arg_rejected_and_model_untouched(self, sim_with_box, kwargs, param):
        """A bare scalar where a numeric *sequence* is expected is rejected.

        ``color`` / ``friction`` / ``size`` are vector-valued. Passing a single
        number (a common agent mistake, e.g. ``color=0.5`` instead of
        ``color=[0.5, 0.5, 0.5, 1.0]``) is not iterable, so it hits the
        sequence-level coercion guard rather than crashing with a bare
        ``TypeError``. The call returns a structured error naming the parameter
        and leaves the model buffers untouched.
        """
        gid = self._geom_id(sim_with_box)
        model = sim_with_box._world._model
        before = {
            "rgba": model.geom_rgba[gid].copy(),
            "friction": model.geom_friction[gid].copy(),
            "size": model.geom_size[gid].copy(),
        }
        res = sim_with_box.set_geom_properties(geom_name="box", **kwargs)
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert param in text
        assert "sequence of numbers" in text
        assert (model.geom_rgba[gid] == before["rgba"]).all()
        assert (model.geom_friction[gid] == before["friction"]).all()
        assert (model.geom_size[gid] == before["size"]).all()
