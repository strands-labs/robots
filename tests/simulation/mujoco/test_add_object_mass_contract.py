"""``add_object`` accepts only a mass the compiled model can carry.

A dynamic body's mass divides every force applied to it, and MuJoCo keeps one
state vector for the whole world, so a mass outside the usable domain is not a
local mistake - it takes the entire scene with it. ``add_object`` used to write
whatever it was handed straight into the spec:

* ``mass=inf`` compiled and reported ``status="success"``. The first ``step``
  produced ``nan`` acceleration, and every *other* body's ``qpos``/``qvel``
  went ``nan`` with it - a 1 kg neighbour added before the bad object stopped
  falling and never moved again.
* ``mass=0`` / negatives / ``nan`` aborted the recompile, surfacing as
  ``"Failed to inject 'x': spec recompile refused."``. MuJoCo's actual reason
  ("mass and inertia of moving bodies must be larger than mjMINVAL") went to
  the log only, so the result named neither the parameter nor the rule.
* ``0 < mass < mjMINVAL`` was refused by the compiler for the same reason and
  reached the same generic message.
* A non-numeric mass raised inside the spec mutation *after* the body had been
  inserted. The injector rolled back only the mesh asset, so the half-built
  body stayed in the spec and every later scene mutation - a valid
  ``add_object``, an ``add_camera`` - failed to recompile. One bad call bricked
  the world.

The same orphan leak applied to an unsupported ``shape``: its type lookup also
raises after the body is inserted, so the name stayed permanently taken and a
corrected retry failed with ``repeated name``.

The domain enforced here is the one
:meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.set_body_properties`
already applied when writing the same ``body_mass`` field (finite, ``> 0``),
plus MuJoCo's compile-time ``mjMINVAL`` floor, so a mass cannot be established
at creation on terms the setter would refuse.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_add_object_mass", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _body_mass(sim: Simulation, name: str) -> float:
    assert sim._world is not None and sim._world._model is not None
    model = sim._world._model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body {name!r} missing from the compiled model"
    return float(model.body_mass[body_id])


class TestAddObjectMassDomain:
    def test_infinite_mass_is_refused_and_the_rest_of_the_world_keeps_stepping(self, sim):
        """The decisive case: ``inf`` used to compile and NaN the whole world.

        A 1 kg neighbour is added first, so the assertion is about the world and
        not just the rejected object: pre-fix the ``inf`` add returned success
        and after a single step every ``qpos``/``qvel`` element was ``nan``,
        freezing the neighbour in mid-air.
        """
        assert sim.add_object("neighbour", shape="box", position=[0.0, 0.0, 0.6], mass=1.0)["status"] == "success"

        result = sim.add_object("blackhole", shape="box", position=[0.4, 0.0, 0.6], mass=float("inf"))
        assert result["status"] == "error", result
        assert "'mass'" in result["content"][0]["text"]
        assert "blackhole" not in sim._world.objects

        assert sim.step(200)["status"] == "success"
        data = sim._world._data
        assert np.all(np.isfinite(data.qpos)), "physics state was poisoned"
        assert np.all(np.isfinite(data.qvel)), "physics state was poisoned"
        # The surviving body actually fell instead of being frozen by NaNs.
        neighbour_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, "neighbour")
        assert float(data.xpos[neighbour_id][2]) < 0.5

    @pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), -float("inf")])
    def test_unusable_mass_names_the_parameter_instead_of_refusing_the_recompile(self, sim, bad):
        result = sim.add_object("crate", shape="box", mass=bad)
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "add_object" in text and "'mass'" in text
        assert "spec recompile refused" not in text
        assert "crate" not in sim._world.objects
        # The scene was never mutated, so the name is still free.
        assert sim.add_object("crate", shape="box", mass=1.0)["status"] == "success"

    def test_mass_below_the_compiler_floor_names_the_floor(self, sim):
        """``0 < mass < mjMINVAL`` is positive yet uncompilable; say so."""
        result = sim.add_object("dust", shape="box", mass=1e-16)
        assert result["status"] == "error", result
        assert "mjMINVAL" in result["content"][0]["text"]
        # The floor itself is accepted, so the boundary is not over-tightened.
        assert sim.add_object("speck", shape="box", mass=float(mujoco.mjMINVAL))["status"] == "success"

    def test_non_numeric_mass_leaves_the_scene_usable(self, sim):
        """Pre-fix this raised mid-mutation and bricked every later scene edit."""
        result = sim.add_object("junk", shape="box", mass="heavy")
        assert result["status"] == "error", result
        assert "'mass'" in result["content"][0]["text"]

        assert sim.add_object("cube", shape="box", position=[0.2, 0.0, 0.1], mass=1.0)["status"] == "success"
        assert sim.add_camera("side", position=[1.0, 0.0, 0.5], target=[0.0, 0.0, 0.1])["status"] == "success"

    def test_valid_mass_reaches_the_compiled_model(self, sim):
        assert sim.add_object("crate", shape="box", position=[0.0, 0.0, 0.3], mass=2.5)["status"] == "success"
        assert _body_mass(sim, "crate") == pytest.approx(2.5)

    def test_static_object_ignores_mass_by_design(self, sim):
        """A static body's mass never comes from the parameter.

        Without a freejoint no explicit inertial block is written, so MuJoCo
        derives the body mass from the geom's default density (1000 kg/m^3 x a
        5 cm cube = 0.125 kg) whatever ``mass`` said. The guard therefore fires
        only where the value reaches ``body_mass`` - a dynamic body - and a
        static add keeps reporting ``static`` rather than quoting a mass, so
        nothing is silently dishonored: the parameter is documented as ignored
        for every value, not just this one.
        """
        result = sim.add_object("floor_tile", shape="box", mass=-1.0, is_static=True)
        assert result["status"] == "success", result
        assert "static" in result["content"][0]["text"]
        assert _body_mass(sim, "floor_tile") == pytest.approx(0.125)


class TestAddObjectRaisePathRollback:
    def test_unsupported_shape_reports_the_shape_and_frees_the_name(self, sim):
        """The shape lookup raises after the body is inserted.

        Pre-fix the actionable message went to the log, the result said only
        "spec recompile refused", and the orphan body kept the name for good -
        the corrected retry failed with ``repeated name``.
        """
        result = sim.add_object("crate", shape="blob")
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "blob" in text
        assert "box" in text and "sphere" in text, "the supported shapes should be listed"

        retry = sim.add_object("crate", shape="box", position=[0.0, 0.0, 0.3], size=[0.2, 0.2, 0.2], mass=1.0)
        assert retry["status"] == "success", retry
        assert "crate" in sim._world.objects
        assert _body_mass(sim, "crate") == pytest.approx(1.0)

    def test_name_collision_with_a_loaded_scene_body_leaves_that_body_intact(self, sim, tmp_path):
        """A colliding add must never delete the body it collides with.

        ``load_scene`` replaces the world with a fresh registry, so a body that
        came from the XML is invisible to ``add_object``'s ``name in
        self._world.objects`` guard. The insert therefore reaches MuJoCo, whose
        ``add_body`` raises ``repeated name`` *but still inserts the duplicate*.
        The rollback used to resolve the name through ``spec.body(name)``, which
        sees the ORIGINAL body (present at the last compile), so it deleted the
        healthy scene body and left the empty orphan holding its name: the next
        mutation recompiled successfully with the original geometry gone - a
        regression from fail-loud to corrupt-quietly. The add is now atomic over
        its own mutation, so the pre-existing body keeps its pose and geom and a
        later add still succeeds.
        """
        scene = tmp_path / "scene.xml"
        scene.write_text(
            '<mujoco model="s"><worldbody>'
            '<body name="table" pos="1 2 0.5"><geom type="box" size="0.2 0.2 0.2"/></body>'
            "</worldbody></mujoco>"
        )
        sim.load_scene(str(scene))
        model = sim._world._model
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        pos_before = list(model.body_pos[table_id])
        geomnum_before = int(model.body_geomnum[table_id])
        assert geomnum_before == 1

        collide = sim.add_object("table", shape="box", position=[0.0, 0.0, 0.6], mass=1.0)
        assert collide["status"] == "error", collide
        assert "repeated name" in collide["content"][0]["text"]

        # A later, unrelated add must recompile - and the original 'table' must
        # still be at its scene pose with its geom, not silently replaced by the
        # empty orphan of the rejected call.
        later = sim.add_object("cube", shape="box", position=[0.3, 0.0, 0.6], mass=1.0)
        assert later["status"] == "success", later

        model = sim._world._model
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        assert list(model.body_pos[table_id]) == pos_before
        assert int(model.body_geomnum[table_id]) == geomnum_before
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0
