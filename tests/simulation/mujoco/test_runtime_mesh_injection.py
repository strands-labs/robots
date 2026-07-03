"""Runtime injection of ``shape="mesh"`` objects into a live MuJoCo scene.

A mesh geom names a mesh asset that must be registered on the ``MjSpec`` before
the geom can compile. The full-scene ``SpecBuilder.build`` registers those
assets in a dedicated pass, but the incremental scene-edit path
(``inject_object_into_scene`` -> ``SpecBuilder.add_object``) historically did
not, so ``Simulation.add_object(shape="mesh", mesh_path=...)`` at runtime always
failed to recompile - even for a valid mesh file - and reported the opaque
"spec recompile refused". These tests pin the behaviour that the incremental
path now registers the mesh, that a missing ``mesh_path`` fails fast with an
actionable message, and that a failed add rolls the mesh asset back out so the
name stays reusable and the scene is not bricked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A minimal but valid triangle mesh (tetrahedron) MuJoCo can load.
_TETRA_OBJ = """v 0.0 0.0 0.0
v 0.05 0.0 0.0
v 0.0 0.05 0.0
v 0.0 0.0 0.05
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


@pytest.fixture
def mesh_file(tmp_path: Path) -> str:
    p = tmp_path / "tetra.obj"
    p.write_text(_TETRA_OBJ)
    return str(p)


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_runtime_mesh", mesh=False)
    s.create_world()
    s.add_robot("so100")  # a compiled world the injector can recompile against
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestRuntimeMeshInjection:
    def test_valid_mesh_object_is_added(self, sim, mesh_file):
        """add_object(shape='mesh', mesh_path=<valid>) succeeds and registers the asset.

        Pre-fix this returned an error ("spec recompile refused") because the
        incremental path never called spec.add_mesh.
        """
        result = sim.add_object(name="widget", shape="mesh", mesh_path=mesh_file, position=[0.3, 0.0, 0.1])
        assert result["status"] == "success"
        assert "widget" in sim._world.objects

        import mujoco as mj

        model = sim._world._model
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "widget") >= 0
        # The mesh asset the geom references is registered on the live spec.
        spec = sim._world._backend_state["spec"]
        assert "mesh_widget" in [m.name for m in spec.meshes]

    def test_mesh_without_path_fails_fast(self, sim):
        """A mesh with no mesh_path is rejected up front with an actionable error."""
        result = sim.add_object(name="nopath", shape="mesh")
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "mesh_path" in text
        # Fail-fast means the scene was never mutated.
        assert "nopath" not in sim._world.objects

    def test_bad_mesh_rolls_back_asset_and_name_is_reusable(self, sim, mesh_file):
        """A mesh file that fails to compile rolls the asset back; the name is reusable.

        A leaked mesh asset would collide on a duplicate mesh name when the same
        object name is retried with a valid file, so a successful mesh retry
        proves the rollback removed the orphan asset.
        """
        bad = sim.add_object(name="widget", shape="mesh", mesh_path="/nonexistent/does-not-exist.stl")
        assert bad["status"] == "error"
        assert "widget" not in sim._world.objects

        spec = sim._world._backend_state["spec"]
        assert "mesh_widget" not in [m.name for m in spec.meshes]

        retry = sim.add_object(name="widget", shape="mesh", mesh_path=mesh_file, position=[0.3, 0.0, 0.1])
        assert retry["status"] == "success"
        assert "widget" in sim._world.objects

    def test_remove_then_readd_mesh_under_same_name(self, sim, mesh_file):
        """Removing a mesh object frees its asset so the same name can be re-added.

        remove_object must delete the f"mesh_{name}" asset, otherwise a re-add
        collides on the duplicate mesh name at recompile.
        """
        assert sim.add_object(name="widget", shape="mesh", mesh_path=mesh_file)["status"] == "success"
        assert sim.remove_object("widget")["status"] == "success"

        spec = sim._world._backend_state["spec"]
        assert "mesh_widget" not in [m.name for m in spec.meshes]

        readd = sim.add_object(name="widget", shape="mesh", mesh_path=mesh_file, position=[0.2, 0.1, 0.1])
        assert readd["status"] == "success"

    def test_primitive_add_after_bad_mesh_still_works(self, sim):
        """A failed mesh add must not brick later primitive adds (scene stays compilable)."""
        bad = sim.add_object(name="bad", shape="mesh", mesh_path="/nope.stl")
        assert bad["status"] == "error"
        ok = sim.add_object(name="cube", shape="box", position=[0.2, 0.1, 0.1])
        assert ok["status"] == "success"
        assert "cube" in sim._world.objects
