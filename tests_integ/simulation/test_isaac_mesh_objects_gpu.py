"""GPU-gated integration: mesh objects on the Isaac stage (#2459).

The behaviour half of the retired ``mesh_path`` refusal: ``add_object``
places a real OBJ-sourced mesh prim (converted to USD, convex-hull
collision), and a LIBERO-shaped MJCF scene realizes a mesh-carrying object
with the mesh as its visual and the collision-AABB cube as its invisible
physics proxy - so a pixel-conditioned policy evaluated on this backend
observes the object it was trained on rather than a gray box.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ tests_integ/simulation/test_isaac_mesh_objects_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

_TETRA_OBJ = "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"

_MESH_SCENE = """
<mujoco model="mesh_scene_probe">
  <asset>
    <mesh name="bowl_mesh" file="{mesh_file}"/>
  </asset>
  <worldbody>
    <body name="fixture_table" pos="0.0 0.0 0.4">
      <geom type="box" size="0.5 0.5 0.4"/>
    </body>
    <body name="bowl_1_main" pos="0.1 0.0 0.9">
      <joint type="free"/>
      <geom type="box" group="0" size="0.05 0.05 0.03"/>
      <geom mesh="bowl_mesh" group="1" type="mesh"/>
    </body>
  </worldbody>
</mujoco>
"""


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _json_payload(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if "json" in c)


class TestIsaacMeshObjectsGPU:
    def test_mesh_add_object_and_scene_visuals_end_to_end(self, tmp_path):
        """One Kit boot covers both journeys (SimulationApp is a process-wide
        singleton, same session-sharing discipline as the other GPU tests):

        1. ``add_object(shape='mesh')`` lands a visible mesh prim whose
           reported extent comes from the asset;
        2. ``load_scene`` on a mesh-carrying MJCF realizes the object with a
           mesh visual + invisible AABB collision cube and reports it under
           ``mesh_visuals`` with no cross-backend caveat for it.
        """
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        asset = tmp_path / "bowl.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            assert sim.create_world()["status"] == "success"

            # Journey 1: a direct mesh add.
            result = sim.add_object(
                "widget",
                shape="mesh",
                mesh_path=str(asset),
                position=[0.4, 0.0, 0.2],
                # Ignored for shape="mesh" - the asset's own units define the
                # extent. Wrong on every axis so the discard is unambiguous, and
                # the same value tests/simulation/
                # test_mesh_size_docs_match_backend_divergence.py uses to prove
                # Newton *does* consume it.
                size=[2.0, 3.0, 4.0],
                is_static=True,
            )
            assert result["status"] == "success", result
            payload = _json_payload(result)
            assert payload["shape"] == "mesh"
            # The asset's extent, not the request.
            assert payload["size"] == pytest.approx([0.1, 0.1, 0.1])

            import omni.usd
            from pxr import UsdGeom  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(payload["prim_path"])
            assert prim.IsValid()
            # The referenced geometry is a real Mesh prim, not a primitive.
            mesh_prims = [p for p in [prim, *prim.GetChildren()] if p.IsA(UsdGeom.Mesh)]
            assert mesh_prims, f"no Mesh prim at/under {payload['prim_path']}"

            # Journey 2: LIBERO-shaped scene realization.
            scene = tmp_path / "scene.xml"
            scene.write_text(_MESH_SCENE.format(mesh_file=str(asset)), encoding="utf-8")
            result = sim.load_scene(str(scene))
            assert result["status"] == "success", result
            payload = _json_payload(result)
            assert payload["mesh_visuals"] == ["bowl_1_main"]
            assert payload["box_proxies"] == ["fixture_table"]

            root = stage.GetPrimAtPath("/World/Objects/bowl_1_main")
            assert root.IsValid()
            visual = stage.GetPrimAtPath("/World/Objects/bowl_1_main/visual")
            assert visual.IsValid()
            collision = stage.GetPrimAtPath("/World/Objects/bowl_1_main/collision")
            assert collision.IsValid()
            # The proxy cube is invisible; the mesh visual is not.
            assert UsdGeom.Imageable(collision).ComputeVisibility() == UsdGeom.Tokens.invisible
            assert UsdGeom.Imageable(visual).ComputeVisibility() != UsdGeom.Tokens.invisible

            # Physics still integrates with the composite prim on stage.
            assert sim.step(5)["status"] == "success"
        finally:
            sim.destroy()
