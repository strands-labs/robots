"""LIBERO scene objects carry their real meshes onto the Isaac stage (#2459).

Two halves, both CPU-only:

* The parse half: :func:`load_mjcf_scene_objects` resolves each body's visual
  mesh asset (path + scale + body-frame pose) from the MJCF ``<asset>``
  registry, prefers a visual-group mesh over a collision one, uses the mesh's
  own bounds when a body has no analytic collision geometry (instead of the
  historical hardcoded 0.05 m box), and raises for a declared-but-missing
  asset file - never a silent box.
* The realization half: ``IsaacSimulation.load_scene`` routes mesh-carrying
  objects through ``_add_scene_mesh_object`` (mesh visual + AABB collision
  proxy) and proxy-only objects through the unchanged ``add_object`` box
  path, and its report separates the two - with the cross-backend
  comparability caveat present exactly when box proxies remain.

The live-Kit half (a mesh actually visible in an RTX camera frame) runs on
GPU in ``tests_integ/``.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects
from strands_robots.simulation.isaac.simulation import IsaacSimulation

_TETRA_OBJ = "v 0 0 0\nv 0.1 0 0\nv 0 0.2 0\nv 0 0 0.3\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def _write_scene(tmp_path, body_xml: str, mesh_scale: str = "1 1 1") -> str:
    (tmp_path / "bowl.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    scene = f"""
    <mujoco model="probe">
      <asset>
        <mesh name="bowl_mesh" file="bowl.obj" scale="{mesh_scale}"/>
      </asset>
      <worldbody>
        {body_xml}
      </worldbody>
    </mujoco>
    """
    p = tmp_path / "scene.xml"
    p.write_text(scene, encoding="utf-8")
    return str(p)


class TestLoaderMeshPassthrough:
    def test_visual_mesh_rides_along_with_collision_aabb(self, tmp_path):
        scene = _write_scene(
            tmp_path,
            """
            <body name="bowl_1_main" pos="0.1 0.2 0.8">
              <joint type="free"/>
              <geom type="box" group="0" size="0.05 0.05 0.02"/>
              <geom mesh="bowl_mesh" group="1" type="mesh" pos="0 0 0.01"/>
            </body>
            """,
        )
        (obj,) = load_mjcf_scene_objects(scene)
        # Collision proxy unchanged: the analytic box AABB, not the mesh.
        assert obj.size == (0.1, 0.1, 0.04)
        assert obj.mesh_path is not None and obj.mesh_path.endswith("bowl.obj")
        assert obj.mesh_pos == (0.0, 0.0, 0.01)
        assert obj.mesh_scale == (1.0, 1.0, 1.0)

    def test_visual_group_mesh_preferred_over_collision_mesh(self, tmp_path):
        (tmp_path / "hull.obj").write_text(_TETRA_OBJ, encoding="utf-8")
        (tmp_path / "bowl.obj").write_text(_TETRA_OBJ, encoding="utf-8")
        scene = tmp_path / "scene.xml"
        scene.write_text(
            """
            <mujoco model="probe">
              <asset>
                <mesh name="hull_mesh" file="hull.obj"/>
                <mesh name="bowl_mesh" file="bowl.obj"/>
              </asset>
              <worldbody>
                <body name="bowl_1_main" pos="0 0 0.5">
                  <joint type="free"/>
                  <geom mesh="hull_mesh" group="0" type="mesh"/>
                  <geom mesh="bowl_mesh" group="1" type="mesh"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        (obj,) = load_mjcf_scene_objects(str(scene))
        # The visual asset is the one a pixel policy was trained on.
        assert obj.mesh_path is not None and obj.mesh_path.endswith("bowl.obj")

    def test_mesh_only_body_gets_mesh_bounds_not_the_5cm_fallback(self, tmp_path):
        scene = _write_scene(
            tmp_path,
            """
            <body name="bowl_1_main" pos="0 0 0.5">
              <joint type="free"/>
              <geom mesh="bowl_mesh" group="1" type="mesh"/>
            </body>
            """,
            mesh_scale="2 1 1",
        )
        (obj,) = load_mjcf_scene_objects(scene)
        # Tetra extents (0.1, 0.2, 0.3) with the asset's scale applied on x,
        # not the historical hardcoded (0.05, 0.05, 0.05) box.
        assert obj.size == pytest.approx((0.2, 0.2, 0.3))
        assert obj.mesh_scale == (2.0, 1.0, 1.0)

    def test_missing_mesh_file_is_an_error_never_a_silent_box(self, tmp_path):
        scene = _write_scene(
            tmp_path,
            """
            <body name="bowl_1_main" pos="0 0 0.5">
              <joint type="free"/>
              <geom type="box" group="0" size="0.05 0.05 0.02"/>
              <geom mesh="bowl_mesh" group="1" type="mesh"/>
            </body>
            """,
        )
        (tmp_path / "bowl.obj").unlink()
        with pytest.raises(ValueError, match="missing on disk"):
            load_mjcf_scene_objects(scene)

    def test_unregistered_mesh_reference_degrades_to_the_box_fallback(self, tmp_path):
        # A mesh reference with no <asset> entry cannot occur in a
        # robosuite-COMPILED scene (MuJoCo refuses to compile it), so it is
        # hand-written scaffolding: the historical degrade-cleanly default
        # box is preserved (see test_description_loader_geometry.py).
        scene = tmp_path / "scene.xml"
        scene.write_text(
            """
            <mujoco model="probe">
              <worldbody>
                <body name="bowl_1_main" pos="0 0 0.5">
                  <geom mesh="ghost_mesh" type="mesh"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        (obj,) = load_mjcf_scene_objects(str(scene))
        assert obj.mesh_path is None
        assert obj.size == (0.05, 0.05, 0.05)

    def test_unconvertible_format_stays_a_box_proxy(self, tmp_path):
        # A format outside the converter's vocabulary (e.g. a COLLADA .dae)
        # cannot be converted; the object keeps
        # its analytic proxy and mesh_path stays None (load_scene reports it
        # among the proxies rather than failing the whole scene).
        (tmp_path / "bowl.dae").write_bytes(b"\x00\x01")
        scene = tmp_path / "scene.xml"
        scene.write_text(
            """
            <mujoco model="probe">
              <asset>
                <mesh name="bowl_mesh" file="bowl.dae"/>
              </asset>
              <worldbody>
                <body name="bowl_1_main" pos="0 0 0.5">
                  <geom type="box" group="0" size="0.05 0.05 0.02"/>
                  <geom mesh="bowl_mesh" group="1" type="mesh"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        (obj,) = load_mjcf_scene_objects(str(scene))
        assert obj.mesh_path is None
        assert obj.size == (0.1, 0.1, 0.04)

    def test_meshdir_resolution(self, tmp_path):
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "bowl.obj").write_text(_TETRA_OBJ, encoding="utf-8")
        scene = tmp_path / "scene.xml"
        scene.write_text(
            """
            <mujoco model="probe">
              <compiler meshdir="assets"/>
              <asset>
                <mesh name="bowl_mesh" file="bowl.obj"/>
              </asset>
              <worldbody>
                <body name="bowl_1_main" pos="0 0 0.5">
                  <geom type="box" group="0" size="0.05 0.05 0.02"/>
                  <geom mesh="bowl_mesh" group="1" type="mesh"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        (obj,) = load_mjcf_scene_objects(str(scene))
        assert obj.mesh_path == str(assets / "bowl.obj")

    def test_scene_without_meshes_is_unchanged(self, tmp_path):
        # The pre-#2459 contract for analytic scenes holds verbatim.
        scene = tmp_path / "scene.xml"
        scene.write_text(
            """
            <mujoco model="probe">
              <worldbody>
                <body name="fixture_table" pos="0 0 0.4">
                  <geom type="box" size="0.5 0.5 0.4"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        (obj,) = load_mjcf_scene_objects(str(scene))
        assert obj.mesh_path is None
        assert obj.mesh_scale == (1.0, 1.0, 1.0)
        assert obj.size == (1.0, 1.0, 0.8)


# --- load_scene dispatch + report ------------------------------------------
#
# Same __new__-skeleton pattern as test_load_scene_physics_view.py: the prim
# mutation surface is mocked, so these pin the routing and the report, not
# the USD authoring (that half is GPU-only and lives in tests_integ/).


def _make_engine() -> tuple[IsaacSimulation, dict[str, list[str]]]:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()

    class _World:
        physics_sim_view = object()

        def play(self) -> None:
            pass

    engine._world = _World()
    engine._world_created = True
    engine._objects = {}
    engine._robots = {}
    engine._scene_objects = set()
    engine._prim_registry = []
    calls: dict[str, list[str]] = {"box": [], "mesh": []}

    def _fake_add_object(name: str, **kwargs: Any) -> dict[str, Any]:
        calls["box"].append(name)
        return {"status": "success", "content": [{"text": f"Object '{name}' added."}]}

    def _fake_add_scene_mesh_object(obj: Any) -> dict[str, Any]:
        calls["mesh"].append(obj.name)
        return {"status": "success", "content": [{"text": f"Scene object '{obj.name}' added."}]}

    setattr(engine, "add_object", _fake_add_object)  # noqa: B010 - mypy-safe method shadow
    setattr(engine, "_add_scene_mesh_object", _fake_add_scene_mesh_object)  # noqa: B010
    return engine, calls


def _mixed_scene(tmp_path) -> str:
    (tmp_path / "bowl.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
        <mujoco model="probe">
          <asset>
            <mesh name="bowl_mesh" file="bowl.obj"/>
          </asset>
          <worldbody>
            <body name="fixture_table" pos="0 0 0.4">
              <geom type="box" size="0.5 0.5 0.4"/>
            </body>
            <body name="bowl_1_main" pos="0.1 0 0.85">
              <joint type="free"/>
              <geom type="box" group="0" size="0.02 0.02 0.02"/>
              <geom mesh="bowl_mesh" group="1" type="mesh"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return str(scene)


class TestLoadSceneMeshDispatch:
    def test_mesh_objects_route_to_the_mesh_realizer(self, tmp_path):
        engine, calls = _make_engine()
        result = engine.load_scene(_mixed_scene(tmp_path))
        assert result["status"] == "success"
        assert calls["mesh"] == ["bowl_1_main"]
        assert calls["box"] == ["fixture_table"]
        payload = result["content"][0]["json"]
        assert payload["mesh_visuals"] == ["bowl_1_main"]
        assert payload["box_proxies"] == ["fixture_table"]
        assert sorted(payload["realized"]) == ["bowl_1_main", "fixture_table"]

    def test_report_carries_the_caveat_while_proxies_remain(self, tmp_path):
        # Eval-integrity interim note (#2459, deliverable 4): while any
        # object renders as a proxy, the report says pixel-conditioned
        # scores are not cross-backend comparable.
        engine, _ = _make_engine()
        result = engine.load_scene(_mixed_scene(tmp_path))
        payload = result["content"][0]["json"]
        assert payload["visual_caveat"] is not None
        assert "not comparable across backends" in payload["visual_caveat"]
        assert payload["visual_caveat"] in result["content"][0]["text"]

    def test_caveat_absent_when_every_object_has_a_mesh_visual(self, tmp_path):
        (tmp_path / "bowl.obj").write_text(_TETRA_OBJ, encoding="utf-8")
        scene = tmp_path / "scene.xml"
        scene.write_text(
            """
            <mujoco model="probe">
              <asset>
                <mesh name="bowl_mesh" file="bowl.obj"/>
              </asset>
              <worldbody>
                <body name="bowl_1_main" pos="0.1 0 0.85">
                  <joint type="free"/>
                  <geom type="box" group="0" size="0.02 0.02 0.02"/>
                  <geom mesh="bowl_mesh" group="1" type="mesh"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        engine, _ = _make_engine()
        result = engine.load_scene(str(scene))
        payload = result["content"][0]["json"]
        assert payload["visual_caveat"] is None
        assert payload["box_proxies"] == []
        assert "not comparable" not in result["content"][0]["text"]

    def test_failed_mesh_realization_is_a_loud_skip(self, tmp_path):
        engine, _calls = _make_engine()

        def _failing(obj: Any) -> dict[str, Any]:
            return {"status": "error", "content": [{"text": f"conversion failed for '{obj.name}'"}]}

        setattr(engine, "_add_scene_mesh_object", _failing)  # noqa: B010
        result = engine.load_scene(_mixed_scene(tmp_path))
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["mesh_visuals"] == []
        skipped = {entry["name"]: entry["reason"] for entry in payload["skipped"]}
        assert "bowl_1_main" in skipped
        assert "conversion failed" in skipped["bowl_1_main"]
