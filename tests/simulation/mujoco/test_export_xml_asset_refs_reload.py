"""``export_xml`` output must be reloadable, by MuJoCo and by strands itself.

MuJoCo resolves a ``<mesh>`` / ``<texture>`` ``file=`` against a base directory
it tracks on the spec (``modelfiledir`` plus ``meshdir`` / ``texturedir``) and
does NOT emit in ``spec.to_xml()``; ``spec.attach()`` does not carry that base
onto the parent either. A scene composed from model files therefore compiled and
stepped normally while ``export_xml`` wrote asset references that resolve
against wherever the XML landed - so the exported "canonical MJCF" could not be
recompiled, by ``MjModel.from_xml_path`` or by ``load_scene``, even though
``describe()`` advertises ``export_xml`` as "the read sibling of
``replace_scene_mjcf``".

A multi-robot scene has no single ``meshdir`` that could paper over it: each
model contributes assets from its own root. The fix makes each reference carry
its own absolute location, so the exported text is self-describing.

The tests build their models on disk (a real one-triangle STL and a real PNG
under a ``meshdir`` / ``texturedir``) so the reference genuinely needs resolving
and nothing is downloaded.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder  # noqa: E402


def _tetrahedron_stl() -> bytes:
    """A minimal solid binary STL: MuJoCo requires at least 4 vertices.

    Binary STL layout: 80-byte header, uint32 facet count, then 50 bytes per
    facet (a normal and three vertices as float32, plus a uint16 attribute
    word). The facet normals are left zero - MuJoCo derives its own.
    """
    corners = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    body = b"".join(struct.pack("<12fH", 0.0, 0.0, 0.0, *corners[a], *corners[b], *corners[c], 0) for a, b, c in faces)
    return b"\x00" * 80 + struct.pack("<I", len(faces)) + body


_STL = _tetrahedron_stl()


def _dims(model: Any) -> tuple[int, ...]:
    """Structural fingerprint of a compiled model, order-independent of naming."""
    return (model.nbody, model.njnt, model.nu, model.nq, model.nv, model.ngeom)


def _write_mesh_model(root: Path, *, meshdir: str = "assets") -> Path:
    """Write a model whose mesh lives under *meshdir*, relative to the model."""
    (root / meshdir).mkdir(parents=True, exist_ok=True)
    (root / meshdir / "block.stl").write_bytes(_STL)
    model = root / "arm.xml"
    model.write_text(
        f"""
        <mujoco model="meshy">
          <compiler meshdir="{meshdir}"/>
          <asset><mesh name="block" file="block.stl"/></asset>
          <worldbody>
            <body name="link" pos="0 0 0.2">
              <joint name="hinge" type="hinge" axis="0 1 0"/>
              <geom name="visual" type="mesh" mesh="block"/>
            </body>
          </worldbody>
          <actuator><position name="drive" joint="hinge" kp="20"/></actuator>
        </mujoco>
        """
    )
    return model


def _write_texture_model(root: Path) -> Path:
    """Write a model carrying a file-backed texture under its own texturedir."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow renders the test texture")
    (root / "tex").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (200, 40, 40)).save(root / "tex" / "skin.png")
    model = root / "painted.xml"
    model.write_text(
        """
        <mujoco model="painted">
          <compiler texturedir="tex"/>
          <asset>
            <texture name="skin" type="2d" file="skin.png"/>
            <material name="painted" texture="skin"/>
          </asset>
          <worldbody>
            <body name="slab" pos="0 0 0.3">
              <joint name="slide" type="slide" axis="0 0 1"/>
              <geom name="face" type="box" size="0.1 0.1 0.1" material="painted"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    return model


@pytest.fixture
def sim():
    s = Simulation(tool_name="export_assets", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestTheExportedSceneReloads:
    def test_a_multi_root_scene_reloads_with_the_same_structure(self, sim: Simulation, tmp_path: Path) -> None:
        """Two models from DIFFERENT roots: no single meshdir could fix this.

        The exported XML is written to a third directory that holds no assets at
        all, which is the whole point - a portable export must not depend on
        where it is written.
        """
        first = _write_mesh_model(tmp_path / "robot_one")
        second = _write_mesh_model(tmp_path / "robot_two", meshdir="geometry")
        sim.create_world()
        assert sim.add_robot(name="a", urdf_path=str(first))["status"] == "success"
        assert sim.add_robot(name="b", urdf_path=str(second))["status"] == "success"
        assert (
            sim.add_object(name="box", shape="box", position=[0.3, 0, 0.05], size=[0.05, 0.05, 0.05])["status"]
            == "success"
        )
        live = _dims(sim.mj_model)

        destination = tmp_path / "elsewhere" / "scene.xml"
        assert sim.export_xml(output_path=str(destination))["status"] == "success"

        reloaded = mujoco.MjModel.from_xml_path(str(destination))
        assert _dims(reloaded) == live, f"exported scene recompiled to {_dims(reloaded)} but the live model is {live}"

    def test_the_export_is_loadable_by_load_scene(self, sim: Simulation, tmp_path: Path) -> None:
        """describe() calls export_xml the read sibling of replace_scene_mjcf.

        strands' own loader is the consumer that pairing implies, so it must be
        able to read what strands wrote.
        """
        model = _write_mesh_model(tmp_path / "src")
        sim.create_world()
        assert sim.add_robot(name="arm", urdf_path=str(model))["status"] == "success"
        live = _dims(sim.mj_model)
        assert sim.mj_model.nmesh == 1, "premise: the scene must actually carry a file-backed mesh"
        destination = tmp_path / "handoff" / "scene.xml"
        sim.export_xml(output_path=str(destination))

        reader = Simulation(tool_name="reader", mesh=False)
        try:
            result = reader.load_scene(scene_path=str(destination))
            assert result["status"] == "success", result["content"][0]["text"]
            assert _dims(reader.mj_model) == live
        finally:
            reader.cleanup(policy_stop_timeout=0.5)

    def test_a_file_backed_texture_reloads_too(self, sim: Simulation, tmp_path: Path) -> None:
        """Textures resolve against texturedir, not meshdir - both are lost."""
        model = _write_texture_model(tmp_path / "painted_src")
        sim.create_world()
        assert sim.add_robot(name="slab", urdf_path=str(model))["status"] == "success"
        live = _dims(sim.mj_model)

        destination = tmp_path / "out" / "scene.xml"
        sim.export_xml(output_path=str(destination))
        assert _dims(mujoco.MjModel.from_xml_path(str(destination))) == live

    def test_a_scene_loaded_from_a_file_exports_reloadably(self, sim: Simulation, tmp_path: Path) -> None:
        """load_scene keeps the model's meshdir but not the directory it is
        relative TO, so the single-spec path loses the assets the same way."""
        model = _write_mesh_model(tmp_path / "scene_src")
        assert sim.load_scene(scene_path=str(model))["status"] == "success"
        live = _dims(sim.mj_model)

        destination = tmp_path / "exported" / "scene.xml"
        assert sim.export_xml(output_path=str(destination))["status"] == "success"
        assert _dims(mujoco.MjModel.from_xml_path(str(destination))) == live


class TestTheRepairStaysWithinItsRemit:
    """Controls: these hold both before and after the fix."""

    def test_a_scene_with_no_file_backed_assets_is_unaffected(self, sim: Simulation, tmp_path: Path) -> None:
        """The common case (primitives only) never had this defect and must
        keep exporting exactly as before."""
        sim.create_world()
        sim.add_object(name="box", shape="box", position=[0, 0, 0.1], size=[0.1, 0.1, 0.1])
        live = _dims(sim.mj_model)
        destination = tmp_path / "plain.xml"
        assert sim.export_xml(output_path=str(destination))["status"] == "success"
        assert _dims(mujoco.MjModel.from_xml_path(str(destination))) == live

    def test_an_unresolvable_reference_is_left_as_authored(self, tmp_path: Path) -> None:
        """A reference that resolves to nothing must not be rewritten.

        MuJoCo's own "Error opening file" names what the model declares; an
        invented absolute path would report a location that was never on disk.
        """
        model = tmp_path / "ghost.xml"
        model.write_text(
            """
            <mujoco model="ghost">
              <compiler meshdir="assets"/>
              <asset><mesh name="absent" file="not_here.stl"/></asset>
              <worldbody><geom name="g" type="plane" size="1 1 0.1"/></worldbody>
            </mujoco>
            """
        )
        spec = SpecBuilder.from_file(str(model))
        assert [m.file for m in spec.meshes] == ["not_here.stl"]

    def test_a_reference_already_absolute_is_preserved(self, tmp_path: Path) -> None:
        """An absolute reference is already self-describing; leave it alone."""
        stl = tmp_path / "solo.stl"
        stl.write_bytes(_STL)
        model = tmp_path / "abs.xml"
        model.write_text(
            f"""
            <mujoco model="abs">
              <asset><mesh name="solo" file="{stl}"/></asset>
              <worldbody><geom name="g" type="mesh" mesh="solo"/></worldbody>
            </mujoco>
            """
        )
        spec = SpecBuilder.from_file(str(model))
        assert [m.file for m in spec.meshes] == [str(stl)]
