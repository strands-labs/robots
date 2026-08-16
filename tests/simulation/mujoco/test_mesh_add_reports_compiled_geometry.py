"""``add_object(shape="mesh")`` reports the geom it compiled, not the request.

A mesh consumes no ``size`` component -- ``_SIZE_LAYOUT["mesh"]`` is ``0``, and
``_normalize_size`` maps it to ``[]`` -- yet the success text echoed ``size=``
back. So the one shape whose extent the request never carries was the one shape
whose extent the result asserted: an omitted ``size`` reported ``[0.05, 0.05,
0.05]`` for an asset of any size (a 3.5 x 10.5 x 12.5 m generated room shell
reported as a 5 cm object), and an explicit vector was echoed as though it had
been honoured. ``_validate_size`` already records why that shape of report is a
defect for primitives -- a short vector "compiled a differently-sized object
while reporting success and echoed the requested ``[0.5]``" -- and the mesh row
is where the echo is wrong unconditionally, because zero components are consumed.

The result now reports the extent read back off the compiled geom
(:func:`~strands_robots.simulation.mujoco.simulation._compiled_geom_extent`,
over MuJoCo's own ``geom_aabb``) and names the collision geometry, which for
every mesh geom is its convex hull rather than the triangles that render. Both
are properties of the asset that no request component defines, and both are what
a caller placing a robot or an object against the asset needs.

Primitive shapes are unchanged: there ``size`` *is* the extent and the geom
compiles to it, so it stays echoed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    Simulation,
    _compiled_geom_extent,
)

# An open-topped channel: a thin floor plate spanned by two tall side walls.
# Concave by construction -- its convex hull is the full 0.3 x 0.4 x 0.4 m box,
# which fills the cavity between the walls. Written as an explicit triangle soup
# so the fixture needs no mesh library.
_CHANNEL_BOXES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((-0.20, -0.20, 0.00), (0.20, 0.20, 0.02)),  # floor plate, 2 cm thick
    ((-0.20, -0.20, 0.00), (-0.18, 0.20, 0.30)),  # left wall, 30 cm tall
    ((0.18, -0.20, 0.00), (0.20, 0.20, 0.30)),  # right wall
)
_CHANNEL_EXTENT = (0.3, 0.4, 0.4)
_CHANNEL_CAVITY_FLOOR_Z = 0.02
_CHANNEL_WALL_TOP_Z = 0.30


def _write_box_soup(
    path: Path,
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> str:
    """Write ``boxes`` (``(lo, hi)`` corner pairs) as one OBJ triangle soup."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for lo, hi in boxes:
        base = len(verts) + 1  # OBJ indices are 1-based
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    verts.append(
                        (
                            hi[0] if sx else lo[0],
                            hi[1] if sy else lo[1],
                            hi[2] if sz else lo[2],
                        )
                    )
        corner = {(sx, sy, sz): base + sx * 4 + sy * 2 + sz for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)}
        for quad in (
            ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
            ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
            ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
            ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
        ):
            a, b, c, d = (corner[k] for k in quad)
            faces.append((a, b, c))
            faces.append((a, c, d))
    path.write_text(
        "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts) + "".join(f"f {a} {b} {c}\n" for a, b, c in faces),
        encoding="utf-8",
    )
    return str(path)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(block["text"] for block in result["content"] if "text" in block)


def _reported_extent(result: dict[str, Any]) -> list[float]:
    """Parse the ``extent=[...]`` vector out of an add_object success text."""
    match = re.search(r"extent=\[([^\]]+)\]", _text(result))
    assert match is not None, f"no extent reported: {_text(result)!r}"
    return [float(part) for part in match.group(1).split(",")]


@pytest.fixture
def channel_mesh(tmp_path: Path) -> str:
    return _write_box_soup(tmp_path / "channel.obj", _CHANNEL_BOXES)


@pytest.fixture
def sim():
    engine = Simulation(tool_name="test_mesh_add_reports_compiled_geometry", mesh=False)
    assert engine.create_world()["status"] == "success"
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


class TestAMeshReportsTheAssetsExtent:
    """The extent comes off the compiled geom, because the request has none."""

    def test_an_omitted_size_reports_the_assets_extent_not_the_5cm_default(self, sim, channel_mesh: str) -> None:
        """Pre-fix: ``size=[0.05, 0.05, 0.05]`` for a 0.3 x 0.4 x 0.4 m asset."""
        result = sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)
        assert result["status"] == "success", _text(result)
        assert _reported_extent(result) == pytest.approx(_CHANNEL_EXTENT, abs=1e-3)
        assert "size=" not in _text(result)

    def test_an_explicit_size_is_not_echoed_as_though_it_were_honoured(self, sim, channel_mesh: str) -> None:
        """A mesh consumes no component, so an echoed vector reports a fiction."""
        result = sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, size=[2.0, 2.0, 2.0])
        assert result["status"] == "success", _text(result)
        assert "2.0" not in _text(result)
        assert _reported_extent(result) == pytest.approx(_CHANNEL_EXTENT, abs=1e-3)

    def test_the_message_names_the_convex_hull_as_the_collision_geometry(self, sim, channel_mesh: str) -> None:
        """The property that makes a concave asset behave unlike it renders."""
        result = sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)
        assert "convex hull" in _text(result)


class TestPrimitiveShapesAreUnchanged:
    """The no-overreach control: ``size`` is the extent for every other shape."""

    def test_a_box_still_reports_the_size_it_compiled_to(self, sim) -> None:
        result = sim.add_object("crate", shape="box", size=[0.2, 0.3, 0.4])
        assert result["status"] == "success", _text(result)
        assert "size=[0.2, 0.3, 0.4]" in _text(result)
        assert "convex hull" not in _text(result)

    def test_a_sphere_still_reports_its_one_component_size(self, sim) -> None:
        result = sim.add_object("ball", shape="sphere", size=[0.06])
        assert result["status"] == "success", _text(result)
        assert "size=[0.06]" in _text(result)


class TestTheReportedGeometryIsTheGeometryThatActsOnObjects:
    """Both halves of the report are checked against the compiled model."""

    def test_the_reported_extent_matches_the_geoms_own_bounding_box(self, sim, channel_mesh: str) -> None:
        assert sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)["status"] == "success"
        extent = _compiled_geom_extent(sim._mj, sim.mj_model, "channel_geom")
        assert extent == pytest.approx(_CHANNEL_EXTENT, abs=1e-3)

    def test_a_primitives_extent_reads_back_as_the_size_that_was_asked_for(self, sim) -> None:
        """The same read is correct for every shape, which is why it is trusted."""
        assert sim.add_object("crate", shape="box", size=[0.2, 0.3, 0.4])["status"] == "success"
        extent = _compiled_geom_extent(sim._mj, sim.mj_model, "crate_geom")
        assert extent == pytest.approx([0.2, 0.3, 0.4], abs=1e-3)

    def test_an_unresolvable_geom_reports_no_extent_rather_than_a_wrong_one(self, sim) -> None:
        assert _compiled_geom_extent(sim._mj, sim.mj_model, "never_added_geom") is None

    def test_a_non_string_geom_name_reports_no_extent_rather_than_crashing(self, sim) -> None:
        """The lookup routes through ``mj_name_to_id``, so it inherits its guard.

        Reaching ``mujoco.mj_name2id`` with a non-string name terminates the
        interpreter with SIGSEGV rather than raising, which no envelope can
        recover from.
        """
        not_a_name: Any = None
        assert _compiled_geom_extent(sim._mj, sim.mj_model, not_a_name) is None

    def test_a_concave_asset_collides_as_its_filled_hull(self, sim, channel_mesh: str) -> None:
        """The documented consequence, pinned so the warning cannot go stale.

        Dropped into the channel's open cavity, a ball comes to rest on the
        convex hull that spans the wall tops rather than on the interior floor
        2 cm up. This is MuJoCo's mesh-collision contract, not a defect -- the
        test exists so the sentence that documents it stays true.
        """
        import mujoco

        assert sim.add_object("channel", shape="mesh", mesh_path=channel_mesh, is_static=True)["status"] == "success"
        assert (
            sim.add_object("ball", shape="sphere", size=[0.06], position=[0.0, 0.0, 0.45], mass=0.05)["status"]
            == "success"
        )
        model, data = sim.mj_model, sim.mj_data
        for _ in range(3000):
            mujoco.mj_step(model, data)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        rest_z = float(data.xpos[body_id][2])
        assert rest_z > _CHANNEL_WALL_TOP_Z - 0.05, f"expected a rest on the hull near the wall tops, got z={rest_z}"
        assert rest_z > _CHANNEL_CAVITY_FLOOR_Z + 0.1, f"the cavity is not load-bearing, yet z={rest_z}"
