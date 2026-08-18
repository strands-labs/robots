"""The reference-motion bridge patches in a floor only when the model has none.

:func:`~strands_robots.policies.protomotions.bridge.qpos_to_motion_data` runs
forward kinematics on a caller-supplied G1 MJCF, and the tracker MJCF it targets
references a geom named ``floor`` from ``<contact><pair>`` elements, so the
bridge appends one when the model does not already declare a ground. The name is
fixed by those references, which makes "does this model already have a ground?"
load-bearing: answer it wrongly and the appended geom is a second ground plane
competing for the ``floor`` name, and MuJoCo refuses the whole model.

The question is about the model MuJoCo builds from the file. MuJoCo merges every
``<worldbody>`` a file declares, splices ``<include>``d content in, and compiles
geoms nested inside bodies - so a check that reads only the first
``<worldbody>``'s direct ``<geom>`` children answers a narrower question, and
refuses G1 MJCFs that MuJoCo itself loads without complaint.

These tests cover the three ways a declared ground escapes that narrower read,
and pin the cases where a floor still has to be appended.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.policies.protomotions.bridge import _patch_and_load_mjcf

mujoco = pytest.importorskip("mujoco")

# A single hinge body, enough for MuJoCo to compile a model. Never a plane, so a
# fixture's ground is only ever the one the test itself declares.
_ROBOT_BODY = '<body name="link"><joint name="j" type="hinge" axis="0 0 1"/><geom type="box" size=".1 .1 .1"/></body>'
_FLOOR = '<geom name="floor" type="plane" size="0 0 0.05"/>'


def _write(tmp_path: Path, body: str, name: str = "model.xml") -> Path:
    """Write an MJCF wrapping ``body`` and return its path."""
    path = tmp_path / name
    path.write_text(f'<mujoco model="probe">{body}</mujoco>', encoding="utf-8")
    return path


def _ground_planes(model) -> list[str | None]:
    """Names of every plane geom in a compiled model."""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
    ]


def _load_or_fail(path: Path):
    """Load ``path`` through the bridge, failing with MuJoCo's own complaint."""
    try:
        return _patch_and_load_mjcf(path)
    except ValueError as exc:
        raise AssertionError(
            f"the bridge refused {path.name}, which MuJoCo loads on its own, "
            f"because it appended a floor the model already declares: {exc}"
        ) from exc


class TestAGroundTheNarrowReadMisses:
    """A declared ground must be honoured however the file declares it."""

    def test_a_second_worldbody_sections_floor_is_not_duplicated(self, tmp_path: Path) -> None:
        """MuJoCo merges every ``<worldbody>``; the ground may be in a later one.

        This is how the ``unitree_ros`` G1 descriptions are authored: the robot
        tree in one ``<worldbody>``, the scene's ``floor`` plane in a second.
        """
        path = _write(tmp_path, f"<worldbody>{_ROBOT_BODY}</worldbody><worldbody>{_FLOOR}</worldbody>")
        reference = mujoco.MjModel.from_xml_path(str(path))
        declared = _ground_planes(reference)
        assert declared == ["floor"], "premise: MuJoCo merges the second worldbody's floor in"

        _, model, _ = _load_or_fail(path)

        compiled = _ground_planes(model)
        assert compiled == ["floor"]
        assert model.ngeom == reference.ngeom

    def test_a_floor_nested_inside_a_body_is_not_duplicated(self, tmp_path: Path) -> None:
        """MuJoCo compiles geoms wherever they are nested, not only at the top."""
        nested = f'<body name="ground_mount">{_FLOOR}</body>'
        path = _write(tmp_path, f"<worldbody>{nested}{_ROBOT_BODY}</worldbody>")
        reference = mujoco.MjModel.from_xml_path(str(path))
        declared = _ground_planes(reference)
        assert declared == ["floor"], "premise: the nested plane is compiled in"

        _, model, _ = _load_or_fail(path)

        compiled = _ground_planes(model)
        assert compiled == ["floor"]
        assert model.ngeom == reference.ngeom

    def test_an_included_floor_is_not_duplicated(self, tmp_path: Path) -> None:
        """MuJoCo splices ``<include>``d content in before compiling."""
        (tmp_path / "ground.xml").write_text(f"<mujoco><worldbody>{_FLOOR}</worldbody></mujoco>", encoding="utf-8")
        path = _write(tmp_path, f'<include file="ground.xml"/><worldbody>{_ROBOT_BODY}</worldbody>')
        reference = mujoco.MjModel.from_xml_path(str(path))
        declared = _ground_planes(reference)
        assert declared == ["floor"], "premise: the included floor is compiled in"

        _, model, _ = _load_or_fail(path)

        compiled = _ground_planes(model)
        assert compiled == ["floor"]
        assert model.ngeom == reference.ngeom


class TestAFloorIsStillAppendedWhenTheModelHasNone:
    """The append is what lets the tracker MJCF compile at all - keep it firing."""

    def test_a_model_declaring_no_ground_gets_a_floor(self, tmp_path: Path) -> None:
        """The tracker MJCF declares no ground, so the append must still happen."""
        path = _write(tmp_path, f"<worldbody>{_ROBOT_BODY}</worldbody>")
        reference = mujoco.MjModel.from_xml_path(str(path))
        declared = _ground_planes(reference)
        assert declared == [], "premise: the fixture declares no ground of its own"

        _, model, _ = _load_or_fail(path)

        compiled = _ground_planes(model)
        assert compiled == ["floor"]
        assert model.ngeom == reference.ngeom + 1

    def test_the_appended_geom_is_named_floor(self, tmp_path: Path) -> None:
        """``<contact><pair geom2="floor">`` in the tracker MJCF resolves by name."""
        path = _write(tmp_path, f"<worldbody>{_ROBOT_BODY}</worldbody>")

        _, model, _ = _load_or_fail(path)

        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert floor_id >= 0


class TestTheRecognisedShapesAreUnchanged:
    """What the previous read already accepted must keep being accepted."""

    def test_a_floor_in_the_first_worldbody_is_not_duplicated(self, tmp_path: Path) -> None:
        """The common case: one ``<worldbody>``, ground as a direct child."""
        path = _write(tmp_path, f"<worldbody>{_FLOOR}{_ROBOT_BODY}</worldbody>")
        reference = mujoco.MjModel.from_xml_path(str(path))

        _, model, _ = _load_or_fail(path)

        compiled = _ground_planes(model)
        assert compiled == ["floor"]
        assert model.ngeom == reference.ngeom

    def test_a_geom_named_floor_that_is_not_a_plane_is_not_duplicated(self, tmp_path: Path) -> None:
        """A finite ground is recognised by name, not only by being a plane."""
        slab = '<geom name="floor" type="box" size="5 5 .01" pos="0 0 -.01"/>'
        path = _write(tmp_path, f"<worldbody>{slab}{_ROBOT_BODY}</worldbody>")
        reference = mujoco.MjModel.from_xml_path(str(path))

        _, model, _ = _load_or_fail(path)

        assert model.ngeom == reference.ngeom
        compiled = _ground_planes(model)
        assert compiled == []
