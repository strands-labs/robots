"""``add_geom``'s ``type`` domain: the one shape this patch op cannot compile.

``patch_scene_mjcf``'s ``add_geom`` op resolves its ``type`` through the same
shape vocabulary ``add_object`` uses, and ``"mesh"`` is in that vocabulary - but
:data:`~strands_robots.simulation.mujoco.scene_ops._PATCH_OP_KEYS` gives the op
no key that could name a mesh asset. So a mesh geom it adds carries no meshid,
and MuJoCo refuses it at the batch's recompile.

That refusal is unusable in both directions:

* It names a MuJoCo element id, not the op or the key that is missing:
  ``Error: mesh geom 'part' (id = 1) must have valid meshid``.
* The recompile runs *outside* the ``try`` that rolls a rejected batch back, so
  the mutated spec stays installed. Measured before the fix, on a default world:
  after one such patch, a second valid patch, a third, and an unrelated
  ``add_object`` all returned that same meshid error - the world was unusable for
  the rest of its life.

The op therefore refuses ``type="mesh"`` before it touches the spec. This module
pins the three things that makes observable, plus the two contract facts the
deletion of ``_normalize_size``'s mesh branch rests on: that the documented mesh
route is unaffected, and that no caller normalizes a mesh ``size`` any more.

Regression test for #2310.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("mujoco")


from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    _UNSUPPORTED_GEOM_SHAPES,
    _geom_shape_error,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import (  # noqa: E402
    _SIZE_LAYOUT,
    _geom_type,
    _normalize_size,
)

#: The op that used to poison a world: well-formed, every key accepted, every
#: numeric field in domain - and fatal at recompile.
MESH_GEOM_OP: dict[str, object] = {"op": "add_geom", "body": "world", "type": "mesh", "name": "part"}

#: A ``size`` each accepted shape actually consumes, so "accepted" can be
#: asserted by compiling rather than by reading the refusal's own list back.
SIZE_FOR_SHAPE: dict[str, list[float]] = {
    "box": [0.1, 0.1, 0.1],
    "ellipsoid": [0.1, 0.2, 0.3],
    "sphere": [0.1],
    "cylinder": [0.1, 0.0, 0.4],
    "capsule": [0.1, 0.0, 0.4],
    "plane": [1.0],
}

#: A minimal tetrahedron, so the documented mesh route can be exercised without
#: shipping a binary asset.
TETRAHEDRON_OBJ = """v 0 0 0
v 0.1 0 0
v 0 0.1 0
v 0 0 0.1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


@pytest.fixture
def sim() -> Iterator[Simulation]:
    s = Simulation(tool_name="devx_patch_op_geom_shape", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _text(result: dict) -> str:
    return str(result["content"][0]["text"])


class TestAMeshGeomOpIsRefused:
    def test_the_refusal_names_the_value_and_the_surface_that_supports_it(self, sim: Simulation) -> None:
        """A refusal whose only remedy is unreachable is not a remedy.

        ``add_object(shape="mesh", mesh_path=...)`` registers the asset itself,
        which is the whole reason it can do what this op cannot - so it is what
        the message has to name.
        """
        result = sim.patch_scene_mjcf([dict(MESH_GEOM_OP)])

        assert result["status"] == "error"
        message = _text(result)
        assert "'mesh'" in message, f"the refused value is not named: {message}"
        assert "add_object" in message and "mesh_path" in message, f"no working route offered: {message}"
        assert "meshid" not in message, f"MuJoCo's own refusal reached the caller: {message}"

    def test_the_refusal_identifies_which_op_failed(self, sim: Simulation) -> None:
        """A batch is only debuggable if the rejection says which op broke it."""
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "rig", "pos": [0, 0, 1]},
                dict(MESH_GEOM_OP, body="rig"),
            ]
        )

        assert result["status"] == "error"
        assert "patch op #2" in _text(result), f"the failing op is not identified: {_text(result)}"

    def test_the_world_survives_the_refusal(self, sim: Simulation) -> None:
        """The property the door refusal exists for.

        Before it, the mutated spec stayed installed and every later mutation
        re-failed on the leftover geom. Two patches and an ``add_object`` after
        the refusal cover both mutation doors.
        """
        assert sim.patch_scene_mjcf([dict(MESH_GEOM_OP)])["status"] == "error"

        first = sim.patch_scene_mjcf([{"op": "add_body", "name": "crate", "pos": [0, 0, 1]}])
        second = sim.patch_scene_mjcf([{"op": "add_body", "name": "crate2", "pos": [0, 0, 2]}])
        injected = sim.add_object("ball", shape="sphere", size=[0.1], position=[0, 0, 3])

        assert first["status"] == "success", _text(first)
        assert second["status"] == "success", _text(second)
        assert injected["status"] == "success", _text(injected)


class TestTheRefusalsAcceptedListIsTheRealVocabulary:
    def test_every_shape_the_message_lists_compiles_through_the_op(self, sim: Simulation) -> None:
        """A list of accepted values is worth nothing unless they are accepted.

        The message builds its list from the shape table rather than from a
        literal, so this asserts the table and the op agree.
        """
        accepted = sorted(set(_SIZE_LAYOUT) - _UNSUPPORTED_GEOM_SHAPES)
        assert accepted, "the refusal would list no alternative at all"

        listed = _geom_shape_error("mesh")
        assert listed is not None
        for shape in accepted:
            assert shape in listed, f"the refusal does not offer {shape!r}: {listed}"
            result = sim.patch_scene_mjcf(
                [
                    {"op": "add_body", "name": f"body_{shape}", "pos": [0, 0, 1]},
                    {
                        "op": "add_geom",
                        "body": f"body_{shape}",
                        "type": shape,
                        "size": SIZE_FOR_SHAPE[shape],
                        "name": f"geom_{shape}",
                    },
                ]
            )
            assert result["status"] == "success", f"{shape} is offered but refused: {_text(result)}"

    def test_the_shape_table_and_the_geom_type_map_share_one_vocabulary(self) -> None:
        """Otherwise the refusal could omit a shape the op does support.

        ``_geom_type`` is what actually decides whether a ``type`` resolves;
        ``_SIZE_LAYOUT`` is what the message enumerates. A shape in one and not
        the other makes the message wrong in one direction or the other.
        """
        for shape in _SIZE_LAYOUT:
            assert _geom_type(shape) is not None, f"{shape!r} has a size layout but no geom type"
        with pytest.raises(ValueError, match="Unsupported shape"):
            _geom_type("hyperboloid")
        assert "hyperboloid" not in _SIZE_LAYOUT

    def test_a_shape_outside_the_vocabulary_is_left_to_geom_type(self, sim: Simulation) -> None:
        """Its own message already names it, and it raises where the rollback is."""
        assert _geom_shape_error("hyperboloid") is None

        result = sim.patch_scene_mjcf([{"op": "add_geom", "body": "world", "type": "hyperboloid"}])

        assert result["status"] == "error"
        assert "Unsupported shape" in _text(result), _text(result)


class TestNoCallSiteNormalizesAMeshSize:
    def test_normalize_size_refuses_a_mesh(self) -> None:
        """The branch that answered ``[0.0, 0.0, 0.0]`` was reachable from one
        call site only, and that site now refuses a mesh before it is reached.

        Its own docstring documented a third answer (``[]``), so the value was
        the one statement of this contract nothing agreed with. With the branch
        gone the fall-through raise says what is true: nothing should be asking.
        """
        with pytest.raises(ValueError, match="Cannot normalize size"):
            _normalize_size("mesh", [2.0, 3.0, 4.0])

    def test_the_documented_mesh_route_still_takes_its_extent_from_the_asset(self, sim: Simulation, tmp_path) -> None:
        """``add_object`` never routed a mesh through ``_normalize_size`` - it
        passes ``meshname`` instead - so the deletion must not touch it."""
        asset = tmp_path / "tetrahedron.obj"
        asset.write_text(TETRAHEDRON_OBJ, encoding="utf-8")

        result = sim.add_object("part", shape="mesh", mesh_path=str(asset), position=[0, 0, 1])

        assert result["status"] == "success", _text(result)
        assert "from the asset" in _text(result), _text(result)
