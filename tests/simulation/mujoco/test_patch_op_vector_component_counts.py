"""``patch_scene_mjcf`` holds a fixed-width op field to the library's component count.

These ops write the very buffers ``add_object``, ``add_camera`` and
``move_object`` write, so a component count either surface refuses has to be
refused by the other. Leaving the count to MuJoCo did not deliver that, in two
different directions.

``set_body_pos`` / ``set_body_quat`` assign the field as a spec ATTRIBUTE
(``body.pos = ...``) instead of passing it as a constructor keyword. pybind11
reports a width mismatch there by dumping its C++ overload table and the
receiving object's address::

    patch set_body_pos pos=[1.0, 2.0]
    status="error"
    (): incompatible function arguments. The following argument types are supported:
        1. (arg0: mujoco._specs.MjsBody,
            arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None
    Invoked with: <mujoco._specs.MjsBody object at 0xffff74ecf670>, [1.0, 2.0]

Neither ``set_body_pos`` nor ``pos`` appears in it, while ``add_body`` and
``add_geom`` - writing the same two fields through a keyword - reported cleanly.

In the other direction a three-component ``rgba`` was refused outright, though it
is the RGB that ``add_object(color=...)`` accepts and completes with an opaque
alpha. One backend, two surfaces, one ``geom_rgba`` buffer, opposite verdicts on
the same colour.

These tests pin the count on every fixed-width field of every op, hold each to
the same verdict as its scene-construction sibling, pin that a present-but-``None``
field is refused rather than read as an omission asking for the default, and pin
that the batch stays atomic when a count is refused mid-sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    _OP_FIELD_DOMAINS,
    _PATCH_OP_VECTOR_FIELDS,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# The pybind11 signature dump this change replaces. Asserting its absence keeps
# the fix from regressing into "the library refuses, but MuJoCo answers first".
PYBIND_DUMP = "incompatible function arguments"

CRATE_POS = [0.4, 0.0, 0.5]

# (op body, field, the width the library defines) for every fixed-width field.
FIXED_WIDTH_CASES: list[tuple[dict[str, Any], str, int]] = [
    ({"op": "add_body", "parent": "world", "name": "fresh"}, "pos", 3),
    ({"op": "add_body", "parent": "world", "name": "fresh"}, "quat", 4),
    ({"op": "add_geom", "body": "crate", "type": "box", "size": [0.1, 0.1, 0.1]}, "pos", 3),
    ({"op": "add_geom", "body": "crate", "type": "box", "size": [0.1, 0.1, 0.1]}, "quat", 4),
    ({"op": "add_geom", "body": "crate", "type": "box", "size": [0.1, 0.1, 0.1]}, "rgba", 4),
    ({"op": "add_site", "body": "crate", "name": "tip"}, "pos", 3),
    ({"op": "add_site", "body": "crate", "name": "tip"}, "rgba", 4),
    ({"op": "set_body_pos", "name": "crate"}, "pos", 3),
    ({"op": "set_body_quat", "name": "crate"}, "quat", 4),
]

# Counts no fixed-width field can consume: never enough, and never RGB either.
UNUSABLE_COUNTS = [0, 1, 2, 5]


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch_counts", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _seeded_world(sim) -> None:
    """Compile a world holding a free-floating body ``crate`` at :data:`CRATE_POS`.

    ``sim`` is left un-annotated throughout this module: ``Simulation._world`` is
    typed ``SimWorld | None``, so annotating it would make every ``_world._model``
    probe below read as a possible ``None`` even though the fixture guarantees a
    compiled world.
    """
    sim.create_world()
    assert sim.add_object(name="crate", shape="box", size=[0.2, 0.2, 0.2], position=CRATE_POS)["status"] == "success"


def _text(result: dict[str, Any]) -> str:
    return " ".join(part.get("text", "") for part in result.get("content", []))


def _op_with(body: dict[str, Any], field: str, value: Any) -> dict[str, Any]:
    op = dict(body)
    op[field] = value
    return op


def _body_pos(sim, name: str) -> list[float]:
    model = sim._world._model
    return [float(v) for v in model.body_pos[model.body(name).id]]


def _geom_rgba(sim, name: str) -> list[float]:
    model = sim._world._model
    return [round(float(v), 4) for v in model.geom_rgba[model.geom(name).id]]


class TestAnUnusableComponentCountIsRefused:
    """Every fixed-width field refuses a count it cannot consume, naming both."""

    @pytest.mark.parametrize(("body", "field", "width"), FIXED_WIDTH_CASES, ids=str)
    def test_the_message_names_the_op_and_the_field(self, sim, body, field, width) -> None:
        _seeded_world(sim)
        for count in UNUSABLE_COUNTS:
            if field == "rgba" and count == 3:  # RGB is a usable colour, not a mismatch
                continue
            result = sim.patch_scene_mjcf([_op_with(body, field, [0.1] * count)])
            text = _text(result)
            assert result["status"] == "error", f"count={count}: {text}"
            assert body["op"] in text, text
            assert f"'{field}'" in text, text
            assert PYBIND_DUMP not in text, text

    @pytest.mark.parametrize(("body", "field", "width"), FIXED_WIDTH_CASES, ids=str)
    def test_the_declared_width_is_accepted(self, sim, body, field, width) -> None:
        _seeded_world(sim)
        value = [1.0, 0.0, 0.0, 0.0] if field == "quat" else [0.3] * width
        result = sim.patch_scene_mjcf([_op_with(body, field, value)])
        assert result["status"] == "success", _text(result)

    def test_a_numpy_vector_of_the_declared_width_is_accepted(self, sim) -> None:
        """The natural product of pose arithmetic still reaches the spec."""
        _seeded_world(sim)
        result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": np.array([0.1, 0.2, 0.3])}])
        assert result["status"] == "success", _text(result)
        assert _body_pos(sim, "crate") == pytest.approx([0.1, 0.2, 0.3])


class TestTheAttributeWritingOpsNoLongerDumpASignature:
    """The two ops MuJoCo could only answer for are the point of the change."""

    @pytest.mark.parametrize(
        ("op", "field"),
        [
            ({"op": "set_body_pos", "name": "crate", "pos": [1.0, 2.0]}, "pos"),
            ({"op": "set_body_quat", "name": "crate", "quat": [1.0, 0.0, 0.0]}, "quat"),
        ],
        ids=["set_body_pos", "set_body_quat"],
    )
    def test_the_caller_is_told_which_field_of_which_op(self, sim, op, field) -> None:
        _seeded_world(sim)
        text = _text(sim.patch_scene_mjcf([op]))
        assert PYBIND_DUMP not in text, text
        assert "mujoco._specs" not in text, text
        assert f"{op['op']}: '{field}' must be a" in text, text

    def test_a_refused_width_leaves_the_body_where_it_was(self, sim) -> None:
        _seeded_world(sim)
        assert sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": [1.0, 2.0]}])["status"] == "error"
        assert _body_pos(sim, "crate") == pytest.approx(CRATE_POS)


class TestTheWidthVerdictMatchesTheSceneConstructionSibling:
    """``set_body_pos`` and ``move_object`` write ``body_pos``; they must agree."""

    def test_pos_agrees_with_move_object(self, sim) -> None:
        _seeded_world(sim)
        for count in [*UNUSABLE_COUNTS, 3]:
            value = [0.3] * count
            patched = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": list(value)}])
            moved = sim.move_object(name="crate", position=list(value))
            assert patched["status"] == moved["status"], f"count={count}: {_text(patched)} vs {_text(moved)}"

    def test_quat_agrees_with_move_object(self, sim) -> None:
        _seeded_world(sim)
        for count in [*UNUSABLE_COUNTS, 4]:
            value = [1.0] + [0.0] * (count - 1) if count else []
            patched = sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "crate", "quat": list(value)}])
            moved = sim.move_object(name="crate", orientation=list(value))
            assert patched["status"] == moved["status"], f"count={count}: {_text(patched)} vs {_text(moved)}"


class TestTheColourVerdictMatchesAddObject:
    """``add_geom`` and ``add_object`` write ``geom_rgba``; they must agree."""

    COLOURS: list[Any] = [
        [0.9, 0.3, 0.1],
        [0.9, 0.3, 0.1, 1.0],
        [0.9, 0.3],
        [0.9, 0.3, 0.1, 1.0, 0.5],
        [],
        np.array([0.9, 0.3, 0.1]),
        np.array([0.9, 0.3, 0.1, 1.0]),
    ]

    def test_add_geom_agrees_with_add_object(self, sim) -> None:
        _seeded_world(sim)
        for index, colour in enumerate(self.COLOURS):
            patched = sim.patch_scene_mjcf(
                [
                    {
                        "op": "add_geom",
                        "body": "crate",
                        "name": f"patched_{index}",
                        "type": "box",
                        "size": [0.1, 0.1, 0.1],
                        "rgba": colour,
                    }
                ]
            )
            built = sim.add_object(name=f"built_{index}", shape="box", size=[0.1, 0.1, 0.1], color=colour)
            assert patched["status"] == built["status"], f"{colour!r}: {_text(patched)} vs {_text(built)}"

    def test_an_rgb_triple_is_completed_with_an_opaque_alpha(self, sim) -> None:
        """The RGB spelling was refused; now it paints the colour it names."""
        _seeded_world(sim)
        result = sim.patch_scene_mjcf(
            [
                {
                    "op": "add_geom",
                    "body": "crate",
                    "name": "painted",
                    "type": "box",
                    "size": [0.1, 0.1, 0.1],
                    "rgba": [0.9, 0.3, 0.1],
                }
            ]
        )
        assert result["status"] == "success", _text(result)
        assert _geom_rgba(sim, "painted") == pytest.approx([0.9, 0.3, 0.1, 1.0], abs=1e-4)

    def test_an_rgba_quadruple_is_stored_verbatim(self, sim) -> None:
        _seeded_world(sim)
        result = sim.patch_scene_mjcf(
            [
                {
                    "op": "add_geom",
                    "body": "crate",
                    "name": "painted",
                    "type": "box",
                    "size": [0.1, 0.1, 0.1],
                    "rgba": [0.9, 0.3, 0.1, 0.25],
                }
            ]
        )
        assert result["status"] == "success", _text(result)
        assert _geom_rgba(sim, "painted") == pytest.approx([0.9, 0.3, 0.1, 0.25], abs=1e-4)

    def test_a_site_rgb_triple_is_also_completed(self, sim) -> None:
        _seeded_world(sim)
        result = sim.patch_scene_mjcf(
            [{"op": "add_site", "body": "crate", "name": "tip", "size": [0.02, 0.02, 0.02], "rgba": [0.0, 1.0, 0.0]}]
        )
        assert result["status"] == "success", _text(result)
        model = sim._world._model
        stored = [round(float(v), 4) for v in model.site_rgba[model.site("tip").id]]
        assert stored == pytest.approx([0.0, 1.0, 0.0, 1.0], abs=1e-4)


class TestAPresentFieldIsAValueNotAnOmission:
    """A key that is present carries a value, so ``None`` is refused."""

    def test_a_none_pose_is_refused(self, sim) -> None:
        _seeded_world(sim)
        for field in ["pos", "quat"]:
            op = {"op": f"set_body_{field}", "name": "crate", field: None}
            text = _text(sim.patch_scene_mjcf([op]))
            assert f"'{field}'" in text, text

    def test_a_none_colour_is_refused_rather_than_painted_grey(self, sim) -> None:
        _seeded_world(sim)
        result = sim.patch_scene_mjcf(
            [
                {
                    "op": "add_geom",
                    "body": "crate",
                    "name": "painted",
                    "type": "box",
                    "size": [0.1, 0.1, 0.1],
                    "rgba": None,
                }
            ]
        )
        assert result["status"] == "error", _text(result)
        assert "'rgba'" in _text(result), _text(result)


class TestTheBatchStaysAtomic:
    """A count refused mid-batch rolls the whole patch back."""

    def test_an_earlier_op_is_not_left_applied(self, sim) -> None:
        _seeded_world(sim)
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "parent": "world", "name": "rig", "pos": [0.0, 0.0, 1.0]},
                {"op": "set_body_pos", "name": "crate", "pos": [1.0, 2.0]},
            ]
        )
        assert result["status"] == "error", _text(result)
        assert "op #2" in _text(result), _text(result)
        assert mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, "rig") == -1
        assert _body_pos(sim, "crate") == pytest.approx(CRATE_POS)


class TestEveryDeclaredFieldHasADecidedDomain:
    """A field added to an op cannot reach the model without a domain."""

    def test_the_field_set_is_the_one_the_ops_declare(self) -> None:
        declared = {field for fields in _PATCH_OP_VECTOR_FIELDS.values() for field in fields}
        assert declared == {"pos", "quat", "rgba", "size"}, declared
        assert declared == set(_OP_FIELD_DOMAINS), (declared, set(_OP_FIELD_DOMAINS))

    def test_a_field_with_no_domain_is_detected(self) -> None:
        """The parity check above is not vacuous on an undeclared field."""
        planted = {*_PATCH_OP_VECTOR_FIELDS.values(), frozenset({"friction"})}
        declared = {field for fields in planted for field in fields}
        assert declared != set(_OP_FIELD_DOMAINS)
