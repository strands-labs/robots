"""``patch_scene_mjcf`` refuses a numeric op field that is not finite numbers.

MuJoCo's compiler does not reject a ``nan``/``inf`` pose, extent or colour - its
one exception is a ``nan`` geom size, and even that lets ``inf`` through - so an
unchecked component was written verbatim into the compiled model while the patch
reported success. On a body owning a freejoint the corruption then spread into
the physics state, with every call in the chain still reporting success::

    patch set_body_pos pos=[nan, 0, 0.3]   status="success"
    model.body_pos                         [nan, 0.0, 0.3]
    step(n_steps=20)                       status="success"
    data.qpos / data.qvel                  non-finite

The same fields written through ``move_object``, ``add_object`` and
``add_camera`` were already refused, so one identical write to ``body_pos`` had
two opposite verdicts depending on which method issued it. The fixed-width fields
reach that guard through the wrapper that also pins their component count, so a
value with no length at all is reported as "must be a list/tuple of 3 numbers" -
the count is pinned in the companion module for component counts. These tests pin the
finiteness domain on every numeric field of every op, hold it to the same verdict
as its ``move_object`` sibling, and pin that the values MuJoCo does define - an
integer component, a NumPy real scalar, and the all-zero quaternion it reads as
identity - are still accepted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    _PATCH_OP_KEYS,
    _PATCH_OP_VECTOR_FIELDS,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

NON_FINITE = [float("nan"), float("inf"), -float("inf")]
CRATE_POS = [0.4, 0.0, 0.5]


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch_numeric", mesh=False)
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

    The freejoint matters: it is what gives the body ``qpos``/``qvel`` rows, so a
    non-finite ``body_pos`` can reach the physics state rather than only the
    static kinematics.
    """
    sim.create_world()
    assert sim.add_object(name="crate", shape="box", size=[0.2, 0.2, 0.2], position=CRATE_POS)["status"] == "success"


def _text(result: dict[str, Any]) -> str:
    return " ".join(part.get("text", "") for part in result.get("content", []))


def _body_pos(sim, name: str) -> list[float]:
    model = sim._world._model
    return [float(v) for v in model.body_pos[model.body(name).id]]


def _state_is_finite(sim) -> bool:
    data = sim._world._data
    return bool(np.all(np.isfinite(data.qpos))) and bool(np.all(np.isfinite(data.qvel)))


# Every (op, field) pair the tables declare, with an op body that is otherwise
# valid so the only reason to refuse is the non-finite component.
def _op_with(kind: str, field: str, value: Any) -> dict[str, Any]:
    bodies: dict[str, dict[str, Any]] = {
        "add_body": {"op": "add_body", "parent": "world", "name": "fresh"},
        "add_geom": {"op": "add_geom", "body": "crate", "type": "box", "size": [0.1, 0.1, 0.1]},
        "add_site": {"op": "add_site", "body": "crate", "name": "tip"},
        "set_body_pos": {"op": "set_body_pos", "name": "crate"},
        "set_body_quat": {"op": "set_body_quat", "name": "crate"},
    }
    lengths = {"pos": 3, "quat": 4, "rgba": 4, "size": 3}
    op = dict(bodies[kind])
    op[field] = [value] + [0.1] * (lengths[field] - 1)
    return op


ALL_NUMERIC_FIELDS = sorted((kind, field) for kind, fields in _PATCH_OP_VECTOR_FIELDS.items() for field in fields)


class TestNonFiniteComponentsAreRefused:
    """A non-finite component is refused on every numeric field of every op."""

    @pytest.mark.parametrize(("kind", "field"), ALL_NUMERIC_FIELDS)
    def test_every_numeric_op_field_refuses_a_non_finite_component(self, sim, kind: str, field: str) -> None:
        _seeded_world(sim)
        for bad in NON_FINITE:
            result = sim.patch_scene_mjcf([_op_with(kind, field, bad)])
            assert result["status"] == "error", f"{kind}.{field}={bad} was accepted"
            message = _text(result)
            assert kind in message, f"{bad!r}"
            assert f"'{field}'" in message, f"{bad!r}"
            assert "finite" in message, f"{bad!r}"

    def test_a_refused_pose_leaves_the_body_and_the_physics_state_untouched(self, sim) -> None:
        """The headline consequence: the model keeps the old pose and the state stays usable.

        Pre-fix this reported success, wrote the component into ``body_pos`` and
        left ``qpos``/``qvel`` non-finite after the next step.
        """
        _seeded_world(sim)
        for bad in NON_FINITE:
            sim.step(n_steps=5)
            assert _state_is_finite(sim), "fixture premise: the scene starts with a usable state"

            result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": [bad, 0.0, 0.3]}])

            assert result["status"] == "error", f"{bad!r}"
            assert _body_pos(sim, "crate") == pytest.approx(CRATE_POS), f"{bad!r}"
            assert sim.step(n_steps=20)["status"] == "success", f"{bad!r}"
            assert _state_is_finite(sim), "a refused patch must not poison qpos/qvel"

    def test_a_non_finite_op_does_not_half_apply_the_batch(self, sim) -> None:
        """A valid op ahead of a bad one is rolled back, matching the documented atomicity."""
        _seeded_world(sim)
        before = sim._world._model.nbody

        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "parent": "world", "name": "good", "pos": [0.1, 0.0, 0.2]},
                {"op": "set_body_pos", "name": "crate", "pos": [float("inf"), 0.0, 0.3]},
            ]
        )

        assert result["status"] == "error"
        assert sim._world._model.nbody == before
        assert _body_pos(sim, "crate") == pytest.approx(CRATE_POS)
        with pytest.raises(KeyError):
            sim._world._model.body("good")


class TestNonNumericComponentsAreRefused:
    """The other axis of the shared guard: a component that is not a number at all."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (["a", "b", "c"], "must be numbers"),
            ([None, 0.0, 0.3], "must be numbers"),
            ([[0.1], 0.0, 0.3], "must be numbers"),
            ([True, 0.0, 0.3], "must be numbers"),
            # A fixed-width field names the width it needs: "of 3 numbers".
            (0.4, "must be a list/tuple of 3 numbers"),
        ],
        ids=["strings", "none", "nested", "bool", "scalar"],
    )
    def test_a_non_numeric_pos_is_refused_with_a_structured_error(self, sim, value: Any, expected: str) -> None:
        _seeded_world(sim)
        result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": value}])
        assert result["status"] == "error"
        assert expected in _text(result)
        assert _body_pos(sim, "crate") == pytest.approx(CRATE_POS)


class TestSameVerdictAsTheMoveObjectSibling:
    """One write to ``body_pos``, one verdict, whichever method issues it."""

    def test_set_body_pos_and_move_object_agree_on_a_position(self, sim) -> None:
        _seeded_world(sim)
        poses: list[list[Any]] = [
            [float("nan"), 0.0, 0.3],
            [float("inf"), 0.0, 0.3],
            [-float("inf"), 0.0, 0.3],
            ["a", "b", "c"],
            [True, 0.0, 0.3],
            [0.2, 0.1, 0.4],
        ]
        for pose in poses:
            via_move = sim.move_object(name="crate", position=list(pose))["status"]
            via_patch = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": list(pose)}])["status"]
            assert via_patch == via_move, f"verdicts differ for pos={pose!r}"

    def test_set_body_quat_and_move_object_agree_on_an_orientation(self, sim) -> None:
        """Includes the all-zero quaternion, which both sides accept.

        MuJoCo reads an all-zero ``quat`` as the identity rotation rather than
        normalising it into ``nan``, so it is a defined input and is deliberately
        not refused - the guard is finiteness, not unit norm.
        """
        _seeded_world(sim)
        for quat in [
            [float("nan"), 0.0, 0.0, 0.0],
            [float("inf"), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]:
            via_move = sim.move_object(name="crate", orientation=list(quat))["status"]
            via_patch = sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "crate", "quat": list(quat)}])["status"]
            assert via_patch == via_move, f"verdicts differ for quat={quat!r}"


class TestUsableValuesStillCompile:
    """No false positives: the numeric shapes MuJoCo defines are still accepted."""

    def test_a_finite_pose_is_applied(self, sim) -> None:
        _seeded_world(sim)
        positions: list[list[Any]] = [[0.2, 0.1, 0.4], [0, 0, 1], [np.float64(0.2), np.float32(0.1), 0.4]]
        for pos in positions:
            result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": list(pos)}])
            assert result["status"] == "success", f"{pos!r}"
            assert _body_pos(sim, "crate") == pytest.approx([float(v) for v in pos]), f"{pos!r}"

    def test_a_full_op_vocabulary_batch_still_applies(self, sim) -> None:
        """Every op writing every numeric field it accepts, with usable values."""
        _seeded_world(sim)
        result = sim.patch_scene_mjcf(
            [
                {
                    "op": "add_body",
                    "parent": "world",
                    "name": "rig",
                    "pos": [0.1, 0.0, 0.2],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "op": "add_geom",
                    "body": "rig",
                    "type": "sphere",
                    "size": [0.06, 0.06, 0.06],
                    "rgba": [0.2, 0.7, 0.3, 1.0],
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "op": "add_site",
                    "body": "rig",
                    "name": "tip",
                    "pos": [0.0, 0.0, 0.1],
                    "size": [0.01],
                    "rgba": [1.0, 0.0, 0.0, 1.0],
                },
                {"op": "set_body_quat", "name": "crate", "quat": [1.0, 0.0, 0.0, 0.0]},
            ]
        )
        assert result["status"] == "success", _text(result)
        assert sim._world._model.body("rig") is not None


class TestTheTwoOpTablesCannotDrift:
    """The numeric-field table is pinned against the accepted-key table."""

    def test_every_op_declares_its_numeric_fields(self) -> None:
        assert set(_PATCH_OP_VECTOR_FIELDS) == set(_PATCH_OP_KEYS)

    @pytest.mark.parametrize("kind", sorted(_PATCH_OP_KEYS))
    def test_every_numeric_field_is_an_accepted_key_of_its_op(self, kind: str) -> None:
        assert _PATCH_OP_VECTOR_FIELDS[kind] <= _PATCH_OP_KEYS[kind]
