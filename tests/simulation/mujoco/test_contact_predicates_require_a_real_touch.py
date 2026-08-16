"""Contact predicates must fire on physics, not on proximity.

``mjData.contact`` lists every geom pair inside the *detection* range, which
is the pair's ``margin`` plus its ``gap``. MuJoCo hands only the pairs inside
``margin`` to the constraint solver, so a pair between the two thresholds is a
proximity report that carries **no force at all**.

``get_contacts`` reports the solver's own decision as ``active``, and every
contact predicate resolves it through
:func:`~strands_robots.simulation.predicates.contact_is_active`. Without that
flag the predicates answered on geometry alone, so ``contact_any``,
``contact_between``, ``grasped`` and ``body_on(require_contact=True)`` all
reported contact for a cube held 25 mm clear of a plate.

``dist`` cannot substitute for the flag, which is why the payload needs it: a
pair with a wide ``margin`` is load-bearing at a *positive* distance, and the
near miss here is positive too, so the sign of ``dist`` separates neither case.

Two fixture requirements, both load-bearing for these tests: the moving body
needs a freejoint (a static/static pair is dropped at collision time, leaving
``ncon == 0`` and nothing to assert), and gravity must be off, or the cube
simply falls into real contact.

The engine-backed tests additionally require the ``margin``/``gap`` semantics
MuJoCo 3.9.0 introduced; see :data:`_GAP_SEMANTICS_MUJOCO`. The tests that only
read the flag need no engine and run on every version the project supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.predicates import (  # noqa: E402
    contact_is_active,
    make_predicate,
)

# A plate and a cube, both declaring a narrow ``margin`` with a wide ``gap`` so
# the detection range reaches far past the force-generation threshold. The cube
# sits 25 mm above the plate's top face (plate half 0.02 + cube half 0.02 =
# 0.04 of contact height; the cube's origin is at 0.065).
_SCENE = """
<mujoco model="proximity_vs_touch">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1"/>
    <body name="plate" pos="0 0 0">
      <geom name="plate_g0" type="box" size="0.10 0.10 0.02" margin="0.001" gap="0.05"/>
    </body>
    <body name="cube" pos="0 0 {cube_z}">
      <freejoint/>
      <geom name="cube_g0" type="box" size="0.02 0.02 0.02" margin="0.001" gap="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_GAP_SEMANTICS_MUJOCO = (3, 9)
"""First MuJoCo release whose ``margin``/``gap`` semantics this scene assumes.

3.9.0 (upstream commit ``a4e49f2d``) redesigned both attributes: detection moved
from ``dist < margin`` to ``dist < margin + gap``, and force generation from
``dist < margin - gap`` to ``dist < margin``. A pair that is *reported yet
carries no force* is therefore spelled differently on either side of that
release - upstream's own migration transform is ``margin_new = margin_old -
gap_old`` - so no single scene expresses it on both and rewriting the fixture
cannot make it portable.

Measured with the scene above:

===========  =========================  =========================
mujoco       clear (``z=0.065``)        touching (``z=0.0399``)
===========  =========================  =========================
3.8.1        ``ncon=0`` - no pair       ``exclude=1``, zero force
3.9 - 3.11   ``exclude=1``, zero force  ``exclude=0``, force > 0
===========  =========================  =========================

Under the old semantics the clear scene reports nothing at all, so the premise
"a pair is reported yet carries no force" is unreachable and the
predicate-is-False cases would pass *vacuously*, while the touching scene is
rejected by the solver so every predicate-is-True case fails. The project's
bound (``mujoco>=3.5.0,<4.0.0``) still admits those releases, so these fixtures
skip there rather than fail a suite the production code handles correctly: the
``active`` flag agrees with zero normal force on 3.8.1 too.
"""

# 25 mm of clear air: inside the detection range, outside ``margin``.
_CLEAR_Z = 0.065
# 0.1 mm of penetration: the solver takes the pair and pushes back.
_TOUCHING_Z = 0.0399

_PREDICATES: list[tuple[str, str, dict[str, Any]]] = [
    ("contact_any", "contact_any", {}),
    ("contact_between", "contact_between", {"geom_a": "cube_g0", "geom_b": "plate_g0"}),
    ("grasped", "grasped", {"body": "cube", "gripper_prefix": "plate_g"}),
    (
        "body_on",
        "body_on",
        {
            "body_a": "cube",
            "body_b": "plate",
            "z_offset": 0.01,
            "xy_tol": 0.2,
            "require_contact": True,
        },
    ),
]


def _require_gap_semantics() -> None:
    """Skip when the running engine predates the ``margin``/``gap`` redesign.

    ``pytest.importorskip(..., minversion=...)`` also skips, but it reports only
    the two version numbers - it honours ``reason`` solely when the *import*
    fails - so the log would not say why an older engine cannot run these
    scenes.
    """
    running = tuple(int(part) for part in mujoco.__version__.split(".")[:2])
    if running < _GAP_SEMANTICS_MUJOCO:
        wanted = ".".join(str(part) for part in _GAP_SEMANTICS_MUJOCO)
        pytest.skip(
            f"mujoco {mujoco.__version__} predates the margin/gap redesign of "
            f"{wanted}: this scene's clear case reports no pair at all there, so "
            "the premise these tests assert on is unreachable"
        )


def _build(tmp_path: Path, cube_z: float, name: str) -> Any:
    _require_gap_semantics()
    scene = tmp_path / f"{name}.xml"
    scene.write_text(_SCENE.format(cube_z=cube_z))
    sim = Simulation()
    assert sim.load_scene(str(scene))["status"] == "success"
    return sim


@pytest.fixture
def clear_sim(tmp_path: Path):
    """Cube held 25 mm clear of the plate - reported pairs, zero force."""
    sim = _build(tmp_path, _CLEAR_Z, "clear")
    try:
        yield sim
    finally:
        sim.cleanup()


@pytest.fixture
def touching_sim(tmp_path: Path):
    """Cube genuinely resting on the plate - the solver pushes back."""
    sim = _build(tmp_path, _TOUCHING_Z, "touching")
    try:
        yield sim
    finally:
        sim.cleanup()


def _records(sim: Any) -> list[dict[str, Any]]:
    result = sim.get_contacts()
    assert result["status"] == "success"
    payload = next(b["json"] for b in result["content"] if "json" in b)
    contacts: list[dict[str, Any]] = payload["contacts"]
    return contacts


def _total_normal_force(sim: Any) -> float:
    """Sum ``get_contact_forces``' normal force - an independent oracle.

    Reading the load from the sibling method rather than recomputing it here
    keeps the premise of these tests honest: the claim "this pair carries no
    force" is measured by the engine surface that exists to report force.
    """
    result = sim.get_contact_forces()
    assert result["status"] == "success"
    blocks = [b["json"] for b in result["content"] if "json" in b]
    if not blocks:  # "No active contacts." - no json block at all.
        return 0.0
    return sum(abs(float(c["normal_force"])) for c in blocks[0]["contacts"])


def _cube_height(sim: Any) -> float:
    result = sim.get_body_state(body_name="cube")
    assert result["status"] == "success"
    state = next(b["json"] for b in result["content"] if "json" in b)
    return float(state["position"][2])


# Fixture premises. Without these the predicate assertions below could pass
# for the wrong reason - "no contact reported at all" is not the case under
# test, and a fixture that quietly fell into real contact would test nothing.


def test_the_clear_fixture_reports_pairs_that_carry_no_force(clear_sim: Any) -> None:
    """The cube is 25 mm clear, pairs ARE reported, and the load is exactly zero."""
    records = _records(clear_sim)
    assert records, "the wide gap must still put the pair in the detection set"
    # 0.02 plate half + 0.02 cube half = 0.04 of contact height.
    assert _cube_height(clear_sim) - 0.04 == pytest.approx(0.025, abs=1e-6)
    assert all(r["dist"] > 0 for r in records), "surfaces are separated"
    assert _total_normal_force(clear_sim) == 0.0


def test_the_touching_fixture_reports_a_pair_that_carries_load(touching_sim: Any) -> None:
    """The resting cube's pair is admitted by the solver and carries real force."""
    assert _records(touching_sim), "a penetrating pair must be reported"
    assert _total_normal_force(touching_sim) > 0.0


# The payload carries the solver's decision.


def test_get_contacts_marks_a_zero_force_pair_inactive(clear_sim: Any) -> None:
    """A pair inside the gap but outside ``margin`` is reported, flagged inactive."""
    records = _records(clear_sim)
    assert [r["active"] for r in records] == [False] * len(records)


def test_get_contacts_marks_a_load_bearing_pair_active(touching_sim: Any) -> None:
    """A pair the solver admits is flagged active."""
    records = _records(touching_sim)
    assert records
    assert all(r["active"] for r in records)


def test_get_contacts_text_separates_touching_from_proximity(clear_sim: Any) -> None:
    """The agent-readable summary must not call a zero-force pair a contact."""
    result = clear_sim.get_contacts()
    text = next(b["text"] for b in result["content"] if "text" in b)
    n = len(_records(clear_sim))
    assert f"{n} contacts (0 touching)" in text
    assert "proximity only - no force" in text


# The predicates.


@pytest.mark.parametrize(("label", "name", "kwargs"), _PREDICATES, ids=[p[0] for p in _PREDICATES])
def test_predicate_is_false_for_a_pair_that_carries_no_force(
    clear_sim: Any, label: str, name: str, kwargs: dict[str, Any]
) -> None:
    """No predicate may report contact for a body held 25 mm clear."""
    assert make_predicate(name, **kwargs)(clear_sim) is False


@pytest.mark.parametrize(("label", "name", "kwargs"), _PREDICATES, ids=[p[0] for p in _PREDICATES])
def test_predicate_is_true_when_the_bodies_really_touch(
    touching_sim: Any, label: str, name: str, kwargs: dict[str, Any]
) -> None:
    """Filtering the detection set must not cost a real touch its verdict."""
    assert make_predicate(name, **kwargs)(touching_sim) is True


# The shared reading of the flag.


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"active": True}, True),
        ({"active": False}, False),
        ({"active": 1}, True),
        ({"active": 0}, False),
        ({}, True),
        ({"active": None}, True),
    ],
)
def test_contact_is_active_reads_the_flag(record: dict[str, Any], expected: bool) -> None:
    """Present-and-falsy means proximity; absent means "cannot tell, assume touch"."""
    assert contact_is_active(record) is expected


class _StubContactSim:
    """Sim whose ``get_contacts`` payload is supplied verbatim by the test."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get_contacts(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"json": self._payload}]}

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        z = {"cube": 0.1, "plate": 0.0}[body_name]
        return {
            "status": "success",
            "content": [{"json": {"position": [0.0, 0.0, z], "quaternion": [1.0, 0.0, 0.0, 0.0]}}],
        }


@pytest.mark.parametrize(("label", "name", "kwargs"), _PREDICATES, ids=[p[0] for p in _PREDICATES])
def test_a_payload_without_the_flag_keeps_its_verdict(label: str, name: str, kwargs: dict[str, Any]) -> None:
    """A backend that cannot report the distinction must not lose every contact.

    Answering ``False`` for a payload that simply omits the key would turn an
    unreported capability into "nothing is ever touching", which is the silent
    degradation the predicate module treats as a bug.
    """
    sim = _StubContactSim({"contacts": [{"geom1": "cube_g0", "geom2": "plate_g0", "dist": -0.001}]})
    assert make_predicate(name, **kwargs)(sim) is True  # type: ignore[arg-type]


def test_contact_any_falls_back_to_the_count_without_a_record_list() -> None:
    """``n_contacts`` remains the fallback for payloads that report nothing else."""
    assert make_predicate("contact_any")(_StubContactSim({"n_contacts": 2})) is True  # type: ignore[arg-type]
    assert make_predicate("contact_any")(_StubContactSim({"n_contacts": 0})) is False  # type: ignore[arg-type]
