"""The particle-proxy pour predicates score containment over a SET of bodies.

``particles_inside`` / ``particles_spilled`` / ``particles_inside_fraction``
are the predicates the articulated-container pouring tasks score with:
contents are proxied by rigid bodies ("particles"), and "poured" / "spilled"
are fractions and counts of them inside / outside named containers. These
tests pin the counting semantics, the compile-time domain of every parameter,
the degradation direction when a name does not resolve (a missing particle
may lower a success fraction but must never fire a failure), and the
``stop_when`` entity collection for the list-valued kwargs.

Containment is the same axis-aligned box as ``body_inside``, so the stubs
here mirror the envelope shape ``tests/simulation/test_benchmark_predicates.py``
uses.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import (
    DeclarativeBenchmark,
    stop_when_referenced_entities,
)
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    make_predicate,
    predicate_kind,
)


class _BodySim:
    """Stub sim resolving body positions from a dict, via the tool-result envelope."""

    def __init__(self, positions: dict[str, list[float]]):
        self._positions = positions

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        pos = self._positions.get(body_name)
        if pos is None:
            return {"status": "error", "content": [{"text": f"no body {body_name}"}]}
        return {
            "status": "success",
            "content": [{"json": {"position": list(pos)}}],
        }


def _pour_sim(**overrides: list[float] | None) -> _BodySim:
    """Tray at origin; three beads - two inside the tray box, one far away."""
    positions: dict[str, list[float]] = {
        "tray": [0.0, 0.0, 0.0],
        "carton": [0.0, 0.0, 0.35],
        "bead_a": [0.05, 0.0, 0.03],
        "bead_b": [-0.04, 0.06, 0.05],
        "bead_c": [1.0, 1.0, 0.0],
    }
    for name, pos in overrides.items():
        if pos is None:
            positions.pop(name, None)
        else:
            positions[name] = pos
    return _BodySim(positions)


BEADS = ["bead_a", "bead_b", "bead_c"]


class TestRegistry:
    def test_pour_predicates_registered(self):
        assert {"particles_inside", "particles_spilled", "particles_inside_fraction"} <= set(PREDICATE_REGISTRY)

    def test_kinds(self):
        assert predicate_kind("particles_inside") == "bool"
        assert predicate_kind("particles_spilled") == "bool"
        assert predicate_kind("particles_inside_fraction") == "float"


class TestParticlesInside:
    def test_fraction_met(self):
        pred = make_predicate("particles_inside", particles=BEADS, container="tray", min_fraction=0.6)
        assert pred(_pour_sim()) is True  # 2 of 3 inside

    def test_fraction_not_met(self):
        pred = make_predicate("particles_inside", particles=BEADS, container="tray", min_fraction=1.0)
        assert pred(_pour_sim()) is False

    def test_all_inside(self):
        pred = make_predicate("particles_inside", particles=BEADS, container="tray")
        assert pred(_pour_sim(bead_c=[0.0, 0.0, 0.1])) is True

    def test_tolerances_bound_the_box(self):
        pred = make_predicate(
            "particles_inside", particles=["bead_a"], container="tray", min_fraction=1.0, xy_tol=0.04, z_tol=0.15
        )
        assert pred(_pour_sim()) is False  # bead_a x=0.05 > xy_tol=0.04

    def test_unresolved_container_degrades_to_false(self):
        pred = make_predicate("particles_inside", particles=BEADS, container="tray", min_fraction=0.1)
        assert pred(_pour_sim(tray=None)) is False

    def test_unresolved_particle_counts_as_not_inside(self):
        # 1 of the 2 remaining resolvable beads is inside; the missing one
        # lowers the fraction (2/3 required, only 1/3 provable).
        pred = make_predicate("particles_inside", particles=BEADS, container="tray", min_fraction=0.66)
        assert pred(_pour_sim(bead_b=None)) is False

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_min_fraction_domain(self, bad):
        with pytest.raises(ValueError, match="min_fraction"):
            make_predicate("particles_inside", particles=BEADS, container="tray", min_fraction=bad)

    def test_min_fraction_must_be_finite(self):
        with pytest.raises(ValueError):
            make_predicate("particles_inside", particles=BEADS, container="tray", min_fraction=float("nan"))

    def test_empty_particles_refused(self):
        with pytest.raises(ValueError, match="at least one body"):
            make_predicate("particles_inside", particles=[], container="tray")

    def test_bare_string_particles_refused(self):
        with pytest.raises(ValueError, match="single string"):
            make_predicate("particles_inside", particles="bead_a", container="tray")

    def test_duplicate_particles_refused(self):
        with pytest.raises(ValueError):
            make_predicate("particles_inside", particles=["bead_a", "bead_a"], container="tray")


class TestParticlesSpilled:
    def test_no_spill_with_both_containers(self):
        # bead_c is far away, but a second sim where it sits in the carton.
        pred = make_predicate("particles_spilled", particles=BEADS, containers=["tray", "carton"], max_spilled=0)
        assert pred(_pour_sim(bead_c=[0.02, 0.0, 0.30])) is False

    def test_one_spill_over_zero_tolerance_fires(self):
        pred = make_predicate("particles_spilled", particles=BEADS, containers=["tray", "carton"], max_spilled=0)
        assert pred(_pour_sim()) is True  # bead_c is in neither

    def test_tolerated_spill_does_not_fire(self):
        pred = make_predicate("particles_spilled", particles=BEADS, containers=["tray", "carton"], max_spilled=1)
        assert pred(_pour_sim()) is False

    def test_unresolved_container_degrades_to_false(self):
        pred = make_predicate("particles_spilled", particles=BEADS, containers=["tray", "carton"], max_spilled=0)
        assert pred(_pour_sim(carton=None)) is False

    def test_unresolved_particle_cannot_be_shown_spilled(self):
        # bead_c (the spilled one) does not resolve -> not countable -> no fire.
        pred = make_predicate("particles_spilled", particles=BEADS, containers=["tray", "carton"], max_spilled=0)
        assert pred(_pour_sim(bead_c=None)) is False

    @pytest.mark.parametrize("bad", [-1, 1.5])
    def test_max_spilled_domain(self, bad):
        with pytest.raises(ValueError, match="max_spilled"):
            make_predicate("particles_spilled", particles=BEADS, containers=["tray"], max_spilled=bad)

    def test_empty_containers_refused(self):
        with pytest.raises(ValueError, match="at least one body"):
            make_predicate("particles_spilled", particles=BEADS, containers=[], max_spilled=0)

    def test_bare_string_containers_refused(self):
        with pytest.raises(ValueError, match="single string"):
            make_predicate("particles_spilled", particles=BEADS, containers="tray", max_spilled=0)


class TestParticlesInsideFraction:
    def test_weighted_fraction(self):
        term = make_predicate("particles_inside_fraction", particles=BEADS, container="tray", weight=3.0)
        assert term(_pour_sim()) == pytest.approx(2.0)  # 3.0 * (2/3)

    def test_unresolved_container_yields_zero(self):
        term = make_predicate("particles_inside_fraction", particles=BEADS, container="tray")
        assert term(_pour_sim(tray=None)) == 0.0

    def test_fraction_tracks_the_pour(self):
        term = make_predicate("particles_inside_fraction", particles=BEADS, container="tray")
        before = term(_pour_sim(bead_a=[1.0, 0.0, 0.3], bead_b=[1.0, 0.1, 0.3]))
        after = term(_pour_sim(bead_c=[0.0, 0.05, 0.02]))
        assert before == pytest.approx(0.0)
        assert after == pytest.approx(1.0)

    def test_empty_particles_refused(self):
        with pytest.raises(ValueError, match="at least one body"):
            make_predicate("particles_inside_fraction", particles=[], container="tray")


class TestStopWhenCollection:
    def test_list_kwargs_are_collected_for_probing(self):
        bodies, joints = stop_when_referenced_entities(
            {
                "predicate": "particles_inside",
                "particles": ["bead_a", "bead_b"],
                "container": "tray",
            }
        )
        assert bodies == ["bead_a", "bead_b", "tray"]
        assert joints == []

    def test_containers_list_is_collected(self):
        bodies, _ = stop_when_referenced_entities(
            {
                "predicate": "particles_spilled",
                "particles": ["bead_a"],
                "containers": ["tray", "carton"],
                "max_spilled": 0,
            }
        )
        assert bodies == ["bead_a", "tray", "carton"]


class TestDeclarativeSpecIntegration:
    def test_pour_spec_compiles_and_scores(self):
        spec = {
            "name": "pour_test",
            "default_robot": "so100",
            "success": {
                "all": [
                    {
                        "predicate": "particles_inside",
                        "particles": BEADS,
                        "container": "tray",
                        "min_fraction": 0.6,
                    }
                ]
            },
            "failure": {
                "any": [
                    {
                        "predicate": "particles_spilled",
                        "particles": BEADS,
                        "containers": ["tray", "carton"],
                        "max_spilled": 2,
                    }
                ]
            },
            "dense_reward": [
                {"predicate": "particles_inside_fraction", "particles": BEADS, "container": "tray"},
            ],
        }
        benchmark = DeclarativeBenchmark.from_dict(spec)
        sim = _pour_sim()
        assert benchmark.is_success(sim) is True  # 2/3 >= 0.6
        assert benchmark.is_failure(sim) is False  # 1 spilled <= 2
        info = benchmark.on_step(sim, {}, {})
        assert info.reward == pytest.approx(2.0 / 3.0)

    def test_float_term_is_refused_in_a_success_clause(self):
        spec = {
            "name": "pour_bad",
            "default_robot": "so100",
            "success": {
                "all": [
                    {
                        "predicate": "particles_inside_fraction",
                        "particles": BEADS,
                        "container": "tray",
                    }
                ]
            },
        }
        with pytest.raises(ValueError, match="particles_inside_fraction"):
            DeclarativeBenchmark.from_dict(spec)
