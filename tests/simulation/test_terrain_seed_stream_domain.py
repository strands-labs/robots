"""``generate_heightfield``'s seed names exactly one value-noise stream.

:func:`strands_robots.simulation.terrain.generate_heightfield` documents itself
as "deterministic given ``(kind, resolution, seed)``", and the module docstring
promises that "a benchmark that evaluates a policy on ``terrain="rough"``
regenerates the identical field on every reset". ``resolution`` was measured
against the shared positive-discrete domain; ``seed`` was handed straight to
:class:`random.Random`, which does not seed from the value it is given - it
seeds an int from ``abs(value)`` and anything else from ``hash(value)``. So the
documented triple was neither injective nor total:

* ``seed=-1``, ``seed=True`` and ``seed=1`` were three distinct triples drawing
  **one** field, because the sign is discarded and ``bool`` is an ``int``
  subclass. A curriculum stepping the seed across resets to draw fresh ground
  silently re-drew ground it had already evaluated on.
* ``seed=float("nan")`` was **irreproducible**: ``hash(nan)`` has been derived
  from the object's identity since Python 3.10, so two ``float("nan")`` seeds
  draw two different fields within a single process - the one input for which
  the module's headline promise is false rather than merely surprising.
* ``seed=2.5`` and ``seed="1"`` were accepted outright, the same fractional and
  string axes the ``resolution`` domain already closed.

The fix measures ``seed`` against
:func:`~strands_robots.utils.non_negative_whole_number_error`, the shared domain
for a seed that is spread into a stream key. Deliberately
scoped two ways, and both boundaries are pinned below: the domain is applied
only on the ``"rough"`` branch, because the seed-independent kinds use no rng
and must not be refused for a value they never read; and no upper bound is
imposed, because ``random.Random`` consumes an arbitrarily large int directly.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from strands_robots.simulation import terrain
from strands_robots.utils import non_negative_whole_number_error

#: The kind whose field is drawn from an rng, and so the only kind that reads
#: the seed. Derived from the generator rather than hardcoded: a kind that later
#: starts drawing must either join this list or fail the closure test below.
SEEDED_KIND = "rough"

#: Kinds that use no rng. ``SUPPORTED_TERRAINS`` minus the seeded one, so a new
#: terrain lands in one of the two buckets on arrival.
UNSEEDED_KINDS = tuple(k for k in terrain.SUPPORTED_TERRAINS if k != SEEDED_KIND)

#: Values outside the domain, each paired with what it did before the fix.
REFUSED: tuple[tuple[str, Any], ...] = (
    ("negative", -1),
    ("negative_large", -12345),
    ("true", True),
    ("false", False),
    ("fractional", 2.5),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("neg_inf", float("-inf")),
    ("str_digits", "1"),
    ("none", None),
)

#: Values the domain accepts. ``2.0`` is deliberately here: an integral float
#: hashes to its int, so it names the same stream and is not an ambiguity.
ACCEPTED: tuple[tuple[str, Any], ...] = (
    ("zero", 0),
    ("one", 1),
    ("integral_float", 2.0),
    ("far_above_a_32_bit_stream_range", 10**30),
)

# At or above every kind's minimum grid, so no cell here is ever refused for a
# resolution the kind cannot draw - the subject of this file is the seed. Read
# from the module rather than fixed at 8, which a pyramid cannot express: eight
# cells hold four concentric rings, not the five plateaus the kind declares.
RESOLUTION = max(terrain.TERRAIN_MIN_RESOLUTION.values())


class TestASeedOutsideTheDomainIsRefused:
    """The regression: every value that could not name one stream is refused."""

    def test_the_seeded_kind_refuses_it(self) -> None:
        for label, value in REFUSED:
            with pytest.raises(ValueError) as excinfo:
                terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION, seed=value)
            message = str(excinfo.value)
            assert "seed" in message, f"{label}: {message}"
            assert "generate_heightfield" in message, f"{label}: {message}"

    def test_the_refusal_is_the_shared_domain_verbatim(self) -> None:
        """Single-sourced: no second spelling of the rule to drift from."""
        expected = non_negative_whole_number_error(-1, "seed", "generate_heightfield")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION, seed=-1)
        assert str(excinfo.value) == expected

    def test_the_kind_is_answered_before_the_seed(self) -> None:
        """An unknown kind must not be reported as a seed problem."""
        with pytest.raises(ValueError) as excinfo:
            terrain.generate_heightfield("not-a-terrain", resolution=RESOLUTION, seed=-1)
        assert "seed" not in str(excinfo.value)


class TestWhyTheDomainIsNonNegativeAndWhole:
    """The premise, in :class:`random.Random` itself.

    Each clause states the language behaviour that makes a refused value an
    alias for an accepted one. Without them the domain reads as arbitrary
    strictness rather than as the exact set of spellings that name one stream.
    """

    @pytest.mark.parametrize("aliased", [-1, True], ids=["negative_of_one", "true"])
    def test_a_refused_int_like_seed_aliases_an_accepted_one(self, aliased: Any) -> None:
        """``abs()`` for ints, and ``bool`` is an ``int``: both collapse onto 1."""
        assert non_negative_whole_number_error(aliased, "seed", "premise") is not None
        assert non_negative_whole_number_error(1, "seed", "premise") is None
        assert random.Random(aliased).random() == random.Random(1).random()

    def test_two_nan_seeds_draw_two_different_streams(self) -> None:
        """``hash(nan)`` is identity-derived, so nan cannot name a stream."""
        first, second = float("nan"), float("nan")
        assert math.isnan(first) and math.isnan(second)
        assert hash(first) != hash(second)
        assert random.Random(first).random() != random.Random(second).random()

    def test_an_accepted_integral_float_aliases_its_int_on_purpose(self) -> None:
        """The complement: ``2.0`` is accepted because it is not an ambiguity."""
        assert non_negative_whole_number_error(2.0, "seed", "premise") is None
        assert random.Random(2.0).random() == random.Random(2).random()

    def test_a_huge_int_is_a_usable_stream_name(self) -> None:
        """Why no upper bound is imposed on top of the shared domain."""
        assert 0.0 <= random.Random(10**30).random() <= 1.0


class TestASeedInsideTheDomainStillDrawsItsField:
    """Controls: the guard refuses the domain, not the callers."""

    def test_it_draws_a_normalized_field_of_the_documented_length(self) -> None:
        for label, value in ACCEPTED:
            field = terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION, seed=value)
            assert len(field) == RESOLUTION * RESOLUTION, label
            assert all(0.0 <= height <= 1.0 for height in field), label

    def test_the_module_default_satisfies_its_own_domain(self) -> None:
        assert non_negative_whole_number_error(terrain.TERRAIN_SEED, "seed", "terrain") is None
        assert len(terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION)) == RESOLUTION * RESOLUTION

    def test_distinct_accepted_seeds_still_draw_distinct_fields(self) -> None:
        """Injectivity, on the set that survives the domain."""
        fields = [
            tuple(terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION, seed=seed)) for seed in range(6)
        ]
        assert len(set(fields)) == len(fields)

    def test_one_accepted_seed_still_redraws_the_identical_field(self) -> None:
        """The promise the domain exists to make true."""
        first = terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION, seed=7)
        second = terrain.generate_heightfield(SEEDED_KIND, resolution=RESOLUTION, seed=7)
        assert first == second


class TestTheSeedIndependentKindsAreNotRefused:
    """The scope boundary: a kind that never reads the seed never judges it.

    ``_stairs`` / ``_pyramid`` / ``_slope`` take no seed argument at all, so
    refusing them for one would refuse a caller a value the requested kind
    cannot act on. Pinned so the boundary is a stated scope rather than an
    omission, and so a kind that later starts drawing cannot quietly inherit
    the exemption.
    """

    @pytest.mark.parametrize("kind", UNSEEDED_KINDS)
    def test_a_value_the_domain_refuses_is_ignored(self, kind: str) -> None:
        for label, value in REFUSED:
            field = terrain.generate_heightfield(kind, resolution=RESOLUTION, seed=value)
            assert len(field) == RESOLUTION * RESOLUTION, label

    @pytest.mark.parametrize("kind", UNSEEDED_KINDS)
    def test_the_field_is_the_same_whatever_the_seed_says(self, kind: str) -> None:
        """The reason it is ignored: the field does not depend on it."""
        assert terrain.generate_heightfield(kind, resolution=RESOLUTION, seed=0) == terrain.generate_heightfield(
            kind, resolution=RESOLUTION, seed=99
        )

    def test_the_two_buckets_are_exhaustive_and_neither_is_empty(self) -> None:
        """Non-vacuity: a new terrain kind must land in one of them."""
        assert SEEDED_KIND in terrain.SUPPORTED_TERRAINS
        assert UNSEEDED_KINDS
        assert {SEEDED_KIND, *UNSEEDED_KINDS} == set(terrain.SUPPORTED_TERRAINS)

    def test_only_the_seeded_kind_reads_a_seed(self) -> None:
        """Closure: the split is read off the generators, not asserted about.

        A kind that gains an rng gains a ``seed`` parameter, which lands it here
        rather than in the exempt bucket - so the exemption cannot outlive the
        reason for it.
        """
        import inspect

        drawing = {
            kind
            for kind in terrain.SUPPORTED_TERRAINS
            if "seed" in inspect.signature(getattr(terrain, f"_{kind}")).parameters
        }
        assert drawing == {SEEDED_KIND}
