"""Unit tests for the procedural terrain heightfield generator.

``strands_robots.simulation.terrain.generate_heightfield`` is the backend- and
MuJoCo-independent ground-generation primitive behind
``create_world(terrain="rough")``. It must be deterministic given
``(kind, resolution, seed)`` (so a benchmark regenerates the identical field on
every reset), produce a genuinely non-flat normalized ``[0, 1]`` field for a
rough kind, and reject an unknown kind with an actionable error. These tests
are pure stdlib (no mujoco / numpy) and exercise the module in isolation.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from strands_robots.simulation import terrain
from strands_robots.utils import positive_whole_number_error


def test_rough_field_has_correct_length_and_range() -> None:
    n = 16
    h = terrain.generate_heightfield("rough", resolution=n, seed=terrain.TERRAIN_SEED)
    assert len(h) == n * n
    assert all(0.0 <= v <= 1.0 for v in h)


def test_rough_field_is_deterministic_for_same_seed() -> None:
    a = terrain.generate_heightfield("rough", resolution=20, seed=3)
    b = terrain.generate_heightfield("rough", resolution=20, seed=3)
    assert a == b


def test_rough_field_varies_with_seed() -> None:
    a = terrain.generate_heightfield("rough", resolution=20, seed=1)
    b = terrain.generate_heightfield("rough", resolution=20, seed=2)
    assert a != b


def test_rough_field_is_genuinely_non_flat() -> None:
    h = terrain.generate_heightfield("rough", resolution=32, seed=0)
    # A rough field must span most of the [0, 1] range (normalization pins the
    # min to 0 and max to 1), i.e. it is not a near-flat plane.
    assert max(h) - min(h) > 0.5


def test_default_resolution_matches_module_constant() -> None:
    h = terrain.generate_heightfield("rough")
    assert len(h) == terrain.TERRAIN_RESOLUTION * terrain.TERRAIN_RESOLUTION


@pytest.mark.parametrize("bad", ["flat", "ROUGH", "", "spiral", "steps"])
def test_unknown_terrain_kind_is_rejected_actionably(bad: str) -> None:
    with pytest.raises(ValueError) as exc:
        terrain.generate_heightfield(bad)
    msg = str(exc.value)
    assert "Supported" in msg and "rough" in msg  # actionable: lists what IS valid


def test_none_kind_is_rejected_by_generator() -> None:
    with pytest.raises(ValueError):
        terrain.generate_heightfield(None)  # type: ignore[arg-type]


def test_resolution_below_two_is_rejected() -> None:
    with pytest.raises(ValueError):
        terrain.generate_heightfield("rough", resolution=1)


def test_validate_terrain_accepts_none_and_supported_rejects_unknown() -> None:
    terrain.validate_terrain(None)  # flat ground; no raise
    for kind in terrain.SUPPORTED_TERRAINS:
        terrain.validate_terrain(kind)  # no raise
    with pytest.raises(ValueError):
        terrain.validate_terrain("bogus")


def test_stairs_field_has_correct_length_and_discrete_levels() -> None:
    n = 40
    h = terrain.generate_heightfield("stairs", resolution=n)
    assert len(h) == n * n
    assert all(0.0 <= v <= 1.0 for v in h)
    assert min(h) == 0.0 and max(h) == 1.0
    # A staircase is DISCRETE: exactly TERRAIN_STAIR_STEPS distinct plateau
    # levels (this is what distinguishes it from the continuous "rough" field).
    assert len(set(h)) == terrain.TERRAIN_STAIR_STEPS


def test_stairs_field_is_deterministic_and_seed_independent() -> None:
    # Stairs are fully deterministic (no rng), so the seed must not change them.
    a = terrain.generate_heightfield("stairs", resolution=24, seed=0)
    b = terrain.generate_heightfield("stairs", resolution=24, seed=99)
    assert a == b


def test_stairs_climbs_along_x_and_is_constant_across_y() -> None:
    n = 40
    h = terrain.generate_heightfield("stairs", resolution=n)
    rows = [h[i * n : (i + 1) * n] for i in range(n)]
    # MuJoCo hfield userdata is row-major (row 0 -> min y, col 0 -> min x): the
    # staircase rises along +x (columns), so every row is identical...
    assert all(rows[i] == rows[0] for i in range(n))
    # ...and each row is a monotonically non-decreasing step function of x.
    row0 = rows[0]
    assert all(row0[j] <= row0[j + 1] for j in range(n - 1))
    assert row0[0] == 0.0 and row0[-1] == 1.0


def test_stairs_is_genuinely_stepped_not_smooth() -> None:
    # Distinguish stairs (few discrete plateaus) from rough (near-continuous).
    stairs = terrain.generate_heightfield("stairs", resolution=32)
    rough = terrain.generate_heightfield("rough", resolution=32, seed=0)
    assert len(set(stairs)) == terrain.TERRAIN_STAIR_STEPS
    assert len(set(rough)) > terrain.TERRAIN_STAIR_STEPS * 10


def test_pyramid_field_has_correct_length_and_discrete_levels() -> None:
    n = 40
    h = terrain.generate_heightfield("pyramid", resolution=n)
    assert len(h) == n * n
    assert all(0.0 <= v <= 1.0 for v in h)
    assert min(h) == 0.0 and max(h) == 1.0
    # Like stairs, a pyramid is DISCRETE: exactly TERRAIN_PYRAMID_STEPS distinct
    # plateau levels (this distinguishes it from the continuous "rough" field).
    assert len(set(h)) == terrain.TERRAIN_PYRAMID_STEPS


def test_pyramid_field_is_deterministic_and_seed_independent() -> None:
    # A stepped pyramid uses no rng, so the seed must not change it.
    a = terrain.generate_heightfield("pyramid", resolution=24, seed=0)
    b = terrain.generate_heightfield("pyramid", resolution=24, seed=99)
    assert a == b


def test_pyramid_peaks_at_center_and_descends_to_the_outer_ring() -> None:
    n = 40
    h = terrain.generate_heightfield("pyramid", resolution=n)
    grid = [h[i * n : (i + 1) * n] for i in range(n)]
    ci = n // 2
    # Highest at the central plateau, flush with z=0 (0.0) on the outer ring, so
    # a robot spawns on the top and never falls below the nominal floor.
    assert grid[ci][ci] == 1.0
    assert grid[0][0] == 0.0 and grid[0][-1] == 0.0 and grid[-1][0] == 0.0 and grid[-1][-1] == 0.0


def test_pyramid_is_radially_isotropic_unlike_the_plus_x_staircase() -> None:
    # The defining property vs terrain="stairs": the pyramid's level depends only
    # on the distance from the centre, so the height profile through the centre
    # along +x and along +y are IDENTICAL (an omnidirectional climb). The +x-only
    # staircase cannot express this (there the +y profile is flat).
    n = 40
    ci = n // 2
    p = terrain.generate_heightfield("pyramid", resolution=n)
    pg = [p[i * n : (i + 1) * n] for i in range(n)]
    p_row = pg[ci]  # height vs x at fixed y=centre
    p_col = [pg[i][ci] for i in range(n)]  # height vs y at fixed x=centre
    assert p_row == p_col  # omnidirectional: +x and +y profiles match
    assert p_row == p_row[::-1]  # symmetric inverted-V about the centre
    assert p_row[0] == 0.0 and p_row[-1] == 0.0 and max(p_row) == 1.0

    # Contrast: the staircase's +x profile rises but its +y profile is flat.
    s = terrain.generate_heightfield("stairs", resolution=n)
    sg = [s[i * n : (i + 1) * n] for i in range(n)]
    s_row = sg[ci]
    s_col = [sg[i][ci] for i in range(n)]
    assert s_row != s_col


def test_rough_field_collapses_to_flat_when_noise_is_degenerate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Normalization divides by (max - min); a uniform value-noise field (every
    # cell identical) has a zero span and would divide by zero. The generator
    # guards that degenerate case by returning a flat field of zeros instead of
    # crashing. Simulate uniform noise with a constant rng and assert the field
    # is well-formed and flush with the floor.
    class _ConstRandom:
        def __init__(self, seed: int) -> None:
            self._seed = seed

        def random(self) -> float:
            return 0.42

    monkeypatch.setattr(terrain, "random", types.SimpleNamespace(Random=_ConstRandom))
    n = 8
    h = terrain.generate_heightfield("rough", resolution=n, seed=0)
    assert len(h) == n * n
    assert h == [0.0] * (n * n)  # flat, no NaN / no ZeroDivisionError


def test_slope_field_has_correct_length_and_range() -> None:
    n = 24
    h = terrain.generate_heightfield("slope", resolution=n)
    assert len(h) == n * n
    assert all(0.0 <= v <= 1.0 for v in h)
    assert min(h) == 0.0 and max(h) == 1.0


def test_slope_field_is_deterministic_and_seed_independent() -> None:
    # A ramp is fully deterministic (no rng), so the seed must not change it.
    a = terrain.generate_heightfield("slope", resolution=20, seed=0)
    b = terrain.generate_heightfield("slope", resolution=20, seed=99)
    assert a == b


def test_slope_climbs_along_x_and_is_constant_across_y() -> None:
    n = 24
    h = terrain.generate_heightfield("slope", resolution=n)
    rows = [h[i * n : (i + 1) * n] for i in range(n)]
    # MuJoCo hfield userdata is row-major (row 0 -> min y, col 0 -> min x): the
    # ramp rises along +x (columns), so every row is identical...
    assert all(rows[i] == rows[0] for i in range(n))
    # ...and each row is a STRICTLY increasing function of x (a ramp, not a
    # step function that has equal-height plateaus).
    row0 = rows[0]
    assert all(row0[j] < row0[j + 1] for j in range(n - 1))
    assert row0[0] == 0.0 and row0[-1] == 1.0


def test_slope_is_constant_grade_not_stepped() -> None:
    # A slope is a CONSTANT-grade ramp: the per-column rise (first difference) is
    # uniform. This is what distinguishes it from "stairs" (whose first
    # difference is 0 across a plateau and jumps at a riser).
    n = 16
    h = terrain.generate_heightfield("slope", resolution=n)
    row0 = h[:n]
    diffs = [row0[j + 1] - row0[j] for j in range(n - 1)]
    step = 1.0 / (n - 1)
    assert all(abs(d - step) < 1e-12 for d in diffs)


def test_slope_is_distinct_from_rough_and_stairs() -> None:
    n = 32
    slope = terrain.generate_heightfield("slope", resolution=n)
    stairs = terrain.generate_heightfield("stairs", resolution=n)
    rough = terrain.generate_heightfield("rough", resolution=n, seed=0)
    # slope: n distinct evenly-spaced levels per row (continuous linear ramp) --
    # more than the few discrete stairs plateaus...
    assert len(set(slope)) == n
    assert len(set(slope)) > terrain.TERRAIN_STAIR_STEPS
    # ...and unlike rough, slope is strictly monotonic (rough value noise is not).
    row0 = slope[:n]
    assert all(row0[j] < row0[j + 1] for j in range(n - 1))
    rough_row0 = rough[:n]
    assert any(rough_row0[j] >= rough_row0[j + 1] for j in range(n - 1))
    # sanity: all three kinds produce a full-size field of the same footprint.
    assert len(slope) == len(stairs) == len(rough) == n * n


def test_terrain_elevation_default_is_the_module_constant() -> None:
    # difficulty=1.0 (the default) is the full-height terrain, unchanged.
    assert terrain.terrain_elevation() == terrain.TERRAIN_ELEVATION
    assert terrain.terrain_elevation(1.0) == terrain.TERRAIN_ELEVATION


def test_terrain_elevation_scales_linearly_with_difficulty() -> None:
    # The curriculum knob: peak elevation is a linear multiple of difficulty, so
    # a trainer ramps terrain magnitude across resets without changing the kind.
    assert terrain.terrain_elevation(0.5) == terrain.TERRAIN_ELEVATION * 0.5
    assert terrain.terrain_elevation(2.0) == terrain.TERRAIN_ELEVATION * 2.0


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.01, float("inf"), float("nan")])
def test_validate_difficulty_rejects_non_positive_or_nonfinite(bad: float) -> None:
    with pytest.raises(ValueError) as exc:
        terrain.validate_difficulty(bad)
    assert "difficulty" in str(exc.value)  # actionable: names the offending argument


@pytest.mark.parametrize("good", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_validate_difficulty_accepts_positive_finite(good: float) -> None:
    terrain.validate_difficulty(good)  # no raise
    assert terrain.terrain_elevation(good) > 0.0


class TestResolutionIsMeasuredAgainstTheSharedDiscreteDomain:
    """``resolution`` sizes the grid, so it is checked before the grid is built.

    ``generate_heightfield`` is the exported, backend-independent generator and
    ``resolution`` is the count its documented ``resolution * resolution`` output
    length is squared from. A bare ``int(resolution)`` truncated a fractional
    count, accepted a string, and let ``TypeError`` / ``OverflowError`` out of a
    module whose error contract is ``ValueError`` - the same three axes
    :func:`~strands_robots.simulation.terrain.validate_difficulty` documents for
    the continuous knob in this file, which is why both now report through the
    shared numeric domains.

    The expected verdicts are read from
    :func:`~strands_robots.utils.positive_whole_number_error` itself rather than
    from a copied value list, so a value the shared domain starts accepting or
    refusing is covered here without an edit.
    """

    # Values whose verdict the shared domain owns. Each is a spelling a caller
    # can plausibly arrive at: a config/argv string, a cell count computed by
    # division, a NumPy size, a boolean flag passed to the wrong parameter.
    DOMAIN_CASES: tuple[Any, ...] = ("40", 2.5, 39.7, True, False, None, "abc", [], float("nan"), float("inf"), -5, 0)

    @pytest.mark.parametrize("value", DOMAIN_CASES)
    def test_a_value_the_shared_domain_refuses_is_refused_here(self, value: object) -> None:
        assert positive_whole_number_error(value, "resolution", "terrain") is not None, (
            f"premise: the shared domain must own the verdict for {value!r}"
        )
        with pytest.raises(ValueError) as exc:
            terrain.generate_heightfield("rough", resolution=value)  # type: ignore[arg-type]
        assert "resolution" in str(exc.value), f"the refusal must name the parameter: {exc.value}"

    @pytest.mark.parametrize("value", DOMAIN_CASES)
    def test_the_refusal_is_a_value_error_and_not_a_type_or_overflow_error(self, value: object) -> None:
        """``ValueError`` is the contract; ``int()`` raised outside it for three spellings.

        ``None`` and ``[]`` raised ``TypeError`` and ``inf`` raised
        ``OverflowError`` from inside ``int()``, so a caller narrowing to
        ``ValueError`` - which is what every terrain refusal in this module is -
        never saw them as a refusal at all.
        """
        try:
            terrain.generate_heightfield("rough", resolution=value)  # type: ignore[arg-type]
        except ValueError:
            return
        except Exception as exc:  # noqa: BLE001 - the point is which class escapes
            raise AssertionError(
                f"resolution={value!r} raised {type(exc).__name__}, which a ValueError-only "
                f"caller does not see as a refusal: {exc}"
            ) from exc
        raise AssertionError(f"resolution={value!r} was accepted rather than refused")

    def test_a_fractional_resolution_is_refused_rather_than_truncated(self) -> None:
        """The documented length is ``resolution * resolution``, so a truncation breaks it.

        ``39.7`` used to return a 39x39 field - 1521 floats for a number whose
        square is 1576.09 - and the only place that surfaced was the consumer,
        where MuJoCo reports ``elevation data length must match nrow*ncol``
        without naming ``resolution`` or the truncation.
        """
        try:
            heights = terrain.generate_heightfield("rough", resolution=39.7)  # type: ignore[arg-type]
        except ValueError:
            return
        raise AssertionError(
            f"resolution=39.7 was accepted and returned {len(heights)} floats, so the documented "
            f"resolution * resolution length (1576.09) does not hold"
        )

    def test_a_string_resolution_is_refused_rather_than_parsed(self) -> None:
        """``"40"`` built a 40x40 field, so a config/argv string was refused nowhere."""
        try:
            heights = terrain.generate_heightfield("rough", resolution="40")  # type: ignore[arg-type]
        except ValueError:
            return
        raise AssertionError(f"resolution='40' was accepted and returned {len(heights)} floats")

    # --- controls: what the domain must not change -------------------------

    @pytest.mark.parametrize("value", [4, 4.0, terrain.TERRAIN_RESOLUTION])
    def test_a_usable_whole_count_still_builds_that_grid(self, value: object) -> None:
        """An int, an integral float and the module default are unaffected.

        The shared domain deliberately accepts an integral float (a count
        computed from a config float), so ``4.0`` must still build the same 4x4
        field ``4`` does.
        """
        heights = terrain.generate_heightfield("rough", resolution=value, seed=0)  # type: ignore[arg-type]
        n = int(value)  # type: ignore[call-overload]
        from_int = terrain.generate_heightfield("rough", resolution=n, seed=0)
        assert len(heights) == n * n
        assert heights == from_int

    def test_the_floor_at_two_keeps_its_own_refusal(self) -> None:
        """``1`` is a positive whole number, so the ``>= 2`` floor is still this module's.

        This fails if the shared domain is treated as the whole check and the
        floor is dropped, or if the floor's message is folded into the domain's.
        """
        assert positive_whole_number_error(1, "resolution", "terrain") is None, (
            "premise: the shared domain accepts 1, so only this module refuses it"
        )
        with pytest.raises(ValueError, match=r">= 2"):
            terrain.generate_heightfield("rough", resolution=1)

    def test_the_kind_is_still_checked_before_the_resolution(self) -> None:
        """An unknown kind keeps its own actionable message even with a bad resolution.

        Ordering matters: a caller who got both wrong should be told about the
        kind, which is the one this module can enumerate a remedy for.
        """
        with pytest.raises(ValueError) as exc:
            terrain.generate_heightfield("bogus", resolution=2.5)  # type: ignore[arg-type]
        assert "Supported" in str(exc.value)
