# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``derive_key_light``'s search-region flag is checked, not read by truthiness.

``upper_hemisphere`` selects which region of the environment map is searched
for the dominant light. It shares its signature with ``brightest_fraction``,
which is checked four ways (a number, not a ``bool``, finite, inside
``(0, 1]``); the flag had no check at all and was read by truthiness at the
region selection.

Two things follow from that, and both are pinned here.

Every spelling of *off* a caller reaches for is a truthy string, so
``upper_hemisphere="false"`` searched *above* the horizon -- the region the
value says to leave out -- while an undeclared falsy value (``None``, ``0``,
``""``) searched the full sphere, which is the footgun
``test_bright_floor_does_not_hijack_the_key_light`` exists to keep callers out
of: a bright floor under the bake point out-weighs the real light sources and
aims the key light from underneath. The two outcomes are a whole hemisphere
apart, and nothing distinguished them.

The refusal raised when the searched region is empty also branched on the same
truthiness, so a caller who wrote ``upper_hemisphere="false"`` was told the map
is black *above the horizon* -- a region they had asked not to search -- and
advised to "pass upper_hemisphere=False", which is what they believed they had
passed. Checking the flag makes that branch reachable only when the search
really was restricted, where the advice is actionable.

The domain is :func:`~strands_robots.utils.boolean_flag_error`, the shared one
for a flag that selects a posture rather than scaling a quantity, so the
spellings this refuses cannot diverge from the ones its sibling call sites
refuse. It is also the authority these tests parametrize over, so a spelling
added there is covered here without an edit.
"""

import numpy as np
import pytest

from strands_robots.rendering import derive_key_light
from strands_robots.utils import boolean_flag_error

# Values that read as "off" to a human but are truthy to Python, so truthiness
# selects the *restricted* search -- the opposite of what they say.
TRUTHY_OFF_SPELLINGS: list[str] = ["false", "False", "no", "off", "0"]

# Falsy values that are not a declared spelling of anything, so truthiness
# silently selects the full-sphere search this argument's docstring warns about.
UNDECLARED_FALSY: list[object] = [None, 0, "", [], 0.0]


def _window_above_a_dim_floor(H: int = 64, W: int = 128) -> np.ndarray:
    """A small bright warm window above the horizon, over a broad dim floor.

    The floor is dimmer per texel but covers far more solid angle, so the two
    search regions disagree by more than a rounding: restricted finds the
    window above the horizon, full-sphere is dragged below it.
    """
    env = np.zeros((H, W, 3), dtype=np.uint8)
    env[int(H * 0.62) :, :, :] = (70, 78, 95)
    env[int(H * 0.22) : int(H * 0.34), int(W * 0.18) : int(W * 0.30), :] = (255, 205, 130)
    return env


def _black_above_the_horizon(H: int = 64, W: int = 128) -> np.ndarray:
    """Light only from a bounce floor: the restricted search finds nothing."""
    env = np.zeros((H, W, 3), dtype=np.uint8)
    env[int(H * 0.60) :, :, :] = (120, 128, 150)
    return env


class TestASpellingOfOffDoesNotSelectTheOnPosture:
    """A value that reads as *off* must not restrict the search."""

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
    def test_a_truthy_spelling_of_off_does_not_search_above_the_horizon(self, spelling) -> None:
        assert boolean_flag_error(spelling, "upper_hemisphere", "derive_key_light"), (
            f"premise: {spelling!r} is not a boolean, so the shared domain refuses it"
        )
        env = _window_above_a_dim_floor()
        restricted = derive_key_light(env, upper_hemisphere=True).elevation_deg
        full_sphere = derive_key_light(env, upper_hemisphere=False).elevation_deg
        assert restricted > 0.0 > full_sphere, "premise: the two search regions disagree on this map"
        try:
            got = derive_key_light(env, upper_hemisphere=spelling).elevation_deg
        except ValueError as exc:
            assert "upper_hemisphere" in str(exc), f"refused, but not for the flag: {exc}"
            return
        assert got != pytest.approx(restricted, abs=1e-6), (
            f"upper_hemisphere={spelling!r} reads as off, yet it searched above the horizon "
            f"(elevation {got:.1f} deg, the same region upper_hemisphere=True searches); "
            f"the full sphere it asks for derives {full_sphere:.1f} deg, "
            f"{abs(restricted - full_sphere):.1f} deg away"
        )

    @pytest.mark.parametrize("value", UNDECLARED_FALSY)
    def test_an_undeclared_falsy_value_does_not_silently_search_the_full_sphere(self, value) -> None:
        assert boolean_flag_error(value, "upper_hemisphere", "derive_key_light"), (
            f"premise: {value!r} is not a boolean, so the shared domain refuses it"
        )
        env = _window_above_a_dim_floor()
        full_sphere = derive_key_light(env, upper_hemisphere=False).elevation_deg
        assert full_sphere < 0.0, "premise: the full sphere is dragged below the horizon on this map"
        try:
            got = derive_key_light(env, upper_hemisphere=value).elevation_deg
        except ValueError as exc:
            assert "upper_hemisphere" in str(exc), f"refused, but not for the flag: {exc}"
            return
        assert got != pytest.approx(full_sphere, abs=1e-6), (
            f"upper_hemisphere={value!r} is not a declared spelling of anything, yet it selected "
            f"the full-sphere search, deriving a key light from {got:.1f} deg -- from underneath"
        )


class TestTheRefusalNamesSomethingTheCallerAskedFor:
    """An empty-region refusal must describe a region the caller requested."""

    @pytest.mark.parametrize("spelling", TRUTHY_OFF_SPELLINGS)
    def test_an_off_spelling_is_not_refused_for_a_region_it_asked_to_skip(self, spelling) -> None:
        env = _black_above_the_horizon()
        assert derive_key_light(env, upper_hemisphere=False).elevation_deg < 0.0, (
            "premise: the full sphere this value asks for does derive a key light"
        )
        with pytest.raises(ValueError) as excinfo:
            derive_key_light(env, upper_hemisphere=spelling)
        message = str(excinfo.value)
        assert "black above the horizon" not in message, (
            f"upper_hemisphere={spelling!r} asks for the full sphere, yet the refusal reports the map "
            f"black above the horizon and advises passing upper_hemisphere=False -- "
            f"what the caller wrote: {message}"
        )
        assert "upper_hemisphere" in message

    def test_the_full_sphere_advice_still_reaches_a_genuinely_restricted_search(self) -> None:
        # The advice is actionable exactly when the search really was
        # restricted, so checking the flag must not remove it.
        env = _black_above_the_horizon()
        with pytest.raises(ValueError, match="pass upper_hemisphere=False"):
            derive_key_light(env, upper_hemisphere=True)


class TestTheCheckedSpellingsStillWork:
    """Controls: the declared spellings, and the sibling refusals, are unchanged."""

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_a_boolean_is_accepted(self, value) -> None:
        assert boolean_flag_error(value, "upper_hemisphere", "derive_key_light") is None, (
            "premise: the shared domain accepts python and numpy booleans"
        )
        est = derive_key_light(_window_above_a_dim_floor(), upper_hemisphere=value)
        assert np.isfinite(est.elevation_deg)

    def test_the_two_declared_spellings_still_select_the_two_regions(self) -> None:
        env = _window_above_a_dim_floor()
        assert derive_key_light(env, upper_hemisphere=True).elevation_deg > 0.0
        assert derive_key_light(env, upper_hemisphere=False).elevation_deg < 0.0

    def test_the_default_is_the_restricted_search(self) -> None:
        env = _window_above_a_dim_floor()
        omitted = derive_key_light(env)
        stated = derive_key_light(env, upper_hemisphere=True)
        assert omitted.elevation_deg == pytest.approx(stated.elevation_deg, abs=1e-9)

    def test_the_brightest_fraction_refusal_is_unchanged(self) -> None:
        env = _window_above_a_dim_floor()
        with pytest.raises(ValueError) as excinfo:
            derive_key_light(env, brightest_fraction=0.0)
        assert str(excinfo.value) == ("derive_key_light: brightest_fraction must be a number in (0, 1], got 0.0.")

    def test_the_env_map_shape_refusal_is_unchanged(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            derive_key_light(np.zeros((8, 8), np.uint8), upper_hemisphere=True)
        assert str(excinfo.value) == "derive_key_light: env_map must be an (H, W, 3) image, got shape (8, 8)."
