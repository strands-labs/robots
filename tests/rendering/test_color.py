# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color-pipeline contract tests (issue #2323, stage 3).

The sRGB transfer functions are light arithmetic's entry and exit points --
the shadow-catcher multiply and the optional linear seam blend both go
through them -- so their round trip and reference values are pinned here as
pure numpy.

``srgb_to_linear`` also owns the pipeline's only *scale* ambiguity: it is the
one helper that accepts both byte codes and unit floats, so it is the one that
can be handed a value whose scale it cannot know. That contract is pinned here
too, including the boundary -- the two unit-only helpers must keep accepting an
integer ``0``/``1``, which is unambiguous for them.
"""

import numpy as np
import pytest

from strands_robots.rendering import linear_to_srgb, relative_luminance, srgb_to_linear


def test_round_trip_is_identity_for_every_uint8_code() -> None:
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    back = np.rint(linear_to_srgb(srgb_to_linear(codes)) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(back, codes)


def test_reference_values_match_iec_61966() -> None:
    # The standard's own anchor points: the knee (0.04045 -> 0.0031308) and
    # 50% sRGB gray (~21.4% linear -- not 25%, which a bare 2.2 power gives
    # nor 50%, which no conversion gives).
    assert srgb_to_linear(np.float32(0.04045)) == pytest.approx(0.0031308, abs=1e-6)
    assert srgb_to_linear(np.float32(0.5)) == pytest.approx(0.21404114, abs=1e-6)
    assert linear_to_srgb(np.float32(0.21404114)) == pytest.approx(0.5, abs=1e-6)
    # Endpoints are exact to float32 precision.
    assert srgb_to_linear(np.uint8(0)) == 0.0
    assert srgb_to_linear(np.uint8(255)) == 1.0
    assert linear_to_srgb(np.float32(1.0)) == pytest.approx(1.0, abs=1e-6)


def test_uint8_and_float_inputs_agree() -> None:
    u8 = np.array([0, 51, 128, 255], dtype=np.uint8)
    as_float = u8.astype(np.float32) / 255.0
    np.testing.assert_allclose(srgb_to_linear(u8), srgb_to_linear(as_float), atol=1e-7)


def test_out_of_range_linear_is_clipped_not_wrapped() -> None:
    out = linear_to_srgb(np.array([-0.5, 2.0], dtype=np.float32))
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)


def test_relative_luminance_uses_rec709_weights() -> None:
    # Pure primaries report their coefficient; white sums to 1.
    prim = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(relative_luminance(prim), [0.2126, 0.7152, 0.0722], atol=1e-6)
    assert relative_luminance(np.ones(3, np.float32)) == pytest.approx(1.0, abs=1e-6)


# Byte codes spanning the curve: 0 and 255 are the endpoints, 128 is the
# reference midpoint, and every code above 1 is one the unit domain cannot hold.
_SRGB_CODES = [0, 64, 128, 192, 255]


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.int32, np.int64])
def test_a_non_uint8_integer_array_states_its_scale_instead_of_saturating(dtype) -> None:
    """An integer array that is not ``uint8`` carries no scale of its own.

    Resolving that ambiguity by assuming the unit domain reads every byte code
    above 1 as "already fully lit" and clips it, so a 16-bit capture -- or any
    array a caller widened -- decodes to pure white with nothing said.
    """
    reference = srgb_to_linear(np.array(_SRGB_CODES, np.uint8))
    assert not np.allclose(reference, 1.0), "premise: these codes decode to a range, not to white"

    try:
        decoded = srgb_to_linear(np.array(_SRGB_CODES, dtype))
    except ValueError as exc:
        # A caller can only act on this if it names both accepted spellings.
        assert "uint8" in str(exc)
        assert "[0, 255]" in str(exc) and "[0, 1]" in str(exc)
        return

    raise AssertionError(
        f"the same codes as {np.dtype(dtype).name} decoded to {np.round(decoded, 4).tolist()} "
        f"instead of {np.round(reference, 4).tolist()}: every code above 1 was read as unit "
        "linear light and clipped to white."
    )


def test_a_python_list_of_byte_codes_states_its_scale_too() -> None:
    """``rgb`` is annotated ``ArrayLike`` and documented as accepting ``[0, 255]``.

    ``np.asarray`` of a list of ints is ``int64``, so the plainest spelling the
    annotation invites is the same ambiguity -- and must reach the same refusal
    rather than the all-white decode.
    """
    with pytest.raises(ValueError, match="cannot tell what scale"):
        srgb_to_linear(_SRGB_CODES)


def test_both_documented_spellings_are_still_accepted() -> None:
    """The guard must not narrow the contract to one of the two scales."""
    codes = np.array(_SRGB_CODES, np.uint8)
    np.testing.assert_allclose(srgb_to_linear(codes), srgb_to_linear(codes.astype(np.float32) / 255.0), atol=1e-7)
    assert srgb_to_linear(np.uint8(255)) == pytest.approx(1.0)
    assert srgb_to_linear(np.float32(1.0)) == pytest.approx(1.0)
    # A bool mask is unit light by construction, not a byte code.
    np.testing.assert_allclose(srgb_to_linear(np.array([True, False])), [1.0, 0.0], atol=1e-6)


def test_the_unit_only_helpers_accept_an_integer_zero_one_array() -> None:
    """The scale guard belongs only on the helper with two scales.

    ``linear_to_srgb`` and ``relative_luminance`` take unit linear light and
    nothing else, so an integer ``0``/``1`` is unambiguous for them; refusing it
    there would reject a legitimate value.
    """
    np.testing.assert_allclose(linear_to_srgb(np.array([0, 1])), [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(relative_luminance(np.array([[0, 1, 0]])), [0.7152], atol=1e-6)
