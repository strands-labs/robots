# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color-pipeline contract tests (issue #2323, stage 3).

The sRGB transfer functions are light arithmetic's entry and exit points --
the shadow-catcher multiply and the optional linear seam blend both go
through them -- so their round trip and reference values are pinned here as
pure numpy.
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
