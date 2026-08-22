"""The frame-level transform surfaces refuse an unusable source-episode value.

The structural sweep in ``tests/simulation/test_replay_episode_index_domain.py``
pins that :func:`~strands_robots.transforms.base.derive_variant_seed` and each
backend's ``transform_frames`` apply the shared non-negative whole-number rule
to ``source_episode``. This file drives the real functions, because what the
sweep protects is behaviour: ``derive_variant_seed(seed, True, 0)`` is
``derive_variant_seed(seed, 1, 0)`` unrefused, so a variant of episode ``True``
would silently share its determinism key - and therefore its pixels - with a
variant of episode 1. ``transform_frames`` is the documented seam a backend
author calls directly, so it refuses the same domain rather than disagreeing
with the orchestration that validated ``spec.episodes`` upstream.
"""

import numpy as np
import pytest

from strands_robots.transforms.base import TransformSpec, derive_variant_seed
from strands_robots.transforms.cosmos_transfer import CosmosTransferTransform
from strands_robots.transforms.mock import MockTransform
from strands_robots.utils import non_negative_whole_number_error

UNUSABLE = [
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
    pytest.param(-1, id="negative"),
    pytest.param(2.5, id="fractional"),
    pytest.param("0", id="str"),
    pytest.param(None, id="None"),
]

_FRAMES = np.zeros((2, 4, 4, 3), dtype=np.uint8)


class TestDeriveVariantSeed:
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_source_episode_is_refused_with_the_shared_verdict(self, value):
        expected = non_negative_whole_number_error(value, "source_episode", "derive_variant_seed")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            derive_variant_seed(7, value, 0)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_does_not_depend_on_a_seed_being_set(self, value):
        """``seed=None`` opts out of determinism, not of the value domain."""
        with pytest.raises(ValueError):
            derive_variant_seed(None, value, 0)

    def test_a_bool_no_longer_derives_episode_ones_seed(self):
        """The load-bearing row: ``True`` would collide with episode 1."""
        with pytest.raises(ValueError):
            derive_variant_seed(7, True, 0)

    @pytest.mark.parametrize("value,expected_episode", [(0, 0), (2, 2), (np.int64(1), 1), (2.0, 2)])
    def test_an_accepted_index_derives_that_episodes_seed(self, value, expected_episode):
        assert derive_variant_seed(7, value, 0) == derive_variant_seed(7, expected_episode, 0)

    def test_none_seed_still_opts_out_for_an_accepted_index(self):
        assert derive_variant_seed(None, 0, 0) is None


class TestTransformFramesRefusesTheSameDomain:
    """The backend seam agrees with the orchestration, whichever way it is reached."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_mock_backend_refuses_before_any_pixel_work(self, value):
        with pytest.raises(ValueError, match="non-negative whole number"):
            MockTransform().transform_frames("cam", _FRAMES, TransformSpec(), source_episode=value, variant=0)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_explicit_shift_mode_refuses_too(self, value):
        """A fixed ``pixel_shift`` never reads the index; the domain still holds."""
        with pytest.raises(ValueError, match="non-negative whole number"):
            MockTransform(pixel_shift=3).transform_frames(
                "cam", _FRAMES, TransformSpec(), source_episode=value, variant=0
            )

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_cosmos_backend_refuses_before_the_pipeline_binding_check(self, value):
        """The domain refusal outranks the unbound-pipeline RuntimeError."""
        with pytest.raises(ValueError, match="non-negative whole number"):
            CosmosTransferTransform().transform_frames("cam", _FRAMES, TransformSpec(), source_episode=value, variant=0)

    def test_an_accepted_index_still_transforms(self):
        """Control: the guard refuses the domain, not the callers."""
        out = MockTransform(pixel_shift=3).transform_frames(
            "cam", _FRAMES, TransformSpec(), source_episode=0, variant=0
        )
        assert out.shape == _FRAMES.shape and out.dtype == _FRAMES.dtype
        assert (out == 3).all()
