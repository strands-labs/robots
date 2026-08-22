"""The frame-level transform surfaces refuse an unusable determinism-key input.

:func:`~strands_robots.transforms.base.derive_variant_seed` spreads the triple
``(seed, source_episode, variant)`` through one
:class:`~numpy.random.SeedSequence`, so all three name the stream a generated
variant is rendered from. The structural sweep in
``tests/simulation/test_replay_episode_index_domain.py`` pins the
``source_episode`` half - that sweep's scope is the episode-index quantity - and
this file drives the real functions, because what is protected is behaviour:
``derive_variant_seed(seed, True, 0)`` is ``derive_variant_seed(seed, 1, 0)``
unrefused, so a variant of episode ``True`` would silently share its
determinism key - and therefore its pixels - with a variant of episode 1.

The other two inputs to that one expression reached NumPy unchecked and
degraded exactly the same way, measured on mid-grey frames through
``MockTransform``:

==================  ==============================  =====================
input               pre-fix outcome                 collided with
==================  ==============================  =====================
``variant=True``    pixel 110, no refusal           ``variant=1``
``variant=False``   pixel 121, no refusal           ``variant=0``
``variant="1"``     pixel 110, no refusal           ``variant=1``
``seed=True``       key 1835504127, no refusal      ``seed=1``
``seed="1"``        key 1835504127, no refusal      ``seed=1``
``variant=-1``      ``ValueError`` from NumPy       -
``variant=2.5``     ``TypeError`` from NumPy        -
``variant=None``    ``TypeError`` from NumPy        -
==================  ==============================  =====================

Three classes in one parameter, the same three the episode-index sweep records:
a **bool named a stream another variant already owned** (and so did a str
spelling of a whole number, which NumPy coerces), leaving two "distinct"
variants of one episode written as two episodes whose pixels are
byte-identical; the values NumPy refuses on its own **named neither the
parameter nor the surface**; and two of them raised ``TypeError``, which is not
the ``ValueError`` these surfaces document as their refusal channel.

``transform_frames`` is the documented seam a backend author calls directly, so
it refuses the same domain rather than disagreeing with the orchestration that
validated ``spec.episodes`` / ``spec.variants_per_episode`` upstream - and
``variant`` has to be refused there as well as inside ``derive_variant_seed``,
because a backend need never reach that function: ``MockTransform``'s explicit
``pixel_shift`` mode derives no key at all, so pre-fix it accepted every value
in the table above without reading the counter once.
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

#: ``seed`` alone admits ``None`` - the documented opt-out from determinism,
#: not a stream name - so it is graded by its own control test instead.
UNUSABLE_SEED = [param for param in UNUSABLE if param.values[0] is not None]

_FRAMES = np.zeros((2, 4, 4, 3), dtype=np.uint8)


class _RecordingPipeline:
    """Minimal video2video pipeline that records whether it saw pixels at all."""

    version = "recording"

    def __init__(self) -> None:
        self.calls = 0
        self.seeds: list[int | None] = []

    def generate(self, video, prompt="", seed=None):
        self.calls += 1
        self.seeds.append(seed)
        return video


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


class TestTheWholeDeterminismKeyIsHeldToTheDomain:
    """Every input the key is spread from, not one of the three."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_variant_is_refused_with_the_shared_verdict(self, value):
        expected = non_negative_whole_number_error(value, "variant", "derive_variant_seed")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            derive_variant_seed(7, 0, value)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("value", UNUSABLE_SEED)
    def test_an_unusable_seed_is_refused_with_the_shared_verdict(self, value):
        expected = non_negative_whole_number_error(value, "seed", "derive_variant_seed")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            derive_variant_seed(value, 0, 0)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_variant_refusal_does_not_depend_on_a_seed_being_set(self, value):
        """``seed=None`` opts out of determinism, not of the value domain."""
        with pytest.raises(ValueError):
            derive_variant_seed(None, 0, value)

    @pytest.mark.parametrize("unusable,peer", [(True, 1), (False, 0), ("1", 1), ("0", 0)])
    def test_a_variant_no_longer_names_another_variants_stream(self, unusable, peer):
        """The load-bearing rows: pre-fix these two derived one key."""
        with pytest.raises(ValueError):
            derive_variant_seed(7, 0, unusable)
        assert derive_variant_seed(7, 0, peer) is not None

    @pytest.mark.parametrize("unusable,peer", [(True, 1), (False, 0), ("1", 1)])
    def test_a_seed_no_longer_names_another_seeds_stream(self, unusable, peer):
        with pytest.raises(ValueError):
            derive_variant_seed(unusable, 0, 0)
        assert derive_variant_seed(peer, 0, 0) is not None

    def test_none_seed_is_still_the_documented_opt_out(self):
        """Control: the one spelling that is absence rather than a stream."""
        assert derive_variant_seed(None, 0, 0) is None
        assert derive_variant_seed(None, 3, 2) is None

    @pytest.mark.parametrize(
        "triple,key",
        [
            ((7, 0, 0), 2083679832),
            ((7, 0, 1), 2028854884),
            ((7, 0, 2), 3835300069),
            ((7, 2, 1), 1060258595),
            ((11, 2, 1), 2549836301),
        ],
    )
    def test_an_accepted_triple_derives_the_key_it_derived_before(self, triple, key):
        """Control: widening the guard did not move any accepted stream.

        The values are the pre-fix outputs, so this fails if the coercion
        moved to before the guard (or changed what is spread into the
        sequence) rather than after it.
        """
        assert derive_variant_seed(*triple) == key

    @pytest.mark.parametrize("integral_float,whole", [(2.0, 2), (0.0, 0)])
    def test_an_integral_float_is_honored_on_every_input(self, integral_float, whole):
        """The shared rule accepts one, so all three inputs do - as ``source_episode`` already did."""
        assert derive_variant_seed(7, 0, integral_float) == derive_variant_seed(7, 0, whole)
        assert derive_variant_seed(integral_float, 0, 0) == derive_variant_seed(whole, 0, 0)

    def test_accepted_variants_still_name_distinct_streams(self):
        """Control: the guard refuses the domain, not the variance the key exists for."""
        keys = {derive_variant_seed(7, 0, v) for v in range(4)}
        assert len(keys) == 4


class TestTransformFramesRefusesAnUnusableVariant:
    """The backend seam agrees with the orchestration on the counter too."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_mock_backend_refuses_before_any_pixel_work(self, value):
        with pytest.raises(ValueError, match="variant must be a non-negative whole number"):
            MockTransform().transform_frames("cam", _FRAMES, TransformSpec(seed=7), source_episode=0, variant=value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_explicit_shift_mode_refuses_too(self, value):
        """The row that needs a guard here rather than in the shared helper.

        A fixed ``pixel_shift`` derives no key, so pre-fix this path shifted
        the pixels without reading the counter for any value in the table.
        """
        with pytest.raises(ValueError, match="variant must be a non-negative whole number"):
            MockTransform(pixel_shift=3).transform_frames(
                "cam", _FRAMES, TransformSpec(), source_episode=0, variant=value
            )

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_cosmos_backend_refuses_before_the_pipeline_binding_check(self, value):
        """The domain refusal outranks the unbound-pipeline RuntimeError.

        Pre-fix an unusable counter was reported as a wiring problem: "no
        video2video pipeline is bound", naming the seam the caller had got right.
        """
        with pytest.raises(ValueError, match="variant must be a non-negative whole number"):
            CosmosTransferTransform().transform_frames(
                "cam", _FRAMES, TransformSpec(seed=7), source_episode=0, variant=value
            )

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_bound_pipeline_never_sees_pixels_for_an_unusable_variant(self, value):
        pipeline = _RecordingPipeline()
        with pytest.raises(ValueError, match="variant must be a non-negative whole number"):
            CosmosTransferTransform(pipeline=pipeline).transform_frames(
                "cam", _FRAMES, TransformSpec(seed=7), source_episode=0, variant=value
            )
        assert pipeline.calls == 0

    def test_an_accepted_variant_still_transforms(self):
        """Control: the guard refuses the domain, not the callers."""
        out = MockTransform(pixel_shift=3).transform_frames(
            "cam", _FRAMES, TransformSpec(), source_episode=0, variant=2
        )
        assert out.shape == _FRAMES.shape and out.dtype == _FRAMES.dtype
        assert (out == 3).all()

    def test_two_accepted_variants_of_one_episode_still_differ(self):
        """Control: two variants are two renderings, which is the point of the key."""
        grey = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
        spec = TransformSpec(seed=7)
        first = MockTransform().transform_frames("cam", grey, spec, source_episode=0, variant=0)
        second = MockTransform().transform_frames("cam", grey, spec, source_episode=0, variant=1)
        assert not np.array_equal(first, second)

    def test_a_bound_pipeline_still_sees_an_accepted_variants_pixels(self):
        pipeline = _RecordingPipeline()
        out = CosmosTransferTransform(pipeline=pipeline).transform_frames(
            "cam", _FRAMES, TransformSpec(seed=7), source_episode=0, variant=1
        )
        assert pipeline.calls == 1
        assert out.shape == _FRAMES.shape
        assert pipeline.seeds == [derive_variant_seed(7, 0, 1)]
