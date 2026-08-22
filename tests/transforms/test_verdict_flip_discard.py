"""The re-validation gate discards a generated episode whose verdict flips.

The acceptance gate of the transform surface: when
:attr:`~strands_robots.transforms.base.TransformSpec.revalidate` supplies a
deterministic verdict function, each generated episode is scored against its
SOURCE episode's verdict and a flip discards the variant - measured, not
assumed. This module engineers a flip on purpose: the reference transform's
explicit ``pixel_shift`` moves a dark episode's mean pixel across a threshold
verdict while a bright episode saturates in place, so exactly one of two
generated episodes must be discarded and exactly one written.
"""

import numpy as np
import pytest

from tests.transforms.conftest import episode_column

lerobot = pytest.importorskip("lerobot")

from strands_robots.transforms import TransformSpec, load_provenance  # noqa: E402
from strands_robots.transforms.mock import MockTransform  # noqa: E402


def _mean_pixel_below_50(episode: dict) -> bool:
    """Deterministic pixel verdict: True for a dark episode."""
    return float(episode["observation.images.cam"].mean()) < 50.0


class TestVerdictFlipDiscard:
    @pytest.fixture
    def result_and_output(self, record_source_dataset, tmp_path):
        """Transform a dark episode (verdict flips) and a bright one (stable)."""
        # Episode 0: pixels 0 -> verdict True; +100 shift -> mean 100 -> False (FLIP).
        # Episode 1: pixels 200 -> verdict False; +100 clips to 255 -> False (stable).
        source_root = record_source_dataset([0, 200])
        output_root = str(tmp_path / "gated")
        spec = TransformSpec(
            source_root=source_root,
            output_root=output_root,
            revalidate=_mean_pixel_below_50,
        )
        result = MockTransform(pixel_shift=100).transform(spec)
        assert result.status == "success", result.message
        return result, output_root

    def test_the_flip_is_discarded_and_counted(self, result_and_output):
        result, _ = result_and_output
        assert result.revalidated is True
        assert result.episodes_read == 2
        assert result.episodes_discarded == 1
        assert result.episodes_written == 1

    def test_only_the_stable_episode_reaches_the_output(self, result_and_output):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        result, output_root = result_and_output
        output = LeRobotDataset(repo_id="local/augmented", root=output_root)
        assert output.meta.total_episodes == 1
        records = load_provenance(output_root)
        assert len(records) == 1
        assert records[0]["source_episode_index"] == 1
        assert records[0]["synthetic"] is True
        # The surviving episode is the bright one, saturated by the shift.
        pixels = episode_column(output, 0, "observation.images.cam")
        assert float(pixels.mean()) > 0.8  # decoded floats in [0, 1]

    def test_a_verdict_stable_in_the_true_direction_is_kept_too(self, record_source_dataset, tmp_path):
        """The gate compares verdicts; it does not require the verdict be True."""
        source_root = record_source_dataset([0])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "gated_true"),
            revalidate=_mean_pixel_below_50,
        )
        # A shift of 10 keeps the dark episode's mean below the threshold.
        result = MockTransform(pixel_shift=10).transform(spec)
        assert result.status == "success", result.message
        assert result.episodes_written == 1
        assert result.episodes_discarded == 0

    def test_all_variants_discarded_is_a_measured_success(self, record_source_dataset, tmp_path):
        """Every variant flipping is reported as counts, never masked."""
        source_root = record_source_dataset([0])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "gated_all"),
            variants_per_episode=2,
            revalidate=_mean_pixel_below_50,
        )
        result = MockTransform(pixel_shift=100).transform(spec)
        assert result.status == "success", result.message
        assert result.episodes_written == 0
        assert result.episodes_discarded == 2
        assert result.output_root is None  # nothing was written anywhere

    def test_verdicts_see_the_pass_through_columns(self, record_source_dataset, tmp_path):
        """The verdict function receives actions/state alongside the pixels."""
        seen: list[set] = []

        def verdict(episode: dict) -> bool:
            seen.append(set(episode))
            assert isinstance(episode["action"], np.ndarray)
            return True

        source_root = record_source_dataset([40])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "gated_keys"),
            revalidate=verdict,
        )
        result = MockTransform(pixel_shift=10).transform(spec)
        assert result.status == "success", result.message
        assert seen and all({"action", "observation.state", "observation.images.cam", "task"} <= keys for keys in seen)


class TestVacuousGateIsReported:
    """A verdict that reads no pixel column can never flip - and must say so.

    Guarantee 1 holds every non-image column byte-identical between a source
    episode and each generated variant, so the gate's whole discriminating
    power lives in the ``observation.images.*`` columns: for any verdict that
    consults none of them, ``f(generated) == f(source)`` identically, and
    ``revalidated=True`` with ``episodes_discarded=0`` would render exactly
    like "checked, nothing flipped" over pixels the transform destroyed.
    These tests pin that such a run refuses the gated label instead.
    """

    def test_a_state_only_verdict_cannot_claim_a_gated_run(self, record_source_dataset, tmp_path):
        """Pixel-destroying shift + state-only verdict: written, but NOT a gated pass."""

        def state_verdict(episode: dict) -> bool:
            return float(episode["observation.state"].mean()) < 100.0

        source_root = record_source_dataset([10, 10])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "vacuous"),
            variants_per_episode=3,
            seed=0,
            revalidate=state_verdict,
        )
        # pixel_shift=245 saturates every frame to pure white - the scene is gone.
        result = MockTransform(pixel_shift=245).transform(spec)
        assert result.status == "success", result.message
        # The variants still pass through (nothing could be discarded)...
        assert result.episodes_written == 6
        assert result.episodes_discarded == 0
        # ...but the run does NOT report a clean gated pass, and names the cause.
        assert result.revalidated is False
        assert "observation.images" in result.message
        assert "NOT gated" in result.message

    def test_a_pixel_verdict_still_claims_the_gate(self, record_source_dataset, tmp_path):
        """Control: a verdict that indexes an image column keeps ``revalidated=True``."""
        source_root = record_source_dataset([10])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "pixel_gated"),
            revalidate=_mean_pixel_below_50,
        )
        result = MockTransform(pixel_shift=10).transform(spec)
        assert result.status == "success", result.message
        assert result.revalidated is True
        assert "NOT gated" not in result.message

    def test_a_verdict_reading_pixels_through_items_is_not_accused(self, record_source_dataset, tmp_path):
        """A verdict iterating ``episode.items()`` received the pixel values too."""

        def items_verdict(episode: dict) -> bool:
            for key, value in episode.items():
                if key.startswith("observation.images."):
                    return float(value.mean()) < 50.0
            return True

        source_root = record_source_dataset([10])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "items_gated"),
            revalidate=items_verdict,
        )
        result = MockTransform(pixel_shift=10).transform(spec)
        assert result.status == "success", result.message
        assert result.revalidated is True

    def test_a_verdict_comparing_the_whole_mapping_is_not_accused(self, record_source_dataset, tmp_path):
        """``episode == other`` reads every column's value at C level - recorded too.

        ``dict.__eq__`` is a bulk read exactly like ``items()``: the probe
        must record every key on it, or an equality-based verdict would be
        falsely accused of consulting no pixels (CodeQL py/missing-equals on
        the probe adding ``consulted`` without overriding ``__eq__``).
        """

        def equality_verdict(episode: dict) -> bool:
            # Compares the whole mapping; deterministic (False) on source and
            # every generated variant alike, so nothing is discarded.
            return episode == {"not": "the episode"}

        source_root = record_source_dataset([10])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "eq_gated"),
            revalidate=equality_verdict,
        )
        result = MockTransform(pixel_shift=10).transform(spec)
        assert result.status == "success", result.message
        assert result.revalidated is True
        assert "NOT gated" not in result.message

    @pytest.mark.parametrize(
        ("spelling", "verdict"),
        [
            ("dict(episode)", lambda ep: float(dict(ep)["observation.images.cam"].mean()) < 50.0),
            ("{**episode}", lambda ep: float({**ep}["observation.images.cam"].mean()) < 50.0),
            ("episode.copy()", lambda ep: float(ep.copy()["observation.images.cam"].mean()) < 50.0),
            ("episode | {}", lambda ep: float((ep | {})["observation.images.cam"].mean()) < 50.0),
        ],
        ids=["dict_call", "splat", "copy_method", "union"],
    )
    def test_a_verdict_reading_pixels_through_a_defensive_copy_is_not_accused(
        self, record_source_dataset, tmp_path, spelling, verdict
    ):
        """Copying the mapping before reading it is still receiving every value.

        Each verdict here is the same predicate as ``_mean_pixel_below_50``,
        spelled with the defensive copy a verdict makes before touching the
        caller's mapping. The copy hands over every column, pixels included, so
        the run is gated - and it demonstrably is, because the shift flips the
        verdict and the variant is discarded. Reporting such a run as ungated
        would put "discarded 1 on the re-validation gate" and "this verdict
        cannot flip" in one payload.
        """
        source_root = record_source_dataset([0])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "copy_gated"),
            revalidate=verdict,
        )
        result = MockTransform(pixel_shift=100).transform(spec)
        assert result.status == "success", result.message
        assert result.episodes_discarded == 1, f"premise: {spelling} must flip on this shift - {result.message}"
        assert result.revalidated is True, (
            f"a verdict reading pixels through {spelling} received them, yet the run refuses the gated label"
        )
        assert "NOT gated" not in result.message
        assert "observation.images" not in result.message

    @pytest.mark.parametrize(
        ("spelling", "verdict"),
        [
            ("episode[key]", _mean_pixel_below_50),
            ("dict(episode)", lambda ep: float(dict(ep)["observation.images.cam"].mean()) < 50.0),
            ("{**episode}", lambda ep: float({**ep}["observation.images.cam"].mean()) < 50.0),
            ("episode.items()", lambda ep: any(float(v.mean()) < 50.0 for k, v in ep.items() if k.endswith(".cam"))),
        ],
        ids=["subscript", "dict_call", "splat", "items"],
    )
    def test_a_discarding_run_never_reports_itself_ungated(self, record_source_dataset, tmp_path, spelling, verdict):
        """The invariant: a gate that discarded provably discriminated.

        ``episodes_discarded > 0`` is a measurement that the verdict answered
        differently on a generated variant than on its source. A run reporting
        that alongside ``revalidated=False`` - whose message says the verdict
        cannot flip - contradicts itself, whichever read path the verdict used.
        """
        source_root = record_source_dataset([0])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "no_contradiction"),
            revalidate=verdict,
        )
        result = MockTransform(pixel_shift=100).transform(spec)
        assert result.status == "success", result.message
        assert result.episodes_discarded > 0, f"premise: {spelling} must flip on this shift - {result.message}"
        assert result.revalidated is True, (
            f"{spelling}: discarded {result.episodes_discarded} on the gate and still reports it ungated - "
            f"{result.message}"
        )

    def test_the_vacuous_gate_still_flips_on_a_pixel_verdict_elsewhere(self, record_source_dataset, tmp_path):
        """The discard machinery is untouched: the pixel verdict still discards a flip."""
        source_root = record_source_dataset([0])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "still_flips"),
            revalidate=_mean_pixel_below_50,
        )
        result = MockTransform(pixel_shift=100).transform(spec)
        assert result.status == "success", result.message
        assert result.revalidated is True
        assert result.episodes_discarded == 1
