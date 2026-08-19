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
