"""Round trip: transform a recorded dataset, reopen it, and hold the contract.

The acceptance criteria of the transform surface, asserted on real files
(recording round-trip convention: start -> write -> stop -> reopen -> assert):

* **schema parity** - the augmented dataset declares the same features, fps
  and robot type as the source;
* **action-column byte-equality** - the pass-through half of the contract:
  a generated episode is the same trajectory (actions, states, tasks)
  rendered differently, so those columns survive byte-identical;
* **pixels actually changed** - the transformed half really happened;
* **provenance present** - every generated episode carries ``synthetic=true``,
  its source episode, and the transform's name and version.
"""

import numpy as np
import pytest

from tests.transforms.conftest import episode_column

lerobot = pytest.importorskip("lerobot")

from strands_robots.transforms import TransformSpec, create_transform, load_provenance  # noqa: E402


@pytest.fixture
def transformed(record_source_dataset, tmp_path):
    """Record 2 episodes, transform them into 2 variants each, reopen both sides."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = record_source_dataset([40, 160])
    output_root = str(tmp_path / "augmented")

    transform = create_transform("mock")
    spec = TransformSpec(
        source_root=source_root,
        output_root=output_root,
        variants_per_episode=2,
        seed=7,
    )
    result = transform.transform(spec)
    assert result.status == "success", result.message

    source = LeRobotDataset(repo_id="local/source", root=source_root)
    output = LeRobotDataset(repo_id="local/augmented", root=output_root)
    return source, output, result


class TestRoundTrip:
    def test_counts_multiply_the_data(self, transformed):
        source, output, result = transformed
        assert result.episodes_read == 2
        assert result.episodes_written == 4
        assert result.episodes_discarded == 0
        assert output.meta.total_episodes == 4
        # No verdict function was supplied, and the result says so honestly.
        assert result.revalidated is False
        assert "NOT gated" in result.message

    def test_schema_parity(self, transformed):
        source, output, _ = transformed
        src_features = dict(source.meta.features)
        out_features = dict(output.meta.features)
        assert set(src_features) == set(out_features)
        for key in ("observation.state", "action", "observation.images.cam"):
            assert tuple(src_features[key]["shape"]) == tuple(out_features[key]["shape"]), key
            assert src_features[key]["dtype"] == out_features[key]["dtype"], key
        assert source.meta.fps == output.meta.fps
        assert source.meta.robot_type == output.meta.robot_type

    def test_action_and_state_columns_pass_through_byte_identical(self, transformed):
        source, output, _ = transformed
        provenance = {r["episode_index"]: r for r in load_provenance(output.root)}
        for out_ep in range(output.meta.total_episodes):
            src_ep = provenance[out_ep]["source_episode_index"]
            for key in ("action", "observation.state"):
                src_col = episode_column(source, src_ep, key)
                out_col = episode_column(output, out_ep, key)
                assert src_col.tobytes() == out_col.tobytes(), (key, out_ep)
                # Not vacuous: the recorded columns carry non-zero values.
                assert np.any(src_col != 0.0), key

    def test_task_strings_pass_through(self, transformed):
        source, output, _ = transformed
        provenance = {r["episode_index"]: r for r in load_provenance(output.root)}
        for out_ep in range(output.meta.total_episodes):
            src_ep = provenance[out_ep]["source_episode_index"]
            start = int(output.meta.episodes[out_ep]["dataset_from_index"])
            src_start = int(source.meta.episodes[src_ep]["dataset_from_index"])
            assert output[start]["task"] == source[src_start]["task"] == f"task-{src_ep}"

    def test_pixels_changed(self, transformed):
        source, output, _ = transformed
        src_img = episode_column(source, 0, "observation.images.cam")
        out_img = episode_column(output, 0, "observation.images.cam")
        # Decoded pixels are lossy-coded floats; a real shift dwarfs codec noise.
        assert abs(float(src_img.mean()) - float(out_img.mean())) > 0.01

    def test_provenance_marks_every_generated_episode(self, transformed):
        _, output, result = transformed
        records = load_provenance(output.root)
        assert len(records) == 4
        assert result.provenance_path is not None
        for record in records:
            assert record["synthetic"] is True
            assert record["transform"] == "mock"
            assert record["transform_version"] == "1"
            assert record["source_repo_id"] == "local/source"
        assert [r["source_episode_index"] for r in records] == [0, 0, 1, 1]
        assert [r["variant"] for r in records] == [0, 1, 0, 1]
        # Distinct variants generated under distinct derived seeds.
        seeds = [r["seed"] for r in records]
        assert len(set(seeds)) == 4


class TestSelectors:
    def test_episode_subset_transforms_only_the_named_episodes(self, record_source_dataset, tmp_path):
        source_root = record_source_dataset([40, 160])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "aug_subset"),
            episodes=[1],
            seed=3,
        )
        result = create_transform("mock").transform(spec)
        assert result.status == "success", result.message
        assert result.episodes_read == 1
        records = load_provenance(tmp_path / "aug_subset")
        assert [r["source_episode_index"] for r in records] == [1]

    def test_out_of_range_episode_is_an_error_result(self, record_source_dataset, tmp_path):
        source_root = record_source_dataset([40])
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / "aug_oob"),
            episodes=[5],
        )
        result = create_transform("mock").transform(spec)
        assert result.status == "error"
        assert "out of range" in result.message
        assert not (tmp_path / "aug_oob").exists()

    def test_validation_failure_writes_nothing(self, tmp_path):
        result = create_transform("mock").transform(TransformSpec(output_root=str(tmp_path / "aug_none")))
        assert result.status == "error"
        assert "validation failed" in result.message
        assert not (tmp_path / "aug_none").exists()
