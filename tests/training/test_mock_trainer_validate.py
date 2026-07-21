"""Direct unit tests for ``MockTrainer.validate`` required-field preflight.

:class:`~strands_robots.training.mock.MockTrainer` is the dependency-free
reference implementation of the :class:`~strands_robots.training.base.Trainer`
contract, so its ``validate`` is the worked example every backend mirrors.
Each ``Trainer.validate`` is defense-in-depth: it must independently reject an
incomplete ``TrainSpec`` even though the ``train_policy`` tool layer also
screens arguments first. The tool-layer tests therefore never reach the
backend's own empty-``dataset_root`` guard (they either pass a non-empty path
or are short-circuited by the tool's arg check), leaving that branch unpinned.
These tests exercise ``MockTrainer().validate`` directly so the backend-level
required-field contract holds regardless of the caller.
"""

from __future__ import annotations

from strands_robots.training import TrainSpec
from strands_robots.training.mock import MockTrainer


class TestRequiredFieldPreflight:
    def test_bare_spec_reports_every_required_field(self):
        # A default TrainSpec has empty dataset_root / base_model / output_dir,
        # so the backend must flag all three - it cannot assume the tool layer
        # already filled them in.
        problems = MockTrainer().validate(TrainSpec())

        assert "dataset_root is required" in problems
        assert "base_model is required" in problems
        assert "output_dir is required" in problems

    def test_empty_dataset_root_short_circuits_dataset_layout_check(self, tmp_path):
        # With base_model / output_dir supplied, an empty dataset_root must
        # yield the "required" problem and NOT the "not a LeRobotDataset v3
        # root" message: the required-field guard fires before the on-disk
        # layout check, so the two are never reported together.
        problems = MockTrainer().validate(
            TrainSpec(
                dataset_root="",
                base_model="mock/base",
                output_dir=str(tmp_path / "out"),
            )
        )

        assert "dataset_root is required" in problems
        assert not any("not a LeRobotDataset v3 root" in p for p in problems)
