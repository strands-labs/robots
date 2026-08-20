"""Behavior tests for the episode-label sidecar and its verdict precedence.

The contract under test is the two-stage verdict from the episode-judge
design: deterministic benchmark predicates are AUTHORITATIVE for success /
failure, and the judge only annotates on top of them. The precedence pin the
feature ships with lives in :class:`TestDeterministicVerdictPrecedence` - a
judge disagreement is recorded as an annotation (``disputes_verdict``), never
as an overridden verdict.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.episode_labels import (
    FAILURE_MODES,
    LABEL_SCHEMA_VERSION,
    QUALITY_GRADES,
    SIDECAR_FILENAME,
    annotate_episode,
    deterministic_verdict,
    filter_episodes,
    labels_path,
    measure_agreement,
    read_labels,
    record_deterministic_verdicts,
)
from strands_robots.utils import boolean_flag_error, non_negative_whole_number_error

#: Values the shared boolean-flag domain refuses. Asserted against the shared
#: guard inside each test, so a spelling added to the domain is covered here
#: without an edit and the two verdicts cannot drift apart.
UNUSABLE_FLAGS = ["false", "no", "off", "0", "true", 1, 0, "", None, [], float("nan")]


def _verdicts(**overrides):
    """Two-episode verdict list in evaluate_benchmark's per-episode shape."""
    episodes = [
        {"episode": 0, "success": True, "failure": False, "steps": 120, "cumulative_reward": 3.5, "seed": 7},
        {"episode": 1, "success": False, "failure": True, "steps": 40, "cumulative_reward": -1.0, "seed": 8},
    ]
    for entry in episodes:
        entry.update(overrides)
    return episodes


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    return root


class TestRecordDeterministicVerdicts:
    def test_round_trip_preserves_the_verdicts_and_versions_the_schema(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts(), benchmark="go2_walk_short")

        document = read_labels(dataset_root)
        assert document["schema_version"] == LABEL_SCHEMA_VERSION
        assert document["benchmark"] == "go2_walk_short"
        assert deterministic_verdict(dataset_root, 0) == {
            "success": True,
            "failure": False,
            "steps": 120,
            "cumulative_reward": 3.5,
            "seed": 7,
        }
        assert deterministic_verdict(dataset_root, 1)["success"] is False

    def test_sidecar_lands_next_to_the_dataset(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        assert labels_path(dataset_root) == dataset_root / SIDECAR_FILENAME
        assert labels_path(dataset_root).is_file()

    def test_missing_root_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not an existing directory"):
            record_deterministic_verdicts(tmp_path / "nowhere", _verdicts())

    def test_empty_episode_list_is_refused(self, dataset_root):
        with pytest.raises(ValueError, match="non-empty list"):
            record_deterministic_verdicts(dataset_root, [])

    @pytest.mark.parametrize("bad_index", [-1, 2.5, "0", None, float("nan")])
    def test_episode_index_is_checked_on_the_shared_whole_number_domain(self, dataset_root, bad_index):
        assert non_negative_whole_number_error(bad_index, "episode", "test") is not None
        with pytest.raises(ValueError):
            record_deterministic_verdicts(dataset_root, [{"episode": bad_index, "success": True}])

    @pytest.mark.parametrize("bad_flag", UNUSABLE_FLAGS)
    def test_success_is_checked_on_the_shared_boolean_domain(self, dataset_root, bad_flag):
        assert boolean_flag_error(bad_flag, "success", "test") is not None
        with pytest.raises(ValueError):
            record_deterministic_verdicts(dataset_root, [{"episode": 0, "success": bad_flag}])


class TestDeterministicVerdictPrecedence:
    """The acceptance pin: the judge can never overturn a deterministic verdict."""

    def test_a_disagreeing_judge_is_recorded_as_a_dispute_and_the_verdict_stands(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        before = deterministic_verdict(dataset_root, 1)
        assert before["success"] is False

        # The judge believes episode 1 actually succeeded. That opinion is
        # recorded as an annotation - the deterministic block is untouched.
        record = annotate_episode(
            dataset_root,
            1,
            quality="medium",
            note="task completed on the video despite the predicate verdict",
            success_opinion=True,
            model="mock-judge",
        )
        assert record["judge"]["disputes_verdict"] is True
        assert record["judge"]["success_opinion"] is True
        assert record["deterministic"] == before

        # And on disk, re-read: the verdict is byte-identical.
        assert deterministic_verdict(dataset_root, 1) == before

    def test_an_agreeing_judge_records_no_dispute(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        record = annotate_episode(dataset_root, 0, quality="high", success_opinion=True)
        assert record["judge"]["disputes_verdict"] is False

    def test_annotating_an_episode_with_no_verdict_is_refused(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        with pytest.raises(ValueError, match="No deterministic verdict recorded for episode 5"):
            annotate_episode(dataset_root, 5, quality="high")

    def test_annotating_before_any_verdict_exists_is_refused(self, dataset_root):
        with pytest.raises(FileNotFoundError, match="Record deterministic verdicts first"):
            annotate_episode(dataset_root, 0, quality="high")

    def test_re_recording_a_verdict_preserves_the_judge_annotation(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        annotate_episode(dataset_root, 0, quality="high", note="smooth")

        # A re-evaluation replaces the measurement; the annotation is about
        # the recorded frames, which did not change.
        record_deterministic_verdicts(dataset_root, [{"episode": 0, "success": False, "failure": True}])
        document = read_labels(dataset_root)
        assert document["episodes"]["0"]["deterministic"]["success"] is False
        assert document["episodes"]["0"]["judge"]["note"] == "smooth"


class TestAnnotateEpisodeDomains:
    def test_quality_outside_the_grade_vocabulary_is_refused(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        with pytest.raises(ValueError, match="quality must be one of"):
            annotate_episode(dataset_root, 0, quality="excellent")

    def test_failure_mode_outside_the_taxonomy_is_refused(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        with pytest.raises(ValueError, match="failure_mode must be None or one of"):
            annotate_episode(dataset_root, 0, quality="low", failure_mode="sloppy")

    def test_failure_mode_is_legal_on_a_successful_episode(self, dataset_root):
        # near_miss / wrong_but_lucky exist to annotate SUCCESSES worth
        # excluding from training data.
        record_deterministic_verdicts(dataset_root, _verdicts())
        record = annotate_episode(dataset_root, 0, quality="low", failure_mode="wrong_but_lucky")
        assert record["deterministic"]["success"] is True
        assert record["judge"]["failure_mode"] == "wrong_but_lucky"

    @pytest.mark.parametrize("bad_flag", UNUSABLE_FLAGS)
    def test_success_opinion_is_checked_on_the_shared_boolean_domain(self, dataset_root, bad_flag):
        if bad_flag is None:
            pytest.skip("None means 'no opinion offered' and is a documented value here")
        assert boolean_flag_error(bad_flag, "success_opinion", "test") is not None
        record_deterministic_verdicts(dataset_root, _verdicts())
        with pytest.raises(ValueError):
            annotate_episode(dataset_root, 0, quality="high", success_opinion=bad_flag)

    def test_non_string_note_and_model_are_refused(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        with pytest.raises(ValueError, match="note must be a string"):
            annotate_episode(dataset_root, 0, quality="high", note=42)
        with pytest.raises(ValueError, match="model must be a string"):
            annotate_episode(dataset_root, 0, quality="high", model=42)

    def test_every_taxonomy_tag_is_writable(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        for mode in FAILURE_MODES:
            record = annotate_episode(dataset_root, 0, quality="low", failure_mode=mode)
            assert record["judge"]["failure_mode"] == mode


class TestReadLabels:
    def test_missing_sidecar_names_the_remedy(self, dataset_root):
        with pytest.raises(FileNotFoundError, match="record_deterministic_verdicts"):
            read_labels(dataset_root)

    def test_a_future_schema_version_is_refused_not_reinterpreted(self, dataset_root):
        labels_path(dataset_root).write_text(json.dumps({"schema_version": 99, "episodes": {}}))
        with pytest.raises(ValueError, match="schema_version 99"):
            read_labels(dataset_root)

    def test_unparseable_sidecar_is_reported_as_such(self, dataset_root):
        labels_path(dataset_root).write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            read_labels(dataset_root)


class TestFilterEpisodes:
    @pytest.fixture
    def labeled_root(self, dataset_root):
        record_deterministic_verdicts(
            dataset_root,
            [
                {"episode": 0, "success": True, "failure": False},  # high quality
                {"episode": 1, "success": True, "failure": False},  # low quality
                {"episode": 2, "success": False, "failure": True},  # judged high, but failed
                {"episode": 3, "success": True, "failure": False},  # unlabeled
                {"episode": 4, "success": True, "failure": False},  # disputed
            ],
        )
        annotate_episode(dataset_root, 0, quality="high")
        annotate_episode(dataset_root, 1, quality="low")
        annotate_episode(dataset_root, 2, quality="high", success_opinion=True)
        annotate_episode(dataset_root, 4, quality="high", success_opinion=False)
        return dataset_root

    def test_success_filter_reads_the_deterministic_field_not_the_opinion(self, labeled_root):
        # Episode 2 failed deterministically; the judge's opinion that it
        # succeeded does not admit it - precedence applies to filtering too.
        assert filter_episodes(labeled_root, require_success=True, min_quality="high") == [0, 4]

    def test_quality_threshold_orders_the_grades(self, labeled_root):
        assert filter_episodes(labeled_root, min_quality="low") == [0, 1, 4]
        assert filter_episodes(labeled_root, min_quality="high") == [0, 4]

    def test_unlabeled_episodes_never_clear_a_quality_bar(self, labeled_root):
        assert 3 not in filter_episodes(labeled_root, min_quality="low")

    def test_exclude_disputed_drops_the_disputed_success(self, labeled_root):
        assert filter_episodes(labeled_root, min_quality="high", exclude_disputed=True) == [0]

    def test_require_success_false_admits_the_judged_failure(self, labeled_root):
        assert filter_episodes(labeled_root, require_success=False, min_quality="high") == [0, 2, 4]

    @pytest.mark.parametrize("bad_flag", UNUSABLE_FLAGS)
    @pytest.mark.parametrize("flag", ["require_success", "exclude_disputed"])
    def test_flags_are_checked_on_the_shared_boolean_domain(self, labeled_root, flag, bad_flag):
        assert boolean_flag_error(bad_flag, flag, "test") is not None
        with pytest.raises(ValueError):
            filter_episodes(labeled_root, **{flag: bad_flag})

    def test_min_quality_outside_the_vocabulary_is_refused(self, labeled_root):
        assert "excellent" not in QUALITY_GRADES
        with pytest.raises(ValueError, match="min_quality must be one of"):
            filter_episodes(labeled_root, min_quality="excellent")


class TestMeasureAgreement:
    @pytest.fixture
    def judged_root(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        annotate_episode(dataset_root, 0, quality="high", failure_mode=None)
        annotate_episode(dataset_root, 1, quality="low", failure_mode="incomplete")
        return dataset_root

    def test_agreement_fractions_and_disagreement_rows(self, judged_root):
        report = measure_agreement(
            judged_root,
            {
                0: {"quality": "high"},
                1: {"quality": "medium", "failure_mode": "incomplete"},
            },
        )
        assert report["episodes_compared"] == 2
        assert report["quality_agreement"] == 0.5
        assert report["failure_mode_agreement"] == 1.0
        assert report["disagreements"] == [{"episode": 1, "field": "quality", "judge": "low", "human": "medium"}]

    def test_empty_holdout_is_refused(self, judged_root):
        with pytest.raises(ValueError, match="non-empty mapping"):
            measure_agreement(judged_root, {})

    def test_a_holdout_with_no_judged_episode_is_not_a_measurement(self, dataset_root):
        record_deterministic_verdicts(dataset_root, _verdicts())
        with pytest.raises(ValueError, match="agreement over nothing"):
            measure_agreement(dataset_root, {0: {"quality": "high"}})

    def test_a_malformed_holdout_entry_is_refused(self, judged_root):
        with pytest.raises(ValueError, match="'quality'"):
            measure_agreement(judged_root, {0: {"grade": "high"}})
