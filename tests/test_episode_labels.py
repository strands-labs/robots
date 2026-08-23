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


class TestReadLabelsRefusesAnUnreadableRecord:
    """A sidecar record this module cannot read is refused, not reinterpreted.

    :func:`read_labels` is the boundary for a sidecar this module did not
    write - the schema is documented as the interop format so downstream
    training can filter without rewriting the dataset - and it already refuses
    an unparseable document and an unknown ``schema_version`` on the stated
    grounds that projecting one onto this version's field meanings "would hand
    downstream filters wrong answers with nothing raised". These pin that the
    same rule reaches the RECORDS, because the verdict-bearing fields are
    where that harm lands: an out-of-vocabulary ``quality`` reached
    :func:`filter_episodes` as ``tuple.index(x): x not in tuple`` and
    :func:`measure_agreement` as a silently understated calibration, and a
    ``success`` spelled ``"false"`` is truthy, so a deterministically FAILED
    episode cleared ``require_success=True``.

    The split is the one :mod:`strands_robots.transforms.provenance` - the
    other per-episode sidecar - already makes at both ends, citing this
    module's writers for it: the keys a reader turns into a verdict are held
    to their domains, and the descriptive keys are carried through untouched
    (:class:`TestAnUnreadableRecordIsOnlyAVerdictField`).
    """

    def _sidecar(self, root, mutate):
        """Build a valid two-episode sidecar, then mutate it on disk."""
        record_deterministic_verdicts(
            root,
            [
                {"episode": 0, "success": True, "failure": False, "steps": 100},
                {"episode": 1, "success": True, "failure": False, "steps": 110},
            ],
            benchmark="reach",
        )
        for index in (0, 1):
            annotate_episode(root, index, quality="high", model="judge")
        path = labels_path(root)
        document = json.loads(path.read_text())
        mutate(document)
        path.write_text(json.dumps(document))
        return root

    @pytest.mark.parametrize("grade", ["excellent", "HIGH", "Medium", "", "9", None, 2])
    def test_a_grade_outside_the_vocabulary_names_the_sidecar_and_the_vocabulary(self, dataset_root, grade):
        """The refusal names where the grade came from and what it may be.

        ``tuple.index(x): x not in tuple`` named neither the sidecar, the
        episode, the field nor the vocabulary - and ``filter_episodes``
        refuses the CALLER's ``min_quality`` on exactly that vocabulary in the
        same function, so this is the other half of one question.
        """
        assert grade not in QUALITY_GRADES
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"]["judge"].update({"quality": grade}))
        with pytest.raises(ValueError) as excinfo:
            filter_episodes(root, min_quality="medium")
        message = str(excinfo.value)
        assert SIDECAR_FILENAME in message
        assert "'1'" in message
        assert "quality" in message
        assert str(QUALITY_GRADES) in message

    def test_an_unreadable_grade_is_not_a_silently_understated_calibration(self, dataset_root):
        """:func:`measure_agreement` refuses rather than counting a disagreement.

        The measurement it exists for is "should this judge be deciding what a
        policy trains on"; an unreadable grade never equals a valid human one,
        so it was scored as a disagreement and the reported agreement came out
        BELOW the truth - the direction that makes a sound judge look unsound.
        """
        holdout = {0: {"quality": "high"}, 1: {"quality": "high"}}
        clean = dataset_root / "clean"
        clean.mkdir()
        self._sidecar(clean, lambda d: None)
        assert measure_agreement(clean, holdout)["quality_agreement"] == 1.0

        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"]["judge"].update({"quality": "excellent"}))
        with pytest.raises(ValueError, match="quality"):
            report = measure_agreement(root, holdout)
            pytest.fail(
                f"an unreadable grade was scored as a disagreement: quality_agreement came back "
                f"{report['quality_agreement']:.4f} where the same holdout measures 1.0000 against "
                f"every readable grade, and the report names it as "
                f"{report['disagreements']} rather than as an unreadable record."
            )

    def test_a_failed_episode_whose_success_is_a_string_does_not_enter_the_training_set(self, dataset_root):
        """``success: "false"`` is truthy, so the filter kept a failed episode.

        ``require_success`` reads this field as the authoritative verdict. A
        non-boolean spelling of it is the one unreadable value that ADMITS an
        episode rather than raising, which is why the shared boolean-flag
        domain grades it here as well as on the way in.
        """
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"]["deterministic"].update({"success": "false"}))
        with pytest.raises(ValueError) as excinfo:
            kept = filter_episodes(root, require_success=True, min_quality="medium")
            pytest.fail(
                f"episode 1 records success='false' and filter_episodes(require_success=True) "
                f"returned {kept}, admitting it into the training set."
            )
        message = str(excinfo.value)
        assert SIDECAR_FILENAME in message
        assert "success" in message

    @pytest.mark.parametrize("flag", UNUSABLE_FLAGS)
    def test_a_verdict_flag_is_checked_on_the_shared_boolean_domain(self, dataset_root, flag):
        """Both verdict booleans are graded by the same guard the writer uses."""
        assert boolean_flag_error(flag, "f", "c") is not None
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"]["deterministic"].update({"success": flag}))
        with pytest.raises(ValueError, match="success"):
            read_labels(root)

    def test_a_disputes_flag_outside_the_boolean_domain_is_refused(self, dataset_root):
        """``exclude_disputed`` drops on this field, so it is verdict-bearing.

        Every non-empty string is truthy, so ``"no"`` would have excluded an
        UNDISPUTED episode from training data.
        """
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"]["judge"].update({"disputes_verdict": "no"}))
        with pytest.raises(ValueError, match="disputes_verdict"):
            kept = filter_episodes(root, exclude_disputed=True)
            pytest.fail(
                f"episode 1 records disputes_verdict='no' and exclude_disputed=True returned "
                f"{kept}, dropping an episode whose annotation says it is not disputed."
            )

    def test_a_failure_mode_outside_the_taxonomy_is_refused(self, dataset_root):
        """The taxonomy is fixed so filters match on identity, not prose."""
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"]["judge"].update({"failure_mode": "bogus"}))
        with pytest.raises(ValueError) as excinfo:
            read_labels(root)
        assert str(FAILURE_MODES) in str(excinfo.value)

    @pytest.mark.parametrize(
        ("block", "field"),
        [("deterministic", "success"), ("judge", "quality")],
    )
    def test_a_block_that_cannot_state_its_own_verdict_is_refused(self, dataset_root, block, field):
        """A missing verdict field was a ``KeyError`` from inside a filter.

        Both are read by subscript downstream, so absence surfaced as
        ``KeyError: 'quality'`` naming no episode and no file.
        """
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"][block].pop(field))
        with pytest.raises(ValueError, match=field):
            read_labels(root)

    @pytest.mark.parametrize("block", ["deterministic", "judge"])
    @pytest.mark.parametrize("value", ["not-an-object", None, 3, [], True], ids=["str", "null", "int", "list", "bool"])
    def test_a_block_that_is_not_an_object_is_refused(self, dataset_root, block, value):
        """A non-object block reached readers as ``TypeError`` / ``AttributeError``.

        ``null`` is the spelling worth naming: it is the most plausible external-writer
        spelling of "predicates not yet run", and a guard keyed on the value
        (``if block is not None``) admits it while every downstream reader distinguishes
        it from an absent key. Measured on a sidecar this function had accepted,
        ``deterministic_verdict`` raised ``TypeError: 'NoneType' object is not iterable``.
        Keying the guard on key presence refuses both spellings alike, and refuses them
        for ``judge`` too -- the two blocks must not reach different verdicts for one
        spelling.
        """
        root = self._sidecar(dataset_root, lambda d: d["episodes"]["1"].update({block: value}))
        with pytest.raises(ValueError, match=block):
            read_labels(root)

    def test_a_record_that_is_not_an_object_is_refused(self, dataset_root):
        root = self._sidecar(dataset_root, lambda d: d["episodes"].update({"1": "not-an-object"}))
        with pytest.raises(ValueError, match="expected a JSON object"):
            read_labels(root)

    def test_an_episodes_that_is_not_an_object_is_refused(self, dataset_root):
        """``filter_episodes`` iterated it with ``.items()``."""
        root = self._sidecar(dataset_root, lambda d: d.update({"episodes": []}))
        with pytest.raises(ValueError, match="'episodes'"):
            read_labels(root)

    @pytest.mark.parametrize("key", ["abc", "-1", "00", " 1", "1.0", ""])
    def test_a_key_that_does_not_spell_its_episode_index_is_refused(self, dataset_root, key):
        """The mapping is keyed by ``str(episode_index)`` per the schema.

        ``measure_agreement`` looks an episode up as ``str(int(index))`` while
        ``filter_episodes`` returns ``int(key)``, so a key only one of them can
        resolve makes the two disagree about which episode a record is about.
        """
        record = {"episode_index": 9, "deterministic": {"success": True}, "judge": {"quality": "high"}}
        root = self._sidecar(dataset_root, lambda d: d["episodes"].update({key: record}))
        with pytest.raises(ValueError) as excinfo:
            read_labels(root)
        assert repr(key) in str(excinfo.value)


class TestAnUnreadableRecordIsOnlyAVerdictField:
    """What the record guard must NOT refuse, so it cannot over-reach.

    Every test here passes both before and after the guard: they pin the
    boundary rather than the regression. A guard that refused a value the
    module's own writers produce, or that graded a descriptive field no reader
    branches on, would fail them.
    """

    @pytest.mark.parametrize("quality", QUALITY_GRADES)
    @pytest.mark.parametrize("failure_mode", [None, *FAILURE_MODES])
    @pytest.mark.parametrize("success", [True, False])
    def test_everything_the_writers_produce_is_read_back(self, dataset_root, quality, failure_mode, success):
        """Writer/reader parity, over the whole documented vocabulary.

        The reason the guard needs no second call at the write sites: what
        :func:`record_deterministic_verdicts` and :func:`annotate_episode`
        accept on the way in is exactly what :func:`read_labels` accepts on
        the way out.
        """
        record_deterministic_verdicts(
            dataset_root,
            [
                {
                    "episode": 0,
                    "success": success,
                    "failure": not success,
                    "steps": 5,
                    "cumulative_reward": 1.0,
                    "seed": 2,
                }
            ],
            benchmark="reach",
        )
        annotate_episode(
            dataset_root, 0, quality=quality, failure_mode=failure_mode, note="n", success_opinion=None, model="m"
        )
        read_labels(dataset_root)
        filter_episodes(dataset_root, require_success=False, min_quality="low")
        assert measure_agreement(dataset_root, {0: {"quality": quality}})["quality_agreement"] == 1.0

    def test_a_verdict_with_no_judge_yet_is_the_ordinary_mid_pipeline_state(self, dataset_root):
        """Stage one lands before stage two; that document must stay readable."""
        record_deterministic_verdicts(dataset_root, _verdicts(), benchmark="reach")
        assert set(read_labels(dataset_root)["episodes"]) == {"0", "1"}
        assert filter_episodes(dataset_root) == []

    def test_a_descriptive_field_is_carried_through_whatever_it_holds(self, dataset_root):
        """No reader branches on these, so a surprising value is not a verdict.

        Refusing here would make the sidecar a schema this module polices
        rather than a record it reads, and would reject provenance a future
        writer adds.
        """
        record_deterministic_verdicts(dataset_root, [{"episode": 0, "success": True}], benchmark="reach")
        annotate_episode(dataset_root, 0, quality="high", model="judge")
        path = labels_path(dataset_root)
        document = json.loads(path.read_text())
        document["episodes"]["0"]["deterministic"].update({"steps": "many", "cumulative_reward": None, "seed": []})
        document["episodes"]["0"]["judge"].update(
            {"note": 7, "model": None, "labeled_at": "yesterday", "success_opinion": "maybe"}
        )
        document["episodes"]["0"]["notes_from_a_future_writer"] = {"anything": True}
        path.write_text(json.dumps(document))

        assert filter_episodes(dataset_root, min_quality="high") == [0]
        record = read_labels(dataset_root)["episodes"]["0"]
        assert record["deterministic"]["steps"] == "many"
        assert record["judge"]["labeled_at"] == "yesterday"
        assert record["notes_from_a_future_writer"] == {"anything": True}

    def test_the_caller_side_refusals_are_unchanged(self, dataset_root):
        """The other half of each question keeps its own message verbatim."""
        record_deterministic_verdicts(dataset_root, [{"episode": 0, "success": True}], benchmark="reach")
        annotate_episode(dataset_root, 0, quality="high", model="judge")
        with pytest.raises(ValueError) as filter_error:
            filter_episodes(dataset_root, min_quality="excellent")
        assert str(filter_error.value) == (
            f"filter_episodes: min_quality must be one of {QUALITY_GRADES}, got 'excellent'."
        )
        with pytest.raises(ValueError) as annotate_error:
            annotate_episode(dataset_root, 0, quality="excellent")
        assert str(annotate_error.value) == (
            f"annotate_episode: quality must be one of {QUALITY_GRADES}, got 'excellent'."
        )
        with pytest.raises(ValueError) as agreement_error:
            measure_agreement(dataset_root, {0: {"quality": "excellent"}})
        assert str(agreement_error.value) == (
            f"measure_agreement: human_labels[0] must be a dict with 'quality' in {QUALITY_GRADES}."
        )


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
