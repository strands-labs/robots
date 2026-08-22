"""Provenance sidecar: the honesty record round-trips and refuses corruption.

Absence and corruption are different verdicts by design: a dataset with no
``meta/provenance.json`` declares no synthetic episodes (the ordinary state of
a recorded dataset, so :func:`~strands_robots.transforms.provenance.load_provenance`
returns ``[]``), while a present-but-unreadable file is corruption and raises -
reading it as "no synthetic episodes" would be exactly the silent mixing of
generated and recorded data the file exists to prevent.
"""

import json

import pytest

from strands_robots.transforms.provenance import (
    load_provenance,
    provenance_path,
    synthetic_episode_indices,
    write_provenance,
)


def _record(episode_index: int, source: int) -> dict:
    return {
        "episode_index": episode_index,
        "synthetic": True,
        "source_episode_index": source,
        "source_repo_id": "local/source",
        "transform": "mock",
        "transform_version": "1",
        "variant": 0,
        "prompt": "",
        "seed": 7,
    }


class TestRoundTrip:
    def test_write_then_load(self, tmp_path):
        records = [_record(0, 0), _record(1, 0), _record(2, 1)]
        path = write_provenance(tmp_path, records)
        assert path == provenance_path(tmp_path)
        assert path.is_file()
        assert load_provenance(tmp_path) == records

    def test_synthetic_episode_indices(self, tmp_path):
        records = [_record(0, 0), _record(1, 1)]
        records.append({"episode_index": 2, "synthetic": False, "transform": "none"})
        write_provenance(tmp_path, records)
        assert synthetic_episode_indices(tmp_path) == {0, 1}

    def test_absence_means_no_synthetic_episodes(self, tmp_path):
        """A recorded dataset carries no file; that is a verdict, not an error."""
        assert load_provenance(tmp_path) == []
        assert synthetic_episode_indices(tmp_path) == set()


class TestRefusals:
    def test_write_refuses_non_list(self, tmp_path):
        with pytest.raises(ValueError, match="list of dicts"):
            write_provenance(tmp_path, {"episode_index": 0})

    def test_write_refuses_record_missing_mandatory_keys(self, tmp_path):
        with pytest.raises(ValueError, match="mandatory provenance key"):
            write_provenance(tmp_path, [{"episode_index": 0, "synthetic": True}])

    def test_load_refuses_corrupt_json(self, tmp_path):
        path = provenance_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_provenance(tmp_path)

    def test_load_refuses_wrong_shape(self, tmp_path):
        path = provenance_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text('{"episodes": "all of them"}', encoding="utf-8")
        with pytest.raises(ValueError, match="documented shape"):
            load_provenance(tmp_path)


def _write_file(root, records: list[dict]) -> None:
    """Plant a provenance file directly, as a hand edit or another tool would."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "provenance.json").write_text(json.dumps({"version": 1, "episodes": records}), encoding="utf-8")


#: Values a caller reaches for instead of a boolean ``synthetic``. Every one is
#: truthy, so reading the field by truthiness would classify them as generated;
#: reading it with ``is True`` classifies them as recorded. Both are guesses, and
#: the second one is silent.
_NON_BOOLEAN_SYNTHETIC = [1, 1.0, "true", "yes"]

#: Values that cannot name an episode. ``int()`` raises on both, and one of the
#: two raises ``TypeError``, outside anything this module documents.
_UNUSABLE_EPISODE_INDEX = [None, "abc", -1]


class TestARecordThatCannotAnswerIsRefusedNotReadAsRecorded:
    """A verdict key present but unreadable is corruption, not a "no".

    The sidecar exists to keep generated and recorded episodes distinguishable,
    and :func:`~strands_robots.transforms.provenance.synthetic_episode_indices`
    is the one call that distinguishes them: everything it returns was
    generated, everything outside it was recorded. So a record whose
    ``synthetic`` field is present but is not a boolean has exactly one wrong
    outcome available - landing outside the set, i.e. being reported as recorded
    - and that is the silent mixing the file exists to prevent. It is refused at
    both ends instead, by one shared rule.
    """

    @pytest.mark.parametrize("synthetic", _NON_BOOLEAN_SYNTHETIC)
    def test_the_writer_refuses_a_non_boolean_synthetic(self, tmp_path, synthetic):
        record = {**_record(0, 3), "synthetic": synthetic}
        try:
            write_provenance(tmp_path, [record])
        except ValueError as refused:
            assert "synthetic" in str(refused)
            return
        pytest.fail(
            f"write_provenance accepted synthetic={synthetic!r}, and synthetic_episode_indices then "
            f"reports {synthetic_episode_indices(tmp_path)} - the generated episode reads as recorded"
        )

    @pytest.mark.parametrize("synthetic", _NON_BOOLEAN_SYNTHETIC)
    def test_the_reader_refuses_a_non_boolean_synthetic(self, tmp_path, synthetic):
        _write_file(tmp_path, [{**_record(0, 3), "synthetic": synthetic}])
        try:
            got = synthetic_episode_indices(tmp_path)
        except ValueError as refused:
            assert "synthetic" in str(refused)
            return
        pytest.fail(
            f"a record carrying synthetic={synthetic!r} was read as {got} - the episode it declares "
            "generated is reported as recorded, with nothing raised"
        )

    def test_a_missing_synthetic_field_is_refused_by_the_reader_too(self, tmp_path):
        """The writer already refused this; a planted file must not slip past it."""
        _write_file(tmp_path, [{"episode_index": 0, "transform": "mock"}])
        with pytest.raises(ValueError, match="missing mandatory provenance key"):
            synthetic_episode_indices(tmp_path)

    @pytest.mark.parametrize("index", _UNUSABLE_EPISODE_INDEX)
    def test_an_unusable_episode_index_is_named_rather_than_crashing_the_read(self, tmp_path, index):
        """The refusal names the file and the field, not ``int()``'s own complaint."""
        _write_file(tmp_path, [{**_record(0, 3), "episode_index": index}])
        with pytest.raises(ValueError, match="episode_index"):
            synthetic_episode_indices(tmp_path)


class TestBothEndsApplyOneRule:
    """The writer and the reader cannot disagree about which records are readable.

    A record the writer refuses to store must be a record the reader refuses to
    trust, or the surface has two notions of provenance and a file can be
    unreadable only after it exists.
    """

    @pytest.mark.parametrize(
        "record",
        [
            _record(0, 3),
            {**_record(0, 3), "synthetic": False},
            {**_record(0, 3), "synthetic": "true"},
            {**_record(0, 3), "synthetic": 1},
            {**_record(0, 3), "episode_index": None},
            {**_record(0, 3), "episode_index": "abc"},
            {"episode_index": 0, "transform": "mock"},
            {"synthetic": True, "transform": "mock"},
        ],
        ids=str,
    )
    def test_the_writer_and_the_reader_reach_the_same_verdict(self, tmp_path, record):
        writer_refused = False
        try:
            write_provenance(tmp_path / "w", [record])
        except ValueError:
            writer_refused = True

        _write_file(tmp_path / "r", [record])
        reader_refused = False
        try:
            load_provenance(tmp_path / "r")
        except ValueError:
            reader_refused = True

        assert writer_refused == reader_refused, (
            f"writer {'refused' if writer_refused else 'accepted'} but reader "
            f"{'refused' if reader_refused else 'accepted'} {record!r}"
        )


class TestTheStoredRecordHoldsTheTypeTheSchemaDocuments:
    """What the writer graded is what a reader loads."""

    def test_a_numpy_boolean_verdict_round_trips_as_json_true(self, tmp_path):
        """A verdict computed with NumPy is a boolean, and must reach the file as one."""
        numpy = pytest.importorskip("numpy")
        write_provenance(tmp_path, [{**_record(0, 3), "synthetic": numpy.True_}])
        on_disk = json.loads(provenance_path(tmp_path).read_text(encoding="utf-8"))
        assert on_disk["episodes"][0]["synthetic"] is True
        assert synthetic_episode_indices(tmp_path) == {0}

    def test_the_shipped_records_are_stored_unchanged(self, tmp_path):
        """The transform surface's own records already hold the documented types."""
        records = [_record(0, 3), _record(1, 3)]
        before = json.dumps(records, sort_keys=True)
        write_provenance(tmp_path, records)
        on_disk = json.loads(provenance_path(tmp_path).read_text(encoding="utf-8"))
        assert json.dumps(on_disk["episodes"], sort_keys=True) == before
        assert json.dumps(records, sort_keys=True) == before  # the caller's list is untouched

    def test_a_descriptive_key_is_carried_through_whatever_its_type(self, tmp_path):
        """Only the keys a reader turns into a verdict are graded."""
        write_provenance(tmp_path, [{**_record(0, 3), "transform_version": 3, "prompt": None}])
        loaded = load_provenance(tmp_path)
        assert loaded[0]["transform_version"] == 3
        assert loaded[0]["prompt"] is None

    def test_synthetic_false_is_a_readable_answer_outside_the_set(self, tmp_path):
        """``false`` says "recorded"; that is an answer, not a failure to answer."""
        write_provenance(tmp_path, [{**_record(0, 3), "synthetic": False}, _record(1, 3)])
        assert synthetic_episode_indices(tmp_path) == {1}
