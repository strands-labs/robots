"""Provenance sidecar: the honesty record round-trips and refuses corruption.

Absence and corruption are different verdicts by design: a dataset with no
``meta/provenance.json`` declares no synthetic episodes (the ordinary state of
a recorded dataset, so :func:`~strands_robots.transforms.provenance.load_provenance`
returns ``[]``), while a present-but-unreadable file is corruption and raises -
reading it as "no synthetic episodes" would be exactly the silent mixing of
generated and recorded data the file exists to prevent.
"""

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
