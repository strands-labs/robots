"""Unit tests for ``strands_robots.dataset_recorder.DatasetRecorder``.

These tests exercise the wrapper logic that does NOT require a real LeRobot
dataset by injecting a fake dataset object, so they run on a minimal env
(``lerobot`` not installed). They pin the partial-episode discard behaviour
introduced for the #366 recording pipeline follow-up.
"""

from strands_robots.dataset_recorder import DatasetRecorder


class _FakeDatasetWithClear:
    """Fake LeRobot dataset exposing ``clear_episode_buffer`` (preferred path)."""

    def __init__(self):
        self.repo_id = "local/fake"
        self.cleared = 0

    def clear_episode_buffer(self):
        self.cleared += 1


class _FakeDatasetWithCreate:
    """Fake dataset exposing only ``create_episode_buffer`` (fallback path)."""

    def __init__(self):
        self.repo_id = "local/fake"
        self.episode_buffer = {"frames": [1, 2, 3]}
        self.created = 0

    def create_episode_buffer(self):
        self.created += 1
        return {}


class _FakeDatasetNoClear:
    """Fake dataset exposing no buffer-reset surface (warn-only path)."""

    def __init__(self):
        self.repo_id = "local/fake"


def _recorder_for(dataset) -> DatasetRecorder:
    rec = DatasetRecorder(dataset=dataset)
    rec.episode_frame_count = 5  # simulate 5 frames buffered for the open episode
    rec.frame_count = 5
    return rec


def test_clear_episode_buffer_prefers_native_clear():
    ds = _FakeDatasetWithClear()
    rec = _recorder_for(ds)

    assert rec.clear_episode_buffer() is True
    assert ds.cleared == 1
    # Next episode starts at frame 0; cumulative frame_count is untouched
    # (those frames were only ever in the open buffer, not flushed to disk).
    assert rec.episode_frame_count == 0


def test_clear_episode_buffer_falls_back_to_create_buffer():
    ds = _FakeDatasetWithCreate()
    rec = _recorder_for(ds)

    assert rec.clear_episode_buffer() is True
    assert ds.created == 1
    assert ds.episode_buffer == {}
    assert rec.episode_frame_count == 0


def test_clear_episode_buffer_warns_when_no_surface(caplog):
    ds = _FakeDatasetNoClear()
    rec = _recorder_for(ds)

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        result = rec.clear_episode_buffer()

    assert result is False
    # Counter still resets so reporting does not carry over the discarded frames.
    assert rec.episode_frame_count == 0
    assert any("partial episode" in r.message for r in caplog.records)


def test_clear_episode_buffer_swallows_dataset_error(caplog):
    """A failure inside the dataset's clear must not mask the original abort."""

    class _Boom:
        repo_id = "local/fake"

        def clear_episode_buffer(self):
            raise RuntimeError("buffer is wedged")

    rec = _recorder_for(_Boom())

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        result = rec.clear_episode_buffer()

    assert result is False
    assert rec.episode_frame_count == 0


def test_clear_episode_buffer_ascii_only_warnings(caplog):
    """Recorder log strings must be plain ASCII (project string-hygiene rule)."""
    rec = _recorder_for(_FakeDatasetNoClear())

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        rec.clear_episode_buffer()

    for record in caplog.records:
        record.getMessage().encode("ascii")  # raises if any non-ASCII glyph leaked
