"""``DatasetRecorder`` frame counters must describe what is on disk.

``add_frame`` counts a frame the moment it is buffered, but frames only reach
disk when ``save_episode`` flushes the buffer. ``clear_episode_buffer``
discards a buffered (aborted) episode, so the frames it throws away never
become parquet rows and must not stay counted.

Two surfaces define ``frame_count`` as the on-disk total, which is what makes
this a correctness contract rather than a cosmetic one:

* ``resume()`` seeds ``frame_count`` from ``dataset.meta.total_frames``, i.e.
  straight from disk truth;
* ``RecordingMixin.stop_recording`` refuses to finalize an empty dataset by
  asking ``frame_count`` whether anything was ever captured, so an inflated
  count reports success for a dataset holding only ``meta/info.json``.

The dataset here is a stub that models the buffer/disk split the contract lives
in, so these run without the ``lerobot`` extra. Its ``save_episode`` refuses an
empty buffer exactly as ``LeRobotDataset`` does.
"""

from __future__ import annotations

import pytest

from strands_robots.dataset_recorder import DatasetRecorder

JOINTS = ["j1", "j2"]


class _BufferedDataset:
    """Stub dataset that distinguishes buffered frames from written frames."""

    def __init__(self) -> None:
        self.repo_id = "local/counters"
        self.root = "/tmp/counters"
        self.features = {
            "observation.state": {"names": list(JOINTS)},
            "action": {"names": list(JOINTS)},
        }
        self.buffered = 0
        self.disk_frames = 0
        self.episodes = 0

    def add_frame(self, frame) -> None:
        self.buffered += 1

    def save_episode(self) -> None:
        if self.buffered == 0:
            # LeRobotDataset raises here; a stub that silently accepted an
            # empty flush would hide the state this contract is about.
            raise RuntimeError("You must add one or several frames with `add_frame`")
        self.disk_frames += self.buffered
        self.buffered = 0
        self.episodes += 1

    def clear_episode_buffer(self) -> None:
        self.buffered = 0


class _WedgedDataset(_BufferedDataset):
    """Dataset whose discard fails, leaving the frames buffered and pending."""

    def clear_episode_buffer(self) -> None:
        raise RuntimeError("buffer is wedged")


def _record(rec: DatasetRecorder, n: int) -> None:
    """Buffer ``n`` frames through the real ``add_frame`` path."""
    for i in range(n):
        rec.add_frame({j: 0.01 * i for j in JOINTS}, {j: 0.02 * i for j in JOINTS})


@pytest.fixture
def recorder() -> DatasetRecorder:
    return DatasetRecorder(dataset=_BufferedDataset(), task="probe", strict=True)


class TestDiscardedFramesAreUncounted:
    """A discarded episode never reached disk, so it must not stay counted."""

    def test_reported_total_matches_the_frames_on_disk_after_an_abort(self, recorder):
        ds = recorder.dataset
        _record(recorder, 10)
        recorder.clear_episode_buffer()  # episode aborted: 10 frames discarded
        _record(recorder, 5)
        result = recorder.save_episode()

        assert result["status"] == "success"
        assert result["episode_frames"] == 5
        assert result["total_frames"] == ds.disk_frames, (
            f"reported total_frames={result['total_frames']} but only "
            f"{ds.disk_frames} frames reached disk; the "
            f"{10} discarded frames are still counted"
        )

    def test_every_frame_discarded_leaves_nothing_counted(self, recorder):
        """The input ``stop_recording``'s empty-dataset refusal reads."""
        _record(recorder, 12)
        recorder.clear_episode_buffer()

        assert recorder.dataset.disk_frames == 0
        assert recorder.frame_count == 0, (
            "nothing reached disk, but frame_count still reports "
            f"{recorder.frame_count} - stop_recording asks this counter whether "
            "anything was captured before refusing an empty dataset"
        )

    def test_a_recorder_seeded_from_disk_stays_consistent_across_an_abort(self, recorder):
        """``resume()`` seeds ``frame_count`` from ``meta.total_frames``."""
        ds = recorder.dataset
        ds.disk_frames = 6  # a previous session's on-disk frames
        recorder.frame_count = 6  # exactly what resume() does

        _record(recorder, 4)
        recorder.clear_episode_buffer()

        assert recorder.frame_count == ds.disk_frames

    def test_each_abort_leaves_no_residue(self, recorder):
        """Drift must not accumulate over repeated aborts."""
        ds = recorder.dataset
        for _ in range(3):
            _record(recorder, 4)
            recorder.clear_episode_buffer()
        _record(recorder, 2)
        recorder.save_episode()

        assert recorder.frame_count == ds.disk_frames == 2

    def test_repeated_discards_do_not_uncount_twice(self, recorder):
        _record(recorder, 5)
        recorder.clear_episode_buffer()
        recorder.clear_episode_buffer()  # nothing buffered the second time

        assert recorder.frame_count == 0


class TestAFailedDiscardKeepsItsFramesCounted:
    """Nothing was thrown away, so nothing may be un-counted."""

    @pytest.fixture
    def wedged(self) -> DatasetRecorder:
        return DatasetRecorder(dataset=_WedgedDataset(), task="probe", strict=True)

    def test_the_frames_stay_counted_for_the_recommended_drain(self, wedged):
        ds = wedged.dataset
        _record(wedged, 7)
        assert wedged.clear_episode_buffer() is False
        assert ds.buffered == 7, "premise: the discard failed, so nothing was dropped"

        assert wedged.frame_count == 7
        assert wedged.episode_frame_count == 7

    def test_the_recommended_drain_reports_the_frames_it_wrote(self, wedged):
        """The warning tells the caller to drain with save_episode."""
        ds = wedged.dataset
        _record(wedged, 7)
        wedged.clear_episode_buffer()
        result = wedged.save_episode()

        assert ds.disk_frames == 7
        assert result["episode_frames"] == 7
        assert result["total_frames"] == ds.disk_frames

    def test_a_later_successful_discard_still_uncounts_them(self, wedged):
        _record(wedged, 7)
        wedged.clear_episode_buffer()  # fails
        wedged.dataset.__class__ = _BufferedDataset  # discard surface recovers
        assert wedged.clear_episode_buffer() is True

        assert wedged.frame_count == 0


class TestTheOrdinaryPathIsUnchanged:
    """Controls: recording without an abort must behave exactly as before."""

    def test_a_saved_episode_reports_the_frames_it_wrote(self, recorder):
        _record(recorder, 5)
        result = recorder.save_episode()

        assert result["episode_frames"] == 5
        assert result["total_frames"] == recorder.dataset.disk_frames == 5

    def test_frames_accumulate_across_saved_episodes(self, recorder):
        for n in (3, 4):
            _record(recorder, n)
            result = recorder.save_episode()

        assert result["total_frames"] == recorder.dataset.disk_frames == 7

    def test_discarding_an_empty_buffer_changes_no_counter(self, recorder):
        _record(recorder, 5)
        recorder.save_episode()
        recorder.clear_episode_buffer()  # nothing buffered

        assert recorder.frame_count == recorder.dataset.disk_frames == 5
