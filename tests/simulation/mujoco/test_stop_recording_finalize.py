"""stop_recording finalize / bucket-sync / hub-publish contract.

``Simulation.stop_recording`` closes an in-progress LeRobotDataset recording.
Beyond saving the episode it has three contractual side effects that an agent
relies on in the physical-AI data loop:

* the episode is saved and the dataset finalized (meta/ written) BEFORE any
  upload, so downstream streaming/training sees a complete dataset;
* when a ``bucket`` is given it syncs to the mutable HF Storage Bucket and the
  reported text reflects success or failure;
* when ``push_to_hub`` is set (per-call or from ``start_recording``) it publishes
  the versioned dataset repo and the text reflects success or failure.

These tests drive a fake recorder so the contract is pinned without the
``lerobot`` extra or any real Hub I/O - only the orchestration in
``recording.py`` runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


class _FakeRecorder:
    """Minimal stand-in for ``DatasetRecorder`` capturing orchestration order."""

    def __init__(
        self,
        *,
        sync_result=None,
        push_result=None,
        save_result=None,
        frame_count=7,
        episode_frame_count=7,
        meta_total_episodes=None,
    ):
        self.repo_id = "local/finalize_test"
        self.frame_count = frame_count
        # Frames captured since the last save_episode (the pending trailing
        # episode). stop_recording only flushes a final save_episode when this
        # is > 0; see RecordingMixin.stop_recording.
        self.episode_frame_count = episode_frame_count
        self.episode_count = 1
        self.root = "/tmp/finalize_test"
        self.calls: list[str] = []
        self._sync_result = sync_result
        self._push_result = push_result
        self._save_result = save_result
        self.sync_args: tuple | None = None
        self.push_tags = None
        # stop_recording's #708 parquet-truth gate reads
        # ``recorder.dataset.meta.total_episodes`` as the ground truth. Only
        # expose ``dataset`` when a caller wants to exercise that gate so the
        # other tests keep the no-dataset (gate-skipped) path.
        if meta_total_episodes is not None:
            self.dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=meta_total_episodes))

    def save_episode(self):
        self.calls.append("save_episode")
        return self._save_result

    def finalize(self):
        self.calls.append("finalize")

    def sync_to_bucket(self, bucket, run_id=None):
        self.calls.append("sync_to_bucket")
        self.sync_args = (bucket, run_id)
        return self._sync_result

    def push_to_hub(self, tags=None):
        self.calls.append("push_to_hub")
        self.push_tags = tags
        return self._push_result


@pytest.fixture
def recording_sim():
    s = Simulation(tool_name="stop_finalize_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _arm(sim, recorder, *, push_to_hub=False):
    """Put the sim into a recording state backed by ``recorder``."""
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["dataset_recorder"] = recorder
    sim._world._backend_state["push_to_hub"] = push_to_hub


class TestStopRecordingFinalize:
    def test_not_recording_is_idempotent(self, recording_sim):
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        assert "Was not recording" in result["content"][0]["text"]

    def test_missing_recorder_reports_error(self, recording_sim):
        # recording flagged on but no recorder object present
        recording_sim._world._backend_state["recording"] = True
        recording_sim._world._backend_state["dataset_recorder"] = None
        result = recording_sim.stop_recording()
        assert result["status"] == "error"
        assert "No dataset recorder active" in result["content"][0]["text"]

    def test_saves_and_finalizes_before_any_upload(self, recording_sim):
        rec = _FakeRecorder()
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        # save happens, then finalize; no upload requested.
        assert rec.calls == ["save_episode", "finalize"]
        text = result["content"][0]["text"]
        assert "7 frames" in text
        assert rec.repo_id in text
        # state cleared so a subsequent stop is a no-op.
        assert recording_sim._world._backend_state["dataset_recorder"] is None
        assert recording_sim._world._backend_state["recording"] is False

    def test_bucket_sync_success_reports_uri_and_runs_after_finalize(self, recording_sim):
        rec = _FakeRecorder(sync_result={"status": "success", "bucket_uri": "hf://org/buck/run1"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(bucket="org/buck", run_id="run1")
        assert result["status"] == "success"
        # finalize must precede the bucket sync.
        assert rec.calls.index("finalize") < rec.calls.index("sync_to_bucket")
        assert rec.sync_args == ("org/buck", "run1")
        assert "Synced to bucket: hf://org/buck/run1" in result["content"][0]["text"]

    def test_bucket_sync_failure_is_surfaced(self, recording_sim):
        rec = _FakeRecorder(sync_result={"status": "error", "message": "bucket unreachable"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(bucket="org/buck")
        assert result["status"] == "success"
        assert "Bucket sync FAILED: bucket unreachable" in result["content"][0]["text"]

    def test_push_to_hub_per_call_publishes_with_tags(self, recording_sim):
        rec = _FakeRecorder(push_result={"status": "success"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(push_to_hub=True)
        assert result["status"] == "success"
        assert "push_to_hub" in rec.calls
        assert rec.push_tags == ["strands-robots", "sim"]
        assert "Pushed to HuggingFace Hub" in result["content"][0]["text"]

    def test_push_to_hub_inherited_from_start_recording(self, recording_sim):
        rec = _FakeRecorder(push_result={"status": "success"})
        # push not requested per-call, but armed at start_recording.
        _arm(recording_sim, rec, push_to_hub=True)
        result = recording_sim.stop_recording()
        assert "push_to_hub" in rec.calls
        assert "Pushed to HuggingFace Hub" in result["content"][0]["text"]

    def test_push_to_hub_failure_is_surfaced(self, recording_sim):
        rec = _FakeRecorder(push_result={"status": "error", "message": "auth denied"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(push_to_hub=True)
        assert result["status"] == "success"
        assert "push_to_hub FAILED: auth denied" in result["content"][0]["text"]


class TestStopRecordingIdleKwargs:
    """Idle-path kwargs are never silently dropped (#1498).

    ``stop_recording(bucket=...)`` on a sim with no open recording session used
    to return the bare idempotent ``"Was not recording."`` success and skip the
    bucket sync entirely - the agent believed the upload happened when nothing
    ran. Contract now:

    * bare ``stop_recording()`` stays the idempotent success no-op;
    * ``bucket=`` syncs the last-finalized dataset (``last_dataset_root``
      stashed at start_recording), the documented daily-sync workflow;
    * ``bucket=`` with no prior dataset is a structured error, not success;
    * ``push_to_hub=True`` / ``run_id=`` without ``bucket=`` are structured
      errors (publishing needs an open session; run_id needs bucket).
    """

    def test_idle_bucket_never_returns_bare_was_not_recording_success(self, recording_sim, monkeypatch):
        """Regression pin: bucket= on an idle sim must not be a silent no-op."""
        result = recording_sim.stop_recording(bucket="org/buck")
        text = result["content"][0]["text"]
        assert not (result["status"] == "success" and text == "Was not recording.")

    def test_idle_bare_stop_stays_idempotent_success(self, recording_sim):
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        assert "Was not recording" in result["content"][0]["text"]

    def test_idle_bucket_with_no_prior_dataset_errors(self, recording_sim):
        # Never recorded on this sim: nothing to sync -> loud error.
        assert recording_sim._world._backend_state.get("last_dataset_root") is None
        result = recording_sim.stop_recording(bucket="org/buck")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "nothing was uploaded" in text
        assert "start_recording" in text

    def test_idle_bucket_syncs_last_finalized_dataset(self, recording_sim, monkeypatch):
        # The daily-sync workflow: session already closed, bucket= re-syncs the
        # dataset stashed as last_dataset_root at start_recording.
        from strands_robots import dataset_recorder as dr

        recording_sim._world._backend_state["last_dataset_root"] = "/tmp/last_ds"
        seen = {}

        def _fake_sync(local_root, bucket, run_id=None, **kwargs):
            seen["args"] = (local_root, bucket, run_id)
            return {"status": "success", "bucket_uri": f"hf://buckets/{bucket}/run7"}

        monkeypatch.setattr(dr, "sync_dataset_to_bucket", _fake_sync)

        result = recording_sim.stop_recording(bucket="org/buck", run_id="run7")
        assert result["status"] == "success"
        assert seen["args"] == ("/tmp/last_ds", "org/buck", "run7")
        text = result["content"][0]["text"]
        assert "Synced to bucket: hf://buckets/org/buck/run7" in text
        assert "/tmp/last_ds" in text

    def test_idle_bucket_sync_failure_is_error(self, recording_sim, monkeypatch):
        # Unlike the open-session path (where the dataset was still saved), the
        # ONLY requested effect on the idle path is the sync - a failed sync
        # fails the call.
        from strands_robots import dataset_recorder as dr

        recording_sim._world._backend_state["last_dataset_root"] = "/tmp/last_ds"
        monkeypatch.setattr(
            dr,
            "sync_dataset_to_bucket",
            lambda *a, **k: {"status": "error", "message": "bucket unreachable"},
        )

        result = recording_sim.stop_recording(bucket="org/buck")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "FAILED" in text
        assert "bucket unreachable" in text

    def test_idle_push_to_hub_errors(self, recording_sim):
        # Publishing a versioned dataset repo needs the open session's recorder.
        result = recording_sim.stop_recording(push_to_hub=True)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "push_to_hub" in text
        assert "no recording session" in text

    def test_idle_run_id_without_bucket_errors(self, recording_sim):
        result = recording_sim.stop_recording(run_id="run7")
        assert result["status"] == "error"
        assert "run_id" in result["content"][0]["text"]

    def test_idle_bucket_after_real_stop_cycle_resyncs(self, recording_sim, monkeypatch):
        # Full repro from #1498: record -> stop (session closed) -> explicit
        # stop_recording(bucket=...) must sync, not silently succeed.
        from strands_robots import dataset_recorder as dr

        rec = _FakeRecorder()
        _arm(recording_sim, rec)
        recording_sim._world._backend_state["last_dataset_root"] = rec.root
        first = recording_sim.stop_recording()
        assert first["status"] == "success"
        assert recording_sim._world._backend_state["recording"] is False

        synced = {}

        def _fake_sync(root, bucket, run_id=None, **_k):
            synced["args"] = (root, bucket)
            return {"status": "success", "bucket_uri": f"hf://buckets/{bucket}/x"}

        monkeypatch.setattr(dr, "sync_dataset_to_bucket", _fake_sync)
        second = recording_sim.stop_recording(bucket="org/buck")
        assert second["status"] == "success"
        assert synced["args"] == (rec.root, "org/buck")
        assert "Synced to bucket" in second["content"][0]["text"]


class TestStopRecordingEmptyDataset:
    """stop_recording must fail loudly when no frames were captured, but must
    NOT fail a dataset that was filled via per-episode save_episode.

    Regression for the silent empty-dataset bug. A recording driven by a path
    that never feeds the dataset recorder (eval_policy / evaluate /
    replay_episode / a bare step loop - only run_policy's on_frame hook calls
    add_frame) leaves the recorder with zero frames. Previously stop_recording
    called save_episode unconditionally, discarded its error return, and
    reported success with "0 frames, 0 episode(s)", producing a dataset with
    only meta/info.json (no parquet/video).

    The fix distinguishes three cases by frame counters:
      1. pending unsaved frames -> flush them (save_episode), surface errors;
      2. no pending frames but dataset non-empty -> finalize only (do not
         re-call save_episode on an empty buffer);
      3. nothing ever captured -> structured error, no empty dataset.
    """

    def test_empty_recording_reports_error(self, recording_sim):
        # frame_count == 0 and no pending frames -> loud empty-dataset error.
        rec = _FakeRecorder(frame_count=0, episode_frame_count=0)
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "captured no frames" in text
        assert "0 frames" in text
        # actionable guidance: point to the only path that records frames.
        assert "run_policy" in text
        # save_episode must NOT be called on the empty buffer.
        assert rec.calls == []

    def test_empty_recording_does_not_finalize_or_upload(self, recording_sim):
        rec = _FakeRecorder(
            frame_count=0,
            episode_frame_count=0,
            push_result={"status": "success"},
        )
        _arm(recording_sim, rec, push_to_hub=True)
        result = recording_sim.stop_recording(push_to_hub=True)
        assert result["status"] == "error"
        # No finalize / no upload after an empty recording.
        assert rec.calls == []
        assert "finalize" not in rec.calls
        assert "push_to_hub" not in rec.calls

    def test_empty_recording_clears_state_for_clean_retry(self, recording_sim):
        rec = _FakeRecorder(frame_count=0, episode_frame_count=0)
        _arm(recording_sim, rec)
        recording_sim.stop_recording()
        # Recorder + buffer cleared so a subsequent stop is the idempotent no-op.
        assert recording_sim._world._backend_state["dataset_recorder"] is None
        assert recording_sim._world._backend_state["recording"] is False
        second = recording_sim.stop_recording()
        assert second["status"] == "success"
        assert "Was not recording" in second["content"][0]["text"]

    def test_trailing_save_episode_failure_is_surfaced(self, recording_sim):
        # Pending frames exist but the final save_episode flush fails.
        rec = _FakeRecorder(
            frame_count=12,
            episode_frame_count=12,
            save_result={"status": "error", "message": "writer broke"},
        )
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "final episode" in text
        assert "writer broke" in text
        # Flush was attempted; no finalize/upload after a failed flush.
        assert rec.calls == ["save_episode"]

    def test_dataset_filled_per_episode_finalizes_without_re_saving(self, recording_sim):
        # Caller saved each episode already (episode_frame_count == 0) and the
        # dataset has frames. stop_recording must finalize WITHOUT calling
        # save_episode again - the previous bug re-saved the empty buffer and
        # wrongly errored an otherwise-complete dataset.
        rec = _FakeRecorder(frame_count=90, episode_frame_count=0)
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        assert rec.calls == ["finalize"]
        assert "save_episode" not in rec.calls
        assert "90 frames" in result["content"][0]["text"]


class TestStopRecordingParquetTruthGate:
    """stop_recording reconciles the recorder's episode bookkeeping against the
    on-disk dataset (the #708 silent-collapse gate).

    ``recorder.episode_count`` is author-side bookkeeping incremented by every
    ``save_episode`` call. The dataset's own ``meta.total_episodes`` (backed by
    the parquet rowcount) is what downstream consumers - the HF hub, training
    loaders, audit tooling - actually trust. When they disagree the on-disk
    dataset wins, and stop_recording must surface the divergence in both the
    structured JSON payload and the human-readable text so a caller (or CI that
    parses the status dict) can fail loudly instead of shipping a dataset whose
    episodes silently collapsed.

    When the recorder exposes no ``dataset`` handle (e.g. a backend without the
    ``lerobot`` extra) the gate is skipped and the recorder's own count stands.
    """

    def test_episode_count_matches_parquet_reports_no_mismatch(self, recording_sim):
        # recorder.episode_count == dataset.meta.total_episodes -> gate is quiet.
        rec = _FakeRecorder(meta_total_episodes=1)  # episode_count defaults to 1
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        payload = result["content"][1]["json"]
        assert payload["parquet_episode_count"] == 1
        assert payload["episode_count_mismatch"] is False
        assert payload["episode_count"] == 1
        # No gate banner in the human-readable text when counts agree.
        assert "#708 gate" not in result["content"][0]["text"]

    def test_episode_count_mismatch_trusts_parquet_and_surfaces_divergence(self, recording_sim):
        # recorder thinks it saved 5 episodes but the parquet only has 3:
        # the on-disk dataset is the source of truth, so the reported
        # episode_count must collapse to 3 and the divergence must be flagged.
        rec = _FakeRecorder(meta_total_episodes=3)
        rec.episode_count = 5
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        payload = result["content"][1]["json"]
        # parquet wins: the canonical episode_count is the on-disk value.
        assert payload["parquet_episode_count"] == 3
        assert payload["episode_count"] == 3
        assert payload["episode_count_mismatch"] is True
        # The divergence is named in the human-readable text (both counts).
        text = result["content"][0]["text"]
        assert "#708 gate" in text
        assert "5 episodes" in text
        assert "parquet has 3" in text

    def test_missing_dataset_handle_skips_gate(self, recording_sim):
        # No ``dataset`` attribute (no lerobot extra) -> gate is skipped, the
        # recorder's own count stands, and parquet_episode_count stays None.
        rec = _FakeRecorder()  # meta_total_episodes=None -> no .dataset attr
        assert not hasattr(rec, "dataset")
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        payload = result["content"][1]["json"]
        assert payload["parquet_episode_count"] is None
        assert payload["episode_count_mismatch"] is False
        assert "#708 gate" not in result["content"][0]["text"]

    def test_parquet_probe_failure_never_aborts_finalize(self, recording_sim):
        # A broken meta probe (total_episodes that cannot be coerced to int)
        # must be swallowed: the gate is best-effort and must never fail an
        # otherwise-complete finalize. The recorder's own count then stands.
        rec = _FakeRecorder(meta_total_episodes=1)
        rec.dataset.meta.total_episodes = "not-a-number"
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        assert "finalize" in rec.calls
        payload = result["content"][1]["json"]
        # Probe failed before assigning -> stays at the safe defaults.
        assert payload["parquet_episode_count"] is None
        assert payload["episode_count_mismatch"] is False
