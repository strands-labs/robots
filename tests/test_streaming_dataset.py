"""Unit tests for ``strands_robots.streaming_dataset.StreamingDatasetReader``
and ``DatasetRecorder.sync_to_bucket``.

Mirrors test_dataset_recorder.py: inject fakes so tests run WITHOUT lerobot or
the ``hf`` CLI installed. Covers version-tolerant kwarg forwarding, the
proprio-only ``drop_videos`` path, delta-grid validation, and the bucket-sync
CLI construction + meta/ guard.
"""

import logging
import os
import subprocess

import pytest

import strands_robots.dataset_recorder as dr
import strands_robots.streaming_dataset as sd


class _FakeStreaming:
    """Fake StreamingLeRobotDataset capturing the kwargs it was built with."""

    def __init__(self, repo_id, **kw):
        self.repo_id = repo_id
        self.kw = kw
        self.num_frames = 1000
        self.num_episodes = 10
        self.fps = 30

    def __iter__(self):
        yield {"observation.state": [0.0], "action": [0.0], "task": "t"}


def test_open_forwards_supported_kwargs(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open(
        "org/ds",
        buffer_size=256,
        shuffle=False,
        max_num_shards=8,
        validate_deltas=False,
    )
    assert r.dataset.repo_id == "org/ds"
    assert r.dataset.kw["buffer_size"] == 256
    assert r.dataset.kw["shuffle"] is False
    assert r.dataset.kw["max_num_shards"] == 8
    assert r.num_episodes == 10
    assert r.fps == 30


def test_open_drops_unknown_kwargs(monkeypatch):
    """A narrow constructor (only repo_id) must not raise on extra kwargs."""

    class _Narrow:
        def __init__(self, repo_id):
            self.repo_id = repo_id
            self.num_frames = self.num_episodes = self.fps = 0

        def __iter__(self):
            yield {}

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", buffer_size=999, shuffle=True, validate_deltas=False)
    assert r.dataset.repo_id == "org/ds"


def test_repo_type_forwarded_when_supported(monkeypatch):
    """repo_type reaches a StreamingLeRobotDataset that declares the parameter."""

    class _WithRepoType:
        def __init__(self, repo_id, repo_type="dataset", **kw):
            self.repo_id = repo_id
            self.repo_type = repo_type
            self.num_frames = self.num_episodes = self.fps = 0

        def __iter__(self):
            yield {}

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _WithRepoType, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)
    assert r.dataset.repo_type == "bucket"


def test_repo_type_bucket_raises_when_unsupported(monkeypatch):
    """repo_type='bucket' on a constructor without the parameter must raise,
    never silently open the versioned dataset namespace instead (a different
    storage system - the forbidden silent-kwarg-drop class)."""

    class _Narrow:
        def __init__(self, repo_id):
            raise AssertionError("constructor must never be reached")

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
    with pytest.raises(RuntimeError, match=r"repo_type='bucket' is not supported by any released lerobot"):
        sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)


def test_bucket_guard_message_survives_unresolvable_lerobot_version(monkeypatch):
    """The repo_type='bucket' fail-fast must surface as the actionable
    RuntimeError even when lerobot's version metadata is unresolvable: the
    version lookup that enriches the message must not raise a secondary
    PackageNotFoundError that masks the primary, upgrade-actionable error."""
    import importlib.metadata as md

    class _Narrow:
        def __init__(self, repo_id):
            raise AssertionError("constructor must never be reached")

    def _raise(_name):
        raise md.PackageNotFoundError("lerobot")

    monkeypatch.setattr(md, "version", _raise)
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
    with pytest.raises(RuntimeError, match=r"installed: unknown") as exc:
        sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)
    assert "repo_type='bucket' is not supported by any released lerobot" in str(exc.value)


def test_repo_type_bucket_forwarded_via_var_kwargs(monkeypatch):
    """A constructor with **kwargs accepts repo_type; the guard must not fire."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)
    assert r.dataset.kw["repo_type"] == "bucket"


def test_repo_type_dataset_default_ok_when_unsupported(monkeypatch):
    """The 'dataset' default is semantics-preserving on an old lerobot: it is
    skipped by tolerant forwarding and open() succeeds without error."""

    class _Narrow:
        def __init__(self, repo_id):
            self.repo_id = repo_id
            self.num_frames = self.num_episodes = self.fps = 0

        def __iter__(self):
            yield {}

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", repo_type="dataset", validate_deltas=False)
    assert r.dataset.repo_id == "org/ds"


def test_return_uint8_drop_warns_when_unsupported(monkeypatch, caplog):
    """return_uint8=True on a lerobot whose StreamingLeRobotDataset lacks the
    parameter is dropped (semantics unchanged) but streams float32 - ~4x the
    bandwidth of uint8. That cost must be surfaced as a warning, not silent."""

    class _Narrow:
        def __init__(self, repo_id):
            self.repo_id = repo_id
            self.num_frames = self.num_episodes = self.fps = 0

        def __iter__(self):
            yield {}

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
    with caplog.at_level(logging.WARNING, logger=sd.logger.name):
        sd.StreamingDatasetReader.open("org/ds", return_uint8=True, validate_deltas=False)
    assert any("return_uint8=True dropped" in r.message for r in caplog.records)


def test_return_uint8_no_warn_when_supported(monkeypatch, caplog):
    """No bandwidth warning when the constructor accepts return_uint8 (**kwargs
    here): the kwarg is forwarded and honored, so nothing is dropped."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    with caplog.at_level(logging.WARNING, logger=sd.logger.name):
        r = sd.StreamingDatasetReader.open("org/ds", return_uint8=True, validate_deltas=False)
    assert r.dataset.kw["return_uint8"] is True
    assert not any("return_uint8=True dropped" in rec.message for rec in caplog.records)


def test_drop_videos_strips_camera_deltas(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open(
        "org/ds",
        delta_timestamps={
            "observation.images.front": [-0.1, 0.0],
            "observation.state": [0.0],
            "action": [0.0],
        },
        drop_videos=True,
        validate_deltas=False,
    )
    dt = r.dataset.kw["delta_timestamps"]
    assert "observation.images.front" not in dt
    assert "observation.state" in dt and "action" in dt


def test_drop_videos_all_camera_keys_raises(monkeypatch):
    """If stripping camera keys leaves NO deltas, drop_videos would be a silent
    no-op (every feature, videos included, would stream) - it must raise."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    with pytest.raises(ValueError, match="drop_videos=True requires delta_timestamps"):
        sd.StreamingDatasetReader.open(
            "org/ds",
            delta_timestamps={"observation.images.front": [-0.1, 0.0]},
            drop_videos=True,
            validate_deltas=False,
        )


def test_drop_videos_without_deltas_raises(monkeypatch):
    """drop_videos=True with no delta_timestamps at all previously did nothing
    (video decode still ran); the documented behavior is now to raise."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    with pytest.raises(ValueError, match="drop_videos=True requires delta_timestamps"):
        sd.StreamingDatasetReader.open("org/ds", drop_videos=True, validate_deltas=False)


def test_dataloader_ignores_shuffle(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", validate_deltas=False)

    captured = {}

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, num_workers, **kw):
            captured["shuffle_in_kw"] = "shuffle" in kw
            captured["batch_size"] = batch_size

    class _FakeTorchUtilsData:
        DataLoader = _FakeDataLoader

    class _FakeTorch:
        utils = type("u", (), {"data": _FakeTorchUtilsData})

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    r.dataloader(batch_size=32, shuffle=True)  # shuffle must be swallowed
    assert captured["shuffle_in_kw"] is False
    assert captured["batch_size"] == 32


# ── sync_to_bucket ─────────────────────────────────────────────────────────


class _FakeDataset:
    def __init__(self, root):
        self.repo_id = "org/pick"
        self.root = root


def _recorder(tmp_path):
    rec = dr.DatasetRecorder(dataset=_FakeDataset(str(tmp_path)))
    rec.episode_count = 3
    rec.frame_count = 300
    return rec


def test_sync_to_bucket_builds_cli(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()  # satisfy the meta/ guard
    rec = _recorder(tmp_path)

    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = rec.sync_to_bucket("my-org/robot-fave", run_id="run-021")
    assert res["status"] == "success"
    assert res["bucket_uri"] == "hf://buckets/my-org/robot-fave/run-021"
    assert any(c[:3] == ["hf", "buckets", "create"] for c in calls)
    assert any(c[:2] == ["hf", "sync"] and c[-1].startswith("hf://buckets/") for c in calls)


def test_sync_to_bucket_requires_meta(tmp_path, monkeypatch):
    rec = _recorder(tmp_path)  # NO meta/ dir
    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")
    res = rec.sync_to_bucket("my-org/robot-fave")
    assert res["status"] == "error"
    assert "meta/" in res["message"]


def test_sync_to_bucket_missing_hf_cli(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    monkeypatch.setattr(dr, "_hf_executable", lambda: None)
    res = rec.sync_to_bucket("my-org/robot-fave")
    assert res["status"] == "error"
    assert "hf` CLI" in res["message"] or "hf CLI" in res["message"]


def _guard_recorder(tmp_path, monkeypatch):
    """Recorder whose hf CLI + meta/ guards pass, so validation runs and any
    subprocess call would be a security regression (the fake raises)."""
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")

    def boom(*a, **k):  # subprocess must never run with a rejected target
        raise AssertionError(f"subprocess.run reached with {a!r}")

    monkeypatch.setattr(subprocess, "run", boom)
    return rec


@pytest.mark.parametrize(
    "bucket",
    [
        "../escape",
        "org/../escape",
        "my-org/robot;rm -rf /",
        "my-org/robot fave",
        "a/b/c",  # too many path segments
        "$(whoami)",
        "-leading-dash",
        "",
    ],
)
def test_sync_to_bucket_rejects_unsafe_bucket(tmp_path, monkeypatch, bucket):
    """Agent-reachable bucket names with traversal / metacharacters / extra
    segments must be rejected before any `hf` subprocess (LLM input safety)."""
    rec = _guard_recorder(tmp_path, monkeypatch)
    res = rec.sync_to_bucket(bucket)
    assert res["status"] == "error"
    assert "bucket" in res["message"]


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "sub/dir",
        "run;rm -rf /",
        "run id",
        "$(id)",
    ],
)
def test_sync_to_bucket_rejects_unsafe_run_id(tmp_path, monkeypatch, run_id):
    """run_id reaches the hf:// URI + argv; reject traversal/metacharacters
    and any path separator before constructing the destination."""
    rec = _guard_recorder(tmp_path, monkeypatch)
    res = rec.sync_to_bucket("my-org/robot-fave", run_id=run_id)
    assert res["status"] == "error"
    assert "run_id" in res["message"]


def test_sync_to_bucket_bucket_create_failure_surfaces_error(tmp_path, monkeypatch):
    """A failing ``hf buckets create`` must abort with the CLI's stderr and
    never fall through to ``hf sync`` (a silent success would upload nowhere)."""
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)

        class R:
            returncode = 1
            stdout = ""
            stderr = "permission denied for bucket"

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = rec.sync_to_bucket("my-org/robot-fave", run_id="run-021")
    assert res["status"] == "error"
    assert "bucket create failed" in res["message"]
    assert "permission denied" in res["message"]
    # sync must not run once create has failed.
    assert not any(c[:2] == ["hf", "sync"] for c in calls)


def test_sync_to_bucket_existing_bucket_proceeds_to_sync(tmp_path, monkeypatch):
    """A non-zero ``hf buckets create`` whose output says the bucket already
    exists is idempotent: sync proceeds and the call succeeds."""
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        is_create = cmd[:3] == ["hf", "buckets", "create"]

        class R:
            returncode = 1 if is_create else 0
            stdout = ""
            stderr = "Error: bucket already exists" if is_create else ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = rec.sync_to_bucket("my-org/robot-fave", run_id="run-021")
    assert res["status"] == "success"
    assert res["bucket_uri"] == "hf://buckets/my-org/robot-fave/run-021"
    assert any(c[:2] == ["hf", "sync"] for c in calls)


def test_sync_to_bucket_sync_failure_surfaces_stderr(tmp_path, monkeypatch):
    """A failing ``hf sync`` must surface as an error carrying the CLI stderr,
    not a false success."""
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")

    def fake_run(cmd, capture_output=True, text=True):
        is_create = cmd[:3] == ["hf", "buckets", "create"]

        class R:
            returncode = 0 if is_create else 1
            stdout = ""
            stderr = "" if is_create else "authentication token expired"

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = rec.sync_to_bucket("my-org/robot-fave", run_id="run-021")
    assert res["status"] == "error"
    assert "authentication token expired" in res["message"]


def test_sync_to_bucket_delete_flag_forwarded(tmp_path, monkeypatch):
    """``delete=True`` appends ``--delete`` to the sync argv so the bucket is
    mirrored (removed-locally files are pruned remotely)."""
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    monkeypatch.setattr(dr, "_hf_executable", lambda: "hf")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = rec.sync_to_bucket("my-org/robot-fave", run_id="run-021", delete=True)
    assert res["status"] == "success"
    sync_cmds = [c for c in calls if c[:2] == ["hf", "sync"]]
    assert sync_cmds and "--delete" in sync_cmds[0]


# ── stream_dataset facade ──────────────────────────────────────────────────


def test_module_level_stream_dataset_delegates(monkeypatch):
    """stream_dataset(...) is a thin alias for StreamingDatasetReader.open -
    dataset read-back must not require constructing a simulator."""
    captured = {}

    def fake_open(repo_id, **kw):
        captured["repo_id"] = repo_id
        captured["kw"] = kw
        return "READER"

    monkeypatch.setattr(sd.StreamingDatasetReader, "open", staticmethod(fake_open), raising=True)

    out = sd.stream_dataset("org/ds", root="/tmp/x", shuffle=False, drop_videos=True)
    assert out == "READER"
    assert captured["repo_id"] == "org/ds"
    assert captured["kw"] == {"root": "/tmp/x", "shuffle": False, "drop_videos": True}


def test_recording_mixin_stream_dataset_delegates(monkeypatch):
    """sim.stream_dataset(...) must delegate to StreamingDatasetReader.open,
    keeping streaming a native facade method (not user-side plumbing)."""
    from strands_robots.simulation.mujoco.recording import RecordingMixin

    captured = {}

    def fake_open(repo_id, **kw):
        captured["repo_id"] = repo_id
        captured["kw"] = kw
        return "READER"

    monkeypatch.setattr(sd.StreamingDatasetReader, "open", staticmethod(fake_open), raising=True)

    mixin = RecordingMixin()
    out = mixin.stream_dataset("org/ds", root="/tmp/x", shuffle=False, drop_videos=True)
    assert out == "READER"
    assert captured["repo_id"] == "org/ds"
    assert captured["kw"]["root"] == "/tmp/x"
    assert captured["kw"]["shuffle"] is False
    assert captured["kw"]["drop_videos"] is True


# ── macOS dyld shim ────────────────────────────────────────────────────────


def test_dyld_shim_noop_off_macos(monkeypatch):
    """On non-macOS the shim is a pure no-op (returns False, no env change)."""
    from strands_robots import _dyld

    monkeypatch.setattr(_dyld.sys, "platform", "linux")
    monkeypatch.delenv(_dyld._DYLD_VAR, raising=False)
    assert _dyld.ensure_ffmpeg_on_dyld_path() is False
    assert _dyld._DYLD_VAR not in os.environ


def test_dyld_shim_opt_out(monkeypatch):
    from strands_robots import _dyld

    monkeypatch.setattr(_dyld.sys, "platform", "darwin")
    monkeypatch.setenv(_dyld._OPT_OUT_ENV, "1")
    assert _dyld.ensure_ffmpeg_on_dyld_path() is False


def test_dyld_shim_noop_without_torchcodec(monkeypatch):
    from strands_robots import _dyld

    monkeypatch.setattr(_dyld.sys, "platform", "darwin")
    monkeypatch.delenv(_dyld._OPT_OUT_ENV, raising=False)
    monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: False)
    assert _dyld.ensure_ffmpeg_on_dyld_path() is False


def test_dyld_shim_sets_env_and_skips_reexec_when_unsafe(monkeypatch, tmp_path):
    """When torchcodec + ffmpeg are present but it's NOT safe to re-exec
    (e.g. under pytest), the shim sets DYLD for child procs and does NOT
    re-exec — it warns instead."""
    from strands_robots import _dyld

    monkeypatch.setattr(_dyld.sys, "platform", "darwin")
    monkeypatch.delenv(_dyld._OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(_dyld._GUARD_ENV, raising=False)
    monkeypatch.delenv(_dyld._DYLD_VAR, raising=False)
    monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
    monkeypatch.setattr(_dyld, "_find_ffmpeg_lib_dir", lambda: str(tmp_path))
    # Under pytest, _is_safe_to_reexec() is False → must NOT call os.execv.
    called = {"execv": False}
    monkeypatch.setattr(_dyld.os, "execv", lambda *a: called.__setitem__("execv", True))

    with pytest.warns(RuntimeWarning):
        result = _dyld.ensure_ffmpeg_on_dyld_path()

    assert result is False
    assert called["execv"] is False  # never re-exec under pytest
    # but child-process env IS set
    assert str(tmp_path) in os.environ[_dyld._DYLD_VAR]


def test_dyld_shim_noop_when_already_set(monkeypatch, tmp_path):
    from strands_robots import _dyld

    monkeypatch.setattr(_dyld.sys, "platform", "darwin")
    monkeypatch.delenv(_dyld._OPT_OUT_ENV, raising=False)
    monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
    monkeypatch.setattr(_dyld, "_find_ffmpeg_lib_dir", lambda: str(tmp_path))
    monkeypatch.setenv(_dyld._DYLD_VAR, str(tmp_path))  # already present
    called = {"execv": False}
    monkeypatch.setattr(_dyld.os, "execv", lambda *a: called.__setitem__("execv", True))
    assert _dyld.ensure_ffmpeg_on_dyld_path() is True
    assert called["execv"] is False


# ── import-resolution branches (has_streaming_dataset / _get_streaming_cls) ─
#
# These exercise the real ``from lerobot.datasets import StreamingLeRobotDataset``
# path. lerobot itself is import-order fragile in some envs, so we inject a
# stand-in ``lerobot.datasets`` module rather than depend on the real package —
# the code under test only cares that the symbol resolves (or doesn't).


def _install_fake_lerobot_datasets(monkeypatch, *, with_streaming):
    """Put a fake ``lerobot.datasets`` in sys.modules; optionally expose
    StreamingLeRobotDataset on it. Returns the fake class (or None)."""
    import sys as _sys

    mod = type(_sys)("lerobot.datasets")
    cls = _FakeStreaming if with_streaming else None
    if with_streaming:
        mod.StreamingLeRobotDataset = _FakeStreaming
    monkeypatch.setitem(_sys.modules, "lerobot.datasets", mod)
    return cls


def test_has_streaming_dataset_true_when_importable(monkeypatch):
    """The probe reports True when the streaming symbol resolves
    (exercises the real import branch, not the fakes-only path)."""
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=True)
    monkeypatch.setattr(sd, "_HAS_STREAMING_DATASET", [])
    assert sd.has_streaming_dataset() is True


def test_has_streaming_dataset_false_when_import_breaks(monkeypatch):
    """If the streaming class cannot be imported, the probe returns False and
    swallows the error (offline / partial-install resilience)."""
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=False)
    monkeypatch.setattr(sd, "_HAS_STREAMING_DATASET", [])
    assert sd.has_streaming_dataset() is False


def test_has_streaming_dataset_does_not_negatively_cache(monkeypatch):
    """A failed availability probe must NOT be cached: once a transient import
    failure clears, the probe must report True again. A frozen first ``False``
    would permanently disable streaming for the rest of the process."""
    monkeypatch.setattr(sd, "_HAS_STREAMING_DATASET", [], raising=False)
    # Transient failure: the streaming symbol does not resolve.
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=False)
    assert sd.has_streaming_dataset() is False
    # Failure clears: the symbol resolves again on the very next call.
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=True)
    assert sd.has_streaming_dataset() is True


def test_has_streaming_dataset_positively_caches(monkeypatch):
    """A successful availability probe is memoized: once the streaming symbol
    resolves, later calls short-circuit and never re-import lerobot.

    The mirror of :func:`test_has_streaming_dataset_does_not_negatively_cache`.
    Hot-loop callers (agent / eval / replay loops) call this repeatedly, and on
    Jetson the ``from lerobot.datasets import ...`` probe is expensive (numpy /
    torch ABI). The contract is proven behaviorally by breaking the import
    AFTER the first success: a fresh probe of that broken state returns False
    (see the negative test above), so a warmed cache returning True can only be
    the memoized short-circuit - not a re-probe.
    """
    monkeypatch.setattr(sd, "_HAS_STREAMING_DATASET", [], raising=False)
    # First probe resolves the symbol and caches the positive result.
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=True)
    assert sd.has_streaming_dataset() is True
    # Break the import. A cold probe of this exact state returns False; the
    # warmed cache must short-circuit past it and still report True.
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=False)
    assert sd.has_streaming_dataset() is True


def test_get_streaming_cls_resolves_via_import(monkeypatch):
    """With no test-injected attribute override, _get_streaming_cls falls
    through to the real import and returns the resolved class."""
    monkeypatch.delattr(sd, "StreamingLeRobotDataset", raising=False)
    cls = _install_fake_lerobot_datasets(monkeypatch, with_streaming=True)
    assert sd._get_streaming_cls() is cls


def test_get_streaming_cls_raises_actionable_error_when_unavailable(monkeypatch):
    """When neither an override nor an import is available, the resolver raises
    ImportError with install guidance (never a bare AttributeError)."""
    monkeypatch.delattr(sd, "StreamingLeRobotDataset", raising=False)
    _install_fake_lerobot_datasets(monkeypatch, with_streaming=False)
    with pytest.raises(ImportError, match="StreamingLeRobotDataset unavailable"):
        sd._get_streaming_cls()


# ── delta-grid validation parity (check_delta_timestamps) ──────────────────


def _install_fake_checker(monkeypatch):
    """Inject a fake lerobot.datasets.feature_utils.check_delta_timestamps that
    enforces the on-grid rule (multiples of 1/fps within tolerance)."""
    import sys as _sys

    def check_delta_timestamps(delta_timestamps, fps, tolerance_s, raise_value_error=True):
        for key, deltas in delta_timestamps.items():
            for ts in deltas:
                if abs(ts * fps - round(ts * fps)) / fps > tolerance_s:
                    if raise_value_error:
                        raise ValueError(f"{key} delta {ts} off the 1/{fps} grid")
                    return False
        return True

    mod = type(_sys)("lerobot.datasets.feature_utils")
    mod.check_delta_timestamps = check_delta_timestamps
    monkeypatch.setitem(_sys.modules, "lerobot.datasets.feature_utils", mod)


def test_open_validates_aligned_deltas(monkeypatch):
    """Deltas that are integer multiples of 1/fps pass the parity grid-check
    (validate_deltas defaults on) and the reader is returned."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    _install_fake_checker(monkeypatch)
    # _FakeStreaming.fps == 30 → 0.0, 1/30, 2/30 are all on-grid.
    r = sd.StreamingDatasetReader.open(
        "org/ds",
        delta_timestamps={"observation.state": [0.0, 1 / 30, 2 / 30]},
    )
    assert r.dataset.kw["delta_timestamps"]["observation.state"] == [0.0, 1 / 30, 2 / 30]


def test_open_rejects_misaligned_deltas(monkeypatch):
    """Deltas off the 1/fps grid raise ValueError, matching the materialized
    dataset's check (the streaming path otherwise skips it)."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    _install_fake_checker(monkeypatch)
    with pytest.raises(ValueError, match="grid"):
        sd.StreamingDatasetReader.open(
            "org/ds",
            delta_timestamps={"observation.state": [0.017]},  # 0.017*30 = 0.51, off-grid
        )


def test_open_skips_validation_when_checker_unavailable(monkeypatch):
    """If check_delta_timestamps cannot be imported, validation is skipped
    silently and open still succeeds (validation is best-effort parity)."""
    import sys as _sys

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    broken = type(_sys)("lerobot.datasets.feature_utils")  # lacks check_delta_timestamps
    monkeypatch.setitem(_sys.modules, "lerobot.datasets.feature_utils", broken)
    r = sd.StreamingDatasetReader.open(
        "org/ds",
        delta_timestamps={"observation.state": [0.017]},  # off-grid but unchecked
    )
    assert r.dataset.kw["delta_timestamps"]["observation.state"] == [0.017]


# ── reader metadata + iteration passthrough ────────────────────────────────


def test_reader_exposes_metadata_and_iterates(monkeypatch):
    """num_frames / meta proxy the wrapped dataset and iteration yields its
    frames unchanged."""

    class _WithMeta(_FakeStreaming):
        meta = {"stats": {"action": {"mean": [0.0]}}}

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _WithMeta, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", validate_deltas=False)
    assert r.num_frames == 1000
    assert r.meta == {"stats": {"action": {"mean": [0.0]}}}
    frames = list(r)
    assert frames == [{"observation.state": [0.0], "action": [0.0], "task": "t"}]
