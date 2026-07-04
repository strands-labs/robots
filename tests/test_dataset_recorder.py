"""Unit tests for ``strands_robots.dataset_recorder.DatasetRecorder``.

These tests exercise the wrapper logic that does NOT require a real LeRobot
dataset by injecting a fake dataset object, so they run on a minimal env
(``lerobot`` not installed). They cover the partial-episode discard behaviour
and the add_frame() control-loop transform (schema-ordered flattening, camera
normalization, drop accounting), plus episode/finalize/push lifecycle.
"""

import logging

import numpy as np
import pytest

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


# add_frame() control-loop transformation behaviour
#
# These tests inject a fake LeRobot dataset that captures every frame handed to
# add_frame(), so they assert the recorder's observable output (the frame dict
# shape, vector ordering, camera normalization, drop accounting) without a real
# lerobot install.


class _CapturingDataset:
    """Fake LeRobot dataset that records frames and exposes a feature schema."""

    def __init__(self, features: dict, *, fail_add: bool = False):
        self.repo_id = "local/fake"
        self.root = "/tmp/local-fake"
        self.features = features
        self.frames: list[dict] = []
        self.saved = 0
        self.finalized = 0
        self.pushed: dict | None = None
        self._fail_add = fail_add

    def add_frame(self, frame):
        if self._fail_add:
            raise RuntimeError("disk full")
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1

    def finalize(self):
        self.finalized += 1

    def push_to_hub(self, tags=None, private=False):
        self.pushed = {"tags": tags, "private": private}


def _state_action_features(state_names, action_names) -> dict:
    return {
        "observation.state": {"dtype": "float32", "names": state_names},
        "action": {"dtype": "float32", "names": action_names},
    }


def test_add_frame_orders_state_and_action_by_schema():
    """State/action vectors follow the feature-schema order, not dict order."""
    ds = _CapturingDataset(_state_action_features(["j1", "j2", "j3"], ["j1", "j2", "j3"]))
    rec = DatasetRecorder(dataset=ds, task="pick")

    # Pass observation/action keys in a deliberately scrambled order.
    rec.add_frame(
        observation={"j3": 3.0, "j1": 1.0, "j2": 2.0},
        action={"j2": 0.2, "j3": 0.3, "j1": 0.1},
    )

    assert rec.frame_count == 1
    assert rec.episode_frame_count == 1
    frame = ds.frames[0]
    assert np.allclose(frame["observation.state"], [1.0, 2.0, 3.0])
    assert np.allclose(frame["action"], [0.1, 0.2, 0.3])
    assert frame["observation.state"].dtype == np.float32
    assert frame["task"] == "pick"


def test_add_frame_fills_missing_keys_with_zero():
    """A joint absent from the observation contributes 0.0 at its schema slot."""
    ds = _CapturingDataset(_state_action_features(["j1", "j2", "j3"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    rec.add_frame(observation={"j1": 1.0, "j3": 3.0}, action={"j1": 0.5}, task="t")

    frame = ds.frames[0]
    assert np.allclose(frame["observation.state"], [1.0, 0.0, 3.0])
    assert np.allclose(frame["action"], [0.5])


def test_add_frame_flattens_vector_valued_entries():
    """List / ndarray observation values are flattened into the state vector."""
    ds = _CapturingDataset(_state_action_features(["pose", "grip"], ["cmd"]))
    rec = DatasetRecorder(dataset=ds)

    rec.add_frame(
        observation={"pose": [1.0, 2.0, 3.0], "grip": 0.5},
        action={"cmd": np.array([0.1, 0.2])},
        task="t",
    )

    frame = ds.frames[0]
    assert np.allclose(frame["observation.state"], [1.0, 2.0, 3.0, 0.5])
    assert np.allclose(frame["action"], [0.1, 0.2])


def test_add_frame_records_numpy_scalar_state_and_action():
    """numpy scalars (np.float32 from indexing a qpos/ctrl array) are recorded
    as scalar columns, not silently dropped.

    MuJoCo observations and policy actions routinely hand back numpy scalar
    types (the element type you get from ``np.asarray(qpos)[i]``), which are
    neither Python ``float`` nor 0-dim ``np.ndarray``. They must still land in
    the flattened state/action vectors - dropping them would record
    wrong-length (or entirely missing) vectors with no error raised.
    """
    ds = _CapturingDataset(_state_action_features(["j1", "j2", "j3"], ["j1", "j2", "j3"]))
    rec = DatasetRecorder(dataset=ds, task="pick")

    rec.add_frame(
        observation={"j1": np.float32(1.0), "j2": np.float64(2.0), "j3": np.int32(3)},
        action={"j1": np.float32(0.1), "j2": np.float32(0.2), "j3": np.float32(0.3)},
    )

    frame = ds.frames[0]
    assert "observation.state" in frame, "numpy-scalar state was dropped"
    assert "action" in frame, "numpy-scalar action was dropped"
    assert np.allclose(frame["observation.state"], [1.0, 2.0, 3.0])
    assert np.allclose(frame["action"], [0.1, 0.2, 0.3])
    assert frame["observation.state"].dtype == np.float32
    assert frame["action"].dtype == np.float32


def test_add_frame_converts_float_images_to_uint8():
    """A float image in [0, 1] is scaled to uint8 HWC for LeRobot."""
    feats = {"observation.images.cam": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.ones((4, 4, 3), dtype=np.float32)  # all-white in [0, 1]
    rec.add_frame(observation={"cam": img}, action={}, task="t")

    out = ds.frames[0]["observation.images.cam"]
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)
    assert np.array_equal(out, np.full((4, 4, 3), 255, dtype=np.uint8))


def test_add_frame_normalizes_namespaced_camera_keys():
    """A 'arm0/wrist' camera key is rewritten to the declared 'arm0__wrist'."""
    feats = {"observation.images.arm0__wrist": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    rec.add_frame(observation={"arm0/wrist": img}, action={}, task="t")

    frame = ds.frames[0]
    assert "observation.images.arm0__wrist" in frame
    assert "observation.images.arm0/wrist" not in frame


def test_add_frame_strips_undeclared_cameras():
    """A camera not in the feature schema is dropped to avoid 'extra features'."""
    feats = {"observation.images.declared": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    rec.add_frame(
        observation={"declared": img, "ghost": img},
        action={},
        task="t",
    )

    frame = ds.frames[0]
    assert "observation.images.declared" in frame
    assert "observation.images.ghost" not in frame


def test_add_frame_strict_reraises_on_dataset_error():
    """strict=True (default) propagates a dataset add_frame failure."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]), fail_add=True)
    rec = DatasetRecorder(dataset=ds, strict=True)

    with pytest.raises(RuntimeError, match="disk full"):
        rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")
    assert rec.frame_count == 0


def test_add_frame_non_strict_counts_drops_without_raising(caplog):
    """strict=False swallows the error, counts the drop, and logs ASCII-only."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]), fail_add=True)
    rec = DatasetRecorder(dataset=ds, strict=False)

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")

    assert rec.dropped_frame_count == 1
    assert rec.frame_count == 0
    for record in caplog.records:
        record.getMessage().encode("ascii")


def test_add_frame_noop_when_closed():
    """A closed recorder ignores add_frame instead of corrupting the dataset."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec._closed = True

    rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")

    assert ds.frames == []
    assert rec.frame_count == 0


def test_save_episode_reports_per_episode_frame_count():
    """save_episode reports frames since the last save and resets the counter."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")
    rec.add_frame(observation={"j1": 2.0}, action={"j1": 0.2}, task="t")
    result = rec.save_episode()

    assert result["status"] == "success"
    assert result["episode"] == 1
    assert result["episode_frames"] == 2
    assert result["total_frames"] == 2
    assert rec.episode_frame_count == 0  # reset for the next episode

    # A second episode reports only its own frames; total stays monotonic.
    rec.add_frame(observation={"j1": 3.0}, action={"j1": 0.3}, task="t")
    result2 = rec.save_episode()
    assert result2["episode_frames"] == 1
    assert result2["total_frames"] == 3


def test_save_episode_poisons_recorder_on_failure():
    """A failed save closes the recorder so later frames cannot corrupt data."""

    class _BadSave(_CapturingDataset):
        def save_episode(self):
            raise RuntimeError("encode failed")

    ds = _BadSave(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")

    result = rec.save_episode()

    assert result["status"] == "error"
    assert rec._closed is True
    # save_episode on a closed recorder returns a clean error, not a crash.
    assert rec.save_episode()["status"] == "error"


def test_finalize_is_idempotent_and_closes():
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    rec.finalize()
    rec.finalize()  # second call is a no-op

    assert rec._closed is True
    assert ds.finalized == 1


def test_push_to_hub_success_and_failure():
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.episode_count = 2
    rec.frame_count = 50

    ok = rec.push_to_hub(tags=["sim"], private=True)
    assert ok == {
        "status": "success",
        "repo_id": "local/fake",
        "episodes": 2,
        "frames": 50,
    }
    assert ds.pushed == {"tags": ["sim"], "private": True}

    class _BadPush(_CapturingDataset):
        def push_to_hub(self, tags=None, private=False):
            raise RuntimeError("network down")

    bad = DatasetRecorder(dataset=_BadPush(_state_action_features(["j1"], ["j1"])))
    bad.episode_count = 1
    bad.frame_count = 10
    err = bad.push_to_hub()
    assert err["status"] == "error"
    assert "network down" in err["message"]


def test_push_to_hub_refuses_empty_dataset_no_frames():
    """A recorder that captured no frames must not hit the Hub.

    Regression: push_to_hub used to publish unconditionally, so a rollout that
    never fed the recorder (e.g. driven by eval_policy / a bare step loop)
    produced a Hub repo containing only meta/info.json. The guard now returns a
    structured error and makes no Hub call.
    """
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    # frame_count and episode_count are 0 by construction.
    result = rec.push_to_hub(tags=["sim"])

    assert result["status"] == "error"
    assert "empty dataset" in result["message"]
    assert ds.pushed is None  # the underlying dataset.push_to_hub was never called


def test_push_to_hub_refuses_frames_without_saved_episode():
    """Frames buffered but no episode flushed is still an empty dataset on disk."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.frame_count = 30
    rec.episode_count = 0  # save_episode was never called (or it failed)

    result = rec.push_to_hub()

    assert result["status"] == "error"
    assert "empty dataset" in result["message"]
    assert ds.pushed is None


def test_repo_id_root_and_repr_properties():
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.episode_count = 1
    rec.frame_count = 7

    assert rec.repo_id == "local/fake"
    assert rec.root == "/tmp/local-fake"
    rep = repr(rec)
    assert "local/fake" in rep and "episodes=1" in rep and "frames=7" in rep


# ---------------------------------------------------------------------------
# load_lerobot_episode: frame-range resolution for replay
# ---------------------------------------------------------------------------
# load_lerobot_episode() resolves the [start, start+length) frame window for a
# given episode index across three LeRobot dataset shapes:
#   1. episode_data_index present (the fast path: from/to index tensors),
#   2. no episode_data_index but meta.episodes carries per-episode lengths,
#   3. neither usable -> last-resort frame scan keyed on episode_index.
# It guards out-of-range and empty episodes with ValueError. These tests inject
# a fake LeRobotDataset so the real body runs without lerobot installed.


class _IndexCell:
    """Mimics a tensor cell exposing ``.item()`` (as torch index tensors do)."""

    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


class _IndexColumn:
    """Indexable column of _IndexCell, mimicking dataset.episode_data_index['from']."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def __getitem__(self, idx: int) -> _IndexCell:
        return _IndexCell(self._values[idx])


class _FakeMeta:
    def __init__(self, total_episodes=None, episodes=None) -> None:
        if total_episodes is not None:
            self.total_episodes = total_episodes
        if episodes is not None:
            self.episodes = episodes


class _FakeDatasetWithIndex:
    """Fast path: exposes episode_data_index with from/to columns."""

    def __init__(self, repo_id, root=None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.meta = _FakeMeta(total_episodes=3)
        self.episode_data_index = {
            "from": _IndexColumn([0, 10, 25]),
            "to": _IndexColumn([10, 25, 40]),
        }


class _FakeDatasetMetaLengths:
    """No episode_data_index; lengths live in meta.episodes."""

    def __init__(self, repo_id, root=None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.meta = _FakeMeta(
            episodes=[{"length": 10}, {"length": 15}, {"length": 5}],
        )


class _RaisingIndex:
    """Subscriptable that raises on lookup, simulating an unusable index.

    Raises ``KeyError`` (the idiomatic ``LookupError`` for a failed
    subscription) so the helper's broad ``except Exception`` still catches it
    and falls through to the frame-scan path, mirroring a real dataset whose
    ``episode_data_index`` is present but missing the expected columns.
    """

    def __getitem__(self, key):
        raise KeyError(key)


class _FakeDatasetScan:
    """Neither fast path usable: forces the frame-scan fallback.

    episode_data_index access raises, so the helper falls through to scanning
    frames by their episode_index field.
    """

    def __init__(self, repo_id, root=None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.meta = _FakeMeta(total_episodes=2)
        # episode 0 -> frames [0,1,2]; episode 1 -> frames [3,4]
        self._frames = [
            {"episode_index": _IndexCell(0)},
            {"episode_index": _IndexCell(0)},
            {"episode_index": _IndexCell(0)},
            {"episode_index": _IndexCell(1)},
            {"episode_index": _IndexCell(1)},
        ]

        # episode_data_index exists (so hasattr passes) but subscripting it
        # raises, forcing the helper into the frame-scan fallback.
        self.episode_data_index = _RaisingIndex()

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]


def _patch_lerobot_dataset(monkeypatch, fake_cls) -> None:
    """Inject a fake LeRobotDataset into the import target used by the helper."""
    import sys
    import types

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    setattr(module, "LeRobotDataset", fake_cls)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)


def test_load_episode_uses_episode_data_index_fast_path(monkeypatch):
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetWithIndex)
    ds, start, length = dr.load_lerobot_episode("user/data", episode=1)

    assert start == 10
    assert length == 15
    assert ds.repo_id == "user/data"


def test_load_episode_first_episode_window(monkeypatch):
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetWithIndex)
    _, start, length = dr.load_lerobot_episode("user/data", episode=0)

    assert start == 0
    assert length == 10


def test_load_episode_falls_back_to_meta_lengths(monkeypatch):
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetMetaLengths)
    # episode 2 starts after episodes 0 (10) + 1 (15) = 25, with length 5.
    _, start, length = dr.load_lerobot_episode("user/data", episode=2)

    assert start == 25
    assert length == 5


def test_load_episode_scans_frames_as_last_resort(monkeypatch):
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetScan)
    _, start, length = dr.load_lerobot_episode("user/data", episode=1)

    # episode 1 occupies frames 3 and 4.
    assert start == 3
    assert length == 2


def test_load_episode_rejects_out_of_range(monkeypatch):
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetWithIndex)
    with pytest.raises(ValueError, match="out of range"):
        dr.load_lerobot_episode("user/data", episode=3)


def test_load_episode_rejects_empty_episode(monkeypatch):
    from strands_robots import dataset_recorder as dr

    class _EmptyEpisode:
        def __init__(self, repo_id, root=None) -> None:
            self.meta = _FakeMeta(episodes=[{"length": 0}])

    _patch_lerobot_dataset(monkeypatch, _EmptyEpisode)
    with pytest.raises(ValueError, match="no frames"):
        dr.load_lerobot_episode("user/data", episode=0)


def test_load_episode_scan_breaks_after_target_episode(monkeypatch):
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetScan)
    # episode 0 occupies frames 0,1,2; the scan must stop at frame 3 (episode 1).
    _, start, length = dr.load_lerobot_episode("user/data", episode=0)

    assert start == 0
    assert length == 3


class TestBuildFeaturesSchema:
    """Behavioral tests for ``DatasetRecorder._build_features`` -- the function
    that turns robot/action feature descriptors into the LeRobot v3 ``features``
    schema every recording is created with.

    The contract under test is the *shape of the emitted schema dict*: which
    keys appear, their ``dtype``/``shape``/``names``, and how the state and
    action dimensions are derived from the several mutually-exclusive input
    sources (explicit feature dicts, a flat ``joint_names`` list, or the
    action-mirrors-state fallback). These are pure-logic assertions -- no
    LeRobot install is required because ``_build_features`` is a classmethod
    that only manipulates plain dicts.
    """

    def test_camera_keys_emit_video_features_with_default_dims(self):
        """Each camera key becomes an ``observation.images.<name>`` video
        feature; absent per-camera dims fall back to the global video size."""
        features = DatasetRecorder._build_features(
            camera_keys=["top", "wrist"],
            video_height=480,
            video_width=640,
        )

        assert features["observation.images.top"] == {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
        assert features["observation.images.wrist"]["shape"] == (3, 480, 640)

    def test_camera_dims_override_per_camera_resolution(self):
        """A per-camera entry in ``camera_dims`` overrides the global size for
        that camera only; others keep the fallback."""
        features = DatasetRecorder._build_features(
            camera_keys=["top", "wrist"],
            camera_dims={"top": (240, 320)},
            video_height=480,
            video_width=640,
        )

        assert features["observation.images.top"]["shape"] == (3, 240, 320)
        assert features["observation.images.wrist"]["shape"] == (3, 480, 640)

    def test_use_videos_false_emits_image_dtype(self):
        """``use_videos=False`` records still frames (``image`` dtype) rather
        than encoded video."""
        features = DatasetRecorder._build_features(camera_keys=["top"], use_videos=False)

        assert features["observation.images.top"]["dtype"] == "image"

    def test_robot_features_drive_state_excluding_cameras(self):
        """Scalar entries in ``robot_features`` become the ``observation.state``
        vector; image/video entries are excluded from the state count."""
        features = DatasetRecorder._build_features(
            robot_features={
                "shoulder.pos": {"dtype": "float32"},
                "elbow.pos": {"dtype": "float32"},
                "front_cam": {"dtype": "video"},
            },
        )

        state = features["observation.state"]
        assert state["dtype"] == "float32"
        assert state["shape"] == (2,)
        assert state["names"] == ["shoulder.pos", "elbow.pos"]

    def test_joint_names_drive_state_when_no_robot_features(self):
        """With no ``robot_features``, a flat ``joint_names`` list defines the
        state dimension and names."""
        features = DatasetRecorder._build_features(joint_names=["j1", "j2", "j3"])

        assert features["observation.state"]["shape"] == (3,)
        assert features["observation.state"]["names"] == ["j1", "j2", "j3"]

    def test_action_features_drive_action_excluding_cameras(self):
        """Explicit ``action_features`` define the action vector, excluding any
        image/video columns."""
        features = DatasetRecorder._build_features(
            action_features={
                "shoulder.pos": {"dtype": "float32"},
                "gripper.pos": {"dtype": "float32"},
                "debug_cam": {"dtype": "image"},
            },
        )

        action = features["action"]
        assert action["shape"] == (2,)
        assert action["names"] == ["shoulder.pos", "gripper.pos"]

    def test_action_names_define_action_columns_independent_of_joints(self):
        """An explicit ``action_names`` list defines the action columns even
        when it diverges from ``joint_names``. This is the actuator-keyed sim
        path: a policy emits actuator short-names (``shoulder_pan_act``) while
        observation.state stays joint-keyed (``shoulder_pan``), so the recorded
        action columns must follow the actuator keys add_frame receives -- not
        the joint names, which would never match and record all zeros."""
        features = DatasetRecorder._build_features(
            joint_names=["shoulder_pan", "shoulder_lift", "elbow"],
            action_names=["shoulder_pan_act", "shoulder_lift_act", "elbow_act"],
        )

        # State stays joint-keyed.
        assert features["observation.state"]["names"] == ["shoulder_pan", "shoulder_lift", "elbow"]
        # Action follows the actuator keys.
        assert features["action"]["shape"] == (3,)
        assert features["action"]["names"] == ["shoulder_pan_act", "shoulder_lift_act", "elbow_act"]

    def test_action_features_take_precedence_over_action_names(self):
        """``action_features`` (explicit typed schema) wins over the
        ``action_names`` fallback when both are supplied."""
        features = DatasetRecorder._build_features(
            action_features={"a": {"dtype": "float32"}, "b": {"dtype": "float32"}},
            action_names=["x", "y", "z"],
            joint_names=["j1", "j2"],
        )

        assert features["action"]["names"] == ["a", "b"]

    def test_action_names_fall_back_to_joint_names_when_absent(self):
        """With no ``action_names`` the action columns mirror ``joint_names``
        (the actuator-mirrors-joints case for stock arms, unchanged behaviour)."""
        features = DatasetRecorder._build_features(joint_names=["j1", "j2", "j3"])

        assert features["action"]["names"] == ["j1", "j2", "j3"]

    def test_action_mirrors_state_when_only_robot_features_given(self):
        """With state from ``robot_features`` but no action source, the action
        feature mirrors the state dimension and names."""
        features = DatasetRecorder._build_features(
            robot_features={"a": {"dtype": "float32"}, "b": {"dtype": "float32"}},
        )

        assert features["action"]["shape"] == (2,)
        assert features["action"]["names"] == ["a", "b"]
        # The mirror is a copy, not an alias of the state names list.
        assert features["action"]["names"] is not features["observation.state"]["names"]

    def test_non_dict_feature_values_count_as_scalar_state(self):
        """A feature whose value is not a dict (e.g. a bare descriptor) is
        treated as a scalar state column, not skipped."""
        features = DatasetRecorder._build_features(
            robot_features={"raw_scalar": "float32", "cam": {"dtype": "video"}},
        )

        assert features["observation.state"]["shape"] == (1,)
        assert features["observation.state"]["names"] == ["raw_scalar"]

    def test_empty_inputs_emit_empty_schema(self):
        """With no cameras, features, or joint names there is nothing to record
        -- the schema is empty rather than carrying zero-dim entries."""
        features = DatasetRecorder._build_features()

        assert features == {}


# ---------------------------------------------------------------------------
# DatasetRecorder.resume() -- the multi-episode append entry point.
#
# resume() opens an EXISTING on-disk LeRobotDataset for appending (the plain
# constructor returns a read-only dataset). Its body must:
#   1. hard-fail with a clear RuntimeError on LeRobot versions lacking resume(),
#   2. route the requested vcodec version-tolerantly -- pass ``vcodec=`` when the
#      installed resume() takes it, else wrap it in a ``VideoEncoderConfig`` for
#      the ``camera_encoder=`` kwarg (0.5.2+), warning if that config is absent,
#   3. forward the optional streaming/threads/backend kwargs only when supported,
#   4. seed episode/frame counters from the existing dataset so totals report
#      correctly, swallowing a malformed meta rather than crashing.
# These tests inject fake LeRobotDataset classes whose ``resume`` classmethods
# expose different signatures, so the real body runs without lerobot installed.


class _ResumeMeta:
    def __init__(self, total_episodes, total_frames) -> None:
        self.total_episodes = total_episodes
        self.total_frames = total_frames


class _FakeDatasetVcodecResume:
    """resume() accepts the legacy ``vcodec=`` kwarg directly."""

    last_resume_kwargs: dict = {}

    def __init__(self, repo_id, root=None, meta=None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.meta = meta or _ResumeMeta(total_episodes=2, total_frames=40)

    @classmethod
    def resume(cls, repo_id, root=None, vcodec="libsvtav1", streaming_encoding=True):
        cls.last_resume_kwargs = {
            "repo_id": repo_id,
            "root": root,
            "vcodec": vcodec,
            "streaming_encoding": streaming_encoding,
        }
        return cls(repo_id, root=root)


class _FakeDatasetCameraEncoderResume:
    """resume() takes ``camera_encoder=`` (0.5.2+) plus thread/backend kwargs."""

    last_resume_kwargs: dict = {}

    def __init__(self, repo_id, root=None, meta=None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.meta = meta or _ResumeMeta(total_episodes=5, total_frames=123)

    @classmethod
    def resume(
        cls,
        repo_id,
        root=None,
        camera_encoder=None,
        image_writer_threads=4,
        video_backend="auto",
    ):
        cls.last_resume_kwargs = {
            "repo_id": repo_id,
            "root": root,
            "camera_encoder": camera_encoder,
            "image_writer_threads": image_writer_threads,
            "video_backend": video_backend,
        }
        return cls(repo_id, root=root)


def _install_video_encoder_config(monkeypatch):
    """Provide a stub ``lerobot.configs.video.VideoEncoderConfig`` and return a
    list that captures each instance constructed, so a test can assert the
    recorder wrapped its vcodec into the config."""
    import sys
    import types

    constructed = []

    class _VideoEncoderConfig:
        def __init__(self, vcodec) -> None:
            self.vcodec = vcodec
            constructed.append(self)

    module = types.ModuleType("lerobot.configs.video")
    module.VideoEncoderConfig = _VideoEncoderConfig
    monkeypatch.setitem(sys.modules, "lerobot.configs.video", module)
    return constructed


def test_resume_raises_clear_error_when_lerobot_lacks_resume(monkeypatch):
    """Pre-0.5.2 LeRobot has no resume(); the recorder must say so explicitly."""
    from strands_robots import dataset_recorder as dr

    class _NoResumeDataset:
        def __init__(self, repo_id, root=None) -> None:
            self.repo_id = repo_id

    _patch_lerobot_dataset(monkeypatch, _NoResumeDataset)
    with pytest.raises(RuntimeError, match="no LeRobotDataset.resume"):
        dr.DatasetRecorder.resume("user/data")


def test_resume_passes_vcodec_directly_when_supported(monkeypatch):
    """When resume() takes ``vcodec=``, the recorder forwards it as-is and only
    adds the optional kwargs the signature actually declares."""
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetVcodecResume)
    recorder = dr.DatasetRecorder.resume("user/data", root="/tmp/ds", vcodec="libx264", task="pick")

    sent = _FakeDatasetVcodecResume.last_resume_kwargs
    # The flat ``vcodec`` surface validates against the codec-name allowlist
    # (it rejects "libx264"), so the ffmpeg name is normalized to "h264".
    assert sent["vcodec"] == "h264"
    assert sent["repo_id"] == "user/data"
    assert sent["root"] == "/tmp/ds"
    assert sent["streaming_encoding"] is True
    # video_backend / image_writer_threads are NOT in this signature -> not sent.
    assert "video_backend" not in sent
    assert "image_writer_threads" not in sent
    assert recorder.default_task == "pick"


def test_resume_wraps_vcodec_in_camera_encoder_on_052(monkeypatch):
    """On 0.5.2+ (camera_encoder= kwarg), the vcodec is wrapped in a
    VideoEncoderConfig, and thread/backend kwargs are forwarded."""
    from strands_robots import dataset_recorder as dr

    constructed = _install_video_encoder_config(monkeypatch)
    _patch_lerobot_dataset(monkeypatch, _FakeDatasetCameraEncoderResume)
    dr.DatasetRecorder.resume("user/data", vcodec="libsvtav1", video_backend="pyav")

    sent = _FakeDatasetCameraEncoderResume.last_resume_kwargs
    assert sent["video_backend"] == "pyav"
    assert sent["image_writer_threads"] == 4
    # vcodec must have been wrapped, not passed raw.
    assert "vcodec" not in sent
    assert len(constructed) == 1
    assert sent["camera_encoder"] is constructed[0]
    assert constructed[0].vcodec == "libsvtav1"


def test_resume_warns_when_video_encoder_config_missing(monkeypatch, caplog):
    """If resume() wants camera_encoder= but VideoEncoderConfig can't be
    imported, the recorder warns and proceeds with camera_encoder unset."""
    import sys

    from strands_robots import dataset_recorder as dr

    # Ensure the import target is absent so the import raises ImportError.
    monkeypatch.setitem(sys.modules, "lerobot.configs.video", None)
    _patch_lerobot_dataset(monkeypatch, _FakeDatasetCameraEncoderResume)

    with caplog.at_level("WARNING"):
        dr.DatasetRecorder.resume("user/data", vcodec="libsvtav1")

    sent = _FakeDatasetCameraEncoderResume.last_resume_kwargs
    assert sent["camera_encoder"] is None
    assert any("VideoEncoderConfig" in rec.message for rec in caplog.records)


def test_resume_seeds_counters_from_existing_dataset(monkeypatch):
    """Counters are seeded from the dataset meta so reported totals include the
    episodes/frames already on disk."""
    from strands_robots import dataset_recorder as dr

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetVcodecResume)
    recorder = dr.DatasetRecorder.resume("user/data")

    assert recorder.episode_count == 2
    assert recorder.frame_count == 40


def test_resume_tolerates_unreadable_meta_counters(monkeypatch):
    """A dataset whose meta lacks numeric totals must not crash resume(); the
    counters simply stay at their zero defaults."""
    from strands_robots import dataset_recorder as dr

    class _BadMeta:
        @property
        def total_episodes(self):
            raise AttributeError("no totals on this meta")

    class _FakeDatasetBadMeta(_FakeDatasetVcodecResume):
        def __init__(self, repo_id, root=None, meta=None) -> None:
            super().__init__(repo_id, root=root, meta=_BadMeta())

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetBadMeta)
    recorder = dr.DatasetRecorder.resume("user/data")

    assert recorder.episode_count == 0
    assert recorder.frame_count == 0


# ---------------------------------------------------------------------------
# DatasetRecorder.create() -- the new-dataset entry point.
#
# create() builds a LeRobotDataset from auto-detected features and must route
# the requested vcodec across LeRobot versions exactly as resume() does:
#   * 0.5.0/0.5.1: create(..., vcodec="libsvtav1")
#   * 0.5.2+:      create(..., camera_encoder=VideoEncoderConfig(vcodec=...))
# It also forwards the optional streaming_encoding / image_writer_threads /
# video_backend kwargs only when the installed create() signature declares them,
# and returns a recorder bound to the created dataset and default task. These
# tests inject fake LeRobotDataset classes whose ``create`` classmethods expose
# different signatures, so the real body runs without lerobot installed.


class _FakeDatasetVcodecCreate:
    """create() accepts the legacy ``vcodec=`` kwarg directly (0.5.0/0.5.1)."""

    last_create_kwargs: dict = {}

    def __init__(self, repo_id, root=None) -> None:
        self.repo_id = repo_id
        self.root = root

    @classmethod
    def create(
        cls,
        repo_id,
        fps=30,
        root=None,
        robot_type="unknown",
        features=None,
        use_videos=True,
        image_writer_threads=4,
        vcodec="libsvtav1",
        streaming_encoding=True,
    ):
        cls.last_create_kwargs = {
            "repo_id": repo_id,
            "fps": fps,
            "root": root,
            "robot_type": robot_type,
            "features": features,
            "use_videos": use_videos,
            "image_writer_threads": image_writer_threads,
            "vcodec": vcodec,
            "streaming_encoding": streaming_encoding,
        }
        return cls(repo_id, root=root)


class _FakeDatasetCameraEncoderCreate:
    """create() takes ``camera_encoder=`` (0.5.2+) plus backend kwargs."""

    last_create_kwargs: dict = {}

    def __init__(self, repo_id, root=None) -> None:
        self.repo_id = repo_id
        self.root = root

    @classmethod
    def create(
        cls,
        repo_id,
        fps=30,
        root=None,
        robot_type="unknown",
        features=None,
        use_videos=True,
        image_writer_threads=4,
        camera_encoder=None,
        streaming_encoding=True,
        video_backend="auto",
    ):
        cls.last_create_kwargs = {
            "repo_id": repo_id,
            "features": features,
            "image_writer_threads": image_writer_threads,
            "camera_encoder": camera_encoder,
            "streaming_encoding": streaming_encoding,
            "video_backend": video_backend,
        }
        return cls(repo_id, root=root)


def test_create_passes_vcodec_directly_when_supported(monkeypatch):
    """When create() takes ``vcodec=``, the recorder forwards it as-is and does
    not synthesize a camera_encoder; backend kwargs absent from the signature
    are not sent."""
    _patch_lerobot_dataset(monkeypatch, _FakeDatasetVcodecCreate)
    recorder = DatasetRecorder.create(
        "user/data", fps=50, root="/tmp/ds", joint_names=["j1", "j2"], vcodec="libx264", task="pick"
    )

    sent = _FakeDatasetVcodecCreate.last_create_kwargs
    # The flat ``vcodec`` surface validates against the codec-name allowlist
    # (it rejects "libx264"), so the ffmpeg name is normalized to "h264".
    assert sent["vcodec"] == "h264"
    assert sent["repo_id"] == "user/data"
    assert sent["fps"] == 50
    assert sent["streaming_encoding"] is True
    # camera_encoder / video_backend are NOT in this signature -> not sent.
    assert "camera_encoder" not in sent
    assert "video_backend" not in sent
    assert recorder.default_task == "pick"
    assert recorder.dataset.repo_id == "user/data"


def test_create_wraps_vcodec_in_camera_encoder_on_052(monkeypatch):
    """On 0.5.2+ (camera_encoder= kwarg), the vcodec is wrapped in a
    VideoEncoderConfig and the backend kwargs are forwarded."""
    constructed = _install_video_encoder_config(monkeypatch)
    _patch_lerobot_dataset(monkeypatch, _FakeDatasetCameraEncoderCreate)
    DatasetRecorder.create("user/data", joint_names=["j1"], vcodec="libsvtav1", video_backend="pyav")

    sent = _FakeDatasetCameraEncoderCreate.last_create_kwargs
    assert sent["video_backend"] == "pyav"
    assert sent["image_writer_threads"] == 4
    # vcodec must have been wrapped, not passed raw.
    assert "vcodec" not in sent
    assert len(constructed) == 1
    assert sent["camera_encoder"] is constructed[0]
    assert constructed[0].vcodec == "libsvtav1"


def test_create_warns_when_video_encoder_config_missing(monkeypatch, caplog):
    """If create() wants camera_encoder= but VideoEncoderConfig can't be
    imported, the recorder warns and proceeds with camera_encoder unset rather
    than crashing the recording setup."""
    import sys

    monkeypatch.setitem(sys.modules, "lerobot.configs.video", None)
    _patch_lerobot_dataset(monkeypatch, _FakeDatasetCameraEncoderCreate)

    with caplog.at_level("WARNING"):
        DatasetRecorder.create("user/data", joint_names=["j1"], vcodec="libsvtav1")

    sent = _FakeDatasetCameraEncoderCreate.last_create_kwargs
    assert sent["camera_encoder"] is None
    assert any("VideoEncoderConfig" in rec.message for rec in caplog.records)


def test_create_forwards_optional_kwargs_only_when_supported(monkeypatch):
    """A minimal create() lacking streaming_encoding / video_backend must not be
    handed those kwargs (version-tolerant forwarding, no TypeError)."""

    class _FakeDatasetMinimalCreate:
        last_create_kwargs: dict = {}

        def __init__(self, repo_id, root=None) -> None:
            self.repo_id = repo_id
            self.root = root

        @classmethod
        def create(
            cls,
            repo_id,
            fps=30,
            root=None,
            robot_type="unknown",
            features=None,
            use_videos=True,
            image_writer_threads=4,
            vcodec="libsvtav1",
        ):
            cls.last_create_kwargs = {"repo_id": repo_id, "vcodec": vcodec}
            return cls(repo_id, root=root)

    _patch_lerobot_dataset(monkeypatch, _FakeDatasetMinimalCreate)
    recorder = DatasetRecorder.create("user/data", joint_names=["j1"])

    sent = _FakeDatasetMinimalCreate.last_create_kwargs
    # The default codec "h264" is already an allowlist-valid codec name and is
    # forwarded unchanged onto the flat ``vcodec`` surface.
    assert sent == {"repo_id": "user/data", "vcodec": "h264"}
    assert recorder.dataset.repo_id == "user/data"


def test_create_builds_features_from_joints_and_cameras(monkeypatch):
    """create() derives the LeRobot ``features`` schema from joint_names and
    camera_keys and hands it to the underlying dataset constructor."""
    _patch_lerobot_dataset(monkeypatch, _FakeDatasetVcodecCreate)
    DatasetRecorder.create(
        "user/data",
        joint_names=["shoulder", "elbow"],
        camera_keys=["top"],
        video_height=240,
        video_width=320,
    )

    features = _FakeDatasetVcodecCreate.last_create_kwargs["features"]
    assert features["observation.state"]["names"] == ["shoulder", "elbow"]
    assert features["observation.images.top"]["shape"] == (3, 240, 320)


# camera_key_map remap + camera-key-mismatch diagnostic
#
# Regression coverage for the silent data-loss mode where a policy declares
# image_keys (e.g. "image"/"wrist_image") that never match the names the
# sim/hardware camera streams emit (e.g. "front_camera"/"wrist_camera"): every
# image frame is stripped and the dataset records zero video columns with no
# error. The recorder now (a) accepts a camera_key_map remap and (b) warns once
# when all observed cameras are dropped.


def test_add_frame_camera_key_map_remaps_observed_to_declared():
    """camera_key_map rewrites an observed stream name to the declared schema key."""
    feats = {"observation.images.image": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds, camera_key_map={"front_camera": "image"})

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    rec.add_frame(observation={"front_camera": img}, action={}, task="t")

    frame = ds.frames[0]
    assert "observation.images.image" in frame
    assert "observation.images.front_camera" not in frame


def test_add_frame_camera_key_map_accepts_fully_qualified_keys():
    """camera_key_map entries may be written as observation.images.* keys."""
    feats = {"observation.images.wrist_image": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(
        dataset=ds,
        camera_key_map={"observation.images.wrist_camera": "observation.images.wrist_image"},
    )

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    rec.add_frame(observation={"wrist_camera": img}, action={}, task="t")

    frame = ds.frames[0]
    assert "observation.images.wrist_image" in frame
    assert "observation.images.wrist_camera" not in frame


def test_add_frame_warns_once_when_no_camera_matches(caplog):
    """All observed cameras stripped (none declared) -> one loud diagnostic."""
    feats = {"observation.images.image": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        rec.add_frame(observation={"front_camera": img}, action={}, task="t")
        rec.add_frame(observation={"front_camera": img}, action={}, task="t")

    mismatch = [r for r in caplog.records if "match the declared image features" in r.message]
    assert len(mismatch) == 1  # warned once, not per frame
    # The message names the offending observed and declared keys + the remedy.
    assert "front_camera" in mismatch[0].getMessage()
    assert "image" in mismatch[0].getMessage()
    assert "camera_key_map" in mismatch[0].getMessage()
    # The image data was still dropped (no remap supplied).
    assert "observation.images.front_camera" not in ds.frames[0]


def test_add_frame_no_mismatch_warning_on_partial_strip(caplog):
    """A partial strip (one camera matches) is the normal 'ignore extra' path."""
    feats = {"observation.images.declared": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        rec.add_frame(observation={"declared": img, "ghost": img}, action={}, task="t")

    assert not [r for r in caplog.records if "match the declared image features" in r.message]
    assert "observation.images.declared" in ds.frames[0]
    assert "observation.images.ghost" not in ds.frames[0]


def test_add_frame_no_mismatch_warning_when_remap_resolves_it(caplog):
    """With a remap that resolves the mismatch, no diagnostic is emitted."""
    feats = {"observation.images.image": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds, camera_key_map={"front_camera": "image"})

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        rec.add_frame(observation={"front_camera": img}, action={}, task="t")

    assert not [r for r in caplog.records if "match the declared image features" in r.message]


def test_has_lerobot_dataset_does_not_negatively_cache(monkeypatch):
    """A failed lerobot-availability probe must NOT be cached.

    lerobot availability is a process capability that can transiently fail to
    resolve (a slow/locked import, or a temporarily monkeypatched
    ``sys.modules``). Freezing the first ``False`` permanently disabled
    recording for the rest of the process - ``start_recording`` would report
    ``requires strands-robots[lerobot]`` forever even after the condition
    cleared. The probe must re-attempt the import on the next call.
    """
    import importlib
    import sys

    from strands_robots import dataset_recorder as dr

    # Capture the real module so we can swap it back in a fully reversible way
    # (monkeypatch.setitem restores the original entry on teardown; no delitem
    # that could leave the parent package attribute stale for later tests).
    real_module = importlib.import_module("lerobot.datasets.lerobot_dataset")

    # Start from a clean, un-probed state regardless of implementation.
    monkeypatch.setattr(dr, "_HAS_LEROBOT_DATASET", [], raising=False)

    # Transient failure: the lerobot dataset module does not resolve.
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)
    assert dr.has_lerobot_dataset() is False

    # Failure clears: the very next call must re-probe and resolve True.
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", real_module)
    assert dr.has_lerobot_dataset() is True


def test_load_lerobot_episode_rejects_negative_index():
    """A negative episode index is a programming error, not episode N-1.

    ``load_lerobot_episode`` previously only guarded the upper bound
    (``episode >= num_episodes``); a negative index fell through to Python's
    negative list indexing, silently resolving (and replaying) the LAST
    episode while reporting the negative number. Reject it up front, before
    any dataset construction, so the contract holds even without lerobot
    installed and without a network round-trip.
    """
    from strands_robots.dataset_recorder import load_lerobot_episode

    with pytest.raises(ValueError, match="non-negative"):
        load_lerobot_episode("any/repo", episode=-1)


class TestCodecRouting:
    """Cover ``_codec_create_kwargs`` version-tolerant codec routing.

    LeRobot routes the codec onto different surfaces across releases (flat
    ``vcodec`` on 0.5.0/0.5.1; an encoder-config object on later minors), but
    every one validates against the same codec-name allowlist and rejects the
    ffmpeg names "libx264"/"libx265". Routing the codec onto the wrong (or no)
    surface silently drops the caller's request and falls back to the AV1
    default. These tests assert the requested codec actually reaches the
    surface this LeRobot exposes, with name spellings normalized per surface.
    """

    def test_flat_vcodec_uses_codec_name(self):
        from strands_robots.dataset_recorder import _codec_create_kwargs

        # The flat ``vcodec`` surface validates against the codec-name allowlist
        # in every supported LeRobot (>=0.5.0,<0.6.0), so the codec name is
        # forwarded as-is - NOT mapped to the ffmpeg name "libx264".
        assert _codec_create_kwargs({"vcodec": None}, "h264") == {"vcodec": "h264"}
        # An ffmpeg name is normalized to its allowlist codec name.
        assert _codec_create_kwargs({"vcodec": None}, "libx264") == {"vcodec": "h264"}
        assert _codec_create_kwargs({"vcodec": None}, "libx265") == {"vcodec": "hevc"}
        # AV1 and HW encoders are already codec names and pass through.
        assert _codec_create_kwargs({"vcodec": None}, "libsvtav1") == {"vcodec": "libsvtav1"}
        assert _codec_create_kwargs({"vcodec": None}, "h264_nvenc") == {"vcodec": "h264_nvenc"}

    def test_no_known_codec_surface_returns_empty(self):
        from strands_robots.dataset_recorder import _codec_create_kwargs

        # A signature exposing none of the known codec kwargs -> no routing
        # (recorder falls back to the LeRobot default codec).
        assert _codec_create_kwargs({"repo_id": None, "fps": None}, "h264") == {}

    def test_rgb_encoder_normalizes_ffmpeg_name_to_allowlist(self):
        pytest.importorskip("lerobot")
        rgb = pytest.importorskip("lerobot.configs.video")
        if not hasattr(rgb, "RGBEncoderConfig"):
            pytest.skip("this LeRobot has no RGBEncoderConfig surface")
        from strands_robots.dataset_recorder import _codec_create_kwargs

        # ffmpeg "libx264" must be normalized to the allowlist name "h264",
        # otherwise RGBEncoderConfig rejects it and the codec is dropped.
        out = _codec_create_kwargs({"rgb_encoder": None}, "libx264")
        assert set(out) == {"rgb_encoder"}
        assert out["rgb_encoder"].vcodec == "h264"
        # Canonical "h264" is already allowlist-valid and passes through.
        out2 = _codec_create_kwargs({"rgb_encoder": None}, "h264")
        assert out2["rgb_encoder"].vcodec == "h264"
        # AV1 opt-in survives onto the encoder config.
        out3 = _codec_create_kwargs({"rgb_encoder": None}, "libsvtav1")
        assert out3["rgb_encoder"].vcodec == "libsvtav1"

    def test_rgb_encoder_rejects_unknown_codec_loudly(self):
        pytest.importorskip("lerobot")
        rgb = pytest.importorskip("lerobot.configs.video")
        if not hasattr(rgb, "RGBEncoderConfig"):
            pytest.skip("this LeRobot has no RGBEncoderConfig surface")
        from strands_robots.dataset_recorder import _codec_create_kwargs

        # An unsupported codec must surface LeRobot's ValueError rather than
        # silently reverting to the default codec.
        with pytest.raises(ValueError):
            _codec_create_kwargs({"rgb_encoder": None}, "not_a_codec")

    def test_rgb_encoder_falls_back_when_config_class_missing(self, monkeypatch, caplog):
        """A LeRobot exposing ``rgb_encoder=`` but whose ``RGBEncoderConfig`` is
        unimportable must warn and route no codec surface (falling back to the
        default) rather than crash dataset setup.

        This mirrors the ``camera_encoder`` fallback contract: the two encoder-
        config surfaces are equivalent version-drift shims, so a build that
        renamed/moved ``RGBEncoderConfig`` should degrade identically to one that
        dropped ``VideoEncoderConfig`` - a warning, an empty routing dict, and no
        exception.
        """
        import sys

        from strands_robots.dataset_recorder import _codec_create_kwargs

        # Setting the import target to None makes
        # ``from lerobot.configs.video import RGBEncoderConfig`` raise
        # ImportError, exercising the fallback branch without needing a LeRobot
        # build that actually lacks the class.
        monkeypatch.setitem(sys.modules, "lerobot.configs.video", None)

        with caplog.at_level("WARNING"):
            out = _codec_create_kwargs({"rgb_encoder": None}, "libsvtav1", context="create")

        # No codec surface routed -> recorder uses the LeRobot default codec.
        assert out == {}
        # The warning names the missing config class so the cause is actionable.
        assert any("RGBEncoderConfig" in rec.message for rec in caplog.records)


def test_add_frame_fills_missing_action_key_with_zero():
    """An action key absent from the action dict contributes 0.0 at its slot.

    ``add_frame`` flattens the action into feature-schema order. A control step
    that does not populate every declared action key (e.g. a gripper channel the
    policy left unset) must still emit a full-width action vector with the
    missing slots zeroed - a short vector would silently misalign the recorded
    ``action`` column against its declared names.
    """
    ds = _CapturingDataset(_state_action_features(["j1"], ["a", "b", "c"]))
    rec = DatasetRecorder(dataset=ds)

    # "b" is absent from the action dict entirely; the schema still has 3 slots.
    rec.add_frame(observation={"j1": 1.0}, action={"a": 0.5, "c": 0.7}, task="t")

    frame = ds.frames[0]
    assert np.allclose(frame["action"], [0.5, 0.0, 0.7])
    assert frame["action"].dtype == np.float32


def test_finalize_swallows_dataset_error_and_still_closes(caplog):
    """A dataset ``finalize()`` that raises is logged, not propagated.

    Finalization runs at the end of a recording (often during teardown); a
    writer that has already released its parquet handle must not turn a
    best-effort flush into a crash. The recorder logs a warning, marks itself
    closed, and a subsequent ``finalize()`` is a clean no-op.
    """

    class _FinalizeRaises(_CapturingDataset):
        def finalize(self):
            raise RuntimeError("parquet writer already gone")

    ds = _FinalizeRaises(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    with caplog.at_level("WARNING"):
        rec.finalize()  # must not raise

    assert rec._closed is True
    assert any("finalize warning" in rec_.message for rec_ in caplog.records)

    # Idempotent after a failed finalize: the second call early-returns.
    rec.finalize()


def test_get_lerobot_dataset_class_raises_clean_importerror_when_unavailable(monkeypatch):
    """``_get_lerobot_dataset_class`` surfaces a clean, actionable ImportError.

    When lerobot's dataset module does not resolve and no test has injected a
    mock class, the resolver must raise an ImportError that names the install
    remedy rather than leaking the raw import failure to the caller.
    """
    import sys

    from strands_robots import dataset_recorder as dr

    # No test-injected mock class, and the real import target does not resolve.
    monkeypatch.setattr(dr, "LeRobotDataset", None, raising=False)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)

    with pytest.raises(ImportError, match="pip install lerobot"):
        dr._get_lerobot_dataset_class()


def test_read_dataset_episode_indices_dedups_and_skips_columnless_parquet(tmp_path):
    """Episode ground truth dedups repeated ``episode_index`` rows and skips a
    shard carrying no ``episode_index`` column.

    A finalized dataset can spread episodes over several parquet shards; a shard
    may repeat an episode row (first length wins) or be a metadata-only shard
    with no ``episode_index`` column at all. The reader must produce a single
    deduplicated, ordered episode list regardless.
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    from strands_robots.dataset_recorder import read_dataset_episode_indices

    ep_dir = tmp_path / "meta" / "episodes"
    (ep_dir / "chunk-000").mkdir(parents=True)
    (ep_dir / "chunk-001").mkdir(parents=True)

    # Shard 0 (sorts first): episode 0 appears twice - first length (3) wins.
    pq.write_table(
        pa.table({"episode_index": [0, 0, 1], "length": [3, 99, 5]}),
        ep_dir / "chunk-000" / "episodes_000.parquet",
    )
    # Shard 1: a metadata-only shard with no episode_index column is skipped.
    pq.write_table(
        pa.table({"some_other_stat": [1.0, 2.0]}),
        ep_dir / "chunk-001" / "episodes_001.parquet",
    )

    result = read_dataset_episode_indices(tmp_path)

    assert result["episode_indices"] == [0, 1]
    assert result["total_episodes"] == 2
    assert result["frames_per_episode"] == [3, 5]  # first-seen length for ep 0
    assert result["total_frames"] == 8
    assert result["info_total_episodes"] is None  # no meta/info.json written
