"""Isaac dataset-recording correctness (start/stop/save_episode -> parquet).

Verifies the Isaac backend records a LeRobotDataset satisfying the canonical
parquet-correctness contract shared with the MuJoCo/Newton backends: a session
of N rollouts, each flushed with ``save_episode``, must produce a dataset with
``total_episodes == N``, N rows in the episode metadata parquet, and N
distinct ``episode_index`` values (no merged ``episode_index=0``
mega-episode) - round-trip verified by reopening the on-disk dataset.

The engine is exercised through a skeleton ``IsaacSimulation`` built with
``__new__`` (mirroring the Newton unit tests) so the recording lifecycle runs
without the Isaac Sim Kit runtime: the recorder and episode bookkeeping are
engine-independent, and the per-step capture hook is the REAL
``IsaacSimulation._make_run_policy_hook`` closure fed by observation dicts of
the exact shape ``IsaacSimulation.get_observation`` emits (joint scalars +
per-camera RGB ndarrays keyed by raw camera name). The multi-camera
render-product freshness contract and the recording-forces-images override are
pinned against the real ``get_observation`` with stub camera handles.

GPU integration coverage (real SimulationApp + RTX frames) lives in
``tests_integ/simulation/test_isaac_recording_gpu.py``.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import (
    IsaacSimulation,
    _CameraState,
    _RobotState,
)

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


class _FakeCameraHandle:
    """Stub RTX camera handle: ``get_rgba()`` returns a settable RGBA buffer."""

    def __init__(self, rgba: np.ndarray) -> None:
        self.rgba = rgba

    def get_rgba(self) -> np.ndarray:
        return self.rgba


class _StubWorld:
    """Stub Isaac ``World`` counting render ticks from the product refresh."""

    def __init__(self) -> None:
        self.render_steps = 0

    def step(self, render: bool = False) -> None:
        if render:
            self.render_steps += 1


def _make_engine(
    robots: dict[str, _RobotState] | None = None,
    cameras: dict[str, _CameraState] | None = None,
    render_mode: str = "rtx_realtime",
) -> IsaacSimulation:
    """Build a skeleton IsaacSimulation without booting the Isaac Kit runtime.

    ``IsaacSimulation.__init__`` is CPU-safe but seeds pump/env-var state this
    test does not need; ``__new__`` + the exact attributes the recording path
    reads keeps the fixture honest about what the mixin depends on.
    """
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig(render_mode=render_mode)
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = robots if robots is not None else {}
    engine._cameras = cameras if cameras is not None else {}
    engine._objects = {}
    engine._prim_registry = []
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._action_controllers = {}  # cleared by destroy() (#1812)
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._replicated = False
    engine._num_envs_active = 1
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    return engine


def _robot(name: str = "so100", joints: list[str] | None = None) -> _RobotState:
    return _RobotState(
        name=name,
        prim_path=f"/World/Robots/{name}",
        joint_names=list(joints or _SO100_JOINTS),
        data_config="so100",
    )


def _camera(name: str, width: int = 64, height: int = 48, fill: int = 0) -> _CameraState:
    cam = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=width, height=height)
    rgba = np.full((height, width, 4), fill, dtype=np.uint8)
    cam.handle = _FakeCameraHandle(rgba)
    return cam


def _drive_episode(engine: IsaacSimulation, robot_name: str, instruction: str, n_frames: int) -> None:
    """Run one mock rollout: call the real on_frame hook ``n_frames`` times.

    The observation dict mirrors what ``IsaacSimulation.get_observation``
    emits: joint scalars keyed by joint name plus one RGB ndarray per camera
    keyed by the RAW camera name.
    """
    hook = engine._make_run_policy_hook(robot_name, instruction)
    assert hook is not None
    joints = engine._robots[robot_name].joint_names
    for step in range(n_frames):
        obs: dict = {j: float(step) * 0.01 for j in joints}
        for cam_name, cam in engine._cameras.items():
            obs[cam_name] = np.asarray(cam.handle.get_rgba())[..., :3].astype(np.uint8)
        action = {j: float(step) * 0.01 + 0.001 for j in joints}
        hook(step, obs, action)


# --- guards (no lerobot required) -------------------------------------------


def test_start_recording_without_world_errors() -> None:
    engine = _make_engine(robots={"so100": _robot()})
    engine._world_created = False
    result = engine.start_recording(repo_id="local/nope")
    assert result["status"] == "error"
    assert "No world" in result["content"][0]["text"]


def test_start_recording_without_robots_errors() -> None:
    result = _make_engine().start_recording(repo_id="local/nope")
    assert result["status"] == "error"
    assert "add_robot" in result["content"][0]["text"]


def test_save_episode_without_recording_errors() -> None:
    result = _make_engine(robots={"so100": _robot()}).save_episode()
    assert result["status"] == "error"
    assert "not recording" in result["content"][0]["text"].lower()


def test_stop_recording_when_idle_is_graceful() -> None:
    result = _make_engine(robots={"so100": _robot()}).stop_recording()
    assert result["status"] == "success"
    assert "not recording" in result["content"][0]["text"].lower()


def test_hook_is_none_for_unknown_robot() -> None:
    engine = _make_engine(robots={"so100": _robot()})
    assert engine._make_run_policy_hook("typo", "task") is None


def test_recording_state_seam_is_engine_owned_dict() -> None:
    """The Isaac state seam returns the engine dict, never a SimWorld field.

    Pins the DatasetRecordingMixin seam override: Isaac's ``self._world`` is
    the Isaac Sim World handle (here ``None``), so the shared lifecycle must
    read ``self._recording_state_dict`` and report no-world before
    ``create_world``.
    """
    engine = _make_engine(robots={"so100": _robot()})
    assert engine._recording_state() is engine._recording_state_dict
    assert engine._is_recording() is False
    engine._recording_state_dict["recording"] = True
    assert engine._is_recording() is True
    engine._world_created = False
    assert engine._recording_state() is None
    assert engine._is_recording() is False


def test_destroy_resets_recording_state() -> None:
    """destroy() drops any in-flight recorder so no stale session survives.

    A recorder left in the seam dict across destroy()/create_world() would
    reference RTX frames from a torn-down stage; the next session must start
    from a clean dict (mirroring MuJoCo/Newton, whose state dict dies with
    the SimWorld).
    """
    engine = _make_engine(robots={"so100": _robot()})
    engine._recording_state_dict.update({"recording": True, "dataset_recorder": object()})
    destroyed = engine.destroy()
    assert destroyed["status"] == "success"
    assert engine._recording_state() is None  # world gone
    engine._world_created = True  # simulate the next create_world()
    assert engine._recording_state() == {}
    assert engine._is_recording() is False


def test_get_observation_forces_images_while_recording() -> None:
    """skip_images=True still yields camera frames during a recording session.

    Parity with MuJoCo/Newton: a non-image policy (requires_images=False, e.g.
    MockPolicy) makes PolicyRunner pass skip_images=True, but the recorded
    frames must carry the camera images the schema declared.
    """
    engine = _make_engine(
        robots={"so100": _robot()},
        cameras={"front": _camera("front", fill=7)},
    )
    obs = engine.get_observation("so100", skip_images=True)
    assert "front" not in obs  # not recording: skip honoured

    engine._recording_state_dict["recording"] = True
    obs = engine.get_observation("so100", skip_images=True)
    assert "front" in obs, "recording must force camera frames into the observation"
    assert obs["front"].shape == (48, 64, 3)


def test_get_observation_refreshes_render_products_for_multi_camera() -> None:
    """>1 camera triggers the render-product refresh before frame read-back.

    Regression pin for the stale-render-product wrinkle: without the refresh a
    second camera's ``get_rgba`` returns a stale buffer and multi-cam
    recordings duplicate frames. One camera must NOT pay the refresh cost.
    """
    world = _StubWorld()
    engine = _make_engine(
        robots={"so100": _robot()},
        cameras={"front": _camera("front", fill=1), "top": _camera("top", fill=2)},
    )
    engine._world = world
    obs = engine.get_observation("so100")
    assert world.render_steps >= 1, "multi-camera observation must refresh the render products"
    assert not np.array_equal(obs["front"], obs["top"]), "each camera must contribute its own frame"

    single = _make_engine(robots={"so100": _robot()}, cameras={"front": _camera("front")})
    single_world = _StubWorld()
    single._world = single_world
    single.get_observation("so100")
    assert single_world.render_steps == 0, "single-camera observation must skip the refresh"


def test_describe_advertises_recording_family() -> None:
    """describe() exposes the record-and-stream workflow, like MuJoCo/Newton.

    An agent enumerating ``describe()["methods"]`` must discover
    ``start_recording`` -> ``save_episode`` -> ``stop_recording`` ->
    ``stream_dataset`` (and the parquet-truth gate) without guessing names,
    and every advertised name must resolve to a real callable.
    """
    import inspect

    sim = IsaacSimulation()  # CPU-safe: heavy omni imports are deferred
    desc = sim.describe()
    assert desc["backend"] == "isaac"
    methods = desc["methods"]
    for name in (
        "start_recording",
        "save_episode",
        "stop_recording",
        "get_recording_status",
        "stream_dataset",
        "verify_dataset_episodes",
        "start_cameras_recording",
        "stop_cameras_recording",
        "add_camera",
        "remove_camera",
    ):
        assert name in methods, f"describe() omits recording-family method {name!r}"
        assert callable(getattr(sim, name, None)), f"{name!r} advertised but not callable"
    # The advertised start_recording signature names the real parameters so a
    # caller (or run_policy's dataset_cameras= forwarding) can invoke it.
    sig = set(inspect.signature(sim.start_recording).parameters)
    for param in ("repo_id", "task", "fps", "root", "vcodec", "overwrite", "cameras"):
        assert param in methods["start_recording"]
        assert param in sig


# --- LeRobotDataset round-trips (need the lerobot extra) --------------------

lerobot = pytest.importorskip("lerobot")
pq = pytest.importorskip("pyarrow.parquet")

from pathlib import Path  # noqa: E402

from strands_robots.dataset_recorder import read_dataset_episode_indices  # noqa: E402


def test_three_episode_rollout_parquet_correctness(tmp_path) -> None:
    """3 rollouts -> 3 distinct episodes; the reopened parquet is the truth."""
    root = str(tmp_path / "isaac_ds")
    engine = _make_engine(robots={"so100": _robot()})

    started = engine.start_recording(repo_id="local/isaac_rec", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    n_episodes = 3
    frames_per_episode = 5
    for ep in range(n_episodes):
        _drive_episode(engine, "so100", f"episode {ep}", frames_per_episode)
        saved = engine.save_episode()
        assert saved["status"] == "success", saved

    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    verified = engine.verify_dataset_episodes(n_episodes)
    assert verified["status"] == "success", verified

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == n_episodes
    assert sorted(set(info["episode_indices"])) == list(range(n_episodes))
    assert info["total_frames"] == n_episodes * frames_per_episode

    parquets = sorted((Path(root) / "meta" / "episodes").glob("**/*.parquet"))
    num_rows = sum(pq.read_table(p).num_rows for p in parquets)
    assert num_rows == n_episodes


def test_stop_recording_flushes_trailing_episode(tmp_path) -> None:
    """stop_recording flushes the final unsaved rollout (no explicit save)."""
    root = str(tmp_path / "isaac_trailing")
    engine = _make_engine(robots={"so100": _robot()})

    engine.start_recording(repo_id="local/isaac_trail", root=root, fps=30, overwrite=True)
    _drive_episode(engine, "so100", "ep0", 4)
    engine.save_episode()
    _drive_episode(engine, "so100", "ep1", 4)
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8


def test_camera_declared_at_probed_resolution_and_recorded(tmp_path) -> None:
    """The schema declares each camera at the resolution the probe observed.

    RTX cameras render at a DLSS-safe NATIVE size that can differ from the
    requested output size; ``get_observation`` emits the native frame. The
    schema must match that stream or every ``add_frame`` is rejected. Here the
    ``_CameraState`` claims 640x480 but the probe frame is 64x48 - the probe
    must win, and the round-trip must land a real MP4 on disk.
    """
    root = str(tmp_path / "isaac_cam")
    cam = _CameraState(name="front", prim_path="/World/Cameras/front", width=640, height=480)
    cam.handle = _FakeCameraHandle(np.full((48, 64, 4), 9, dtype=np.uint8))
    engine = _make_engine(robots={"so100": _robot()}, cameras={"front": cam})

    started = engine.start_recording(repo_id="local/isaac_cam", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = engine._recording_state_dict["dataset_recorder"]
    feat = recorder.dataset.features["observation.images.front"]
    shape = tuple(feat["shape"]) if isinstance(feat, dict) else tuple(feat.shape)
    assert 48 in shape and 64 in shape, f"probed 64x48 must define the schema, got {shape}"

    _drive_episode(engine, "so100", "with camera", 4)
    engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 4
    mp4s = sorted(Path(root).glob("videos/**/*.mp4"))
    assert mp4s, "camera recording must land per-camera MP4 files on disk"
    assert all(p.stat().st_size > 0 for p in mp4s)


def test_two_cameras_record_two_video_columns(tmp_path) -> None:
    """Both cameras land distinct feature columns and MP4 streams."""
    root = str(tmp_path / "isaac_two_cams")
    engine = _make_engine(
        robots={"so100": _robot()},
        cameras={"front": _camera("front", fill=10), "top": _camera("top", fill=200)},
    )

    started = engine.start_recording(repo_id="local/isaac_two", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = engine._recording_state_dict["dataset_recorder"]
    image_feats = {k for k in recorder.dataset.features if k.startswith("observation.images.")}
    assert image_feats == {"observation.images.front", "observation.images.top"}

    _drive_episode(engine, "so100", "two cams", 3)
    engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    mp4s = sorted(Path(root).glob("videos/**/*.mp4"))
    assert mp4s, "camera recording must land two per-camera MP4 streams on disk"
    keys = {part for p in mp4s for part in p.parts if part.startswith("observation.images.")}
    assert keys == {"observation.images.front", "observation.images.top"}, keys


def test_start_recording_scopes_cameras_to_subset(tmp_path) -> None:
    """``cameras=`` records only the requested subset (parity with MuJoCo/Newton)."""
    root = str(tmp_path / "isaac_scope")
    engine = _make_engine(
        robots={"so100": _robot()},
        cameras={"front": _camera("front"), "top": _camera("top")},
    )

    started = engine.start_recording(repo_id="local/isaac_scope", root=root, fps=30, overwrite=True, cameras=["front"])
    assert started["status"] == "success", started

    recorder = engine._recording_state_dict["dataset_recorder"]
    image_feats = {k for k in recorder.dataset.features if k.startswith("observation.images.")}
    assert image_feats == {"observation.images.front"}, image_feats
    scoped = [tpl[0] for tpl in engine._recording_state_dict["recording_cameras"]]
    assert scoped == ["front"], scoped


def test_start_recording_unknown_camera_fails_loudly(tmp_path) -> None:
    root = str(tmp_path / "isaac_unknown_cam")
    engine = _make_engine(robots={"so100": _robot()}, cameras={"front": _camera("front")})
    result = engine.start_recording(repo_id="local/isaac_bad", root=root, overwrite=True, cameras=["wrist"])
    assert result["status"] == "error"
    assert "wrist" in result["content"][0]["text"]
    assert "front" in result["content"][0]["text"]
    assert engine._is_recording() is False


def test_headless_render_mode_records_proprio_only(tmp_path, caplog) -> None:
    """render_mode='headless' cannot produce RTX frames: proprio-only + warning."""
    import logging

    root = str(tmp_path / "isaac_headless")
    engine = _make_engine(
        robots={"so100": _robot()},
        cameras={"front": _camera("front")},
        render_mode="headless",
    )
    with caplog.at_level(logging.WARNING):
        started = engine.start_recording(repo_id="local/isaac_headless", root=root, overwrite=True)
    assert started["status"] == "success", started
    assert any("headless" in rec.message for rec in caplog.records)

    recorder = engine._recording_state_dict["dataset_recorder"]
    image_feats = {k for k in recorder.dataset.features if k.startswith("observation.images.")}
    assert image_feats == set(), "headless render mode must not declare camera columns"

    _drive_episode(engine, "so100", "proprio", 3)
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3


def test_multi_robot_namespaced_schema(tmp_path) -> None:
    """Two robots -> joint/action ids are namespaced ``robot__joint``."""
    root = str(tmp_path / "isaac_multi")
    engine = _make_engine(robots={"alice": _robot("alice"), "bob": _robot("bob")})

    started = engine.start_recording(repo_id="local/isaac_multi", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = engine._recording_state_dict["dataset_recorder"]
    state_names = recorder.dataset.features["observation.state"]["names"]
    assert "alice__Rotation" in state_names
    assert "bob__Jaw" in state_names
    assert len(state_names) == 2 * len(_SO100_JOINTS)
    action_names = recorder.dataset.features["action"]["names"]
    assert "alice__Rotation" in action_names
    assert "bob__Jaw" in action_names

    for ep, rname in enumerate(("alice", "bob")):
        _drive_episode(engine, rname, f"ep {ep}", 3)
        engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2
