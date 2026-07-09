"""Newton dataset-recording correctness (start/stop/save_episode -> parquet).

Verifies that the Newton backend records a LeRobotDataset that satisfies the
canonical parquet-correctness contract: a session of N rollouts, each flushed
with ``save_episode``, must produce a dataset with ``total_episodes == N``,
exactly ``N`` rows in the episode metadata parquet, and ``N`` distinct
``episode_index`` values (no merged ``episode_index=0`` mega-episode).

The engine is exercised through a hand-built ``SimWorld`` so the recording
lifecycle runs without the optional Newton/Warp physics stack (the recorder and
episode bookkeeping are engine-independent). The per-step capture hook is the
real ``NewtonSimEngine._make_run_policy_hook`` closure.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("pyarrow")

from strands_robots.dataset_recorder import read_dataset_episode_indices
from strands_robots.simulation.models import SimCamera, SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _make_engine(world: SimWorld) -> NewtonSimEngine:
    """Build a NewtonSimEngine bound to ``world`` without the Warp stack.

    ``NewtonSimEngine.__init__`` imports Newton/Warp, which are optional; the
    recording lifecycle does not touch physics, so the engine is constructed
    via ``__new__`` and given just the attributes the recording path reads.
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


def _world_with_robot(name: str = "so100") -> SimWorld:
    world = SimWorld()
    world.robots[name] = SimRobot(
        name=name, urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
    )
    return world


def _drive_episode(engine: NewtonSimEngine, robot_name: str, instruction: str, n_frames: int) -> None:
    """Run one mock rollout: call the real on_frame hook ``n_frames`` times."""
    hook = engine._make_run_policy_hook(robot_name, instruction)
    assert hook is not None
    for step in range(n_frames):
        obs = {j: float(step) * 0.01 for j in _SO100_JOINTS}
        action = {j: float(step) * 0.01 + 0.001 for j in _SO100_JOINTS}
        hook(step, obs, action)


def test_three_episode_rollout_parquet_correctness(tmp_path):
    """3 rollouts -> 3 distinct episodes; parquet is the source of truth."""
    root = str(tmp_path / "newton_ds")
    engine = _make_engine(_world_with_robot())

    started = engine.start_recording(repo_id="local/newton_rec", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    n_episodes = 3
    frames_per_episode = 5
    for ep in range(n_episodes):
        _drive_episode(engine, "so100", f"episode {ep}", frames_per_episode)
        saved = engine.save_episode()
        assert saved["status"] == "success", saved

    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    # verify_dataset_episodes reads the parquet ground truth.
    verified = engine.verify_dataset_episodes(n_episodes)
    assert verified["status"] == "success", verified

    # Canonical correctness contract: total_episodes == N AND episode-parquet
    # num_rows == N AND len(unique(episode_index)) == N.
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == n_episodes
    assert len(set(info["episode_indices"])) == n_episodes
    assert sorted(info["episode_indices"]) == list(range(n_episodes))
    assert info["total_frames"] == n_episodes * frames_per_episode

    from pathlib import Path

    import pyarrow.parquet as pq

    parquets = sorted((Path(root) / "meta" / "episodes").glob("**/*.parquet"))
    num_rows = sum(pq.read_table(p).num_rows for p in parquets)
    assert num_rows == n_episodes


def test_stop_recording_flushes_trailing_episode(tmp_path):
    """stop_recording flushes the final unsaved rollout (no explicit save)."""
    root = str(tmp_path / "newton_trailing")
    engine = _make_engine(_world_with_robot())

    engine.start_recording(repo_id="local/newton_trail", root=root, fps=30, overwrite=True)
    _drive_episode(engine, "so100", "ep0", 4)
    engine.save_episode()
    _drive_episode(engine, "so100", "ep1", 4)
    # No save_episode here: stop_recording must flush the trailing episode.
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8


def test_camera_frames_recorded(tmp_path, monkeypatch):
    """A registered named camera is declared in the schema and captured."""
    root = str(tmp_path / "newton_cam")
    world = _world_with_robot()
    world.cameras["front"] = SimCamera(name="front", width=64, height=48)
    engine = _make_engine(world)

    # Stub render so the capture path runs without the Warp ray tracer.
    fake_img = np.zeros((48, 64, 3), dtype=np.uint8)

    def _fake_render(camera_name="default", width=None, height=None):
        return {
            "status": "success",
            "content": [{"image": {"format": "png", "source": {"bytes": _encode_png(fake_img)}}}],
        }

    monkeypatch.setattr(engine, "render", _fake_render)

    started = engine.start_recording(repo_id="local/newton_cam", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = world._backend_state["dataset_recorder"]
    assert "observation.images.front" in recorder.dataset.features

    _drive_episode(engine, "so100", "with camera", 4)
    engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 4


def test_multi_robot_namespaced_schema(tmp_path):
    """Two robots -> joint ids are namespaced ``robot__joint`` in the schema."""
    root = str(tmp_path / "newton_multi")
    world = SimWorld()
    for name in ("alice", "bob"):
        world.robots[name] = SimRobot(
            name=name, urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
        )
    engine = _make_engine(world)

    started = engine.start_recording(repo_id="local/newton_multi", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = world._backend_state["dataset_recorder"]
    state_names = recorder.dataset.features["observation.state"]["names"]
    assert "alice__Rotation" in state_names
    assert "bob__Jaw" in state_names
    assert len(state_names) == 2 * len(_SO100_JOINTS)

    # Drive both robots and confirm two episodes record cleanly.
    for ep, rname in enumerate(("alice", "bob")):
        _drive_episode(engine, rname, f"ep {ep}", 3)
        engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2


def test_start_recording_without_world_errors():
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = None
    engine._model = None
    result = engine.start_recording(repo_id="local/nope")
    assert result["status"] == "error"
    assert "No world" in result["content"][0]["text"]


def test_save_episode_without_recording_errors():
    engine = _make_engine(_world_with_robot())
    result = engine.save_episode()
    assert result["status"] == "error"
    assert "not recording" in result["content"][0]["text"].lower()


def test_stop_recording_when_idle_is_graceful():
    engine = _make_engine(_world_with_robot())
    result = engine.stop_recording()
    assert result["status"] == "success"
    assert "not recording" in result["content"][0]["text"].lower()


def _encode_png(img: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_start_recording_scopes_cameras_to_subset(tmp_path):
    """``cameras=`` records only the requested subset (parity with MuJoCo).

    Regression: ``run_policy(dataset_cameras=...)`` is a backend-agnostic tool
    that forwards ``start_recording(cameras=...)``. The Newton backend must
    accept the same scope the MuJoCo backend does; before it did not accept a
    ``cameras`` kwarg at all, so a scoped Newton rollout raised
    ``TypeError: start_recording() got an unexpected keyword argument 'cameras'``.
    """
    root = str(tmp_path / "newton_scope")
    world = _world_with_robot()
    world.cameras["front"] = SimCamera(name="front", width=64, height=48)
    world.cameras["top"] = SimCamera(name="top", width=64, height=48)
    engine = _make_engine(world)

    started = engine.start_recording(repo_id="local/newton_scope", root=root, fps=30, overwrite=True, cameras=["front"])
    assert started["status"] == "success", started

    recorder = world._backend_state["dataset_recorder"]
    image_feats = {k for k in recorder.dataset.features if k.startswith("observation.images")}
    assert image_feats == {"observation.images.front"}, image_feats

    # The on_frame hook renders only the scoped camera.
    scoped = [tpl[0] for tpl in world._backend_state["recording_cameras"]]
    assert scoped == ["front"], scoped


def test_start_recording_accepts_schema_safe_camera_name(tmp_path):
    """A namespaced camera can be scoped by its schema-safe ``__`` form."""
    root = str(tmp_path / "newton_safe")
    world = _world_with_robot()
    world.cameras["arm0/wrist"] = SimCamera(name="arm0/wrist", width=64, height=48)
    world.cameras["overview"] = SimCamera(name="overview", width=64, height=48)
    engine = _make_engine(world)

    started = engine.start_recording(
        repo_id="local/newton_safe", root=root, fps=30, overwrite=True, cameras=["arm0__wrist"]
    )
    assert started["status"] == "success", started

    recorder = world._backend_state["dataset_recorder"]
    image_feats = {k for k in recorder.dataset.features if k.startswith("observation.images")}
    assert image_feats == {"observation.images.arm0__wrist"}, image_feats


def test_start_recording_unknown_camera_errors(tmp_path):
    """An unknown ``cameras=`` name fails loudly and lists what exists."""
    root = str(tmp_path / "newton_unknown")
    world = _world_with_robot()
    world.cameras["front"] = SimCamera(name="front", width=64, height=48)
    engine = _make_engine(world)

    result = engine.start_recording(repo_id="local/newton_unknown", root=root, fps=30, overwrite=True, cameras=["nope"])
    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert "unknown camera(s)" in text
    assert "nope" in text
    assert "front" in text  # the available list is surfaced
    # A failed scope must not leave the session flagged as recording.
    assert world._backend_state.get("recording") is False


_BASE_COLS = [
    "base_pos.x",
    "base_pos.y",
    "base_pos.z",
    "base_quat.w",
    "base_quat.x",
    "base_quat.y",
    "base_quat.z",
    "base_lin_vel.x",
    "base_lin_vel.y",
    "base_lin_vel.z",
    "base_ang_vel.x",
    "base_ang_vel.y",
    "base_ang_vel.z",
]


def _recorder_state_names(engine: NewtonSimEngine) -> tuple[list[str], tuple[int, ...] | None]:
    world = engine._world
    assert world is not None
    rec = world._backend_state["dataset_recorder"]
    feat = rec.dataset.features.get("observation.state", {})
    names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
    shape = feat.get("shape") if isinstance(feat, dict) else getattr(feat, "shape", None)
    return list(names), shape


def test_start_recording_preserves_floating_base_state_schema(tmp_path, caplog):
    """A floating-base robot (``_robot_free_base_joint`` populated) records its
    full base kinematics as per-component observation.state columns
    (``base_pos.x`` .. ``base_ang_vel.z``), matching the MuJoCo backend, and no
    longer emits the legacy base-state-dropped warning."""
    import logging

    world = _world_with_robot("humanoid")
    engine = _make_engine(world)
    # NewtonSimEngine.__init__ records the base free joint per robot; the __new__
    # harness skips it, so set what a floating-base build would produce.
    engine._robot_free_base_joint = {"humanoid": "floating_base_joint"}

    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.recording"):
        started = engine.start_recording(
            repo_id="local/newton_fb_preserve", root=str(tmp_path / "ds"), fps=20, overwrite=True
        )
    assert started["status"] == "success", started
    names, shape = _recorder_state_names(engine)
    for col in _BASE_COLS:
        assert col in names, f"{col} missing from recorded observation.state schema"
    # The scalar joints are preserved alongside the base columns, and the flat
    # feature shape matches the per-element name count.
    for j in _SO100_JOINTS:
        assert j in names
    assert tuple(shape)[0] == len(names)
    # The base state is now preserved, so the legacy drop-warning is superseded.
    assert not any("have a floating base" in r.getMessage() for r in caplog.records), (
        "the base-drop warning must not fire now that base state is preserved"
    )
    engine.stop_recording()


def test_start_recording_fixed_arm_has_no_base_columns(tmp_path, caplog):
    """A fixed-base arm (no floating base recorded) gains NO base columns - the
    schema is unchanged - and detection degrades gracefully when
    ``_robot_free_base_joint`` is absent entirely (the __new__ harness leaves it
    unset)."""
    import logging

    engine = _make_engine(_world_with_robot("so100"))
    assert not hasattr(engine, "_robot_free_base_joint")  # attr absent -> getattr default

    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.recording"):
        started = engine.start_recording(
            repo_id="local/newton_arm_nobase", root=str(tmp_path / "ds"), fps=20, overwrite=True
        )
    assert started["status"] == "success", started
    names, _shape = _recorder_state_names(engine)
    assert not any(n.startswith(("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")) for n in names), (
        "fixed-base arm must NOT gain floating-base columns"
    )
    assert not any("have a floating base" in r.getMessage() for r in caplog.records)
    engine.stop_recording()


def test_recorded_floating_base_values_round_trip(tmp_path):
    """End-to-end: a known base pose + twist fed through the real on_frame hook
    is written to the dataset and read back after reopen - the base state is no
    longer silently dropped on the Newton backend."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    world = _world_with_robot("humanoid")
    engine = _make_engine(world)
    engine._robot_free_base_joint = {"humanoid": "floating_base_joint"}
    root = str(tmp_path / "ds")
    started = engine.start_recording(repo_id="local/newton_fb_roundtrip", root=root, fps=20, overwrite=True)
    assert started["status"] == "success", started

    known_pos = [0.11, -0.22, 0.83]  # world x, y, HEIGHT
    known_quat = [0.7071068, 0.0, 0.7071068, 0.0]  # +90deg about Y
    known_linvel = [0.41, -0.52, 0.63]
    known_angvel = [0.13, 0.24, 0.37]
    hook = engine._make_run_policy_hook("humanoid", "t")
    assert hook is not None
    for step in range(3):
        # A floating-base get_observation surfaces the base kinematics as vector
        # keys alongside the scalar joints; feed exactly that shape to the hook.
        obs: dict[str, object] = {j: float(step) * 0.01 for j in _SO100_JOINTS}
        obs["base_pos"] = list(known_pos)
        obs["base_quat"] = list(known_quat)
        obs["base_lin_vel"] = list(known_linvel)
        obs["base_ang_vel"] = list(known_angvel)
        action = {j: 0.0 for j in _SO100_JOINTS}
        hook(step, obs, action)
    engine.save_episode()
    engine.stop_recording()

    ds = LeRobotDataset("local/newton_fb_roundtrip", root=root)
    names = ds.features["observation.state"]["names"]
    idx = {n: i for i, n in enumerate(names)}
    state = np.asarray(ds[0]["observation.state"], dtype=np.float32)
    got_pos = [float(state[idx[f"base_pos.{c}"]]) for c in "xyz"]
    got_quat = [float(state[idx[f"base_quat.{c}"]]) for c in "wxyz"]
    got_linvel = [float(state[idx[f"base_lin_vel.{c}"]]) for c in "xyz"]
    got_angvel = [float(state[idx[f"base_ang_vel.{c}"]]) for c in "xyz"]
    assert np.allclose(got_pos, known_pos, atol=1e-3), f"base_pos not preserved: {got_pos}"
    assert np.allclose(got_quat, known_quat, atol=1e-3), f"base_quat not preserved: {got_quat}"
    assert np.allclose(got_linvel, known_linvel, atol=1e-3), f"base_lin_vel not preserved: {got_linvel}"
    assert np.allclose(got_angvel, known_angvel, atol=1e-3), f"base_ang_vel not preserved: {got_angvel}"
