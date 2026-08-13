"""Isaac ``run_multi_policy`` merged multi-robot recording (one frame per timestep).

Multi-robot recording parity for the Isaac synchronized loop (#2159, part of
the #2122 parity work; the control loop itself landed with #2158): while a
dataset recording session is active, every loop iteration emits exactly ONE
``add_frame`` containing all driven robots' namespaced state/action columns
(``alice__shoulder_pan`` ...) plus all camera images - so a 2-robot dataset has
both arms co-observed in every frame, mirroring the MuJoCo merged-frame
semantics pinned by
``tests/simulation/mujoco/test_recording_paths.py::test_b4_synchronized_multi_robot_recording``.

The engine is a skeleton ``IsaacSimulation`` built with ``__new__`` (the
established no-Kit pattern from ``test_dataset_recording.py``): physics is a
stub ``World``, articulations are stubs, cameras are ``_CameraState`` entries
with fake RGBA handles, ``isaacsim.core.utils.types`` is faked by the shared
``fake_isaacsim_types`` fixture - and the capture path is the REAL
``IsaacRecordingMixin`` schema declaration plus the REAL ``run_multi_policy``
loop writing into a REAL ``DatasetRecorder``. Round-trips reopen the on-disk
``LeRobotDataset`` (the recording round-trip rule). GPU/real-Kit coverage is a
separate ``tests_integ/`` follow-up.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.dataset_recorder import has_lerobot_dataset
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import (
    IsaacSimulation,
    _CameraState,
    _RobotState,
)
from tests.tool_result_contract import tool_json

from .test_backend_parity import fake_isaacsim_types  # noqa: F401 - fixture

_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow"]


class _StubArticulation:
    """Minimal articulation: accepts targets, reports zero joint positions."""

    def __init__(self, n_joints: int) -> None:
        self._n = n_joints
        self.applied: list[Any] = []

    def apply_action(self, action: Any) -> None:
        self.applied.append(action)

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(self._n, dtype=np.float32)


class _StubWorld:
    """Stub Isaac ``World`` counting physics steps."""

    def __init__(self) -> None:
        self.step_calls = 0

    def step(self, render: bool = False) -> None:  # noqa: ARG002 - signature parity
        self.step_calls += 1


class _FakeCameraHandle:
    """Stub RTX camera handle returning a fixed RGBA buffer."""

    def __init__(self, rgba: np.ndarray) -> None:
        self.rgba = rgba

    def get_rgba(self) -> np.ndarray:
        return self.rgba


def _robot(name: str) -> _RobotState:
    return _RobotState(
        name=name,
        prim_path=f"/World/Robots/{name}",
        joint_names=list(_JOINTS),
        data_config="so100",
        articulation=_StubArticulation(len(_JOINTS)),
    )


def _camera(name: str, width: int = 64, height: int = 48, fill: int = 7) -> _CameraState:
    cam = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=width, height=height)
    cam.handle = _FakeCameraHandle(np.full((height, width, 4), fill, dtype=np.uint8))
    return cam


def _make_engine(
    robots: dict[str, _RobotState],
    cameras: dict[str, _CameraState] | None = None,
    render_mode: str = "rtx_realtime",
) -> IsaacSimulation:
    """Skeleton IsaacSimulation (no Kit runtime), per the recording-test pattern."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig(render_mode=render_mode)
    engine._lock = threading.RLock()
    engine._world = _StubWorld()
    engine._world_created = True
    engine._robots = robots
    engine._cameras = cameras if cameras is not None else {}
    engine._objects = {}
    engine._prim_registry = []
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._action_controllers = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._replicated = False
    engine._num_envs_active = 1
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    engine._main_jobs = queue.Queue()
    return engine


@pytest.fixture
def sim_two_robots(fake_isaacsim_types) -> IsaacSimulation:  # noqa: F811, ARG001 - fixture injects the fake module
    """Two stub-articulation robots, headless (proprio-only recording)."""
    return _make_engine(
        {"alice": _robot("alice"), "bob": _robot("bob")},
        render_mode="headless",
    )


if not has_lerobot_dataset():
    pytest.skip("lerobot not installed", allow_module_level=True)

from pathlib import Path  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from strands_robots.dataset_recorder import read_dataset_episode_indices  # noqa: E402


def test_synchronized_two_robot_rollout_records_merged_frames(sim_two_robots, tmp_path) -> None:
    """The direct Isaac mirror of MuJoCo's B4 pin: a synchronized 2-robot
    ``run_multi_policy`` rollout records ONE merged frame per timestep, and
    BOTH robots' action columns are non-zero in EVERY frame.

    Pre-#2159 the loop refused to run inside an active recording session; the
    merged-frame path is the whole point of a multi-robot dataset (bimanual /
    multi-agent policy training needs both arms co-observed per frame).
    """
    sim = sim_two_robots
    root = str(tmp_path / "sync")
    n_steps = 8

    r = sim.start_recording(repo_id="local/isaac_sync_multi", fps=30, root=root, overwrite=True)
    assert r["status"] == "success", r

    r = sim.run_multi_policy(
        policies={"alice": MockPolicy(), "bob": MockPolicy()},
        instructions={"alice": "pour", "bob": "catch"},
        n_steps=n_steps,
        control_frequency=30.0,
    )
    assert r["status"] == "success", r
    assert tool_json(r)["steps"] == n_steps
    assert "(recorded)" in r["content"][0]["text"]
    stopped = sim.stop_recording()
    assert stopped["status"] == "success", stopped

    ds = LeRobotDataset(repo_id="local/isaac_sync_multi", root=root)

    # Merged schema: 2 robots x 3 joints = 6 prefixed action columns.
    af = ds.features["action"]
    names = af["names"] if isinstance(af, dict) else getattr(af, "names", None)
    assert names is not None
    assert [n for n in names if n.startswith("alice__")] == [f"alice__{j}" for j in _JOINTS]
    assert [n for n in names if n.startswith("bob__")] == [f"bob__{j}" for j in _JOINTS]

    # One merged frame per timestep: no interleaving (2N single-robot frames)
    # and no doubling.
    assert len(ds) == n_steps

    # Both robots' action columns are non-zero in EVERY frame (B4 mirror):
    # interleaved single-robot frames would leave the other half zero.
    half = len(names) // 2
    both = 0
    for i in range(len(ds)):
        ac = np.asarray(ds[i]["action"])
        a = float(np.abs(ac[:half]).sum())
        b = float(np.abs(ac[half:]).sum())
        if a > 1e-6 and b > 1e-6:
            both += 1
    assert both == len(ds) and len(ds) > 0, f"only {both}/{len(ds)} frames had both robots co-observed"

    # The state columns are merged and namespaced the same way.
    sf = ds.features["observation.state"]
    state_names = sf["names"] if isinstance(sf, dict) else getattr(sf, "names", None)
    assert state_names is not None
    assert "alice__shoulder_pan" in state_names
    assert "bob__elbow" in state_names
    assert len(state_names) == 2 * len(_JOINTS)


def test_round_trip_episode_and_frame_counts(sim_two_robots, tmp_path) -> None:
    """start_recording -> rollout -> save_episode -> rollout -> stop_recording
    -> reopen: episode/frame counts come from the on-disk parquet (round-trip
    rule), with frame count == timestep count per episode.
    """
    sim = sim_two_robots
    root = str(tmp_path / "roundtrip")

    r = sim.start_recording(repo_id="local/isaac_multi_rt", fps=30, root=root, overwrite=True)
    assert r["status"] == "success", r

    for _ep in range(2):
        r = sim.run_multi_policy(
            policies={"alice": MockPolicy(), "bob": MockPolicy()},
            instructions="handover",
            n_steps=5,
            control_frequency=30.0,
        )
        assert r["status"] == "success", r
        saved = sim.save_episode()
        assert saved["status"] == "success", saved

    stopped = sim.stop_recording()
    assert stopped["status"] == "success", stopped

    verified = sim.verify_dataset_episodes(2)
    assert verified["status"] == "success", verified

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2
    assert sorted(set(info["episode_indices"])) == [0, 1]
    assert info["total_frames"] == 2 * 5

    # Physics advanced exactly once per timestep across both rollouts.
    assert sim._world.step_calls == 2 * 5


def test_recorded_task_is_first_robots_instruction_with_shared_warning(sim_two_robots, tmp_path, caplog) -> None:
    """LeRobot stores ONE task per frame: with distinct per-robot instructions
    the frame records the FIRST robot's instruction, and the shared
    normalization helper emits the one-task-per-frame warning (same behavior
    and warning as MuJoCo - not duplicated by the Isaac loop).
    """
    import logging

    sim = sim_two_robots
    root = str(tmp_path / "task")

    r = sim.start_recording(repo_id="local/isaac_multi_task", fps=30, root=root, overwrite=True)
    assert r["status"] == "success", r
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.isaac.simulation"):
        r = sim.run_multi_policy(
            policies={"alice": MockPolicy(), "bob": MockPolicy()},
            instructions={"alice": "pour", "bob": "catch"},
            n_steps=3,
            control_frequency=30.0,
        )
    assert r["status"] == "success", r
    warnings = [rec.message for rec in caplog.records if "distinct per-robot instructions" in rec.message]
    assert len(warnings) == 1, "the shared helper warns exactly once (not re-warned per frame)"
    assert sim.stop_recording()["status"] == "success"

    ds = LeRobotDataset(repo_id="local/isaac_multi_task", root=root)
    # ``meta.tasks`` is indexed by the task strings themselves: the first
    # robot's instruction is the recorded task and the second's never appears.
    recorded_tasks = [str(t) for t in ds.meta.tasks.index]
    assert recorded_tasks == ["pour"], recorded_tasks


def test_cameras_recorded_once_per_step_into_the_merged_frame(fake_isaacsim_types, tmp_path) -> None:  # noqa: F811
    """A registered RTX camera lands ONE image per merged frame (scene-global,
    read from the first robot's observation), and the round-trip writes a real
    per-camera MP4 alongside the merged proprio columns.
    """
    sim = _make_engine(
        {"alice": _robot("alice"), "bob": _robot("bob")},
        cameras={"front": _camera("front")},
        render_mode="rtx_realtime",
    )
    root = str(tmp_path / "cams")

    r = sim.start_recording(repo_id="local/isaac_multi_cam", fps=30, root=root, overwrite=True)
    assert r["status"] == "success", r
    recorder = sim._recording_state_dict["dataset_recorder"]
    assert "observation.images.front" in recorder.dataset.features

    r = sim.run_multi_policy(
        policies={"alice": MockPolicy(), "bob": MockPolicy()},
        instructions="look",
        n_steps=4,
        control_frequency=30.0,
    )
    assert r["status"] == "success", r
    assert sim.stop_recording()["status"] == "success"

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 4
    mp4s = sorted(Path(root).glob("videos/**/*.mp4"))
    assert mp4s, "camera recording must land a per-camera MP4 on disk"
    assert all(p.stat().st_size > 0 for p in mp4s)


def test_failed_rollout_discards_partial_episode(sim_two_robots, tmp_path) -> None:
    """A mid-loop failure (empty action chunk) discards the partially-recorded
    frames, so the next episode starts at frame 0 rather than appending to a
    dangling half-episode (MuJoCo parity).
    """
    from strands_robots.policies.base import Policy

    class _EmptyAfter(Policy):
        """Returns one real chunk, then an empty one mid-rollout."""

        requires_images = False

        def __init__(self) -> None:
            self.calls = 0
            self._keys: list[str] = list(_JOINTS)

        def set_robot_state_keys(self, keys):
            self._keys = list(keys)

        @property
        def provider_name(self) -> str:
            return "empty_after"

        async def get_actions(self, obs, instruction=""):
            self.calls += 1
            if self.calls > 1:
                return []
            return [{k: 0.1 for k in self._keys}]

    sim = sim_two_robots
    root = str(tmp_path / "partial")
    r = sim.start_recording(repo_id="local/isaac_multi_partial", fps=30, root=root, overwrite=True)
    assert r["status"] == "success", r

    with pytest.raises(RuntimeError, match="empty action chunk"):
        sim.run_multi_policy(
            policies={"alice": _EmptyAfter(), "bob": _EmptyAfter()},
            instructions="doomed",
            n_steps=5,
            control_frequency=30.0,
            action_horizon=1,
        )
    # The dangling frame from the completed first step was discarded.
    recorder = sim._recording_state_dict["dataset_recorder"]
    assert getattr(recorder, "episode_frame_count", 0) == 0

    # A clean rollout afterwards records a well-formed single episode.
    r = sim.run_multi_policy(
        policies={"alice": MockPolicy(), "bob": MockPolicy()},
        instructions="recovered",
        n_steps=3,
        control_frequency=30.0,
    )
    assert r["status"] == "success", r
    assert sim.stop_recording()["status"] == "success"
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3
