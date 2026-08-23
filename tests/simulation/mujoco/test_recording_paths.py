"""Targeted coverage for ``RecordingMixin`` (LeRobotDataset recorder).

Covers:
* ``start_recording`` with no world → graceful error
* ``stop_recording`` with no active recording → graceful error
* ``get_recording_status`` with/without active session
* start_recording twice → second call does NOT crash (overwrite path)
* HF-cache repo_id path (repo_id with '/' and no local root)
* Multi-robot namespace prefix for joint names
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from tests.tool_result_contract import tool_json  # noqa: E402

# Inline MJCF XML to avoid network-dependent so101 model downloads.
_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_with_two_robots():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("alpha", urdf_path=path, position=[-0.2, 0, 0])
    s.add_robot("beta", urdf_path=path, position=[0.2, 0, 0])
    s.step(5)
    yield s
    s.destroy()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_start_recording_no_world_returns_graceful_error(tmp_path):
    from strands_robots.simulation import Simulation

    s = Simulation()
    r = s.start_recording(repo_id="local/nope", root=str(tmp_path / "dataset"), task="t")
    assert r["status"] == "error"
    assert "No world" in r["content"][0]["text"]
    s.destroy()


def test_stop_recording_without_start_is_idempotent(sim_with_two_robots):
    """T16: idempotent - success with 'Was not recording' message."""
    r = sim_with_two_robots.stop_recording()
    assert r["status"] == "success"
    assert "Was not recording" in r["content"][0]["text"]


def test_get_recording_status_shows_active_and_idle(sim_with_two_robots, tmp_path):
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots

    # Idle before any start
    r = sim.get_recording_status()
    assert r["status"] == "success"

    # Start → active
    r = sim.start_recording(repo_id="local/status_probe", fps=20, root=str(tmp_path), overwrite=True)
    assert r["status"] == "success"

    r = sim.get_recording_status()
    assert r["status"] == "success"

    # Stop → idle again
    sim.stop_recording()
    r = sim.get_recording_status()
    assert r["status"] == "success"


def test_start_recording_overwrite_wipes_existing_dir(sim_with_two_robots, tmp_path):
    """The ``overwrite=True`` flag removes any pre-existing dataset dir
    before re-creating it (covers the ``shutil.rmtree`` branch)."""
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    # Pre-create some junk in the target dir
    junk = tmp_path / "stale.txt"
    junk.write_text("stale")
    assert junk.exists()

    r = sim_with_two_robots.start_recording(
        repo_id="local/overwrite_probe",
        fps=20,
        root=str(tmp_path),
        overwrite=True,
    )
    assert r["status"] == "success"
    # The junk should be gone (dir was wiped)
    assert not junk.exists()

    sim_with_two_robots.stop_recording()


def test_start_recording_namespaced_joint_prefix_with_two_robots(sim_with_two_robots, tmp_path):
    """With >1 robot, joint_names are prefixed with the robot's instance name."""
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    r = sim_with_two_robots.start_recording(repo_id="local/namespace_probe", fps=20, root=str(tmp_path), overwrite=True)
    assert r["status"] == "success"

    from strands_robots.policies.mock import MockPolicy

    p = MockPolicy()
    p.set_robot_state_keys(sim_with_two_robots.robot_joint_names("alpha"))
    r = sim_with_two_robots.run_policy("alpha", policy_object=p, duration=0.2, control_frequency=20.0)
    assert r["status"] == "success"

    sim_with_two_robots.stop_recording()

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    joint_names = info["features"]["observation.state"]["names"]
    # Unique joint names - the fix we pushed
    assert len(joint_names) == len(set(joint_names)), f"dup names: {joint_names}"
    # Both robots prefixed
    assert any(jn.startswith("alpha__") for jn in joint_names)
    assert any(jn.startswith("beta__") for jn in joint_names)


def test_b12_multi_episode_resume_appends(sim_with_two_robots, tmp_path):
    """B12 regression: start_recording(overwrite=False) on an EXISTING dataset
    must RESUME (append) instead of crashing with FileExistsError.

    Pre-fix, the 2nd start_recording raised
    ``Dataset init failed: [Errno 17] File exists`` because it always called
    LeRobotDataset.create() (mkdir exist_ok=False). The fix routes to
    DatasetRecorder.resume() when a dataset already exists on disk.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots
    root = str(tmp_path / "multiep")

    # Episode 1 (fresh)
    r = sim.start_recording(repo_id="local/multiep", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r
    sim.run_policy(
        robot_name="alpha",
        policy_provider="mock",
        instruction="ep0",
        duration=0.3,
        control_frequency=20.0,
        fast_mode=True,
    )
    r = sim.stop_recording()
    assert r["status"] == "success", r

    # Episode 2 (append - overwrite=False on existing dir must NOT crash)
    r = sim.start_recording(repo_id="local/multiep", fps=20, root=root, overwrite=False)
    assert r["status"] == "success", f"B12 regression - resume failed: {r}"
    sim.run_policy(
        robot_name="alpha",
        policy_provider="mock",
        instruction="ep1",
        duration=0.3,
        control_frequency=20.0,
        fast_mode=True,
    )
    r = sim.stop_recording()
    assert r["status"] == "success", r

    # Readback: two episodes appended into one dataset.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/multiep", root=root)
    assert ds.meta.total_episodes == 2, f"expected 2 episodes, got {ds.meta.total_episodes}"


def test_resume_at_a_different_fps_is_refused_and_leaves_the_dataset_intact(sim_with_two_robots, tmp_path):
    """Appending to an existing dataset at a different ``fps`` is rejected.

    A resumed dataset keeps the rate it was created at (``LeRobotDataset.resume``
    takes no ``fps``), so a differing request cannot be honored. Pre-fix the
    second ``start_recording`` returned ``status="success"`` and reported the
    requested rate while every appended frame was timestamped at the on-disk one
    - two episodes captured at different cadences became indistinguishable on
    disk. The refusal must also leave the existing dataset untouched.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots
    root = str(tmp_path / "fpsmix")

    r = sim.start_recording(repo_id="local/fpsmix", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r
    sim.run_policy(
        robot_name="alpha",
        policy_provider="mock",
        instruction="ep0",
        duration=0.3,
        control_frequency=20.0,
        fast_mode=True,
    )
    assert sim.stop_recording()["status"] == "success"

    r = sim.start_recording(repo_id="local/fpsmix", fps=40, root=root, overwrite=False)
    assert r["status"] == "error", f"resume at a different fps must be refused: {r}"
    text = r["content"][0]["text"]
    assert "fps" in text and "on-disk=20" in text and "requested=40" in text, text

    # The dataset the caller already recorded is intact and still at its rate.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/fpsmix", root=root)
    assert ds.meta.fps == 20
    assert ds.meta.total_episodes == 1

    # The same rate still appends, so the guard only rejects the unhonorable case.
    r = sim.start_recording(repo_id="local/fpsmix", fps=20, root=root, overwrite=False)
    assert r["status"] == "success", r
    sim.run_policy(
        robot_name="alpha",
        policy_provider="mock",
        instruction="ep1",
        duration=0.3,
        control_frequency=20.0,
        fast_mode=True,
    )
    assert sim.stop_recording()["status"] == "success"
    assert LeRobotDataset(repo_id="local/fpsmix", root=root).meta.total_episodes == 2


def test_b4_synchronized_multi_robot_recording(sim_with_two_robots, tmp_path):
    """B4 fix: run_multi_policy drives BOTH robots in one synchronized loop and
    records them into ONE merged frame per timestep - so every frame co-observes
    both arms (the whole point of a multi-robot dataset).

    Pre-fix, two independent start_policy threads shared the recorder and
    interleaved single-robot frames (each frame had only one robot's state, the
    other's columns zero). Here we assert BOTH robots' action columns are
    non-zero in EVERY frame.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    import numpy as np

    from strands_robots.policies import create_policy

    sim = sim_with_two_robots
    root = str(tmp_path / "sync")
    r = sim.start_recording(repo_id="local/sync_multi", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r

    pols = {"alpha": create_policy("mock"), "beta": create_policy("mock")}
    r = sim.run_multi_policy(
        policies=pols, instructions={"alpha": "a", "beta": "b"}, duration=0.5, control_frequency=20.0
    )
    assert r["status"] == "success", r
    assert tool_json(r)["steps"] > 0
    sim.stop_recording()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/sync_multi", root=root)

    # Schema: 2 robots × 3 joints (the inline test arm) = 6-dim, prefixed.
    af = ds.features["action"]
    names = af["names"] if isinstance(af, dict) else getattr(af, "names", None)
    assert names is not None
    assert any(n.startswith("alpha__") for n in names)
    assert any(n.startswith("beta__") for n in names)

    half = len(names) // 2
    both = 0
    for i in range(len(ds)):
        ac = np.asarray(ds[i]["action"])
        a = float(np.abs(ac[:half]).sum())
        b = float(np.abs(ac[half:]).sum())
        if a > 1e-6 and b > 1e-6:
            both += 1
    # Every frame must co-observe both robots (synchronized recording).
    assert both == len(ds) and len(ds) > 0, f"B4: only {both}/{len(ds)} frames had both robots co-observed"


def test_multi_robot_recording_action_columns_keyed_by_actuators(sim_with_two_robots, tmp_path):
    """Recorded action columns are keyed by each robot's ACTUATORS, not its
    joints, so every frame carries both arms' commanded actions.

    The rollout loops drive policies with ``robot_action_keys`` (actuator
    short-names), which on this arm (joints ``shoulder_pan`` vs actuators
    ``shoulder_pan_act``) differ from the joint names. If the recorder declared
    its action schema from the joint names, ``add_frame`` could not match the
    actuator-keyed action dict and would write all-zero action columns. This
    asserts the action schema follows the actuator keys and that both robots'
    action columns are non-zero in every frame.

    Reads the action column straight from the on-disk parquet so the assertion
    does not depend on a working video decoder (the image features are still
    encoded; only ``action`` is inspected here).
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    pq = pytest.importorskip("pyarrow.parquet")
    import glob

    import numpy as np

    from strands_robots.policies import create_policy

    sim = sim_with_two_robots
    root = str(tmp_path / "act_keys")
    r = sim.start_recording(repo_id="local/act_keys", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r

    # The declared action schema follows the actuator keys (namespaced), not
    # the joint names.
    recorder = sim._world._backend_state["dataset_recorder"]
    feat = recorder.dataset.features.get("action")
    names = feat["names"] if isinstance(feat, dict) else getattr(feat, "names", None)
    assert names == [
        "alpha__shoulder_pan_act",
        "alpha__shoulder_lift_act",
        "alpha__elbow_act",
        "beta__shoulder_pan_act",
        "beta__shoulder_lift_act",
        "beta__elbow_act",
    ], names

    pols = {"alpha": create_policy("mock"), "beta": create_policy("mock")}
    r = sim.run_multi_policy(
        policies=pols, instructions={"alpha": "a", "beta": "b"}, duration=0.5, control_frequency=20.0
    )
    assert r["status"] == "success", r
    assert tool_json(r)["steps"] > 0
    sim.stop_recording()

    # Read the action column directly from parquet (no video decode needed).
    parquets = [p for p in glob.glob(f"{root}/**/*.parquet", recursive=True) if f"{os.sep}data{os.sep}" in p]
    parquets = parquets or glob.glob(f"{root}/**/*.parquet", recursive=True)
    assert parquets, "no parquet data files written"
    actions = np.asarray(pq.read_table(parquets).column("action").to_pylist())
    assert actions.shape[0] > 0 and actions.shape[1] == 6, actions.shape

    half = actions.shape[1] // 2
    both = sum(
        1 for row in actions if float(np.abs(row[:half]).sum()) > 1e-6 and float(np.abs(row[half:]).sum()) > 1e-6
    )
    assert both == actions.shape[0], f"only {both}/{actions.shape[0]} frames had both robots' actions recorded"


def test_run_multi_policy_validates_robots(sim_with_two_robots):
    """run_multi_policy rejects unknown robots and empty policy maps."""
    from strands_robots.policies import create_policy

    sim = sim_with_two_robots
    assert sim.run_multi_policy(policies={})["status"] == "error"
    r = sim.run_multi_policy(policies={"ghost": create_policy("mock")}, duration=0.1)
    assert r["status"] == "error"
    assert "not found" in r["content"][0]["text"].lower()


def test_run_multi_policy_action_horizon_batches_inference(sim_with_two_robots, tmp_path):
    """run_multi_policy re-queries a policy ONLY when its action queue drains,
    so an N-action chunk with action_horizon=N runs inference once per N steps
    (not every step). Critical for expensive VLAs. Also verifies per-robot
    horizon mapping and that batching doesn't break the synchronized-frame
    invariant (both robots still co-observed every frame).
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    import numpy as np

    from strands_robots.policies.base import Policy

    class _ChunkCounter(Policy):
        requires_images = False

        def __init__(self, chunk=10):
            self.calls = 0
            self.chunk = chunk
            self._keys = None

        def set_robot_state_keys(self, keys):
            self._keys = list(keys)

        @property
        def provider_name(self):
            return "chunk_counter"

        async def get_actions(self, obs, instruction=""):
            self.calls += 1
            keys = self._keys or ["shoulder_pan", "shoulder_lift", "elbow"]
            return [{k: 0.05 * (j + 1) for k in keys} for j in range(self.chunk)]

    sim = sim_with_two_robots

    # horizon=10 over 20 steps → ceil(20/10) = 2 inference calls per robot.
    r = sim.start_recording(repo_id="local/hz", fps=20, root=str(tmp_path / "hz"), overwrite=True)
    assert r["status"] == "success", r
    pa, pb = _ChunkCounter(chunk=10), _ChunkCounter(chunk=10)
    r = sim.run_multi_policy(policies={"alpha": pa, "beta": pb}, n_steps=20, control_frequency=20.0, action_horizon=10)
    assert r["status"] == "success", r
    assert pa.calls == 2, f"expected 2 inference calls (20 steps / horizon 10), got {pa.calls}"
    assert pb.calls == 2, f"expected 2, got {pb.calls}"
    sim.stop_recording()

    # Per-robot horizon: alpha closed-loop (every step), beta batched.
    r = sim.start_recording(repo_id="local/hz2", fps=20, root=str(tmp_path / "hz2"), overwrite=True)
    assert r["status"] == "success", r
    pa2, pb2 = _ChunkCounter(chunk=10), _ChunkCounter(chunk=10)
    r = sim.run_multi_policy(
        policies={"alpha": pa2, "beta": pb2},
        n_steps=20,
        control_frequency=20.0,
        action_horizon={"alpha": 1, "beta": 10},
    )
    assert r["status"] == "success", r
    assert pa2.calls == 20, f"alpha horizon=1 → every step, expected 20, got {pa2.calls}"
    assert pb2.calls == 2, f"beta horizon=10 → expected 2, got {pb2.calls}"
    sim.stop_recording()

    # Batching must NOT break the co-observation invariant.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/hz2", root=str(tmp_path / "hz2"))
    af = ds.features["action"]
    names = af["names"] if isinstance(af, dict) else getattr(af, "names", None)
    half = len(names) // 2
    for i in range(len(ds)):
        ac = np.asarray(ds[i]["action"])
        assert float(np.abs(ac[:half]).sum()) > 1e-6 and float(np.abs(ac[half:]).sum()) > 1e-6, (
            f"frame {i}: a robot's action is zero - batching broke co-observation"
        )


def test_run_multi_policy_raises_on_empty_action_chunk(sim_with_two_robots, tmp_path):
    """A policy that returns an empty chunk must fail loudly, not silently record
    all-zero ctrl. Pins the Key-Conventions-#6 fix (no silent zero-valued action
    on failure): pre-fix, an empty deque popleft fell back to ``{}`` -> all-zero
    ctrl -> dead frames in the dataset with no error.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")
    from strands_robots.policies.base import Policy

    class _EmptyChunkPolicy(Policy):
        requires_images = False

        def set_robot_state_keys(self, keys):
            self._keys = list(keys)

        @property
        def provider_name(self):
            return "empty_chunk"

        async def get_actions(self, obs, instruction=""):
            return []  # degenerate: no actions

    sim = sim_with_two_robots
    r = sim.start_recording(repo_id="local/empty", fps=20, root=str(tmp_path / "empty"), overwrite=True)
    assert r["status"] == "success", r
    with pytest.raises(RuntimeError, match="empty action chunk"):
        sim.run_multi_policy(
            policies={"alpha": _EmptyChunkPolicy(), "beta": _EmptyChunkPolicy()},
            n_steps=5,
            control_frequency=20.0,
        )
    sim.stop_recording()


def test_run_multi_policy_discards_partial_episode_on_empty_chunk(sim_with_two_robots, tmp_path):
    """An empty action chunk mid-loop must DISCARD the partial episode so the
    next recording starts at frame 0, not mid-episode.

    Pins the #366 follow-up: pre-fix, frames already add_frame'd before the
    RuntimeError stayed in the open episode buffer, so the next
    start_recording/run_multi_policy appended to that half-episode. A chunk
    returning [] on step 5 of 30 should leave the recorder starting the next
    episode at frame 0, not frame 5.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    from strands_robots.policies.base import Policy

    class _DiesAtStepFive(Policy):
        """Yields a valid 1-action chunk per step, then an empty chunk at step 5."""

        requires_images = False

        def __init__(self):
            self.calls = 0
            self._keys = None

        def set_robot_state_keys(self, keys):
            self._keys = list(keys)

        @property
        def provider_name(self):
            return "dies_at_five"

        async def get_actions(self, obs, instruction=""):
            self.calls += 1
            if self.calls >= 5:
                return []  # empty chunk -> RuntimeError mid-loop
            keys = self._keys or ["shoulder_pan", "shoulder_lift", "elbow"]
            return [{k: 0.05 for k in keys}]

    sim = sim_with_two_robots
    r = sim.start_recording(repo_id="local/partial", fps=20, root=str(tmp_path / "partial"), overwrite=True)
    assert r["status"] == "success", r

    # action_horizon=1 -> the policy is re-queried every step, so the empty
    # chunk surfaces on step 5 after frames 0-3 were recorded.
    with pytest.raises(RuntimeError, match="empty action chunk"):
        sim.run_multi_policy(
            policies={"alpha": _DiesAtStepFive(), "beta": _DiesAtStepFive()},
            n_steps=30,
            control_frequency=20.0,
            action_horizon=1,
        )

    recorder = sim._world._backend_state.get("dataset_recorder")
    assert recorder is not None
    # The partial episode (frames recorded before the abort) was discarded:
    # the next episode starts at frame 0.
    assert recorder.episode_frame_count == 0, (
        f"partial episode not discarded: {recorder.episode_frame_count} frames left in the open buffer"
    )
    sim.stop_recording()


def test_get_recording_status_text_is_ascii_no_world():
    """``get_recording_status`` with no world emits ASCII-only text.

    Previously the no-world branch returned a Unicode status-dot prefix
    ("\\u26aa No world..."), violating the project ASCII-only output
    contract. This needs no lerobot extra - it exercises the world-less
    branch directly.
    """
    from strands_robots.simulation import Simulation

    s = Simulation()
    r = s.get_recording_status()
    assert r["status"] == "success"
    text = r["content"][0]["text"]
    assert [ch for ch in text if ord(ch) > 127] == [], f"non-ASCII in text: {text!r}"
    # Uses the canonical world-less message shared by every action.
    assert "No world" in text
    s.destroy()


def test_get_recording_status_text_is_ascii_idle_and_recording(sim_with_two_robots):
    """``get_recording_status`` idle + active branches emit ASCII-only text.

    Drives the ``_backend_state`` recording flag directly so the assertion
    runs without the lerobot dataset stack. Both the idle and recording
    branches previously carried emoji status dots.
    """
    sim = sim_with_two_robots

    # Idle branch.
    r = sim.get_recording_status()
    text = r["content"][0]["text"]
    assert [ch for ch in text if ord(ch) > 127] == [], f"non-ASCII in idle text: {text!r}"
    assert "[idle]" in text

    # Active branch - flip the flag + seed a trajectory buffer directly.
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["trajectory"] = [object(), object(), object()]
    r = sim.get_recording_status()
    text = r["content"][0]["text"]
    assert [ch for ch in text if ord(ch) > 127] == [], f"non-ASCII in recording text: {text!r}"
    assert "[recording]" in text and "3 steps" in text
    sim._world._backend_state["recording"] = False


# start_recording() pre-create branches - lerobot guidance, dataset-dir
# resolution, and init-failure cleanup.
#
# These exercise the parts of start_recording that run BEFORE the LeRobot
# dataset object is built: the missing-extra guard, the repo_id -> on-disk
# dir resolution (local path vs HuggingFace cache), and the cleanup that runs
# when the recorder constructor fails. They mock DatasetRecorder.create so
# they pass with or without the lerobot extra installed.


def test_start_recording_without_lerobot_points_at_mp4_fallback(sim_with_two_robots, monkeypatch, tmp_path):
    """When the lerobot extra is absent, start_recording does not crash: it
    returns an error that names the lerobot extra and the plain-MP4 fallback."""
    import strands_robots.dataset_recorder as dr

    reason = "lerobot is not installed (ModuleNotFoundError: No module named 'lerobot'). Install lerobot >= 0.6.0 with: pip install 'strands-robots[lerobot]'"
    monkeypatch.setattr(dr, "lerobot_dataset_import_error", lambda: reason)

    r = sim_with_two_robots.start_recording(repo_id="local/no_lerobot", root=str(tmp_path / "dataset"), task="t")
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    # The diagnosis is surfaced verbatim: a caller must be told WHICH
    # dependency is missing, not just that recording is unavailable.
    assert reason in text
    assert "start_cameras_recording" in text


def test_start_recording_resolves_namespaced_repo_id_under_hf_cache(sim_with_two_robots, monkeypatch, tmp_path):
    """With root=None and a 'user/name' repo_id, the dataset dir resolves under
    the HuggingFace lerobot cache. We pin this by pointing the resolver's HF-cache home at a temp
    dir, pre-seeding the resolved cache dir, and asserting overwrite=True wipes
    exactly that resolved path."""
    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "has_lerobot_dataset", lambda: True)
    # Path resolution lives in dataset_recorder.resolve_dataset_dir(); pin the
    # HF-cache home it uses so the resolved dir is deterministic under tmp_path.
    monkeypatch.setattr(dr, "_lerobot_home", lambda: tmp_path / ".cache" / "huggingface" / "lerobot")
    monkeypatch.setattr(dr.DatasetRecorder, "create", classmethod(lambda cls, **kw: object()))

    cache_dir = tmp_path / ".cache" / "huggingface" / "lerobot" / "user" / "name"
    cache_dir.mkdir(parents=True)
    stale = cache_dir / "stale.txt"
    stale.write_text("old")

    r = sim_with_two_robots.start_recording(repo_id="user/name", root=None, overwrite=True)
    assert r["status"] == "success"
    # The resolved HF-cache dir (not cwd) was the one wiped by overwrite.
    assert not stale.exists()


def test_start_recording_resolves_bare_repo_id_as_local_path(sim_with_two_robots, monkeypatch, tmp_path):
    """A repo_id with no namespace slash resolves to a local relative path
    (Path(repo_id)), not the HF cache. Verified via the overwrite wipe."""
    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "has_lerobot_dataset", lambda: True)
    monkeypatch.setattr(dr.DatasetRecorder, "create", classmethod(lambda cls, **kw: object()))
    monkeypatch.chdir(tmp_path)

    local_dir = tmp_path / "bare_local"
    local_dir.mkdir()
    stale = local_dir / "stale.txt"
    stale.write_text("old")

    r = sim_with_two_robots.start_recording(repo_id="bare_local", root=None, overwrite=True)
    assert r["status"] == "success"
    assert not stale.exists()


def test_start_recording_recorder_init_failure_clears_recording_flag(sim_with_two_robots, monkeypatch, tmp_path):
    """If the dataset recorder constructor raises, start_recording reports an
    error AND resets the recording flag so the sim is not left wedged in a
    half-armed recording state."""
    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "has_lerobot_dataset", lambda: True)

    def _boom(cls, **kw):
        raise RuntimeError("codec unavailable")

    monkeypatch.setattr(dr.DatasetRecorder, "create", classmethod(_boom))

    r = sim_with_two_robots.start_recording(repo_id="local/boom", root=str(tmp_path), overwrite=True)
    assert r["status"] == "error"
    assert "codec unavailable" in r["content"][0]["text"]
    assert sim_with_two_robots._world._backend_state.get("recording") is False


@pytest.fixture
def sim_with_one_robot():
    """Single-robot sim for episode-boundary tests (clean, unprefixed joints)."""
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=path)
    s.step(5)
    yield s
    s.destroy()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_episode_delimits_one_episode_per_rollout(sim_with_one_robot, tmp_path):
    """Episode-boundary regression: ``save_episode`` after each ``run_policy``
    rollout produces one LeRobotDataset episode per rollout.

    Pre-fix the Simulation facade had no public episode-boundary primitive, so
    N ``run_policy`` calls in a single recording session all appended into the
    SAME buffer and ``stop_recording`` flushed them as a single
    ``episode_index=0`` (e.g. 3 rollouts -> 1 long episode). Calling
    ``save_episode`` between rollouts now writes each rollout as its own
    episode with a distinct ``episode_index`` / length.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_one_robot
    root = str(tmp_path / "perep")
    r = sim.start_recording(repo_id="local/perep", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r

    n_episodes = 3
    for i in range(n_episodes):
        rp = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            instruction=f"ep{i}",
            n_steps=5,
            control_frequency=20.0,
            fast_mode=True,
        )
        assert rp["status"] == "success", rp
        se = sim.save_episode()
        assert se["status"] == "success", se
        assert f"Episode {i + 1} saved" in se["content"][0]["text"]

    r = sim.stop_recording()
    assert r["status"] == "success", r

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/perep", root=root)
    assert ds.meta.total_episodes == n_episodes, f"expected {n_episodes} episodes, got {ds.meta.total_episodes}"
    # Each episode carries its own length and the frames partition cleanly.
    lengths = [ds.meta.episodes[ep]["length"] for ep in range(n_episodes)]
    assert all(length == 5 for length in lengths), f"episode lengths: {lengths}"
    assert ds.meta.total_frames == sum(lengths) == n_episodes * 5


def test_without_save_episode_rollouts_collapse_into_one_episode(sim_with_one_robot, tmp_path):
    """Documents the contrast: WITHOUT ``save_episode`` between rollouts, all
    frames land in a single episode. This pins the exact behaviour the boundary
    primitive fixes - run_policy alone does not delimit episodes.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_one_robot
    root = str(tmp_path / "collapsed")
    assert sim.start_recording(repo_id="local/collapsed", fps=20, root=root, overwrite=True)["status"] == "success"

    for i in range(3):
        sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            instruction=f"ep{i}",
            n_steps=5,
            control_frequency=20.0,
            fast_mode=True,
        )
    assert sim.stop_recording()["status"] == "success"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/collapsed", root=root)
    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == 15


def test_save_episode_without_recording_is_graceful_error(sim_with_one_robot):
    """save_episode outside an active recording session returns a structured
    error (not a raise), pointing at the correct workflow."""
    r = sim_with_one_robot.save_episode()
    assert r["status"] == "error"
    assert "not recording" in r["content"][0]["text"]


def test_save_episode_empty_buffer_is_idempotent(sim_with_one_robot, tmp_path):
    """Calling save_episode with no frames captured since the last boundary
    succeeds with a 'no frames' message instead of tripping LeRobot's
    'add frames before add_episode' guard - so loops can call it safely."""
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_one_robot
    root = str(tmp_path / "empty")
    assert sim.start_recording(repo_id="local/empty", fps=20, root=root, overwrite=True)["status"] == "success"

    # No run_policy yet -> nothing buffered.
    r = sim.save_episode()
    assert r["status"] == "success"
    assert "no frames" in r["content"][0]["text"]

    sim.stop_recording()


def test_reset_flushes_pending_recording_episode(sim_with_one_robot, tmp_path):
    """Episode-boundary regression: ``reset()`` during an active recording
    flushes the buffered rollout as its own episode.

    A ``run_policy`` + ``reset`` collection loop (the natural multi-episode
    pattern, since ``n_episodes`` is an ``eval_policy`` param, not a
    ``run_policy`` one) must not silently merge every rollout into a single
    ``episode_index=0``. Pre-fix the recorder buffered all N rollouts together
    and ``stop_recording`` flushed them as ONE giant episode
    (``total_episodes == 1`` for 20 rollouts). ``reset()`` now flushes the
    pending frames as an episode boundary first, so the dataset records one
    episode per rollout with clean ``from_index``/``to_index`` ranges.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_one_robot
    root = str(tmp_path / "reset_flush")
    r = sim.start_recording(repo_id="local/reset_flush", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r

    n_episodes = 4
    for i in range(n_episodes):
        rp = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            instruction=f"ep{i}",
            n_steps=5,
            control_frequency=20.0,
            fast_mode=True,
        )
        assert rp["status"] == "success", rp
        # reset() between rollouts is the episode boundary - it flushes the
        # buffered frames and reports the saved episode in its message.
        rr = sim.reset()
        assert rr["status"] == "success", rr
        assert f"Episode {i + 1} saved" in rr["content"][0]["text"]

    # The final reset already flushed the last rollout, so stop_recording has
    # nothing left to flush and just finalizes.
    r = sim.stop_recording()
    assert r["status"] == "success", r

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/reset_flush", root=root)
    assert ds.meta.total_episodes == n_episodes, f"expected {n_episodes} episodes, got {ds.meta.total_episodes}"
    lengths = [ds.meta.episodes[ep]["length"] for ep in range(n_episodes)]
    assert all(length == 5 for length in lengths), f"episode lengths: {lengths}"
    assert ds.meta.total_frames == sum(lengths) == n_episodes * 5


def test_reset_without_recording_does_not_flush(sim_with_one_robot):
    """``reset()`` outside an active recording session behaves exactly as
    before - a plain world reset with no episode-flush note. Guards against the
    auto-flush leaking into the non-recording path.
    """
    sim = sim_with_one_robot
    sim.run_policy(
        robot_name="arm",
        policy_provider="mock",
        n_steps=3,
        control_frequency=20.0,
        fast_mode=True,
    )
    r = sim.reset()
    assert r["status"] == "success", r
    assert r["content"][0]["text"] == "Reset to initial state."


def test_reset_empty_buffer_during_recording_does_not_create_episode(sim_with_one_robot, tmp_path):
    """A ``reset()`` while recording but with NO buffered frames (e.g. two
    resets in a row, or a reset right after start_recording) must not flush a
    spurious empty episode - save_episode is a no-op on an empty buffer.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_one_robot
    root = str(tmp_path / "reset_empty")
    assert sim.start_recording(repo_id="local/reset_empty", fps=20, root=root, overwrite=True)["status"] == "success"

    # reset with empty buffer -> plain reset, no flush note.
    r1 = sim.reset()
    assert r1["status"] == "success"
    assert r1["content"][0]["text"] == "Reset to initial state."

    sim.run_policy(
        robot_name="arm",
        policy_provider="mock",
        n_steps=5,
        control_frequency=20.0,
        fast_mode=True,
    )
    # First reset after a rollout flushes episode 1.
    r2 = sim.reset()
    assert "Episode 1 saved" in r2["content"][0]["text"]
    # Second consecutive reset has nothing buffered -> no new episode.
    r3 = sim.reset()
    assert r3["content"][0]["text"] == "Reset to initial state."

    assert sim.stop_recording()["status"] == "success"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/reset_empty", root=root)
    assert ds.meta.total_episodes == 1, f"expected 1 episode, got {ds.meta.total_episodes}"
    assert ds.meta.total_frames == 5


def test_reset_surfaces_save_episode_failure_without_resetting(sim_with_one_robot, tmp_path):
    """A ``reset()`` that auto-flushes a pending episode must SURFACE a flush
    failure and leave the world untouched - not silently reset into an
    undefined state.

    ``reset()`` flushes buffered recording frames as an episode boundary before
    teleporting the world (see
    :func:`test_reset_flushes_pending_recording_episode`). If that flush fails,
    the facade ``save_episode`` has already marked the recorder poisoned and
    cleared the recording flag; ``reset()`` must propagate that error and abort
    the reset rather than run ``mj_resetData`` on top of a half-written episode.
    Otherwise a data-integrity failure would masquerade as a clean reset and the
    caller would keep collecting into a broken dataset.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_one_robot
    root = str(tmp_path / "reset_flush_fail")
    assert sim.start_recording(repo_id="local/reset_flush_fail", fps=20, root=root, overwrite=True)["status"] == (
        "success"
    )

    # Buffer a real rollout so reset() takes the flush path (episode_frame_count > 0).
    rp = sim.run_policy(
        robot_name="arm",
        policy_provider="mock",
        n_steps=5,
        control_frequency=20.0,
        fast_mode=True,
    )
    assert rp["status"] == "success", rp

    recorder = sim._world._backend_state["dataset_recorder"]
    assert recorder is not None
    assert getattr(recorder, "episode_frame_count", 0) > 0, "expected a non-empty buffer to flush"

    # Force the underlying flush to fail. The facade save_episode turns a
    # recorder error dict into a cleared-recording error return, which reset()
    # must propagate verbatim.
    def _boom() -> dict:
        return {"status": "error", "message": "simulated lerobot flush failure"}

    recorder.save_episode = _boom  # type: ignore[method-assign]

    # Capture pre-reset world bookkeeping; a fail-fast reset must not touch it.
    step_count_before = sim._world.step_count
    sim_time_before = sim._world.sim_time
    assert step_count_before > 0, "rollout should have advanced the step counter"

    rr = sim.reset()

    # 1. The failure is surfaced, not swallowed.
    assert rr["status"] == "error", rr
    assert "save_episode failed" in rr["content"][0]["text"]
    assert "simulated lerobot flush failure" in rr["content"][0]["text"]

    # 2. The world was NOT reset - no mj_resetData ran, so time/step survive.
    assert sim._world.step_count == step_count_before
    assert sim._world.sim_time == sim_time_before

    # 3. The poisoned recorder was dropped so callers cannot append into it.
    assert sim._world._backend_state.get("recording") is False
    assert sim._world._backend_state.get("dataset_recorder") is None


def test_start_recording_existing_empty_root_records(sim_with_two_robots, tmp_path):
    """An EXISTING empty ``root`` (e.g. from tempfile.mkdtemp()) must record.

    Pre-fix, start_recording(overwrite=False) on an existing empty directory
    dead-ended with ``Dataset init failed: [Errno 17] File exists`` because it
    called LeRobotDataset.create() (mkdir exist_ok=False) on a dir that already
    existed. The natural caller pattern is to mkdtemp() a root and pass it in;
    that must produce a valid dataset, not a cryptic errno.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots
    # tmp_path is an EXISTING empty directory - the repro condition.
    root = str(tmp_path)
    assert os.path.isdir(root) and not os.listdir(root)

    r = sim.start_recording(repo_id="local/empty_root_probe", fps=20, root=root, overwrite=False)
    assert r["status"] == "success", f"empty-root start_recording failed: {r}"
    sim.run_policy(
        robot_name="alpha",
        policy_provider="mock",
        instruction="ep0",
        duration=0.3,
        control_frequency=20.0,
        fast_mode=True,
    )
    r = sim.stop_recording()
    assert r["status"] == "success", r

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/empty_root_probe", root=root)
    assert ds.meta.total_episodes == 1, f"expected 1 episode, got {ds.meta.total_episodes}"
    assert ds.meta.total_frames > 0


def test_run_multi_policy_single_robot_records_unnamespaced_columns(tmp_path):
    """A single-robot world driven via ``run_multi_policy`` records BARE
    (un-prefixed) action/observation columns.

    ``run_multi_policy`` accepts a one-entry ``policies`` dict (e.g. a bimanual
    rig temporarily driving a single arm). When the world holds exactly one
    robot, ``multi_robot`` is False and the merged frame must carry the robot's
    plain actuator/joint keys - NOT ``<name>__`` prefixed - so the dataset
    schema is identical to a single-robot ``run_policy`` recording and a policy
    trained on one layout can consume the other. This pins the un-namespaced
    merge branch of the synchronized loop (the multi-robot namespaced branch is
    covered by ``test_b4_synchronized_multi_robot_recording``).
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    import numpy as np

    from strands_robots.policies import create_policy
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    sim = Simulation()
    sim.create_world()
    sim.add_robot("solo", urdf_path=path, position=[0, 0, 0])
    sim.step(5)
    try:
        root = str(tmp_path / "solo")
        r = sim.start_recording(repo_id="local/solo_multi", fps=20, root=root, overwrite=True)
        assert r["status"] == "success", r

        r = sim.run_multi_policy(
            policies={"solo": create_policy("mock")},
            instructions="wave",
            duration=0.5,
            control_frequency=20.0,
        )
        assert r["status"] == "success", r
        assert tool_json(r)["steps"] > 0
        sim.stop_recording()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id="local/solo_multi", root=root)

        af = ds.features["action"]
        names = af["names"] if isinstance(af, dict) else getattr(af, "names", None)
        assert names, names
        # Single-robot world: columns are the bare actuator keys, un-prefixed.
        assert all("__" not in n for n in names), f"expected un-namespaced columns, got {names}"
        assert set(names) == {"shoulder_pan_act", "shoulder_lift_act", "elbow_act"}, names

        # Frames were actually written with non-zero commanded actions.
        assert len(ds) > 0
        moved = 0
        for i in range(len(ds)):
            if float(np.abs(np.asarray(ds[i]["action"])).sum()) > 1e-6:
                moved += 1
        assert moved == len(ds), f"only {moved}/{len(ds)} frames carried a commanded action"
    finally:
        sim.destroy()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_get_state_reports_live_recording_progress(sim_with_two_robots, tmp_path):
    """``get_state`` annotates the recorded step count while a recording is live.

    ``Simulation.get_state`` is the human-readable introspection surface an
    agent polls to monitor a data-collection episode. When a LeRobotDataset
    recording is active it appends a ``[recording] N steps`` line so the caller
    can watch the buffer fill mid-rollout; on an idle world that line is absent.
    The count reflects the trajectory buffer the run_policy hook fills, so it is
    strictly positive once frames have been rolled out under an active recorder.
    """
    import re

    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots

    # Idle world: no recording annotation on the state summary.
    idle = sim.get_state()
    assert idle["status"] == "success"
    assert "[recording]" not in idle["content"][0]["text"]

    r = sim.start_recording(repo_id="local/state_progress", fps=20, root=str(tmp_path), overwrite=True)
    assert r["status"] == "success", r

    from strands_robots.policies.mock import MockPolicy

    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("alpha"))
    r = sim.run_policy("alpha", policy_object=policy, duration=0.5, control_frequency=20.0)
    assert r["status"] == "success", r

    # Recording still active: the summary reports a positive buffered count.
    active_text = sim.get_state()["content"][0]["text"]
    match = re.search(r"\[recording\] (\d+) steps", active_text)
    assert match is not None, active_text
    assert int(match.group(1)) > 0, active_text

    sim.stop_recording()

    # Back to idle: the annotation is gone once the recording is finalized.
    done_text = sim.get_state()["content"][0]["text"]
    assert "[recording]" not in done_text
