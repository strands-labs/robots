"""Multi-robot dual-task integration - the scenario the PR exists for.

Two robots in one world, each given its own instruction via its own policy,
the whole scene captured as a single LeRobotDataset episode.

Guards several invariants at once:
    * ``start_recording`` accepts multi-robot worlds
    * Robot joint names are disambiguated with ``{name}__{joint}`` prefix
      when the scene has >1 robot (avoid schema clashes).
    * Per-robot wrist cameras appear in the dataset as namespaced features
      (``observation.images.alice__wrist_cam``), not as lossy flat names.
    * Running two ``run_policy`` calls sequentially against the same
      ``start_recording`` session writes frames from BOTH robots into the
      same episode.
    * The dataset parquet has ``episode_index=0`` for every row (one
      episode) and has at least one action vector per control step.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

os.environ.setdefault("MUJOCO_GL", "glfw")


@pytest.fixture
def dual_robot_world():
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world(timestep=0.002, gravity=[0, 0, -9.81])
    sim.add_robot("alice", data_config="so101", position=[-0.25, 0.0, 0.0])
    sim.add_robot("bob", data_config="so101", position=[0.25, 0.0, 0.0])
    sim.add_object("red_cube", shape="box", size=[0.025, 0.025, 0.025], position=[-0.15, 0.2, 0.05], rgba=[1, 0, 0, 1])
    sim.add_object("blue_ball", shape="sphere", size=[0.03, 0.03, 0.03], position=[0.15, 0.2, 0.05], rgba=[0, 0, 1, 1])
    sim.add_camera("top", position=[0, 0, 0.9], target=[0, 0.2, 0.05])
    # Per-robot wrist cameras: the "/" namespace separator is normalised to
    # "__" in recorded LeRobot feature names (observation.images.alice__wrist_cam).
    sim.add_camera("alice/wrist_cam", position=[-0.25, 0.2, 0.4], target=[-0.15, 0.2, 0.05])
    sim.add_camera("bob/wrist_cam", position=[0.25, 0.2, 0.4], target=[0.15, 0.2, 0.05])
    sim.step(n_steps=10)
    yield sim
    sim.destroy()


def test_two_robots_two_tasks_recorded_as_single_episode(dual_robot_world, tmp_path):
    from strands_robots.dataset_recorder import has_lerobot_dataset
    from strands_robots.policies.mock import MockPolicy

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = dual_robot_world

    r = sim.start_recording(repo_id="local/dual_task", task="pick_two", fps=20, root=str(tmp_path), overwrite=True)
    assert r["status"] == "success", r

    # Build one policy per robot bound to that robot's joint ordering
    policy_a = MockPolicy()
    policy_a.set_robot_state_keys(sim.robot_joint_names("alice"))
    policy_b = MockPolicy()
    policy_b.set_robot_state_keys(sim.robot_joint_names("bob"))

    # Two sequential rollouts, both feeding the SAME recording
    r = sim.run_policy(
        "alice",
        policy_object=policy_a,
        instruction="grasp the red cube",
        duration=0.3,
        control_frequency=20.0,
    )
    assert r["status"] == "success"

    r = sim.run_policy(
        "bob",
        policy_object=policy_b,
        instruction="grasp the blue ball",
        duration=0.3,
        control_frequency=20.0,
    )
    assert r["status"] == "success"

    stop = sim.stop_recording()
    assert stop["status"] == "success"

    # Dataset-on-disk invariants
    info_path = tmp_path / "meta" / "info.json"
    assert info_path.exists(), "meta/info.json missing"

    info = json.loads(info_path.read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] > 0

    # Features should include the shared 'top' camera and BOTH wrist cams,
    # correctly namespaced with ``__`` separators (no '/' allowed in LeRobot
    # feature names).
    features = info["features"]
    feature_names = set(features.keys())
    assert "observation.images.top" in feature_names
    assert "observation.images.alice__wrist_cam" in feature_names
    assert "observation.images.bob__wrist_cam" in feature_names

    # Joint names must be disambiguated per robot (alice__X / bob__X)
    joint_names = features["observation.state"]["names"]
    assert len(joint_names) == 12, f"2 robots × 6 joints expected, got {joint_names}"
    assert len(set(joint_names)) == 12, f"duplicate joint names: {joint_names}"
    assert any(jn.startswith("alice__") for jn in joint_names)
    assert any(jn.startswith("bob__") for jn in joint_names)

    # Parquet invariants
    import pandas as pd  # type: ignore[import-untyped]

    data_parquets = glob.glob(str(tmp_path / "data" / "chunk-*" / "*.parquet"))
    assert data_parquets, "no data parquet written"
    df = pd.read_parquet(data_parquets[0])

    # Every row is in the same episode
    assert (df["episode_index"] == 0).all()
    # Each row has a 12-D state (alice 6 + bob 6) and 12-D action
    sample_state = df["observation.state"].iloc[0]
    assert len(sample_state) == 12, f"state should be 12-D, got {len(sample_state)}"
    sample_action = df["action"].iloc[0]
    assert len(sample_action) == 12, f"action should be 12-D, got {len(sample_action)}"

    # Two sequential 0.3s @ 20Hz rollouts = ~12 frames total
    assert len(df) >= 6, f"expected >=6 frames, got {len(df)}"

    # Video assets
    video_files = list((tmp_path / "videos").rglob("*.mp4"))
    video_names = {p.parent.parent.name for p in video_files}
    assert "observation.images.top" in video_names
    assert "observation.images.alice__wrist_cam" in video_names
    assert "observation.images.bob__wrist_cam" in video_names
    for v in video_files:
        assert v.stat().st_size > 0, f"empty video: {v}"
