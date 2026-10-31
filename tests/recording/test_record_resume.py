"""A second ``run(record=same_dir)`` resumes: the episode index continues and earlier episodes are untouched."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy import SequencePolicy

pytest.importorskip("lerobot.datasets.lerobot_dataset")

_RUN = """
import sys
from strands_robots import Robot
from strands_robots.policy import SequencePolicy
arm = Robot("so101")
arm.sim.camera("wrist", position=[0.3, -0.3, 0.3], target=[0, 0, 0.1], size=(64, 48), attach=arm)
r = arm.run(SequencePolicy([{"shoulder_pan": -0.3}]), hz=20, steps=7, record=sys.argv[1], instruction="back")
print(r.recorded)
"""

_DUMP = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/x", root=sys.argv[1])
rows = [ds[i] for i in range(ds.num_frames)]
print(json.dumps({
    "episodes": ds.num_episodes,
    "episode_index": [int(r["episode_index"]) for r in rows],
    "action": [[round(float(v), 6) for v in r["action"]] for r in rows],
    "state": [[round(float(v), 6) for v in r["observation.state"]] for r in rows],
    "pixels": [round(float(r["observation.images.wrist"].mean()), 3) for r in rows],
    "tasks": [r["task"] for r in rows],
}))
"""


def _py(code: str, *args) -> str:
    out = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def _arm():
    arm = Robot("so101")
    arm.sim.camera("wrist", position=[0.3, -0.3, 0.3], target=[0, 0, 0.1], size=(64, 48), attach=arm)
    return arm


def test_a_second_run_in_a_new_process_resumes_the_same_directory(tmp_path):
    target = tmp_path / "ds"
    _arm().run(SequencePolicy([{"shoulder_pan": 0.3}]), hz=20, steps=9, record=target, instruction="out")
    before = json.loads(_py(_DUMP, target))
    assert before["episodes"] == 1 and before["episode_index"] == [0] * 9
    assert _py(_RUN, target) == str(target)  # a different interpreter appends to what this one wrote
    after = json.loads(_py(_DUMP, target))
    assert after["episodes"] == 2 and after["episode_index"] == [0] * 9 + [1] * 7
    for k in ("action", "state", "pixels", "tasks"):
        assert after[k][:9] == before[k], f"episode 0 {k} changed on resume"
    assert after["tasks"][9:] == ["back"] * 7 and after["action"][-1][0] == pytest.approx(-0.3)
    print(f"[measured] resume: episodes {before['episodes']} -> {after['episodes']}, frames 9 -> {len(after['tasks'])}")


def test_resume_refuses_other_units_naming_what_differs_before_the_robot_moves(tmp_path):
    from test_replay_units import Degrees  # declares LeRobot degrees, as a checkpoint run under norm= does

    target, arm = tmp_path / "ds", _arm()
    arm.run(SequencePolicy([{"shoulder_pan": 10.0}]), hz=20, steps=3, record=target, units="lerobot")
    before = arm.observe(images=False).joints.positions.tolist()
    in_degrees = (  # only what differs: shoulder_lift's travel is 100 deg either side, so only its mode is named
        "this run records 'lerobot', but meta/units.json there has shoulder_pan.pos range [-100.0, 100.0] (this run: "
        "[-110.0, 110.0]), shoulder_pan.pos mode 'range_m100_100' (this run: 'degrees'), shoulder_lift.pos mode "
    )
    with pytest.raises(ValueError, match=re.escape(in_degrees)):
        arm.run(Degrees(), hz=20, steps=3, record=target)
    path = target / "meta" / "units.json"
    said = json.loads(path.read_text())
    for joint in said["joints"].values():
        del joint["mode"]  # as record= wrote it before it named each joint's norm mode
    path.write_text(json.dumps(said))
    with pytest.raises(ValueError) as refused:
        arm.run(SequencePolicy([{"shoulder_pan": 60.0}]), hz=20, steps=3, record=target, units="lerobot")
    assert str(refused.value) == (
        f"{target}: the dataset there is stored in units 'lerobot'; this run records 'lerobot', but meta/units.json "
        "there has shoulder_pan.pos mode none (this run: 'range_m100_100'), shoulder_lift.pos mode none (this run: "
        "'range_m100_100'), elbow_flex.pos mode none (this run: 'range_m100_100'), wrist_flex.pos mode none (this run: "
        "'range_m100_100'), wrist_roll.pos mode none (this run: 'range_m100_100'), gripper.pos mode none (this run: "
        "'range_0_100')"
    )
    with pytest.raises(ValueError, match=r"stored in units 'lerobot'; this run records 'rad'$"):  # the scheme says it
        arm.run(SequencePolicy([{"shoulder_pan": 0.6}]), hz=20, steps=3, record=target)
    assert arm.observe(images=False).joints.positions.tolist() == before


def test_resume_refuses_a_different_fps_or_robot_before_the_robot_moves(tmp_path):
    target = tmp_path / "ds"
    arm = _arm()
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=3, record=target)
    before = arm.observe(images=False)["shoulder_pan"]
    with pytest.raises(ValueError, match="20 fps; this run records at 10"):
        arm.run(SequencePolicy([{"shoulder_pan": 0.6}]), hz=10, steps=3, record=target)
    assert arm.observe(images=False)["shoulder_pan"] == before
    g1 = Robot("g1", sim=Simulation("mujoco"), keyframe="stand")
    with pytest.raises(ValueError, match="robot"):
        g1.run(lambda o: o.joints.positions, hz=20, steps=3, record=target)
    g1.sim.close()


def test_resume_refuses_a_camera_the_dataset_lacks_before_the_robot_moves(tmp_path):
    """The features come from an observation, so they were checked at the first frame, after tick 1's write."""
    target, arm = tmp_path / "ds", Robot("so101")
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=3, record=target)
    arm.sim.camera("wrist", position=[0.3, -0.3, 0.3], target=[0, 0, 0.1], size=(64, 48), attach=arm)
    before = arm.observe(images=False)["shoulder_pan"]
    with pytest.raises(ValueError, match="features"):
        arm.run(SequencePolicy([{"shoulder_pan": 0.6}]), hz=20, steps=3, record=target)
    assert arm.observe(images=False)["shoulder_pan"] == before


def _files(root):
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_runs_that_end_before_their_first_frame_leave_a_resumed_dataset_as_it_was(tmp_path):
    """The dataset opens before tick 1, so a run that ends before its first frame opens it too: nothing changes."""
    target, arm = tmp_path / "ds", _arm()
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=5, record=target)
    before = _files(target)
    arm.estop()
    assert arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=20, steps=5, record=target).recorded is None
    arm.resume(approved_by="op")
    r = arm.run(SequencePolicy([{"shoulder_pan": 0.3}]), hz=20, steps=5, record=target, stop_when=lambda o: True)
    assert r.recorded is None and _files(target) == before
    arm.sim.close()


def test_non_integer_hz_is_refused_before_step_one(tmp_path):
    with pytest.raises(ValueError, match="integer fps"):
        _arm().run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=12.5, steps=2, record=tmp_path / "x")
    assert not (tmp_path / "x").exists()
