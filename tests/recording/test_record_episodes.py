"""``robot.run(policy, record=dir)``: episodes a fresh process reloads with LeRobot's own ``LeRobotDataset``."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from strands_robots import Robot
from strands_robots.policy import SequencePolicy

pytest.importorskip("lerobot.datasets.lerobot_dataset")

_RELOAD = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/x", root=sys.argv[1])
first, last = ds[0], ds[ds.num_frames - 1]
cam = next(k for k, v in ds.meta.features.items() if v["dtype"] == "video")
print(json.dumps({
    "episodes": ds.num_episodes,
    "frames": ds.num_frames,
    "fps": ds.fps,
    "names": {k: v["names"] for k, v in ds.meta.features.items() if k in ("observation.state", "action")},
    "image": list(first[cam].shape),
    "state": list(first["observation.state"].shape),
    "action": list(first["action"].shape),
    "episode_index": [int(first["episode_index"]), int(last["episode_index"])],
    "tasks": [first["task"], last["task"]],
}))
"""


def reload(root) -> dict:
    """Read a dataset back in a FRESH interpreter, so nothing this process cached can make it look right."""
    out = subprocess.run([sys.executable, "-c", _RELOAD, str(root)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_three_episodes_with_a_camera_reload_in_a_fresh_process(tmp_path):
    arm = Robot("so101")
    arm.sim.camera("wrist", position=[0.3, -0.3, 0.3], target=[0, 0, 0.1], size=(64, 48), attach=arm)
    target = tmp_path / "waves"
    results = [
        arm.run(SequencePolicy([{"shoulder_pan": 0.1 * (i + 1)}]), hz=20, steps=n, record=target, instruction=task)
        for i, (n, task) in enumerate([(10, "wave"), (12, "wave"), (8, "wave back")])
    ]
    assert [r.recorded for r in results] == [str(target)] * 3
    got = reload(target)
    names = [f"{k}.pos" for k in arm.action_keys]
    assert got["episodes"] == 3 and got["frames"] == 30 and got["fps"] == 20
    assert got["names"] == {"observation.state": names, "action": names}
    assert got["image"] == [3, 48, 64] and got["state"] == [6] and got["action"] == [6]
    assert got["episode_index"] == [0, 2] and got["tasks"] == ["wave", "wave back"]
    info = json.loads((target / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["robot_type"] == arm.description.hardware.lerobot_type
    assert info["features"]["observation.images.wrist"]["dtype"] == "video"
    print(f"[measured] {got['episodes']} episodes / {got['frames']} frames, image {got['image']}, fps {got['fps']}")


def test_record_is_part_of_every_run_signature():
    """Whatever wraps ``Robot.run`` must keep ``record=``: a wrapper that re-spells the signature loses it silently."""
    import inspect

    from strands_robots.robot.run import run

    assert "record" in inspect.signature(run).parameters
    arm = Robot("so101")
    with pytest.raises(ValueError, match="integer fps"):
        arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=12.5, steps=1, record="unused")
