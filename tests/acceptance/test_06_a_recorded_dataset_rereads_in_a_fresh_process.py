"""Check 6: ``run(record=)`` writes LeRobot v3 that a fresh interpreter (one that never imported
strands_robots) reads back with LeRobot's own ``LeRobotDataset``; a second run resumes the same directory.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from checks import needs_module

pytest.importorskip("mujoco")

READER = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/acceptance", root=sys.argv[1])
item = ds[0]
print(json.dumps({"episodes": ds.num_episodes, "frames": ds.num_frames, "fps": ds.fps,
                  "image": list(item["observation.images.front"].shape), "action": list(item["action"].shape),
                  "imported": "strands_robots" in sys.modules}))
"""


def test_a_recorded_dataset_rereads_in_a_fresh_process(tmp_path):
    needs_module("strands_robots.recording", "part 9: recording")
    needs_module("lerobot.datasets.lerobot_dataset", "the [lerobot] extra")
    from strands_robots import Robot
    from strands_robots.policy import SequencePolicy

    arm = Robot("so101")
    arm.sim.camera("front", size=(64, 48), attach=arm)
    target = tmp_path / "acceptance"
    for i in range(3):
        r = arm.run(SequencePolicy([{"shoulder_pan": 0.1 * (i + 1)}]), hz=30, steps=30, record=target)
        assert r.recorded == str(target)
    r = arm.run(SequencePolicy([{"shoulder_pan": -0.2}]), hz=30, steps=15, record=target)  # resumes
    out = subprocess.run([sys.executable, "-c", READER, str(target)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {"episodes": 4, "frames": 105, "fps": 30, "image": [3, 48, 64], "action": [6], "imported": False}
