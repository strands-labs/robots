"""Check 15: a plain top-level script (no ``if __name__ == "__main__"`` guard) records an episode with two
cameras and exits 0. On macOS, worker processes start by re-importing ``__main__``; a recording that
needs that guard crashes the first script a newcomer writes.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from checks import needs_module

pytest.importorskip("mujoco")

SCRIPT = """
from strands_robots import Robot
from strands_robots.policy import SequencePolicy
arm = Robot("so101")
arm.sim.camera("front", size=(64, 48), attach=arm)
arm.sim.camera("side", size=(64, 48), position=(0.0, -0.6, 0.3), attach=arm)
r = arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=30, steps=30, record=TARGET)
print("recorded", r.recorded, r.steps)
"""


def test_a_plain_script_records_two_cameras(tmp_path):
    needs_module("strands_robots.recording", "part 9: recording")
    needs_module("lerobot.datasets.lerobot_dataset", "the [lerobot] extra")
    script = tmp_path / "record.py"
    script.write_text(SCRIPT.replace("TARGET", repr(str(tmp_path / "ds"))))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert out.returncode == 0 and "recorded" in out.stdout, out.stderr[-2000:]
