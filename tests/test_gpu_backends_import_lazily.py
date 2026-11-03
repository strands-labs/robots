"""The GPU backends are plain modules on a laptop: importing them loads no Isaac or Warp; asking for one says how."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from strands_robots import Simulation

_PROBE = """
import sys
import strands_robots.simulation.backends.isaac, strands_robots.simulation.backends.newton
print(sorted(m for m in ("isaacsim", "omni", "pxr", "warp", "mujoco_warp", "newton") if m in sys.modules))
"""


def test_importing_the_gpu_backends_imports_no_gpu_stack():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.skipif(importlib.util.find_spec("isaacsim") is not None, reason="Isaac Sim is installed here")
def test_isaac_without_isaac_sim_says_how_to_install():
    with pytest.raises(
        ImportError, match=r"Simulation\('isaac'\) needs Isaac Sim: pip install 'strands-robots\[isaac\]'"
    ):
        Simulation("isaac")
