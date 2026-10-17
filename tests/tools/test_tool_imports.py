"""Gate row 6: strands is imported only when ``.tool`` is used (a fresh interpreter, so nothing is cached)."""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import strands_robots, strands_robots.tools
from strands_robots import Robot
from strands_robots.simulation import Simulation
arm = Robot("so101")
spec = strands_robots.tools.tool_spec(arm)  # the spec alone needs no strands
print("before", "strands" in sys.modules)
arm.tool
print("after", "strands" in sys.modules)
"""


def test_strands_is_imported_only_when_a_tool_is_built():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    assert "before False" in out and "after True" in out, out
