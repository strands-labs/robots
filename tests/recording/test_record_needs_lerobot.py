"""record= without LeRobot's dataset writer refuses by name before the robot moves (no LeRobot needed to run this)."""

from __future__ import annotations

import re
import sys

import pytest

from strands_robots import Robot
from strands_robots._extras import LEROBOT, hint


def test_record_without_lerobot_refuses_before_the_first_step(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)  # the import fails as if not installed
    arm = Robot("so101")
    before = arm.observe(images=False)["shoulder_pan"]
    with pytest.raises(ImportError, match=re.escape(LEROBOT)):
        arm.run(lambda obs: {"shoulder_pan": 0.5}, steps=3, record=tmp_path / "ds")
    assert arm.observe(images=False)["shoulder_pan"] == before
    assert not (tmp_path / "ds").exists()


def test_the_one_lerobot_install_line_also_records():
    """``[lerobot]``'s hint is the line record= names: after following it, record= works (no second install)."""
    assert re.search(r"lerobot\[[a-z,]*\bdataset\b[a-z,]*\]", LEROBOT)
    assert hint("lerobot").endswith(LEROBOT)
