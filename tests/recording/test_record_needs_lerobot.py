"""record= without LeRobot's dataset writer refuses by name before the robot moves (no LeRobot needed to run this)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import tomllib

from strands_robots import Robot
from strands_robots._extras import hint


def test_record_without_lerobot_refuses_before_the_first_step(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)  # the import fails as if not installed
    arm = Robot("so101")
    before = arm.observe(images=False)["shoulder_pan"]
    with pytest.raises(ImportError, match=re.escape(hint("lerobot"))):
        arm.run(lambda obs: {"shoulder_pan": 0.5}, steps=3, record=tmp_path / "ds")
    assert arm.observe(images=False)["shoulder_pan"] == before
    assert not (tmp_path / "ds").exists()


def test_the_one_lerobot_install_line_also_records():
    """``[lerobot]`` pulls LeRobot's ``dataset`` extra, so the line policies name also gives record= (one install)."""
    root = Path(__file__).resolve().parents[2]
    (lerobot_dep,) = (
        d
        for d in tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]["lerobot"]
        if d.startswith("lerobot[")
    )
    assert re.search(r"lerobot\[[a-z,]*\bdataset\b[a-z,]*\]", lerobot_dep)
    assert "pip install 'strands-robots[lerobot]'" in hint("lerobot")
