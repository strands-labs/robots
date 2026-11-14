"""A stalled tick shows in the recording: ``time.measured`` is when each frame was read; ``timestamp`` stays nominal."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pytest

from strands_robots import Robot

pytest.importorskip("lerobot.datasets.lerobot_dataset")

STALL_S, HZ = 0.3, 30

_DUMP = """
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/x", root=sys.argv[1])
rows = [ds[i] for i in range(ds.num_frames)]
print(json.dumps({
    "features": sorted(ds.meta.features),
    "episode_index": [int(r["episode_index"]) for r in rows],
    "timestamp": [float(r["timestamp"]) for r in rows],
    "measured": [float(r["time.measured"].reshape(-1)[0]) for r in rows] if "time.measured" in rows[0] else None,
}))
"""


def _dump(root) -> dict:
    """Read the dataset back in a FRESH interpreter with LeRobot's own loader."""
    out = subprocess.run([sys.executable, "-c", _DUMP, str(root)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _stalling(at: int):
    n = {"i": 0}

    def policy(obs):
        n["i"] += 1
        if n["i"] == at:
            time.sleep(STALL_S)
        return {"shoulder_pan": 0.01 * n["i"]}

    return policy


def _arm() -> Robot:
    return Robot("so101", driver="memory", operator=lambda action: True)


def test_a_stalled_tick_is_a_gap_in_measured_time_while_timestamp_stays_on_the_grid(tmp_path):
    root = tmp_path / "stall"
    r = _arm().run(_stalling(at=10), hz=HZ, steps=30, record=root, on_miss="skip")
    assert r.recorded == str(root)
    d = _dump(root)
    assert "time.measured" in d["features"]
    assert np.allclose(np.diff(d["timestamp"]), 1 / HZ, atol=1e-4)  # LeRobot's grid: video still decodes by it
    gaps = np.diff(d["measured"])
    assert d["measured"][0] == 0.0
    assert int(np.argmax(gaps)) == 9  # the stall in tick 10's policy delays tick 11's read
    assert gaps[9] >= STALL_S * 0.9
    assert np.all(gaps > 0)


def test_a_dataset_recorded_without_measured_time_still_takes_new_episodes(tmp_path, monkeypatch):
    from strands_robots.recording import features

    root = tmp_path / "old"
    with monkeypatch.context() as m:  # what main wrote before frames carried their time
        old = features.features_from
        m.setattr("strands_robots.recording.lerobot_v3.features_from", lambda *a: _without_time(old(*a)))
        _arm().run(lambda o: {"shoulder_pan": 0.1}, hz=HZ, steps=4, record=root)
    r = _arm().run(lambda o: {"shoulder_pan": 0.2}, hz=HZ, steps=5, record=root)
    assert r.recorded == str(root)
    d = _dump(root)
    assert "time.measured" not in d["features"] and d["measured"] is None
    assert d["episode_index"] == [0] * 4 + [1] * 5


def test_each_resumed_episode_measures_from_its_own_first_frame(tmp_path):
    root = tmp_path / "two"
    for _ in range(2):
        _arm().run(lambda o: {"shoulder_pan": 0.1}, hz=HZ, steps=6, record=root)
    d = _dump(root)
    first = [m for m, e in zip(d["measured"], d["episode_index"], strict=True) if e == 1][0]
    assert first == 0.0 and d["measured"][0] == 0.0
    assert all(m < 1.0 for m in d["measured"])


def _without_time(feats: dict) -> dict:
    return {k: v for k, v in feats.items() if k != "time.measured"}
