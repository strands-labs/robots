"""Record from the page's routes: start, save, discard, stop -> a LeRobot v3 dataset LeRobot itself reads back."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("lerobot.datasets.lerobot_dataset")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402
from strands_robots.dashboard.record import targets_for  # noqa: E402


def _wait_frames(u: str, n: int, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = httpx.get(f"{u}/api/record/status").json()
        if st["frames"] >= n or st["state"] == "error":
            return st
        time.sleep(0.05)
    raise AssertionError(f"no {n} frames: {st}")


def test_two_episodes_of_so101_from_the_page_read_back_by_lerobot(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    sim.camera("front", size=(160, 120), attach=arm)
    sim.camera("top", size=(96, 64), position=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0))
    server = Dashboard(sim).start()
    target = tmp_path / "so101-wave"
    try:
        u = server.url
        assert httpx.get(f"{u}/api/record/status").json()["state"] == "idle"
        body = {"robots": ["so101"], "cameras": {"so101": ["front", "top"]}, "target": str(target), "task": "wave"}
        r = httpx.post(f"{u}/api/record/start", json={**body, "fps": 20}, timeout=60)
        assert r.status_code == 201, r.text
        assert httpx.post(f"{u}/api/record/start", json={**body, "fps": 20}).status_code == 409  # one at a time
        for i in range(10):  # a slider moves the arm while it records
            httpx.post(f"{u}/api/robots/so101/act", json={"targets": {"shoulder_pan": 0.05 * i}})
            time.sleep(0.02)
        _wait_frames(u, 10)
        st = httpx.post(f"{u}/api/record/episode/save", json={}, timeout=120).json()
        assert st["state"] == "recording" and st["episode"] == 1 and len(st["episodes"]) == 1
        assert isinstance(st["dropped"], int) and st["dropped"] >= 0
        _wait_frames(u, 5)
        st = httpx.post(f"{u}/api/record/episode/discard", json={}, timeout=60).json()
        assert st["episode"] == 1 and len(st["episodes"]) == 1  # a discarded take is not an episode
        httpx.post(f"{u}/api/robots/so101/act", json={"targets": {"shoulder_pan": -0.3}})
        _wait_frames(u, 8)
        st = httpx.post(f"{u}/api/record/stop", json={"save": True}, timeout=120).json()
        assert st["state"] == "idle" and st["episode"] == 2 and [e["index"] for e in st["episodes"]] == [0, 1]
        assert st["robots"]["so101"]["units"] == "lerobot"
        assert httpx.post(f"{u}/api/record/episode/save", json={}).status_code == 409  # nothing records now
    finally:
        server.stop()
        sim.close()

    ds = LeRobotDataset("local/so101-wave", root=target)
    assert ds.meta.total_episodes == 2 and ds.fps == 20 and ds.meta.robot_type == "so101_follower"
    assert ds.num_frames == sum(e["frames"] for e in st["episodes"]) >= 18
    assert set(ds.meta.video_keys) == {"observation.images.front", "observation.images.top"}
    f = ds[0]
    assert tuple(f["observation.images.front"].shape) == (3, 120, 160)
    assert tuple(f["observation.images.top"].shape) == (3, 64, 96)
    assert f["task"] == "wave" and f["action"].shape == (6,) and f["observation.state"].shape == (6,)
    actions = np.stack([ds[i]["action"].numpy() for i in range(ds.num_frames)])
    assert np.ptp(actions[:, 0]) > 1.0  # the slider's pan sweep is in the actions (LeRobot units, not rad)
    assert json.loads((target / "meta" / "units.json").read_text())["units"] == "lerobot"


def test_bad_bodies_are_400_or_404_and_nothing_is_written(tmp_path: Path) -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    server = Dashboard(sim).start()
    try:
        u = server.url
        ok = {"robots": ["so101"], "target": str(tmp_path / "d"), "fps": 30}
        assert httpx.post(f"{u}/api/record/start", json={**ok, "robots": []}).status_code == 400
        assert httpx.post(f"{u}/api/record/start", json={**ok, "fps": 29.5}).status_code == 400
        assert httpx.post(f"{u}/api/record/start", json={**ok, "robots": ["nope"]}).status_code == 404
        r = httpx.post(f"{u}/api/record/start", json={**ok, "cameras": {"so101": ["nope"]}})
        assert r.status_code == 404 and "camera" in r.json()["detail"]
        assert httpx.post(f"{u}/api/record/episode/explode", json={}).status_code == 404
        assert httpx.get(f"{u}/api/record/status").json()["state"] == "idle"
        assert not (tmp_path / "d").exists()
    finally:
        server.stop()
        sim.close()


def test_several_robots_record_one_dataset_each() -> None:
    assert targets_for("/tmp/x", ["a"]) == {"a": "/tmp/x"}
    assert targets_for("/tmp/x", ["a", "b"]) == {"a": "/tmp/x/a", "b": "/tmp/x/b"}
    assert targets_for("hf://me/set", ["a", "b"]) == {"a": "hf://me/set-a", "b": "hf://me/set-b"}
