"""Recording under a LeRobot checkpoint stores what the policy saw and did, in the checkpoint's units, and says so."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from strands_robots import Policy, Robot
from strands_robots.units import semantics_for, units_for_semantics

pytest.importorskip("lerobot.remote_inference", reason="needs the lerobot extra (Python >= 3.12)")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "policy"))
from test_record_episodes import reload  # noqa: E402
from tiny_act import save_tiny_act  # noqa: E402


def test_a_checkpoint_run_records_in_its_units_and_the_dataset_says_which(tmp_path):
    ckpt = save_tiny_act(tmp_path / "act", 6, camera="front")
    arm = Robot("so101")
    arm.sim.camera("front", size=(64, 48), attach=arm)
    seen: list[np.ndarray] = []
    target = tmp_path / "act-run"
    r = arm.run(
        Policy(str(ckpt), device="cpu"),
        hz=30,
        steps=15,
        record=target,
        stop_when=lambda o: seen.append(o.joints.positions.copy()) and False,
    )
    assert r.recorded == str(target) and r.steps == 15
    units = units_for_semantics(semantics_for(arm.description), arm.description)
    assert units is not None and units.name == "lerobot"

    import pandas as pd

    data = pd.concat(pd.read_parquet(p) for p in sorted((target / "data").rglob("*.parquet")))
    state = np.stack(data["observation.state"].to_numpy())
    action = np.stack(data["action"].to_numpy())
    np.testing.assert_allclose(state, units.from_rad(np.stack(seen[:15])), atol=1e-3)
    assert np.all(state[:, :5] >= -100 - 1e-3) and np.all(state[:, :5] <= 100 + 1e-3)
    assert np.all(action >= units.lo_u - 1e-3) and np.all(action <= units.hi_u + 1e-3)
    assert not np.allclose(state, np.stack(seen[:15]), atol=1e-2)  # not radians

    said = json.loads((target / "meta" / "units.json").read_text())
    assert said["units"] == "lerobot"
    names = [f"{k}.pos" for k in arm.action_keys]
    assert list(said["joints"]) == names
    assert said["joints"]["gripper.pos"]["range"] == [0.0, 100.0]
    assert said["joints"]["shoulder_pan.pos"]["range"] == [-100.0, 100.0]
    assert said["joints"]["shoulder_pan.pos"]["rad"] == list(arm.description.limits["shoulder_pan"])
    assert reload(target)["frames"] == 15  # LeRobot reads it back alongside our file

    rad_target = tmp_path / "rad-run"
    arm.run(lambda o: o.joints.positions, hz=30, steps=3, record=rad_target)
    assert json.loads((rad_target / "meta" / "units.json").read_text())["units"] == "rad"
    with pytest.raises(ValueError, match="units"):
        arm.run(lambda o: o.joints.positions, hz=30, steps=3, record=target)  # radians into a lerobot-units dataset
    print(f"[measured] checkpoint run: 15 frames in 'lerobot' units, state[0] {np.round(state[0], 2).tolist()}")


def test_a_chunked_checkpoint_records_every_camera_frame_without_stop_when(tmp_path):
    ckpt = save_tiny_act(tmp_path / "act", 6, camera="front")
    arm = Robot("so101")
    arm.sim.camera("front", size=(64, 48), attach=arm)
    target = tmp_path / "chunked"
    r = arm.run(Policy(str(ckpt), device="cpu"), hz=30, steps=15, record=target)  # replay steps skip the render
    assert r.recorded == str(target) and r.steps == 15
    assert reload(target)["frames"] == 15
