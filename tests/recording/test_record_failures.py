"""What a recording does when a run does not end well, and what recording costs a run that does not record."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest
from test_record_episodes import reload  # noqa: E402

from strands_robots import Robot
from strands_robots.policy import SequencePolicy
from strands_robots.policy.base import PolicyFault

pytest.importorskip("lerobot.datasets.lerobot_dataset")


def _arm(cams=1):
    arm = Robot("so101")
    for i in range(cams):
        arm.sim.camera(f"c{i}", position=[0.4 + 0.2 * i, -0.3, 0.4], target=[0, 0, 0.1], size=(64, 48), attach=arm)
    return arm


def _failing(after: int, exc: BaseException):
    n = {"i": 0}

    def policy(obs):
        n["i"] += 1
        if n["i"] > after:
            raise exc
        return obs.joints.positions

    return policy


def test_a_top_level_script_records_two_cameras(tmp_path):
    """No ``if __name__ == "__main__":`` guard: encoding must not need a process pool that re-imports the script."""
    script = tmp_path / "record_me.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from strands_robots import Robot
            from strands_robots.policy import SequencePolicy
            arm = Robot("so101")
            arm.sim.camera("wrist", size=(64, 48), attach=arm)
            arm.sim.camera("top", position=[0.6, 0, 0.5], size=(64, 48), attach=arm)
            r = arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=5, record=sys.argv[1])
            print("RECORDED", r.recorded)
            """
        )
    )
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "ds")], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-1500:]
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1 and info["total_frames"] == 5
    assert {"observation.images.wrist", "observation.images.top"} <= set(info["features"])


def test_a_run_that_raises_records_nothing_and_leaves_the_dataset_usable(tmp_path):
    arm, target = _arm(), tmp_path / "ds"
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=10, record=target)
    with pytest.raises(RuntimeError, match="boom"):
        arm.run(_failing(5, RuntimeError("boom")), hz=20, steps=10, record=target)
    arm.run(SequencePolicy([{"shoulder_pan": -0.1}]), hz=20, steps=10, record=target)
    got = reload(target)
    assert got["episodes"] == 2 and got["frames"] == 20 and got["episode_index"] == [0, 1]


def test_a_policy_fault_is_not_saved_as_an_episode(tmp_path):
    arm, target = _arm(), tmp_path / "ds"
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=6, record=target)
    r = arm.run(_failing(3, PolicyFault("lost the server")), hz=20, steps=10, record=target)
    assert r.stopped_by == "fault" and r.recorded is None
    assert reload(target)["frames"] == 6


def test_a_faulted_first_run_leaves_no_dataset_behind(tmp_path):
    r = _arm().run(_failing(2, PolicyFault("x")), hz=20, steps=10, record=tmp_path / "ds")
    assert r.stopped_by == "fault" and r.recorded is None and not (tmp_path / "ds").exists()


def test_lerobot_is_imported_only_when_a_run_records():
    code = textwrap.dedent(
        """
        import sys
        from strands_robots import Robot
        import strands_robots.recording
        Robot("so101").run(lambda o: o.joints.positions, hz=20, steps=3)
        assert not [m for m in sys.modules if m.split(".")[0] == "lerobot"], "lerobot imported without record="
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-1500:]


def test_stop_waits_for_a_slow_save_and_the_episode_is_kept(tmp_path, monkeypatch):
    """A save longer than any join cap still lands: ``stop()`` returns the result, the episode is counted."""
    import time

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    save = LeRobotDataset.save_episode

    def slow_save(self, *a, **kw):
        time.sleep(6.0)
        return save(self, *a, **kw)

    monkeypatch.setattr(LeRobotDataset, "save_episode", slow_save)
    arm = _arm()
    arm.start(lambda o: o.joints.positions, hz=20, seconds=600, record=tmp_path / "ds", realtime=True)
    time.sleep(1.0)
    r = arm.stop()
    assert r is not None and r.stopped_by == "stopped" and r.recorded == str(tmp_path / "ds")
    assert json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())["total_episodes"] == 1
