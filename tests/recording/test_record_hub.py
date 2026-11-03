"""``record="hf://owner/name"`` is a Hub repo id in LeRobot's cache; nothing is uploaded without ``push=True``."""

from __future__ import annotations

import pytest

from strands_robots import Robot
from strands_robots.policy import SequencePolicy
from strands_robots.recording import is_repo_id, repo_id

pytest.importorskip("lerobot.datasets.lerobot_dataset")


@pytest.fixture
def no_hub(tmp_path, monkeypatch):
    """LeRobot's cache in tmp, the Hub offline, and every upload counted instead of sent."""
    import lerobot.utils.constants as constants
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_path / "cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pushed: list[str] = []
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", lambda self, *a, **k: pushed.append(self.repo_id))
    return tmp_path / "cache", pushed


def test_only_hf_prefixed_strings_are_repo_ids(tmp_path):
    assert is_repo_id("hf://cagataycali/so101-waves")
    assert not is_repo_id("cagataycali/so101-waves") and not is_repo_id("data/duck") and not is_repo_id(tmp_path)
    assert repo_id("hf://cagataycali/so101-waves") == "cagataycali/so101-waves"
    for bad in ("hf://nope", "hf://a/b/c", "hf:///b", "hf://a/"):
        with pytest.raises(ValueError, match="hf://owner/name"):
            repo_id(bad)


def test_a_hub_target_records_into_lerobots_cache_and_uploads_nothing(no_hub):
    cache, pushed = no_hub
    arm = Robot("so101")
    r = arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=20, steps=4, record="hf://owner/name")
    assert r.recorded == str(cache / "owner" / "name") and (cache / "owner" / "name" / "meta" / "info.json").is_file()
    assert pushed == []
    arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=20, steps=4, record="hf://owner/name", push=True)
    assert pushed == ["owner/name"]  # the one explicit push, counted by the stub


def test_push_needs_a_hub_target_and_is_refused_before_step_one(tmp_path, no_hub):
    arm = Robot("so101")
    with pytest.raises(ValueError, match="hf://owner/name"):
        arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=20, steps=2, record=tmp_path / "d", push=True)
    with pytest.raises(ValueError, match="record="):
        arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=20, steps=2, push=True)
    with pytest.raises(ValueError, match="hf://owner/name"):
        arm.run(SequencePolicy([{"shoulder_pan": 0.2}]), hz=20, steps=2, record="hf://nope")
    assert not (tmp_path / "d").exists() and no_hub[1] == []
