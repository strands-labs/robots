"""A dataset directory no other account can move: a symlink swapped in on its path, or a directory others can write
on it, is refused before anything is written, and an aborted take deletes only the directory it created.

LeRobot is not needed: a stand-in dataset writes by path, as ``LeRobotDataset`` does."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.recording import Episode, lerobot_v3

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason="directory ownership is POSIX")


class _Dataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.meta = SimpleNamespace(total_episodes=0, features={})

    @classmethod
    def create(cls, repo_id, *, fps, features, root, robot_type, use_videos):
        Path(root).mkdir(parents=True, exist_ok=False)
        (Path(root) / "meta").mkdir()
        (Path(root) / "meta" / "info.json").write_text(json.dumps({"fps": fps}))
        return cls(Path(root))

    def clear_episode_buffer(self) -> None:
        pass

    def finalize(self) -> None:
        pass


@pytest.fixture
def lerobot(monkeypatch):
    monkeypatch.setattr(lerobot_v3, "require_lerobot", lambda: _Dataset)
    monkeypatch.setattr(lerobot_v3, "features_from", lambda first, keys: {})
    monkeypatch.setattr(lerobot_v3, "quiet", contextlib.nullcontext)


@pytest.fixture
def places(tmp_path):
    base, other = tmp_path / "recordings", tmp_path / "other"
    (base / "a").mkdir(parents=True)
    other.mkdir()
    return base, other


def _episode(target: Path) -> Episode:
    description = SimpleNamespace(name="so101", action_keys=[], hardware=SimpleNamespace(lerobot_type="so101"))
    return Episode(target, description=description, hz=30, instruction=None, units=None)  # type: ignore[arg-type]


def test_a_directory_swapped_for_a_symlink_after_the_check_is_refused_before_writing(lerobot, places):
    base, other = places
    ep = _episode(base / "a" / "set")  # the path is checked and resolved here
    shutil.rmtree(base / "a")
    os.symlink(other, base / "a")
    with pytest.raises(PermissionError, match="symlink"):
        ep.open(lambda: None)
    assert list(other.iterdir()) == []


def test_an_aborted_take_keeps_a_directory_swapped_in_under_its_name(lerobot, places):
    base, other = places
    (other / "set").mkdir()
    (other / "set" / "keep.bin").write_text("x")
    ep = _episode(base / "a" / "set")
    ep.open(lambda: None)
    os.rename(base / "a", base / "a-was")
    os.symlink(other, base / "a")
    ep.abort()
    assert (other / "set" / "keep.bin").read_text() == "x"
    assert (base / "a-was" / "set").is_dir()  # its own, left where it went


def test_a_directory_other_accounts_can_write_on_the_path_is_refused(lerobot, places):
    base, _ = places
    base.chmod(0o775)
    with pytest.raises(PermissionError, match=r"another account.*chmod go-w"):
        _episode(base / "a" / "set").open(lambda: None)
    assert not (base / "a" / "set").exists()


def test_a_sticky_shared_directory_holding_a_directory_of_ours_records(lerobot, places):
    base, _ = places
    base.chmod(0o1777)
    ep = _episode(base / "a" / "set")
    ep.open(lambda: None)
    assert (base / "a" / "set" / "meta" / "info.json").is_file()


def test_missing_parents_are_created_and_an_aborted_take_removes_its_own_directory(lerobot, places):
    base, _ = places
    ep = _episode(base / "new" / "deeper" / "set")
    ep.open(lambda: None)
    assert (base / "new" / "deeper" / "set" / "meta" / "info.json").is_file()
    ep.abort()
    assert not (base / "new" / "deeper" / "set").exists()
    assert (base / "new" / "deeper").is_dir()
