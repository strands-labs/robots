"""The page names where a dataset goes, so it may name only a place inside the recordings directory: a dataset is
created or resumed there, and an aborted take deletes it."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard, takes  # noqa: E402


@pytest.fixture
def dash():
    sim = Simulation()
    Robot("so101", sim=sim)
    yield Dashboard(sim)
    sim.close()


def _plan(dash, target, robots=("so101",)):
    return takes.plan(dash, list(robots), {}, target, 30, None, None)


@pytest.mark.parametrize("target", ["/etc/lerobot-set", "../outside", "a/../../outside", "~/somewhere-else", "."])
def test_a_target_outside_the_recordings_directory_is_refused(dash, tmp_path, target):
    with pytest.raises(PermissionError, match="records only inside"):
        _plan(dash, target)


def test_a_symlink_out_of_the_recordings_directory_is_refused(dash, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(PermissionError):
        _plan(dash, "link/set")


def test_names_inside_it_and_hub_names_are_kept(dash, tmp_path):
    assert _plan(dash, "wave")[0].target == str((tmp_path / "wave").resolve())
    assert _plan(dash, str(tmp_path / "a" / "b"))[0].target == str((tmp_path / "a" / "b").resolve())
    assert _plan(dash, "hf://me/set")[0].target == "hf://me/set"


def test_the_default_target_is_inside_it(dash, tmp_path):
    assert takes.options(dash)["target"].startswith(str(tmp_path))
