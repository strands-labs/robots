"""An agent's run on a real body: the policy files the operator approves are the files the run reads."""

import json

import numpy as np
import pytest

import strands_robots.policy.dataset as dataset
from strands_robots import Robot
from strands_robots.robot.run import from_agent
from strands_robots.robot.run.common import files_digest
from strands_robots.robot.safety import RequestRefused


@pytest.fixture
def recording(tmp_path, monkeypatch):
    """A replay dataset whose actions live in ``a.json`` (the parquet reader is LeRobot's)."""
    actions = lambda root, ep: np.asarray(json.loads((root / "a.json").read_text()))  # noqa: E731
    monkeypatch.setattr(dataset, "_episode_actions", actions)
    keys = Robot("so101", driver="memory").description.action_keys
    (tmp_path / "rec" / "meta").mkdir(parents=True)
    info = {"fps": 30, "features": {"action": {"names": [f"{k}.pos" for k in keys]}}}
    (tmp_path / "rec" / "meta" / "info.json").write_text(json.dumps(info))
    (tmp_path / "rec" / "meta" / "units.json").write_text(json.dumps({"units": "rad"}))
    (tmp_path / "rec" / "a.json").write_text(json.dumps([[0.0] * len(keys)] * 3))
    return tmp_path / "rec", len(keys)


def _positions(robot):
    return np.asarray(robot.observe().joints.positions)


def test_the_plan_carries_the_digest_of_the_files_and_an_unchanged_run_goes_ahead(recording):
    rec, _ = recording
    plans = []
    robot = Robot("so101", driver="memory", approve=lambda p: plans.append(p) or True)
    with from_agent():
        res = robot.run(str(rec), steps=3)
    assert res.steps == 3
    assert plans[0].files == files_digest(str(rec)) and len(plans[0].files) == 64


def test_files_changed_while_the_operator_was_asked_refuse_the_run_and_move_nothing(recording):
    rec, n = recording
    robot = Robot("so101", driver="memory")
    before = _positions(robot)

    def approve(plan):
        (rec / "a.json").write_text(json.dumps([[0.4] * n] * 3))  # replaced while the dialog is open
        return True

    robot.approve = approve
    with from_agent(), pytest.raises(RequestRefused, match="changed on disk after the operator approved"):
        robot.run(str(rec), steps=3)
    assert np.allclose(_positions(robot), before)


def test_a_policy_not_on_disk_has_no_files_to_pin(tmp_path):
    assert files_digest(lambda obs: {}) is None
    assert files_digest("lerobot/act_so101_not_here") is None
    assert files_digest(str(tmp_path / "missing")) is None


def test_the_digest_covers_names_and_bytes_and_ignores_the_query(tmp_path):
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "w.bin").write_bytes(b"one")
    first = files_digest(str(tmp_path / "ck") + "?episode=0")
    (tmp_path / "ck" / "w.bin").write_bytes(b"two")
    assert files_digest(str(tmp_path / "ck")) != first
    (tmp_path / "ck" / "w.bin").write_bytes(b"one")
    assert files_digest(str(tmp_path / "ck")) == first
    (tmp_path / "ck" / "w.bin").rename(tmp_path / "ck" / "x.bin")
    assert files_digest(str(tmp_path / "ck")) != first


def test_the_digest_reads_through_a_symlinked_folder_and_survives_a_cycle(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "w.bin").write_bytes(b"one")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "sub").symlink_to(tmp_path / "real")
    (tmp_path / "real" / "loop").symlink_to(tmp_path / "ck")
    first = files_digest(str(tmp_path / "ck"))
    (tmp_path / "real" / "w.bin").write_bytes(b"two")
    assert files_digest(str(tmp_path / "ck")) != first
