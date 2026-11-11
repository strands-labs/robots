"""An agent's run on a real body: the policy files the operator approves are the files the run reads."""

import hashlib
import json
import os

import numpy as np
import pytest

import strands_robots.policy.dataset as dataset
from strands_robots import Robot
from strands_robots.policy.factory import local_root
from strands_robots.robot.run import from_agent, pin
from strands_robots.robot.run.pin import files_digest
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


def _unpin(rec, cwd):
    (rec / "meta" / "info.json").unlink()  # the approved folder is no longer one the loader reads


def _appear(rec, cwd):
    (cwd / "org").mkdir()
    rec.rename(cwd / "org" / "rec")  # a folder now shadows the Hub id the operator approved


@pytest.mark.parametrize(
    "change", [_unpin, _appear], ids=["folder-stops-being-a-dataset", "folder-appears-for-a-hub-id"]
)
def test_a_spec_that_changes_kind_while_the_operator_was_asked_is_refused(recording, monkeypatch, tmp_path, change):
    rec, n = recording
    monkeypatch.setattr(pin, "_commit", lambda repo, revision: "c" * 40)  # the Hub id resolves; no network
    monkeypatch.setenv("STRANDS_POLICY_REPOS", "org/rec")  # and the operator lists it
    (rec / "a.json").write_text(json.dumps([[0.4] * n] * 3))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    spec = str(rec) if change is _unpin else "org/rec"
    robot = Robot("so101", driver="memory")
    before = _positions(robot)
    robot.approve = lambda plan: change(rec, cwd) or True
    with from_agent(), pytest.raises(RequestRefused, match="changed on disk after the operator approved"):
        robot.run(spec, steps=3)
    assert np.allclose(_positions(robot), before)


def test_a_policy_not_on_disk_has_no_files_to_pin_and_a_hub_id_pins_its_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(pin, "_commit", lambda repo, revision: "c" * 40)
    assert files_digest(lambda obs: {}) is None
    assert files_digest("lerobot/act_so101_not_here") == "lerobot/act_so101_not_here@" + "c" * 40
    assert files_digest(str(tmp_path / "missing")) is None
    assert files_digest(str(tmp_path)) is None  # a folder the loader would refuse: nothing it reads to pin


def test_the_digest_covers_names_and_bytes_and_ignores_the_query(tmp_path):
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "config.json").write_text("{}")  # a local checkpoint: a folder the loader reads
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
    (tmp_path / "ck" / "config.json").write_text("{}")
    (tmp_path / "ck" / "sub").symlink_to(tmp_path / "real")
    (tmp_path / "real" / "loop").symlink_to(tmp_path / "ck")
    first = files_digest(str(tmp_path / "ck"))
    (tmp_path / "real" / "w.bin").write_bytes(b"two")
    assert files_digest(str(tmp_path / "ck")) != first


def test_the_digest_needs_nothing_newer_than_python_3_10(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_bytes(b"one")
    one = hashlib.sha256(b"one").hexdigest()
    want = hashlib.sha256(f"config.json\0{one}\n".encode()).hexdigest()
    monkeypatch.delattr(hashlib, "file_digest", raising=False)  # as on 3.10
    assert files_digest(str(tmp_path)) == want


def test_a_file_swapped_only_while_the_policy_loads_never_reaches_the_robot(recording, monkeypatch):
    rec, n = recording
    read = dataset._episode_actions

    def swap_around_the_read(root, ep):  # replaced just before the read, put back just after it
        (rec / "a.json").write_text(json.dumps([[0.4] * n] * 3))
        out = read(root, ep)
        (rec / "a.json").write_text(json.dumps([[0.0] * n] * 3))
        return out

    monkeypatch.setattr(dataset, "_episode_actions", swap_around_the_read)
    robot = Robot("so101", driver="memory", approve=lambda p: True)
    with from_agent():
        robot.run(str(rec), steps=3)
    assert np.allclose(_positions(robot), 0.0)


def test_the_private_copy_is_removed_when_the_run_ends(recording, monkeypatch, tmp_path):
    rec, _ = recording
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "t"))
    (tmp_path / "t").mkdir()
    robot = Robot("so101", driver="memory", approve=lambda p: True)
    with from_agent():
        robot.run(str(rec), steps=3)
    assert list((tmp_path / "t").iterdir()) == []


def test_an_approved_preset_never_runs_a_same_named_folder_in_the_cwd(recording, monkeypatch, tmp_path):
    rec, n = recording
    (rec / "a.json").write_text(json.dumps([[0.4] * n] * 3))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    rec.rename(cwd / "microduck")  # a dataset that happens to share a preset's name
    monkeypatch.chdir(cwd)
    assert local_root("microduck") is None and files_digest("microduck") is None
    robot = Robot("so101", driver="memory", approve=lambda p: True)
    before = _positions(robot)
    with from_agent(), pytest.raises(ValueError, match="Microduck"):  # the preset loads (and needs a duck's body)
        robot.run("microduck", steps=3)
    assert np.allclose(_positions(robot), before)


def test_a_file_swapped_while_the_private_copy_is_taken_is_refused(recording, monkeypatch):
    rec, n = recording
    listing = pin.walk

    def swap_before_the_copy_reads(folder):
        (rec / "a.json").write_text(json.dumps([[0.4] * n] * 3))
        return listing(folder)

    robot = Robot("so101", driver="memory")
    robot.approve = lambda p: monkeypatch.setattr(pin, "walk", swap_before_the_copy_reads) or True
    with from_agent(), pytest.raises(RequestRefused, match="changed on disk"):
        robot.run(str(rec), steps=3)
    assert np.allclose(_positions(robot), 0.0)


def test_a_fifo_and_a_folder_cycle_are_copied_exactly_as_they_are_hashed(recording):
    rec, _ = recording
    os.mkfifo(rec / "meta" / "pipe")  # not a policy file: neither hashed nor copied, so nothing blocks on it
    (rec / "meta" / "loop").symlink_to(rec, target_is_directory=True)
    robot = Robot("so101", driver="memory", approve=lambda p: True)
    with from_agent():
        assert robot.run(str(rec), steps=3).steps == 3


@pytest.mark.parametrize("fail", ["ctrl-c in the load", "an error after the load"])
def test_the_private_copy_is_removed_however_the_run_ends(recording, monkeypatch, tmp_path, fail):
    rec, _ = recording
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "t"))
    (tmp_path / "t").mkdir()
    if fail == "ctrl-c in the load":
        monkeypatch.setattr(dataset, "_episode_actions", lambda root, ep: (_ for _ in ()).throw(KeyboardInterrupt()))
    robot = Robot("so101", driver="memory", approve=lambda p: True)
    with from_agent(), pytest.raises(KeyboardInterrupt if fail == "ctrl-c in the load" else ValueError):
        robot.run(str(rec), steps=3, hz=10)  # the recording runs at 30 Hz: refused after it is loaded
    assert list((tmp_path / "t").iterdir()) == []


def test_under_umask_002_the_private_copy_is_still_this_users_alone(recording):
    """The copy an approved run loads is made 0700/0600 whatever the umask, so the trust check on it passes."""
    rec, _ = recording
    for path in (rec, *rec.rglob("*")):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    old = os.umask(0o002)
    try:
        robot = Robot("so101", driver="memory", approve=lambda p: True)
        with from_agent():
            assert robot.run(str(rec), steps=3).steps == 3
    finally:
        os.umask(old)


@pytest.mark.parametrize("where", ["the folder", "a folder inside it"])
def test_an_approved_real_run_checks_the_original_folder_not_only_the_copy(recording, where):
    """The folder walk runs on what the operator approved, so a folder another user could write is refused."""
    rec, _ = recording
    os.chmod(rec if where == "the folder" else next(p for p in rec.rglob("*") if p.is_dir()), 0o777)
    asked = []
    robot = Robot("so101", driver="memory", approve=lambda p: asked.append(p) or True)
    with from_agent(), pytest.raises(PermissionError, match="writable by group or others"):
        robot.run(str(rec), steps=3)
    assert asked == []  # refused before the operator is asked
