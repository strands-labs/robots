"""An agent loads a policy folder from this disk only if no other user could have written it.

The agent picks the path, but the bytes are whoever could write them: a checkpoint rebuilds what its JSON names and
a dataset is the motion a robot replays. Refused before ``attach``, on every entry point, while the operator's own
folders (and the developer's own ``Policy(path)``) load as before.
"""

from __future__ import annotations

import json
import os

import pytest
from scripted_model import ScriptedModel
from strands import Agent

from strands_robots import Robot
from strands_robots.policy.trust import refuse_untrusted_local
from strands_robots.robot.run.pin import files_digest
from strands_robots.robot.run.source import from_agent
from strands_robots.simulation import Simulation


class _Loaded(Exception):
    pass


@pytest.fixture
def world(monkeypatch):
    from strands_robots.policy.lerobot import local

    loads: list[str] = []

    def attach(self, robot):
        loads.append(self.checkpoint)
        raise _Loaded(self.checkpoint)

    monkeypatch.setattr(local.LeRobotPolicy, "attach", attach)
    sim = Simulation()
    arm = Robot("so101", sim=sim, id="a")
    yield sim, arm, loads
    sim.close()


def _ckpt(folder, mode=0o755, file_mode=0o644):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"type": "act"}))
    os.chmod(folder / "config.json", file_mode)
    os.chmod(folder, mode)
    return folder


def _turn(tool, call):
    agent = Agent(model=ScriptedModel([(tool.tool_name, call)]), tools=[tool], callback_handler=None)
    agent("go")
    (r,) = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
    return r["status"], r["content"][0]["text"]


def test_a_folder_anyone_may_write_is_refused_on_every_entry_point(world, tmp_path):
    from strands_robots.dashboard import runs

    sim, arm, loads = world
    planted = str(_ckpt(tmp_path / "planted", mode=0o777))
    status, text = _turn(arm.tool(), {"rpc": "run", "run": {"policy": planted, "steps": 1}})
    assert status == "error" and "writable by group or others" in text
    status, _ = _turn(sim.tool, {"rpc": "run", "run": {"policies": {"a": planted}, "steps": 1}})
    assert status == "error"
    with from_agent(confirmed=True), pytest.raises(PermissionError, match="no other user could have written"):
        arm.run(planted, steps=1)
    with from_agent(), pytest.raises(PermissionError):
        runs.start(arm, planted, instruction=None, seconds=1, hz=None, confirmed=False)
    assert loads == []


def _file_open(tmp_path):
    return _ckpt(tmp_path / "f", file_mode=0o666)


def _parent_open(tmp_path):
    folder = _ckpt(tmp_path / "open" / "c")
    os.chmod(tmp_path / "open", 0o777)  # whoever may write the parent may rename the folder and put another there
    return folder


def _link_to_open(tmp_path):
    target = _ckpt(tmp_path / "t", mode=0o777)
    (tmp_path / "mine").mkdir()
    os.symlink(target, tmp_path / "mine" / "ckpt")
    return tmp_path / "mine" / "ckpt"


def _dataset_open(tmp_path):
    (tmp_path / "d" / "meta").mkdir(parents=True)
    (tmp_path / "d" / "meta" / "info.json").write_text("{}")
    os.chmod(tmp_path / "d" / "meta", 0o777)
    return tmp_path / "d"


@pytest.mark.parametrize("make", [_file_open, _parent_open, _link_to_open, _dataset_open])
def test_a_writable_file_parent_symlink_target_or_dataset_is_refused(make, tmp_path):
    with from_agent(), pytest.raises(PermissionError, match="writable by group or others"):
        refuse_untrusted_local(str(make(tmp_path)))


def test_a_folder_another_user_owns_is_refused(world, tmp_path, monkeypatch):
    _, arm, loads = world
    mine = str(_ckpt(tmp_path / "ckpt"))
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)  # the agent runs as someone else: the folder is theirs
    with from_agent(), pytest.raises(PermissionError, match="is owned by uid"):
        arm.run(mine, steps=1)
    assert loads == []


def test_the_operators_own_folder_loads_also_under_a_sticky_shared_folder(world, tmp_path):
    _, arm, loads = world
    mine = _ckpt(tmp_path / "ckpt")
    shared = tmp_path / "shared"
    shared.mkdir()
    in_sticky = _ckpt(shared / "ckpt")
    os.chmod(shared, 0o1777)  # like /tmp: no one renames or replaces an entry in it they do not own
    for spec in (mine, in_sticky):
        with from_agent(), pytest.raises(_Loaded):
            arm.run(str(spec), steps=1)
    os.chmod(in_sticky, 0o1777)  # the policy folder itself never: anyone could add a file the loader reads
    with from_agent(), pytest.raises(PermissionError):
        arm.run(str(in_sticky), steps=1)
    assert loads == [str(mine), str(in_sticky)]


def test_the_developers_own_path_is_not_checked(world, tmp_path):
    _, arm, loads = world
    planted = str(_ckpt(tmp_path / "planted", mode=0o777))
    with pytest.raises(_Loaded):
        arm.run(planted, steps=1)
    assert loads == [planted]


def test_the_files_an_operator_approves_are_checked_on_the_open_file(tmp_path):
    folder = _ckpt(tmp_path / "ckpt", file_mode=0o666)
    with pytest.raises(PermissionError, match="writable by group or others"):
        files_digest(str(folder))


@pytest.mark.parametrize("private", [True, False], ids=["private-group", "shared-group"])
def test_a_group_write_bit_counts_only_for_a_group_others_are_in(tmp_path, monkeypatch, private):
    """umask 002 with a private group per user (Debian/Ubuntu) writes 0664/0775: the group is the owner alone."""
    import grp
    import pwd
    from types import SimpleNamespace

    me = pwd.getpwuid(os.getuid())
    group = SimpleNamespace(gr_gid=me.pw_gid, gr_name=me.pw_name if private else "staff", gr_mem=[])
    monkeypatch.setattr(grp, "getgrgid", lambda gid: group)
    folder = _ckpt(tmp_path / "ckpt", mode=0o775, file_mode=0o664)
    for p in (tmp_path, *tmp_path.parents):
        if os.stat(p).st_mode & 0o022 and not os.stat(p).st_mode & 0o1000 and os.stat(p).st_uid == os.getuid():
            pytest.skip(f"{p} is group-writable on this machine")
    if private:
        refuse_untrusted_local(str(folder))
    else:
        with pytest.raises(PermissionError, match="writable by group or others"):
            refuse_untrusted_local(str(folder))
