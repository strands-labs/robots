"""An agent names a Hub checkpoint only if the operator listed it in ``STRANDS_POLICY_REPOS``.

Loading a LeRobot checkpoint rebuilds the processor steps its own JSON names, and some of those load further repos
with remote code enabled: the repo is code its publisher chose. Refused before ``attach``, so nothing is fetched.
"""

from __future__ import annotations

import pytest
from scripted_model import ScriptedModel
from strands import Agent

from strands_robots import Policy, Robot
from strands_robots.policy.factory import spec_help
from strands_robots.policy.hub import allowed_repos, hub_repo
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
    monkeypatch.delenv("STRANDS_POLICY_REPOS", raising=False)
    sim = Simulation()
    arm = Robot("so101", sim=sim, id="a")
    yield sim, arm, loads
    sim.close()


def _turn(tool, call):
    agent = Agent(model=ScriptedModel([(tool.tool_name, call)]), tools=[tool], callback_handler=None)
    agent("go")
    (r,) = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
    return r["status"], r["content"][0]["text"]


def test_an_agent_cannot_load_a_hub_checkpoint_the_operator_did_not_list(world):
    sim, arm, loads = world
    status, text = _turn(arm.tool(), {"rpc": "run", "run": {"policy": "someone/ckpt", "steps": 1}})
    assert status == "error" and "STRANDS_POLICY_REPOS" in text and "someone/ckpt" in text
    status, text = _turn(sim.tool, {"rpc": "run", "run": {"policies": {"a": "someone/ckpt"}, "steps": 1}})
    assert status == "error" and "STRANDS_POLICY_REPOS" in text
    with from_agent(), pytest.raises(PermissionError, match="STRANDS_POLICY_REPOS"):  # every entry point: bind()
        arm.run("someone/ckpt", steps=1)
    with from_agent(confirmed=True), pytest.raises(PermissionError, match="has not allowed"):  # a yes is for motion
        arm.run("someone/ckpt", steps=1)
    assert loads == []


@pytest.mark.parametrize("listed", ["someone/ckpt", "SomeOne/*", " other/x , someone/ckpt "])
def test_a_listed_repo_or_publisher_loads(world, monkeypatch, listed):
    sim, arm, loads = world
    monkeypatch.setenv("STRANDS_POLICY_REPOS", listed)
    with from_agent(), pytest.raises(_Loaded):
        arm.run("someone/ckpt", steps=1)
    with from_agent(), pytest.raises(PermissionError):
        arm.run("someone/other" if "*" not in listed else "nobody/ckpt", steps=1)
    assert loads == ["someone/ckpt"]


def test_the_developer_and_a_registered_handle_are_not_checked(world):
    sim, arm, loads = world
    with pytest.raises(_Loaded):
        arm.run("someone/ckpt", steps=1)
    status, _ = _turn(arm.tool(policies={"pick": Policy("someone/pick")}), {"rpc": "run", "run": {"steps": 1}})
    assert status == "error" and loads == ["someone/ckpt", "someone/pick"]


def test_only_a_download_is_a_hub_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local" / "ckpt").mkdir(parents=True)
    (tmp_path / "local" / "ckpt" / "config.json").write_text("{}")
    assert hub_repo("local/ckpt") is None  # a checkpoint on this disk, as make() reads it
    assert hub_repo("zenoh://h:7447/d") is None and hub_repo(lambda obs: obs) is None
    assert hub_repo("owner/repo") == "owner/repo"


@pytest.mark.parametrize("bad", ["*", "owner", "https://hf.co/a/b", "a/b/c", "*/x"])
def test_a_malformed_entry_is_refused_not_a_wildcard(monkeypatch, bad):
    monkeypatch.setenv("STRANDS_POLICY_REPOS", f"ok/one,{bad}")
    with pytest.raises(ValueError, match="STRANDS_POLICY_REPOS"):
        allowed_repos()


def test_the_model_is_told_only_the_allowed_repos(monkeypatch):
    monkeypatch.delenv("STRANDS_POLICY_REPOS", raising=False)
    assert "Hugging Face" not in spec_help()
    monkeypatch.setenv("STRANDS_POLICY_REPOS", "lerobot/smolvla_base")
    assert "lerobot/smolvla_base" in spec_help() and "STRANDS_POLICY_REPOS" in spec_help()


def test_a_dashboard_run_under_an_agent_is_refused_before_anything_loads(world):
    from strands_robots.dashboard import runs

    _, arm, loads = world
    with from_agent(), pytest.raises(PermissionError, match="STRANDS_POLICY_REPOS"):
        runs.start(arm, "someone/ckpt", instruction=None, seconds=1, hz=None, confirmed=False)
    assert loads == []
