"""An agent's run on a real body loads the bytes the operator approved, for a registered policy object and for a
Hugging Face Hub checkpoint too: a LeRobot checkpoint reads its files when it attaches, after the yes."""

import json
from types import SimpleNamespace

import pytest

from strands_robots import Policy, Robot
from strands_robots.policy.lerobot import local
from strands_robots.robot.run import from_agent
from strands_robots.robot.safety import RequestRefused

A, B = "a" * 40, "b" * 40


class _Loaded(Exception):
    pass


@pytest.fixture
def loads(monkeypatch):
    """``attach`` reports what it would load: the folder and its config.json, or the Hub repo and revision."""
    seen: list[tuple[str, str]] = []

    def attach(self, robot):
        cfg = local.Path(self.checkpoint) / "config.json"
        seen.append((self.checkpoint, json.loads(cfg.read_text())["who"] if cfg.is_file() else self._kw["revision"]))
        raise _Loaded

    monkeypatch.setattr(local.LeRobotPolicy, "attach", attach)
    return seen


@pytest.fixture
def ckpt(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps({"who": "approved"}))
    return tmp_path / "ckpt"


@pytest.fixture
def hub(monkeypatch):
    """The Hub's answer for ``model_info(...).sha``; ``hub.calls`` counts lookups. The operator lists the repo."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("STRANDS_POLICY_REPOS", "someone/ckpt")
    state = SimpleNamespace(sha=A, calls=0, error=None)

    def model_info(self, repo, revision=None):
        state.calls += 1
        if state.error:
            raise state.error
        return SimpleNamespace(sha=state.sha)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    return state


def _arm(approve):
    plans = []
    return Robot("so101", driver="memory", approve=lambda p: plans.append(p) or approve(p)), plans


def test_a_registered_checkpoint_rewritten_after_the_yes_is_refused_and_never_loads(loads, ckpt):
    arm, plans = _arm(lambda p: (ckpt / "config.json").write_text(json.dumps({"who": "swapped"})) or True)
    with from_agent(), pytest.raises(RequestRefused, match="changed on disk after the operator approved"):
        arm.run(Policy(str(ckpt)), steps=1)
    assert len(plans[0].files) == 64 and loads == []


def test_an_unchanged_registered_checkpoint_loads_a_private_copy_and_the_handle_is_untouched(loads, ckpt):
    handle = Policy(str(ckpt))
    arm, _ = _arm(lambda p: True)
    with from_agent(), pytest.raises(_Loaded):
        arm.run(handle, steps=1)
    ((path, who),) = loads
    assert who == "approved" and path != str(ckpt) and not local.Path(path).exists()  # the copy is gone after
    assert handle.inner.checkpoint == str(ckpt)


@pytest.mark.parametrize("policy", ["someone/ckpt", Policy("someone/ckpt"), Policy("someone/ckpt", revision="main")])
def test_a_hub_checkpoint_loads_the_commit_the_operator_saw_even_if_the_branch_moves(loads, hub, policy):
    def approve(plan):
        hub.sha = B  # a push lands while the dialog is open
        return True

    arm, plans = _arm(approve)
    with from_agent(), pytest.raises(_Loaded):
        arm.run(policy, steps=1)
    assert plans[0].files == f"someone/ckpt@{A}" and loads == [("someone/ckpt", A)]


def test_a_full_commit_needs_no_lookup_and_an_unresolvable_one_is_refused_before_anyone_is_asked(loads, hub):
    arm, plans = _arm(lambda p: True)
    with from_agent(), pytest.raises(_Loaded):
        arm.run(Policy("someone/ckpt", revision=B), steps=1)
    assert hub.calls == 0 and loads == [("someone/ckpt", B)]
    hub.error = OSError("offline")
    with from_agent(), pytest.raises(RequestRefused, match="revision=<40-hex commit>"):
        arm.run("someone/ckpt", steps=1)
    hub.error, hub.sha = None, "not-a-commit"
    with from_agent(), pytest.raises(RequestRefused, match="no commit id"):
        arm.run("someone/ckpt", steps=1)
    assert len(plans) == 1 and len(loads) == 1


def test_the_developers_run_and_a_network_spec_from_an_agent_are_never_put_to_the_operator(loads, ckpt):
    arm, plans = _arm(lambda p: True)
    with pytest.raises(_Loaded):
        arm.run(Policy(str(ckpt)), steps=1)  # the developer's own run: no plan, the folder as it is
    assert loads == [(str(ckpt), "approved")]
    with from_agent(), pytest.raises(PermissionError, match="network address"):
        arm.run("zenoh://127.0.0.1:7447/d", steps=1)
    assert plans == []


def test_an_unlisted_hub_checkpoint_from_an_agent_is_refused_before_the_operator_is_asked(loads, hub, monkeypatch):
    monkeypatch.setenv("STRANDS_POLICY_REPOS", "lerobot/*")
    arm, plans = _arm(lambda p: True)
    with from_agent(), pytest.raises(PermissionError, match="STRANDS_POLICY_REPOS"):
        arm.run("someone/ckpt", steps=1)
    assert plans == [] and hub.calls == 0 and loads == []  # nobody asked, nothing looked up, nothing loaded
