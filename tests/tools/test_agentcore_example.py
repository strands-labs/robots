"""notebooks/examples/agentcore_fleet.py with a scripted model. The entrypoint is called directly, so no server and no
network, and a stand-in app is used when bedrock-agentcore is not installed, so CI runs it."""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest
from scripted_model import ScriptedModel

pytest.importorskip("mujoco")
pytest.importorskip("starlette")
from starlette.exceptions import HTTPException  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "notebooks" / "examples" / "agentcore_fleet.py"


class _App:
    """Stands in for BedrockAgentCoreApp when bedrock-agentcore is not installed: only entrypoint and run."""

    def entrypoint(self, fn):
        return fn

    def run(self, **_):
        raise AssertionError("the test calls the entrypoint directly")


@pytest.fixture
def ex(monkeypatch):
    if importlib.util.find_spec("bedrock_agentcore") is None:
        runtime = types.ModuleType("bedrock_agentcore.runtime")
        runtime.BedrockAgentCoreApp = _App
        monkeypatch.setitem(sys.modules, "bedrock_agentcore", types.ModuleType("bedrock_agentcore"))
        monkeypatch.setitem(sys.modules, "bedrock_agentcore.runtime", runtime)
    spec = importlib.util.spec_from_file_location("agentcore_fleet", EXAMPLE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        if "sim" in mod.FLEET:
            mod.FLEET["sim"].close()
        mod.FLEET.clear()
        mod.PENDING.clear()
        mod.HALTED.clear()


def _results(agent):
    return [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]


def test_the_agent_moves_the_arm_and_estops_the_g1_but_cannot_resume(ex):
    model = ScriptedModel(
        [
            ("robot_so101", {"rpc": "act", "act": {"action": {"shoulder_pan": 0.3}, "steps": 400}}),
            ("robot_g1", {"rpc": "estop"}),
            ("robot_g1", {"rpc": "resume", "resume": {"approved_by": "me, the model"}}),
        ]
    )
    fleet = ex.fleet(model)
    assert sorted(fleet["agent"].tool_names) == ["robot_g1", "robot_so101"]
    assert ex.invoke({"prompt": "pan the arm, then stop the humanoid and start it again"}) == {"reply": "done"}
    statuses = [r["status"] for r in _results(fleet["agent"])]
    assert statuses[:2] == ["success", "success"] and statuses[2] == "error"
    assert "operator" in _results(fleet["agent"])[2]["content"][0]["text"]
    assert fleet["robots"]["so101"].observe()["shoulder_pan"] == pytest.approx(0.3, abs=0.05)
    assert fleet["robots"]["g1"].estopped and not fleet["robots"]["so101"].estopped


def test_a_direct_estop_is_not_queued_behind_a_running_turn(ex):
    ex.fleet(ScriptedModel([]))
    with ex.TURN:  # a turn is in progress
        done = threading.Event()
        out = {}

        def stop():
            try:
                out["r"] = ex.invoke({"estop": "all"})
            except BaseException as e:
                out["r"] = e
            finally:
                done.set()

        worker = threading.Thread(target=stop)
        worker.start()
        assert done.wait(10), "the e-stop waited on the running turn"
        worker.join()
        with pytest.raises(HTTPException) as busy:
            ex.invoke({"prompt": "move again"})
        assert busy.value.status_code == 409
    assert out["r"] == {"estopped": ["so101", "g1"]}
    assert all(r.estopped for r in ex.FLEET["robots"].values())


def test_estop_all_still_latches_the_other_robots_when_one_fails(ex, monkeypatch):
    ex.fleet(ScriptedModel([]))
    robots = ex.FLEET["robots"]

    def broken(**_):
        raise RuntimeError("latch hook failed")

    monkeypatch.setattr(robots["so101"], "estop", broken)
    with pytest.raises(HTTPException) as err:
        ex.invoke({"estop": "all"})
    assert err.value.status_code == 500 and "so101" in err.value.detail
    assert robots["g1"].estopped
    with pytest.raises(HTTPException) as refused:
        ex.invoke({"prompt": "move the arm"})
    assert refused.value.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [{}, {"prompt": "  "}, "text", {"estop": "spot"}, {"estop": ["g1"]}, {"estop": "g1", "prompt": "then move"}],
)
def test_a_bad_request_is_a_400(ex, payload):
    ex.fleet(ScriptedModel([]))
    with pytest.raises(HTTPException) as err:
        ex.invoke(payload)
    assert err.value.status_code == 400


def test_a_held_estop_that_cannot_latch_keeps_the_world_from_starting_for_the_session(ex, monkeypatch):
    ex.invoke({"estop": "all"})
    real, built = ex.Robot, []

    class Unlatchable(real):
        def __init__(self, *a, **kw):
            built.append(a[0])
            super().__init__(*a, **kw)

        def estop(self, **kw):
            if self.id == "so101":
                raise RuntimeError("latch hook failed")
            return super().estop(**kw)

    monkeypatch.setattr(ex, "Robot", Unlatchable)
    with pytest.raises(RuntimeError, match="so101"):
        ex.fleet(ScriptedModel([]))
    assert not ex.FLEET and built == ["so101", "g1"]
    for later in ({"prompt": "move the arm"}, {"estop": "g1"}):  # neither rebuilds, nor promises a latch
        with pytest.raises(HTTPException) as err:
            ex.invoke(later)
        assert err.value.status_code == 503
    assert not ex.FLEET and built == ["so101", "g1"]


def test_an_estop_before_the_world_exists_is_held_until_it_is_built(ex):
    assert ex.invoke({"estop": "g1"}) == {"pending": ["g1"]}
    assert not ex.FLEET  # the e-stop did not wait on, or start, the build
    robots = ex.fleet(ScriptedModel([]))["robots"]
    assert robots["g1"].estopped and not robots["so101"].estopped


def test_a_bad_prompt_is_refused_before_the_world_is_built(ex):
    with pytest.raises(HTTPException) as err:
        ex.invoke({"prompt": "  "})
    assert err.value.status_code == 400 and not ex.FLEET
