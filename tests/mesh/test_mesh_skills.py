"""Device skills: a mounted device runs a named policy on itself when the mesh asks, so no control loop crosses the
network (mesh/skills.py). The caller names a skill and a duration; the device's catalogue, safety chain and
operator decide the rest."""

from __future__ import annotations

import time

import numpy as np
import pytest

from strands_robots import Policy, Robot
from strands_robots.mesh import Mesh, RequestRefused


class Hold(Policy):
    """A policy that holds shoulder_pan at 0.2 rad, paced in wall time so a skill lasts as long as it says."""

    units = "rad"
    realtime = True

    def act(self, obs, *, instruction=None):
        return {"shoulder_pan": 0.2}


def _wait(pred, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.01)
    return pred()


def _done(b, did="arm", timeout=10.0):
    assert _wait(lambda: b.invoke(did, "skill_status")["state"] != "running", timeout)
    return b.invoke(did, "skill_status")


@pytest.fixture
def skilled(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101", id="arm")
    a.mount(arm, skills={"pan": lambda o: {"shoulder_pan": 0.3}, "hold": Hold()}, watchdog_ms=300)
    assert b.discover("arm", min_count=1, timeout=3.0)
    yield arm, a, b
    arm.close()


def test_the_manifest_advertises_the_skills_and_their_rpcs(skilled):
    _, _, b = skilled
    d = b.discover("arm")[0]
    assert d.manifest["skills"] == ["hold", "pan"]
    assert {"skill", "skill_status"} <= {r["name"] for r in d.manifest["rpcs"]}


def test_a_skill_runs_on_the_device_and_status_reports_it(skilled):
    arm, _, b = skilled
    out = b.invoke("arm", "skill", name="pan", seconds=0.5)
    assert out == {"started": "pan", "device": "arm", "seconds": 0.5}
    st = _done(b)
    assert st["state"] == "done" and st["stopped_by"] == "seconds" and st["steps"] > 0
    assert arm.observe().joints["shoulder_pan"] == pytest.approx(0.3, abs=0.02)


@pytest.mark.parametrize(
    "args, error, said",
    [
        ({"name": "fly", "seconds": 1.0}, LookupError, "has no skill 'fly'; it offers ['hold', 'pan']"),
        ({"name": "pan", "seconds": 0}, ValueError, "seconds= must be a number"),
        ({"name": "pan", "seconds": 61}, ValueError, "seconds= must be a number"),
        ({"name": "pan", "seconds": True}, ValueError, "seconds= must be a number"),
        ({"name": "pan", "seconds": float("inf")}, ValueError, "cannot send NaN or infinity"),
        ({"name": "pan", "seconds": 1.0, "instruction": 3}, TypeError, "instruction= must be a string"),
    ],
)
def test_a_skill_the_device_cannot_honour_is_refused_and_nothing_moves(skilled, args, error, said):
    arm, _, b = skilled
    before = arm.observe().joints.positions.copy()
    with pytest.raises(error, match=r".*" + said.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(")):
        b.invoke("arm", "skill", **args)
    assert b.invoke("arm", "skill_status")["state"] == "idle"
    assert np.allclose(arm.observe().joints.positions, before)


def test_a_device_mounted_without_skills_refuses_a_skill_by_name(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101", id="arm")
    a.mount(arm, watchdog_ms=300)
    assert b.discover("arm", min_count=1, timeout=3.0)
    assert "skills" not in b.discover("arm")[0].manifest
    with pytest.raises(LookupError, match="offers no skills"):
        b.invoke("arm", "skill", name="pan", seconds=1.0)
    arm.close()


def test_a_second_skill_while_one_runs_is_refused(skilled):
    _, _, b = skilled
    b.invoke("arm", "skill", name="hold", seconds=1.5)
    with pytest.raises(RuntimeError, match="already running skill 'hold'"):
        b.invoke("arm", "skill", name="pan", seconds=0.5)
    assert _done(b)["state"] == "done"


def test_an_estop_ends_a_running_skill_and_a_latched_device_refuses_the_next(skilled):
    _, _, b = skilled
    b.invoke("arm", "skill", name="hold", seconds=5.0)
    time.sleep(0.3)
    b.invoke("arm", "estop", by="alice")
    st = _done(b, timeout=3.0)
    assert st["stopped_by"] == "estop"
    with pytest.raises(RequestRefused, match="e-stop is engaged"):
        b.invoke("arm", "skill", name="pan", seconds=0.5)


def test_stop_ends_a_running_skill(skilled):
    _, _, b = skilled
    b.invoke("arm", "skill", name="hold", seconds=5.0)
    time.sleep(0.3)
    b.invoke("arm", "stop")
    st = _done(b, timeout=3.0)
    assert st["state"] == "done" and st["stopped_by"] == "stopped"


def test_a_quiet_caller_does_not_cut_a_skill_short(skilled):
    """The caller of a skill may poll slowly: the quiet clock pauses while the device is busy with its skill."""
    _, _, b = skilled
    t0 = time.monotonic()
    b.invoke("arm", "skill", name="hold", seconds=1.2)  # four times the 300 ms watchdog, with no rpc in between
    time.sleep(1.4)
    st = b.invoke("arm", "skill_status")
    assert st["state"] == "done" and st["stopped_by"] == "seconds" and time.monotonic() - t0 >= 1.2


def test_a_caller_that_vanishes_mid_skill_ends_it_and_the_device_holds(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101", id="arm")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, skills={"hold": Hold()}, watchdog_ms=5000)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "skill", name="hold", seconds=10.0)
    time.sleep(0.3)
    t0 = time.monotonic()
    b.close()
    assert _wait(lambda: holds, 3.0), "the caller's presence vanished: the skill ends and on_disconnect applies"
    assert holds[0] - t0 < 1.5
    assert _wait(lambda: not arm._task.running, 2.0), "the skill ended well before its 10 s"
    arm.close()


def test_a_real_robot_asks_its_own_operator_on_the_device(mesh_pair):
    a, b = mesh_pair
    plans: list = []
    arm = Robot("so101", id="arm", driver="memory", confirm=lambda t: True, approve=lambda p: plans.append(p) or True)
    a.mount(arm, skills={"pan": lambda o: {"shoulder_pan": 0.2}}, watchdog_ms=2000)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "skill", name="pan", seconds=0.3)
    assert _done(b)["state"] == "done"
    assert [p.robot for p in plans] == ["arm"]
    arm.close()


def test_a_real_robot_whose_operator_declines_reports_the_skill_refused(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101", id="arm", driver="memory", confirm=lambda t: True, approve=lambda p: False)
    a.mount(arm, skills={"pan": lambda o: {"shoulder_pan": 0.2}}, watchdog_ms=2000)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "skill", name="pan", seconds=0.3)
    st = _done(b)
    assert st["state"] == "refused" and "operator refused" in st["error"]
    arm.close()


def test_a_real_robot_without_approve_refuses_a_skill_at_once(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101", id="arm", driver="memory", confirm=lambda t: True)
    a.mount(arm, skills={"pan": lambda o: {"shoulder_pan": 0.2}}, watchdog_ms=2000)
    assert b.discover("arm", min_count=1, timeout=3.0)
    with pytest.raises(RequestRefused, match="pass approve="):
        b.invoke("arm", "skill", name="pan", seconds=0.3)
    arm.close()


@pytest.mark.parametrize(
    "skills, error, said",
    [
        ({"bad name": "x"}, ValueError, "skill name"),
        ({"pan": None}, TypeError, "needs a policy"),
        ({"pan": "zenoh://10.0.0.9:7447/x"}, ValueError, "names a network address"),
        (["pan"], TypeError, "must be a mapping"),
    ],
)
def test_a_catalogue_that_cannot_be_served_is_refused_at_mount(skills, error, said):
    m = Mesh("mem://skill-cat", tenant="t-skill-cat")
    arm = Robot("so101", id="arm")
    with pytest.raises(error, match=said):
        m.mount(arm, skills=skills)
    assert not m._mounted
    m.close()
    arm.close()


def test_the_device_refuses_a_non_finite_duration_even_off_the_wire():
    from strands_robots.mesh.skills import Skills

    arm = Robot("so101", id="arm")
    with pytest.raises(ValueError, match="seconds= must be a number"):
        Skills(arm, "arm", {"pan": lambda o: {}}).start(None, name="pan", seconds=float("nan"))
    arm.close()


strands = pytest.importorskip("strands")


def _call(tool, **inp):
    import json

    res = tool._tool_func({"toolUseId": "t1", "input": inp})
    text = res["content"][0]["text"]
    return res["status"], (json.loads(text) if res["status"] == "success" else text)


def test_the_tool_lists_a_devices_skills_and_runs_one_on_the_device(skilled):
    arm, _, b = skilled
    tool = b.tool
    assert '"skills": ["hold", "pan"]' in tool.tool_spec["description"]
    assert "status" in tool.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    status, out = _call(tool, action="run", device_id="arm", run={"policy": "pan", "seconds": 0.4})
    assert status == "success" and out["started"] == "pan" and not b.tasks  # nothing runs on the caller
    assert _wait(lambda: _call(tool, action="status", device_id="arm")[1]["state"] == "done", 10.0)
    assert arm.observe().joints["shoulder_pan"] == pytest.approx(0.3, abs=0.02)


@pytest.mark.parametrize(
    "run, said",
    [
        ({"policy": "fly", "seconds": 1}, "runs only its own skills"),
        ({"policy": "pan"}, "needs seconds="),
        ({"policy": "pan", "seconds": 1, "hz": 50}, "rate and steps are the device's own"),
    ],
)
def test_the_tool_refuses_a_run_a_skilled_device_cannot_take(skilled, run, said):
    _, _, b = skilled
    status, text = _call(b.tool, action="run", device_id="arm", run=run)
    assert status == "error" and said in text
    assert b.invoke("arm", "skill_status")["state"] == "idle"


def test_a_real_robot_on_another_machine_without_skills_is_not_driven_over_the_network(port, tenant):
    """Before this, the tool ran the policy here and drove the robot one observe and one act rpc per control step."""
    pytest.importorskip("zenoh")
    from strands_robots.mesh.tool import use_mesh

    a = Mesh(f"zenoh://?listen=tcp/127.0.0.1:{port}&trusted=1", tenant=tenant)
    b = Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant)
    arm = Robot("so101", id="arm", driver="memory", confirm=lambda t: True)
    a.mount(arm, watchdog_ms=2000)
    assert b.discover("arm", min_count=1, timeout=5.0)
    tool = use_mesh(b, approve=lambda plan: True)
    status, text = _call(tool, action="run", device_id="arm", run={"policy": "noop", "steps": 3})
    assert status == "error" and "offers no skills" in text and "skills={name: policy}" in text
    assert not b.tasks
    b.close()
    a.close()
    arm.close()


def test_a_skill_the_operator_refuses_leaves_the_watchdog_disarmed(mesh_pair):
    """A refused skill moved nothing, so its quiet caller must not make the device apply on_disconnect, the same as a
    refused act (review on #4938: the run armed the watchdog before the operator was asked, and was never disarmed)."""
    a, b = mesh_pair
    arm = Robot("so101", id="arm", driver="memory", confirm=lambda t: True, approve=lambda p: False)
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, skills={"pan": lambda o: {"shoulder_pan": 0.2}}, watchdog_ms=300)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "skill", name="pan", seconds=0.3)
    assert _done(b)["state"] == "refused"
    time.sleep(0.8)  # the caller goes quiet for well past watchdog_ms
    assert holds == [], "a skill that moved nothing applied on_disconnect"
    arm.close()


def test_a_caller_that_keeps_polling_after_its_skill_keeps_the_watchdog_quiet(mesh_pair):
    """A skill arms the watchdog for its caller like an act does, so its reply carries the beat key too: where the
    transport does not prove a peer, the caller's later requests count as its heartbeat only with that key. Without
    it, a caller still polling skill_status would look quiet and the device would apply on_disconnect after every
    skill."""
    a, b = mesh_pair
    arm = Robot("so101", id="arm")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, skills={"pan": lambda o: {"shoulder_pan": 0.2}}, watchdog_ms=300)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "skill", name="pan", seconds=0.3)
    end = time.monotonic() + 1.5
    while time.monotonic() < end:  # well past watchdog_ms after the skill ends, with the caller still talking
        b.invoke("arm", "skill_status")
        time.sleep(0.1)
    assert holds == [], "a caller that kept polling was taken for a quiet one"
    arm.close()


def test_an_act_from_the_skill_caller_while_the_skill_waits_keeps_the_watchdog_armed():
    """A refused skill restores the watchdog only if nothing armed it since. An act from the same caller while the
    operator is still deciding keeps that caller's peer and key, so comparing those cannot see it: the act moved the
    body, so the caller's silence must still apply on_disconnect (review on #4938). Driven on the watchdog directly,
    in the order that matters: skill armed, act done, skill released as having moved nothing."""
    from strands_robots.mesh.watchdog import MeshWatchdog

    arm = Robot("so101", id="arm")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    dog = MeshWatchdog(arm, 300.0, [], "t", "arm")
    try:
        release, key = dog.busy("caller")
        done = dog.on_rpc("act", "caller", key, False)  # the same caller, showing its key: the act moves the body
        assert done is not None
        done(True)
        release(False)  # the operator refused the skill: it moved nothing
        assert _wait(lambda: holds, 2.0), "the act moved the body, so its quiet caller must still make the device hold"
    finally:
        dog.close()
        arm.close()
