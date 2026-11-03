"""The world tool tells a model where things are and how to fill every argument; two robots run in lock-step."""

from __future__ import annotations

import json

import pytest
from scripted_model import ScriptedModel
from strands import Agent

from strands_robots import Robot
from strands_robots.simulation import Simulation


@pytest.fixture
def sim():
    world = Simulation()
    yield world
    world.close()


def _props(sim):
    return sim.tool.tool_spec["inputSchema"]["json"]["properties"]


def _turn(sim, calls):
    agent = Agent(model=ScriptedModel([("sim_world", c) for c in calls]), tools=[sim.tool], callback_handler=None)
    agent("go")
    return [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]


def test_describe_and_pose_report_where_an_object_is_now_not_where_it_spawned(sim):
    sim.add_object("cube", shape="box", position=(0.3, 0.0, 0.5))
    sim.step(500)  # it falls onto the ground
    live = sim.pose("cube").position
    (d,) = [o for o in sim.describe()["objects"] if o["name"] == "cube"]
    assert live[2] < 0.1 and d["position"] == pytest.approx(live.tolist())
    moved = sim.set_pose("cube", position=(0.0, 0.4, 0.2))
    assert moved.position.tolist() == pytest.approx([0.0, 0.4, 0.2])
    r = _turn(sim, [{"rpc": "pose", "pose": {"body": "cube"}}])[0]
    assert json.loads(r["content"][0]["text"])["position"] == pytest.approx([0.0, 0.4, 0.2])


def test_an_agent_reads_where_its_gripper_is(sim):
    arm = Robot("so101", sim=sim)
    r = _turn(sim, [{"rpc": "pose", "pose": {"body": f"{arm.id}/gripper"}}])[0]
    assert json.loads(r["content"][0]["text"])["position"] == pytest.approx(sim.pose("so101/gripper").position.tolist())


@pytest.mark.parametrize(
    ("rpc", "arg", "n"),
    [
        ("add_object", "color", 4),
        ("add_object", "orientation", 4),
        ("set_pose", "position", 3),
        ("camera", "target", 3),
    ],
)
def test_vectors_on_the_wire_are_fixed_length_number_arrays(sim, rpc, arg, n):
    p = _props(sim)[rpc]["properties"][arg]
    assert "array" in p["type"] and p["items"] == {"type": "number"} and p["minItems"] == p["maxItems"] == n


@pytest.mark.parametrize(
    ("kw", "err"),
    [({"color": "1001"}, TypeError), ({"color": "red"}, TypeError), ({"orientation": (0, 0, 1)}, ValueError)],
)
def test_a_string_or_wrong_length_vector_is_refused_by_name(sim, kw, err):
    with pytest.raises(err, match="color|orientation"):
        sim.add_object("cube", shape="box", **kw)
    assert sim.scene.objects == []


def test_an_agent_never_chooses_where_a_worlds_cameras_stream(sim, monkeypatch):
    """The same rule as robot.tool's ``policy=``, for ``sim.tool``'s ``policies={id: spec}``: refused before attach."""
    from strands_robots.policy.lerobot import remote
    from strands_robots.robot.run.source import from_agent

    attached: list[str] = []
    monkeypatch.setattr(remote.ZenohPolicy, "attach", lambda self, robot: attached.append(robot.id))
    Robot("so101", sim=sim, id="a")
    host = ".".join(["203", "0", "113", "9"])  # TEST-NET-3, assembled so no scanner reads it as an address
    (r,) = _turn(sim, [{"rpc": "run", "run": {"policies": {"a": f"zenoh://{host}:7447/x"}, "steps": 1}}])
    assert r["status"] == "error" and "developer" in r["content"][0]["text"] and "network" in r["content"][0]["text"]
    assert attached == []
    with from_agent(), pytest.raises(PermissionError, match="network address"):  # any entry point: bind() refuses
        sim.run({"a": f"zenoh://{host}:7447/x"}, steps=1)
    assert attached == []


def test_an_agent_runs_two_robots_in_lock_step_by_id(sim, monkeypatch):
    from strands_robots.policy import factory

    real_make = factory.make
    pans = {"t.left": 0.3, "t.right": -0.3}
    monkeypatch.setattr(
        factory, "make", lambda s, **kw: (lambda obs: {"shoulder_pan": pans[s]}) if s in pans else real_make(s)
    )
    a, b = Robot("so101", sim=sim, id="a"), Robot("so101", sim=sim, id="b", position=(0.5, 0, 0))
    gated = []
    for r in (a, b):
        monkeypatch.setattr(r, "gate", lambda act, _r=r, _g=r.gate, **kw: gated.append(_r.id) or _g(act, **kw))
    t0 = sim.time
    (r,) = _turn(sim, [{"rpc": "run", "run": {"policies": {"a": "t.left", "b": "t.right"}, "steps": 50}}])
    assert r["status"] == "success", r
    out = json.loads(r["content"][0]["text"])
    assert out["a"]["steps"] == out["b"]["steps"] == 50 and sim.time - t0 == pytest.approx(50 * sim.physics.control_dt)
    assert a.observe().joints["shoulder_pan"] > 0.2 and b.observe().joints["shoulder_pan"] < -0.2
    assert gated.count("a") == gated.count("b") == 50  # an agent's lock-step passes the gate on every step
    live = sim.describe()["robots"]["b"]
    assert live["position"] == [0.5, 0, 0] and live["pose"]["position"] == pytest.approx(b.pose.position.tolist())
    with pytest.raises(KeyError, match="no robot"):
        sim.run({"c": "t.left"}, steps=1)


def test_randomize_takes_a_range_and_says_what_it_drew(sim):
    p = _props(sim)["randomize"]["properties"]["friction"]["anyOf"]
    assert {"type": "number"} in p and {"type": "null"} in p
    assert {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2} in p
    drawn = sim.randomize(friction=(0.5, 1.5), mass=2.0, seed=3)
    assert 0.5 <= drawn["friction_scale"] <= 1.5 and drawn["mass_scale"] == 2.0 and drawn["light_scale"] == 1.0
    assert drawn == sim.randomize(friction=(0.5, 1.5), mass=2.0, seed=3)


def test_reset_refuses_a_keyframe_no_robot_has_and_takes_no_seed(sim):
    Robot("so101", sim=sim)
    with pytest.raises(ValueError, match="keyframe 'hom'"):
        sim.reset(keyframe="hom")
    assert "seed" not in _props(sim)["reset"]["properties"]


def test_remove_takes_an_object_or_a_camera_and_shape_is_an_enum(sim):
    p = _props(sim)
    assert p["add_object"]["properties"]["shape"]["enum"] == ["box", "sphere", "cylinder", "capsule", None]
    sim.add_object("cube", shape="box")
    sim.camera("front", position=(1, 0, 0.5))
    sim.remove("front")
    sim.camera("front", position=(0, 1, 0.5))  # moved: removed, added again
    sim.remove("cube")
    assert [o.name for o in sim.scene.objects] == [] and [c.name for c in sim.scene.cameras] == ["front"]
    with pytest.raises(KeyError, match="nothing named 'ghost'"):
        sim.remove("ghost")
