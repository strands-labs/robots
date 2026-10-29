"""A Robot is a device: its manifest, rpcs, events and position hold."""

from __future__ import annotations

import json
import time

import pytest
from test_drivers import make_driver

from strands_robots import Robot
from strands_robots.robot.device import Manifest
from strands_robots.robot.safety import RequestRefused


def test_manifest_round_trips_and_describe_is_the_manifest():
    arm = Robot("so101")
    m = arm.manifest
    assert m is arm.manifest and m.device_id == "so101" and m.tenant == "default" and not m.real
    assert Manifest.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    d = arm.describe()
    assert {k: v for k, v in d.items() if k not in ("task", "estop", "reach")} == m.to_dict()
    assert d["task"] is None and d["estop"] is False and d["reach"]["link"]
    assert {r.name for r in m.rpcs} >= {"describe", "observe", "act", "run", "stop", "hold", "estop", "resume"}
    assert {e.name for e in m.events} == {"state", "fault", "estop", "run.started", "run.stopped"}


def test_one_control_per_actuated_joint_range_is_the_limit_and_safe_inside():
    arm = Robot("so101")
    controls = arm.manifest.controls
    assert [c.name for c in controls] == [f"{k}.pos" for k in arm.action_keys]
    for c, k in zip(controls, arm.action_keys, strict=True):
        assert c.range == tuple(arm.description.limits[k])
        assert c.range[0] <= c.safe[0] <= c.safe[1] <= c.range[1]
        assert c.on_disconnect == "hold"


def test_rpc_schema_is_generated_from_the_signature():
    schema = {r.name: r.params for r in Robot("so101").manifest.rpcs}
    assert schema["resume"]["properties"]["approved_by"]["type"] == "string"
    assert schema["act"]["required"] == ["action"]


def test_state_subscription_carries_the_policy_joints_dict():
    arm = Robot("so101")
    got: list[dict] = []
    unsubscribe = arm.subscribe("state", got.append, hz=100)
    time.sleep(0.12)
    unsubscribe()
    n = len(got)
    time.sleep(0.05)
    assert n >= 5 and len(got) == n
    assert [g["seq"] for g in got] == list(range(n)) and got[-1]["t"] > got[0]["t"]
    assert {f"{k}.pos" for k in arm.action_keys} <= set(got[0])
    with pytest.raises(ValueError, match="unknown event"):
        arm.subscribe("nope", print)


def test_rpc_schema_drops_the_receiver_of_a_run_loop_function():
    from strands_robots.robot.device import rpc_schema

    def run(robot, policy, *, hz: float | None = None):
        """A run-loop function bound onto Robot."""

    assert list(rpc_schema(run)["properties"]) == ["policy", "hz"]


def test_position_hold_is_declared_sim_always_holds_hardware_reads_the_registry():
    from strands_robots.description import registry
    from strands_robots.robot.run.bind import bind

    sim = Robot("so101")
    assert sim.supports_position_hold and sim.manifest.position_hold
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    assert real.supports_position_hold is registry()["so101"]["hardware"]["position_hold"] is True
    assert real.manifest.position_hold

    class NeedsHold:
        requires_position_hold = True

        def act(self, obs, instruction=None):
            return obs.joints.positions

        def reset(self):
            pass

    real.__dict__["supports_position_hold"] = False  # a bus that drops its goal
    with pytest.raises(RequestRefused, match="cannot hold position"):
        bind(real, NeedsHold(), None)
    bind(sim, NeedsHold(), None)
    real.close()


def test_a_manifest_that_omits_position_hold_reads_as_not_holding():
    d = Robot("so101").manifest.to_dict()
    del d["position_hold"]
    assert Manifest.from_dict(d).position_hold is False


def test_closing_a_robot_ends_its_state_stream_quietly():
    import threading as th

    arm = Robot("so101")
    errors: list[BaseException] = []
    old = th.excepthook
    th.excepthook = lambda a: errors.append(a.exc_value)
    try:
        got: list[dict] = []
        unsubscribe = arm.subscribe("state", got.append, hz=200)
        time.sleep(0.05)
        arm.close()
        time.sleep(0.05)
        n = len(got)
        time.sleep(0.05)
        unsubscribe()
    finally:
        th.excepthook = old
    assert errors == [] and len(got) == n and n > 0


def test_a_run_emits_run_started_and_run_stopped():
    arm = Robot("so101")
    got: list[tuple[str, dict]] = []
    for e in ("run.started", "run.stopped", "fault"):
        arm.subscribe(e, lambda p, e=e: got.append((e, p)))
    arm.run(lambda o: {"shoulder_pan": 0.1}, steps=3)
    assert [e for e, _ in got] == ["run.started", "run.stopped"]
    assert got[0][1]["device_id"] == "so101" and got[1][1] == {"device_id": "so101", "stopped_by": "steps", "steps": 3}


def test_a_policy_fault_emits_fault_then_run_stopped():
    from strands_robots.policy.base import PolicyFault

    def broken(o):
        raise PolicyFault("server lost")

    arm = Robot("so101")
    got: list[tuple[str, dict]] = []
    for e in ("fault", "run.stopped"):
        arm.subscribe(e, lambda p, e=e: got.append((e, p)))
    assert arm.run(broken, steps=5).stopped_by == "fault"
    assert [e for e, _ in got] == ["fault", "run.stopped"] and "server lost" in got[0][1]["reason"]


def test_lockstep_emits_per_robot():
    from strands_robots import Simulation

    sim = Simulation("mujoco")
    a, b = Robot("so101", sim=sim, id="a"), Robot("so101", sim=sim, id="b")
    stopped: list[str] = []
    for r in (a, b):
        r.subscribe("run.stopped", lambda p: stopped.append(p["device_id"]))
    sim.run({a: lambda o: {}, b: lambda o: {}}, steps=2)
    assert sorted(stopped) == ["a", "b"]
    sim.close()


def test_k4_manifest_sees_a_camera_added_after_it_was_first_read():
    """K4: manifest was a cached_property, so a camera added after the first read never reached the surface."""
    from strands_robots import Robot, Simulation

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    assert arm.manifest.sensors == arm.manifest.sensors  # a read is stable
    before = {s.name for s in arm.manifest.sensors}
    sim.camera("front", attach=arm)
    after = {s.name for s in arm.manifest.sensors}
    assert after - before, (before, after)
    sim.close()
