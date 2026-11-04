"""Services, params, actions, request triplets, and the command gate."""

from __future__ import annotations

import pytest

from .robots import REL


def _server(ros):
    n = ros.node("server")

    def add(req, res):
        res.sum = req.a + req.b
        return res

    n.create_service(type("S", (), {"_type": "example_interfaces/srv/AddTwoInts"}), "/add", add)
    params = {"volume": 0.5}

    def get(req, res):
        res.values = [ros.msg("rcl_interfaces/ParameterValue")(type=3, double_value=params[k]) for k in req.names]
        return res

    def set_(req, res):
        for p in req.parameters:
            params[p.name] = p.value.double_value
        return res

    n.create_service(type("S", (), {"_type": "rcl_interfaces/srv/GetParameters"}), "/audio/get_parameters", get)
    n.create_service(type("S", (), {"_type": "rcl_interfaces/srv/SetParameters"}), "/audio/set_parameters", set_)
    return n


def test_service_call_and_param_set_with_read_back(ros):
    from strands_robots.ros import calls
    from strands_robots.ros.bus import RosBus

    _server(ros)
    bus = RosBus().start()
    assert calls.service_call(bus, "/add", "example_interfaces/srv/AddTwoInts", {"a": 2, "b": 3})["response"] == {
        "sum": 5
    }
    assert calls.get_params(bus, "/audio", ["volume"]) == {"volume": 0.5}
    r = calls.set_param(bus, "/audio", "volume", 0.25)
    assert r["ok"] and r["read_back"] == 0.25
    with pytest.raises(TimeoutError):
        calls.service_call(bus, "/nope", "example_interfaces/srv/AddTwoInts", {}, timeout=0.1)
    bus.close()


def test_action_goal_returns_result_and_caps_feedback(ros):
    from strands_robots.ros import calls
    from strands_robots.ros.bus import RosBus

    from . import fake_rclpy

    def server(goal, fb, cls):
        for i in range(9):
            fb(type("F", (), {"feedback": cls.Feedback(sequence=list(range(i)))})())
        res = fake_rclpy.Future()
        res.set_result(type("R", (), {"status": 4, "result": cls.Result(sequence=[0, 1, 1, 2])})())
        return type("H", (), {"accepted": True, "get_result_async": lambda self: res})()

    fake_rclpy.GRAPH.actions["/fib"] = ("example_interfaces/action/Fibonacci", server)
    bus = RosBus().start()
    out = calls.action_goal(bus, "/fib", "example_interfaces/action/Fibonacci", {"order": 4}, timeout=1.0)
    assert out["accepted"] and out["result"] == {"sequence": [0, 1, 1, 2]}
    assert len(out["feedback"]) == 5 and out["feedback"][-1] == {"sequence": list(range(8))}
    bus.close()


def test_request_topic_triplet_matches_by_id(ros):
    from strands_robots.ros import calls
    from strands_robots.ros.bus import RosBus

    n = ros.node("hri")
    S = ros.msg("std_msgs/Header")
    resp = n.create_publisher(S, "/hri/behavior/response", REL)
    done = n.create_publisher(S, "/hri/behavior/complete", REL)

    def on_req(m):
        resp.publish(S(frame_id="other"))  # someone else's reply first
        resp.publish(S(frame_id=m.frame_id))
        done.publish(S(frame_id=m.frame_id))

    n.create_subscription(S, "/hri/behavior/request", on_req, REL)
    import threading
    import time

    stop = threading.Event()

    def drain():
        while not stop.is_set():
            try:
                cb, msg = n.queue.get(timeout=0.02)
                cb(msg)
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=drain, daemon=True).start()
    bus = RosBus().start()
    out = calls.request_topic(
        bus,
        "/hri/behavior",
        "std_msgs/msg/Header",
        {},
        ack_type="std_msgs/msg/Header",
        id_field="frame_id",
        timeout=1.0,
    )
    stop.set()
    time.sleep(0.03)
    assert out["response"]["frame_id"] == out["id"] and out["complete"]["frame_id"] == out["id"]
    bus.close()


def test_gate_fails_closed_matches_segments_and_allows_exact_names(monkeypatch):
    from strands_robots.ros.gate import CommandGate, CommandRefused, env_allowlist, matches

    assert matches("/robot1/cmd_vel", ["/cmd_vel"]) == "/cmd_vel"
    assert matches("/my_cmd_vel", ["/cmd_vel"]) is None
    assert matches("/arm/joint_trajectory_controller/joint_trajectory", ["/joint_trajectory"])
    g = CommandGate(allow=frozenset())
    assert g.check("publish", "/chatter") == "free"
    with pytest.raises(CommandRefused, match="STRANDS_ROS_ALLOW=/r1/cmd_vel"):
        g.check("publish", "/r1/cmd_vel")
    monkeypatch.setenv("STRANDS_ROS_ALLOW", "/r1/cmd_vel")
    g2 = CommandGate(allow=env_allowlist())
    assert g2.check("publish", "/r1/cmd_vel") == "allowlist"
    with pytest.raises(CommandRefused):
        g2.check("publish", "/r2/cmd_vel")  # exact names only: no fail-open across namespaces
    asked = []
    g3 = CommandGate(allow=frozenset(), approve=lambda v, t, p: asked.append((v, t)) or False)
    with pytest.raises(CommandRefused, match="declined"):
        g3.check("call", "/api/motion_switcher/request")
    assert asked == [("call", "/api/motion_switcher/request")]
    g4 = CommandGate(owners={"/lowcmd": "unitree_g1:abc"}, approve=lambda *a: True)
    with pytest.raises(CommandRefused, match="streamed by robot"):
        g4.check("publish", "/lowcmd")


def test_dispatch_checks_args_before_asking_and_never_gates_reads_or_stop(ros):
    from strands_robots.ros.bus import RosBus
    from strands_robots.ros.gate import CommandGate
    from strands_robots.ros.verbs import dispatch

    asked = []
    gate = CommandGate(allow=frozenset(), approve=lambda v, t, p: asked.append(t) or True)
    bus = RosBus().start()
    with pytest.raises(ValueError, match="needs"):
        dispatch(bus, gate, "call", {"service": "/robot/set_mode"})  # no type: refused, nobody asked
    assert asked == []
    dispatch(bus, gate, "topics", {})
    dispatch(bus, gate, "stop", {"profile": "unitree_g1"})  # STOP: no operator needed
    assert asked == []
    r = dispatch(
        bus, gate, "publish", {"topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist", "fields": {"linear": {"x": 0.1}}}
    )
    assert asked == ["/cmd_vel"] and r["count"] == 1
    bus.close()
