"""strands_robots.ros against a real rclpy graph: connect -> Robot, act, use_ros verbs, presence."""

import json
import subprocess
import sys
import time

import numpy as np

from strands_robots import Robot, ros
from strands_robots.ros.bus import RosBus
from strands_robots.ros.gate import CommandGate, CommandRefused
from strands_robots.ros.verbs import dispatch

results = {}


def check(name, cond, info=""):
    results[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)


t0 = time.monotonic()
arm = ros.connect("ros2_control", timeout=10.0)
check(
    "connect: model from /robot_description (latched), meshes stripped",
    arm.action_keys == ("shoulder", "elbow", "wrist"),
    arm.action_keys,
)
obs = arm.observe(images=False)
check(
    "observe: BEST_EFFORT /joint_states heard via qos=auto, keyed by name",
    np.allclose(obs.joints.positions, [0.1, -0.2, 0.3]),
    obs.joints.positions.tolist(),
)
bus = RosBus.shared()
check("auto QoS resolved to sensor", bus.stats()["/joint_states"]["qos"] == "sensor", bus.stats()["/joint_states"])
arm.act({"elbow": 0.5})
bus.subscribe("/smoke/received", "std_msgs/msg/String", qos="reliable")
got = bus.wait_newer("/smoke/received", 0, 3.0)
check("act: Float64MultiArray reached the robot node", got is not None and "0.5" in got.data, got and got.data)
time.sleep(0.3)
check("robot reports the new pose back", abs(arm.observe(images=False).joints.positions[1] - 0.5) < 1e-6)
time.sleep(0.7)  # past the 500 ms deadman: the stream holds the measured pose
st = arm.body.stream.stats
check("timer stream kept publishing at rate (held after deadman)", st["sent"] > 30 and st["held"] > 0, st)
gate = CommandGate(allow=frozenset())
r = dispatch(bus, gate, "topics", {})
check("use_ros topics sees the graph", "/joint_states" in r and "/robot_description" in r, sorted(r)[:6])
r = dispatch(bus, gate, "info", {"topic": "/joint_states"})
check(
    "use_ros info: type, fields, publisher QoS",
    r["type"] == "sensor_msgs/msg/JointState" and r["publishers"],
    r["publishers"],
)
r = dispatch(bus, gate, "hz", {"topic": "/joint_states", "seconds": 1.0})
check("use_ros hz ~100", 60 < r["hz"] < 140, r)
r = dispatch(bus, gate, "echo", {"topic": "/joint_states", "count": 2, "timeout": 2.0})
check("use_ros echo", len(r["messages"]) == 2 and r["messages"][0]["name"], r["messages"][0]["name"])
r = dispatch(bus, gate, "params", {"node": "/smoke_arm", "names": ["speed_scale"]})
check("use_ros params get", r == {"speed_scale": 0.5}, r)
r = dispatch(
    bus,
    CommandGate(allow=frozenset(), approve=lambda *a: True),
    "set_param",
    {"node": "/smoke_arm", "name": "speed_scale", "value": 0.8},
)
check("use_ros set_param with read-back", r["ok"] and r["read_back"] == 0.8, r)
r = dispatch(bus, gate, "tf", {"target": "base", "source": "tool", "timeout": 3.0})
check("use_ros tf lookup from /tf_static", np.allclose(r["xyz"], [0.4, 0, 0]), r)
try:
    dispatch(
        bus,
        gate,
        "publish",
        {"topic": "/forward_position_controller/commands", "type": "std_msgs/msg/Float64MultiArray"},
    )
    check("use_ros refuses the topic the Robot streams", False)
except CommandRefused as e:
    check("use_ros refuses the topic the Robot streams", "streamed by robot" in str(e), str(e)[:80])
try:
    dispatch(bus, gate, "publish", {"topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist"})
    check("use_ros fails closed on /cmd_vel with no operator", False)
except CommandRefused as e:
    check("use_ros fails closed on /cmd_vel with no operator", "STRANDS_ROS_ALLOW" in str(e))
r = dispatch(bus, gate, "record", {"topics": ["/joint_states"], "seconds": 1.0}, record_dir="/tmp/smoke_bags")
check(
    "use_ros record (rosbag2 mcap or jsonl)", r["counts"]["/joint_states"] > 30, {k: r[k] for k in ("format", "counts")}
)
# presence: a strands robot that is NOT ROS shows up as a native ROS device
toy = Robot(arm.description.model_path, driver="memory", id="toy")
pres = ros.RosPresence(toy, bus=bus, rate_hz=30).start()
time.sleep(1.0)
print("graph topics:", sorted(bus.graph()["topics"]), "presence errors:", pres.errors, pres.last_error)
out = subprocess.run(
    [
        "bash",
        "-lc",
        "source /opt/ros/$ROS_DISTRO/setup.bash; timeout 15 ros2 topic echo --once --no-daemon /toy/joint_states "
        "sensor_msgs/msg/JointState --qos-reliability best_effort",
    ],
    capture_output=True,
    text=True,
)
check(
    "presence: `ros2 topic echo /toy/joint_states` sees the strands robot",
    "shoulder" in out.stdout,
    (out.stdout[:60] + out.stderr[-200:]).replace("\n", " ") + f" presence errors={pres.errors} {pres.last_error}",
)
out = subprocess.run(
    ["bash", "-lc", "source /opt/ros/$ROS_DISTRO/setup.bash; timeout 15 ros2 topic list --no-daemon --spin-time 3"],
    capture_output=True,
    text=True,
)
check("presence: /toy/robot_description on the graph", "/toy/robot_description" in out.stdout)
pres.stop()
arm.close()
print(json.dumps({"passed": sum(results.values()), "total": len(results), "seconds": round(time.monotonic() - t0, 1)}))
sys.exit(0 if all(results.values()) else 1)
