"""use_ros as a Strands tool, RosPresence (strands -> native ROS), MultiOnnxPolicy, and no-ROS behaviour."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from .conftest import wait_until
from .robots import REL, urdf
from .toy import FakeToy, write_profile


def _call(tool, verb, args=None, ctx=None):
    out = tool._tool_func(verb=verb, args=args or {}, tool_context=ctx)
    return out["status"], out["content"][0]["text"]


def test_use_ros_reads_freely_and_asks_through_a_strands_interrupt_for_motion(ros):
    pytest.importorskip("strands")
    from strands_robots.ros import use_ros

    n = ros.node("talker")
    p = n.create_publisher(ros.msg("std_msgs/String"), "/chatter", REL)
    tool = use_ros()
    assert tool.tool_name == "use_ros"
    status, text = _call(tool, "topics")
    assert status == "success" and "/chatter" in json.loads(text)
    import threading

    threading.Timer(0.1, lambda: p.publish(ros.msg("std_msgs/String")(data="hello"))).start()
    status, text = _call(tool, "echo", {"topic": "/chatter", "count": 1, "timeout": 1.0})
    assert json.loads(text)["messages"] == [{"data": "hello"}]
    status, text = _call(tool, "publish", {"topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist"})
    assert status == "error" and "no operator" in text and "STRANDS_ROS_ALLOW=/cmd_vel" in text

    class Ctx:
        asked = []

        def interrupt(self, name, reason=None):
            self.asked.append((name, reason["target"]))
            return "yes"

    ctx = Ctx()
    status, text = _call(
        tool,
        "publish",
        {"topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist", "fields": {"angular": {"z": 0.2}}},
        ctx,
    )
    assert status == "success" and ctx.asked == [("use_ros-command-approval", "/cmd_vel")]
    status, text = _call(tool, "nope")
    assert status == "error" and "unknown verb" in text


def test_use_ros_refuses_a_topic_a_robot_streams(ros, tmp_path, monkeypatch):
    pytest.importorskip("strands")

    from strands_robots import ros as sros

    monkeypatch.setenv("STRANDS_ROBOTS_ROS_PROFILES", str(tmp_path / "addon"))
    write_profile(tmp_path / "addon")
    s = FakeToy()
    try:
        robot = s.connect(lambda: sros.connect("toy_biped", timeout=1.0))
        tool = sros.use_ros(approve=lambda *a: True)
        status, text = _call(tool, "publish", {"topic": "/toy/command", "type": "toy_msgs/msg/Command"})
        assert status == "error" and "streamed by robot" in text
        status, text = _call(tool, "status")
        assert "/toy/command" in json.loads(text)["owned"]
        robot.close()
    finally:
        s.close()


def test_tf_lookup_composes_the_tree(ros):
    from strands_robots.ros.bus import RosBus
    from strands_robots.ros.tf import TfReader

    bus = RosBus().start()
    reader = TfReader(bus)
    n = ros.node("tf_pub")
    from .robots import LATCH

    pub = n.create_publisher(ros.msg("tf2_msgs/TFMessage"), "/tf_static", LATCH)
    T = ros.msg("geometry_msgs/TransformStamped")

    def tr(parent, child, x, qz=0.0, qw=1.0):
        t = T()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x = x
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        return t

    s = 2**-0.5
    pub.publish(ros.msg("tf2_msgs/TFMessage")(transforms=[tr("base", "arm", 1.0, s, s), tr("arm", "hand", 0.5)]))
    out = wait_until(lambda: "hand" in reader.edges and reader.lookup("base", "hand"))
    np.testing.assert_allclose(out["xyz"], [1.0, 0.5, 0.0], atol=1e-6)  # 0.5 along arm-x = base-y after a 90° yaw
    bus.close()


def test_presence_shows_a_strands_robot_on_ros_and_gates_inbound_commands(ros, tmp_path, monkeypatch):
    from strands_robots import Robot
    from strands_robots.ros import RosPresence
    from strands_robots.ros.bus import RosBus

    path = tmp_path / "toy.urdf"
    path.write_text(urdf({"a": (-1.0, 1.0), "b": (-1.0, 1.0)}, "toy"))
    approvals = []
    robot = Robot(path, driver="memory", id="toy", confirm=lambda t: approvals.append(list(t)) or True)
    with pytest.raises(PermissionError, match="STRANDS_ROS_OPEN_COMMANDS"):
        RosPresence(robot, commands=True)
    monkeypatch.setenv("STRANDS_ROS_OPEN_COMMANDS", "1")
    bus = RosBus().start()
    pres = RosPresence(robot, bus=bus, rate_hz=50, commands=True).start()
    seen = []
    n = ros.node("rviz")
    n.create_subscription(ros.msg("sensor_msgs/JointState"), "/toy/joint_states", seen.append, _be())
    desc = []
    from .robots import LATCH

    n.create_subscription(ros.msg("std_msgs/String"), "/toy/robot_description", desc.append, LATCH)
    cmd = n.create_publisher(ros.msg("sensor_msgs/JointState"), "/toy/joint_command", REL)

    def drain():
        while True:
            try:
                cb, m = n.queue.get_nowait()
                cb(m)
            except Exception:  # noqa: BLE001
                return

    assert wait_until(lambda: (drain(), seen)[1])
    drain()
    assert list(seen[-1].name) == ["a", "b"] and "<robot" in desc[0].data
    cmd.publish(ros.msg("sensor_msgs/JointState")(name=["b"], position=[0.5]))
    assert wait_until(lambda: robot.observe().joints.positions[1] == pytest.approx(0.5))
    assert approvals  # the robot's own operator gate was asked
    cmd.publish(ros.msg("sensor_msgs/JointState")(name=["b"], position=[float("nan")]))
    cmd.publish(ros.msg("sensor_msgs/JointState")(name=["zz"], position=[0.1]))
    cmd.publish(ros.msg("sensor_msgs/JointState")(name=["a"], position=[5.0]))  # outside limits: the chain refuses
    assert wait_until(lambda: pres.refused == 3)
    pres.stop()
    bus.close()


def _be():
    from .robots import BE

    return BE


class _Sess:
    """A fake onnxruntime session with a typical walk-export contract: joints + 0.1, gains 30/2, carry."""

    class _IO:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    def __init__(self, n):
        self.n, self.calls = n, []

    def get_inputs(self):
        return [
            self._IO("joint_positions_control", [1, self.n]),
            self._IO("joint_velocities_control", [1, self.n]),
            self._IO("imu_quat", [1, 4]),
            self._IO("base_ang_vel", [1, 3]),
            self._IO("joystick_axes", [1, 3]),
            self._IO("obs_hist", [1, 6]),
            self._IO("last_actions", [1, self.n]),
        ]

    def get_outputs(self):
        return [
            self._IO(x, None)
            for x in (
                "motor_targets_control",
                "stiffness_control",
                "damping_control",
                "updated_obs_hist",
                "updated_last_actions",
            )
        ]

    def run(self, _names, feed):
        self.calls.append(feed)
        q = feed["joint_positions_control"]
        return [q + 0.1, np.full_like(q, 30.0), np.full_like(q, 2.0), feed["obs_hist"] + 1, q]


def test_multi_onnx_policy_runs_in_wire_order_with_carry_gains_and_crossfade(ros, tmp_path, monkeypatch):
    from strands_robots import ros as sros
    from strands_robots.policy.onnx.multi import MultiOnnxPolicy

    monkeypatch.setenv("STRANDS_ROBOTS_ROS_PROFILES", str(tmp_path / "addon"))
    write_profile(tmp_path / "addon")
    s = FakeToy()
    try:
        robot = s.connect(lambda: sros.connect("toy_biped", timeout=1.0))
        sess = _Sess(4)
        pol = MultiOnnxPolicy(
            "unused.onnx",
            inputs={
                "joint_positions_control": "joints.positions",
                "joint_velocities_control": "joints.velocities",
                "imu_quat": "pose.orientation",
                "base_ang_vel": "pose.angular_velocity",
                "joystick_axes": "command",
                "obs_hist": "carry",
                "last_actions": "carry",
            },
            outputs={"targets": "motor_targets_control", "kp": "stiffness_control", "kd": "damping_control"},
            carry={"obs_hist": "updated_obs_hist", "last_actions": "updated_last_actions"},
            joint_order=sros.wire_order(robot),
            command=lambda: [0.5, 0.0, 0.0],
            session=sess,
        )
        pol.blend_steps = 2
        pol.attach(robot)
        obs = robot.observe(images=False)
        a1 = pol.act(obs)
        a2 = pol.act(obs)
        assert sros.wire_order(robot) == ["knee", "neck", "hip", "ankle"]  # hardware ids 1, 2, 3, 7
        feed = sess.calls[0]
        k = list(robot.action_keys).index("hip")
        assert feed["joint_positions_control"][0, 2] == pytest.approx(obs.joints.positions[k])
        np.testing.assert_allclose(feed["imu_quat"][0], [0.7071, 0, 0, 0.7071], atol=1e-4)
        np.testing.assert_allclose(feed["joystick_axes"][0], [0.5, 0, 0])
        assert sess.calls[1]["obs_hist"][0, 0] == 1.0  # carry fed back
        np.testing.assert_allclose(a1, obs.joints.positions + 0.05)  # half-way crossfade
        np.testing.assert_allclose(a2, obs.joints.positions + 0.1)
        assert np.all(robot.body._policy_gains[0] == 30.0) and np.all(robot.body._policy_gains[1] == 2.0)
        robot.close()
    finally:
        s.close()


def test_without_ros_importing_is_free_and_using_says_how_to_get_rclpy(tmp_path):
    code = (
        "import sys; import strands_robots, strands_robots.ros, strands_robots.robot.drivers.ros\n"
        "assert not any(m == 'rclpy' or m.startswith('rclpy.') for m in sys.modules), 'rclpy imported'\n"
        "assert 'strands' not in sys.modules\n"
        "from strands_robots.ros import RosBus, RosUnavailable\n"
        "try:\n    RosBus().start()\nexcept RosUnavailable as e:\n    print('HINT', e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "source /opt/ros/" in out.stdout


def test_profiles_load_validate_and_take_overlays(ros, tmp_path, monkeypatch):
    from strands_robots.ros import profile as P

    assert set(P.available()) == {"unitree_g1", "ros2_control"}  # built in; add-ons bring their own
    g1 = P.load("unitree_g1")
    assert g1.wire_id("left_knee_joint") == 3 and g1.gains("kp", ("left_knee_joint",)) == [100.0]
    assert g1.need("command")["requires_release"] is True
    over = P._OVERLAY.expanduser()
    over.mkdir(parents=True, exist_ok=True)
    (over / "unitree_g1.json").write_text(json.dumps({"odom": {"topic": "/my_odom"}}))
    assert P.load("unitree_g1").need("odom")["topic"] == "/my_odom"
    with pytest.raises(ValueError):
        P.load("unitree_g1", overrides={"state": {"topic": "/bad topic"}})
    with pytest.raises(ValueError, match="family"):
        P.load("unitree_g1", overrides={"family": ""})
    monkeypatch.setenv("STRANDS_ROBOTS_ROS_PROFILES", str(tmp_path / "addon"))
    write_profile(tmp_path / "addon")
    assert "toy_biped" in P.available() and "/toy/mode_request" in P.motion_surfaces()
