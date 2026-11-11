"""RosDriver behind a real strands-robots Robot: a made-up add-on robot, a ros2_control arm and a G1 (fake graph)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from .conftest import wait_until
from .robots import G1_NAMES, FakeArm, FakeG1, urdf
from .toy import FakeToy, write_profile


@pytest.fixture
def toy(ros, tmp_path, monkeypatch):
    """The toy's profile comes from a directory on STRANDS_ROBOTS_ROS_PROFILES; its codec by dotted path."""
    monkeypatch.setenv("STRANDS_ROBOTS_ROS_PROFILES", str(tmp_path / "addon"))
    write_profile(tmp_path / "addon")
    t = FakeToy()
    yield t
    t.close()


def _connect(t, **kw):
    from strands_robots import ros

    return t.connect(lambda: ros.connect("toy_biped", host="toy.local", timeout=1.0, **kw))


def _armed(t, robot):
    t.set_mode(3)
    assert wait_until(lambda: robot.observe(images=False).extras.get("armed"))


def test_an_out_of_tree_robot_connects_with_its_model_from_the_robot(toy):
    from strands_robots.ros import available

    assert "toy_biped" in available()
    robot = _connect(toy)
    assert robot.is_real and robot.id == "toy_biped" and set(robot.action_keys) == {"hip", "knee", "ankle", "neck"}
    obs = robot.observe(images=False)
    assert obs.joints.positions[list(robot.action_keys).index("ankle")] == pytest.approx(0.7)
    np.testing.assert_allclose(obs.pose.orientation, [0.7071, 0, 0, 0.7071])
    np.testing.assert_allclose(obs.pose.angular_velocity, [0, 0, 0.25])
    assert obs.extras["mode"] == "STAND" and obs.extras["armed"] is False  # labels come from the profile


def test_profile_armed_check_refuses_writes_and_sends_nothing(toy):
    robot = _connect(toy)
    with pytest.raises(PermissionError, match="mode=STAND .needs LOWLEVEL."):
        robot.body.write(np.array(robot.observe(images=False).joints.positions))
    toy.frames(5)
    time.sleep(0.05)
    assert toy.commands == []  # not armed: the robot's own controller owns the body
    assert robot.body.stream.stats["skipped_unarmed"] >= 5


def test_frame_synced_stream_answers_every_frame_and_echoes_its_id(toy):
    robot = _connect(toy)
    _armed(toy, robot)
    target = np.array(robot.observe(images=False).joints.positions)
    target[0] += 0.05
    robot.body.write(target)
    ids = toy.frames(30)  # no new target for 30 frames: the stream must keep answering
    assert wait_until(lambda: len(toy.commands) >= 30)
    assert [c.seq for c in toy.commands[-30:]] == ids[-30:]  # one per frame, none skipped
    c = toy.commands[-1]
    assert len(c.cmds) == 4 and c.frame_stamp.sec == 7
    assert all(j.kp > 0 and j.kd > 0 for j in c.cmds)  # gains from the robot's gains topic, never a damp


def test_deadman_holds_the_measured_pose_never_damps(toy):
    robot = _connect(toy)
    _armed(toy, robot)
    t = np.array(robot.observe(images=False).joints.positions)
    t[0] += 0.02
    robot.body.write(t)
    time.sleep(0.35)  # past the 300 ms deadman
    toy.frames(3)
    assert wait_until(lambda: len(toy.commands) >= 3)
    last = {j.id: j for j in toy.commands[-1].cmds}
    assert last[3].q == pytest.approx(toy.q[3], abs=1e-6)  # measured pose, not the stale target
    assert last[3].kp == pytest.approx(23.0) and robot.body.stream.stats["held"] >= 3


def test_estop_sends_the_profile_stop_and_keeps_holding(toy):
    robot = _connect(toy)
    _armed(toy, robot)
    robot.estop()
    toy.frames(3)
    assert wait_until(lambda: len(toy.twists) >= 2)
    assert wait_until(lambda: len(toy.commands) >= 3)
    assert all(j.kp > 0 for j in toy.commands[-1].cmds)


def test_shadow_mode_publishes_nothing_and_measures_the_gap(toy, ros):
    robot = _connect(toy, shadow=True)
    _armed(toy, robot)
    cmd = ros.msg("toy_msgs/JointCmd")
    toy.reference.publish(ros.msg("toy_msgs/Command")(cmds=[cmd(id=i, q=q + 0.1) for i, q in toy.q.items()]))
    time.sleep(0.05)
    robot.body.write(np.array(robot.observe(images=False).joints.positions))
    toy.frames(5)
    time.sleep(0.05)
    assert toy.commands == []
    assert robot.body.stream.stats["shadow_max_err"] == pytest.approx(0.1, abs=1e-5)


def test_policy_gains_reach_the_wire_and_bad_targets_are_refused_whole(toy):
    robot = _connect(toy)
    _armed(toy, robot)
    q = np.array(robot.observe(images=False).joints.positions)
    robot.set_gains(np.full(4, 33.0), np.full(4, 2.5))
    robot.body.write(q)
    toy.frames(2)
    assert wait_until(lambda: toy.commands and {j.kp for j in toy.commands[-1].cmds} == {33.0})
    bad = q.copy()
    bad[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        robot.body.write(bad)
    far = q.copy()
    far[0] = 9.0
    with pytest.raises(ValueError, match="whole frame"):
        robot.body.write(far)


def test_run_loop_drives_the_robot_through_the_stream(toy):
    robot = _connect(toy)
    _armed(toy, robot)
    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            toy.frame()
            time.sleep(0.005)

    threading.Thread(target=feeder, daemon=True).start()
    q0 = np.array(robot.observe(images=False).joints.positions)
    k = list(robot.action_keys).index("neck")

    def policy(obs):
        t = q0.copy()
        t[k] = 0.3
        return t

    result = robot.run(policy, hz=50, seconds=0.4)
    stop.set()
    assert result.steps >= 10
    assert [j.q for j in toy.commands[-1].cmds if j.id == 2][0] == pytest.approx(0.3)


def test_codec_and_profile_from_entry_points(ros, tmp_path, monkeypatch):
    """An installed add-on: a strands_robots.ros_profiles entry point (a directory) and a ros_codecs one."""
    from importlib.metadata import EntryPoint

    from strands_robots.robot.drivers.ros import codec as codec_mod
    from strands_robots.ros import profile as profile_mod

    from . import toy

    write_profile(tmp_path / "pkg", codec="")
    groups = {
        "strands_robots.ros_profiles": [EntryPoint("toy", str(tmp_path / "pkg"), "strands_robots.ros_profiles")],
        "strands_robots.ros_codecs": [EntryPoint("toy", f"{toy.__name__}:ToyCodec", "strands_robots.ros_codecs")],
    }
    monkeypatch.setattr(profile_mod, "entry_points", lambda group: groups.get(group, []))
    monkeypatch.setattr(codec_mod, "entry_points", lambda group: groups.get(group, []))
    monkeypatch.setattr(EntryPoint, "load", lambda self: self.value if ":" not in self.value else toy.ToyCodec)
    prof = profile_mod.load("toy_biped")
    assert prof.family == "toy" and codec_mod.codec_class(prof) is toy.ToyCodec
    assert "/toy/command" in profile_mod.motion_surfaces()


def test_a_codec_that_is_not_a_codec_and_an_unknown_family_are_refused(ros, tmp_path, monkeypatch):
    from strands_robots.robot.drivers.ros import codec_class
    from strands_robots.ros import profile as profile_mod

    monkeypatch.setenv("STRANDS_ROBOTS_ROS_PROFILES", str(tmp_path))
    write_profile(tmp_path, codec="json:dumps")
    with pytest.raises(TypeError, match="not a strands_robots Codec"):
        codec_class(profile_mod.load("toy_biped"))
    write_profile(tmp_path, codec="")
    with pytest.raises(ValueError, match="no codec for family 'toy'"):
        codec_class(profile_mod.load("toy_biped"))


def test_any_ros2_control_arm_with_model_from_robot_description(ros):
    from strands_robots import ros as sros

    arm = FakeArm()
    try:
        robot = sros.connect("ros2_control", timeout=1.0)
        assert robot.action_keys == ("shoulder", "elbow", "wrist")
        np.testing.assert_allclose(robot.observe(images=False).joints.positions, [0.1, -0.2, 0.3])  # keyed by name
        # the hold frame on connect is the measured pose in the controller's slot order (wrist, elbow, shoulder)
        assert wait_until(lambda: arm.commands)
        assert list(arm.commands[0].data) == pytest.approx([0.3, -0.2, 0.1])
        robot.act({"elbow": 0.5})
        assert wait_until(lambda: arm.commands and arm.commands[-1].data[1] == pytest.approx(0.5))
        assert list(arm.commands[-1].data) == pytest.approx([0.3, 0.5, 0.1])  # omitted joints keep their position
    finally:
        arm.close()


def test_a_publish_error_drops_one_frame_never_the_stream(ros):
    from strands_robots import ros as sros

    arm = FakeArm()
    try:
        robot = sros.connect("ros2_control", timeout=1.0)
        stream = robot.body.stream
        real, failed = stream.bus.publish, []

        def flaky(*a, **k):
            if not failed:
                failed.append(1)
                raise RuntimeError("rmw: publish failed")
            return real(*a, **k)

        stream.bus.publish = flaky
        assert wait_until(lambda: failed and stream.stats["errors"] >= 1)
        assert stream._thread.is_alive() and "publish" in stream.stats["last_reason"]
        robot.act({"elbow": 0.5})
        assert wait_until(lambda: arm.commands and arm.commands[-1].data[1] == pytest.approx(0.5))  # still on the wire
        stream.stop()
        stream._running = True  # the thread is gone but nobody said so: set_target must refuse, not accept silently
        with pytest.raises(ConnectionError, match="command stream stopped"):
            stream.set_target(np.zeros(3))
        stream._running = False
    finally:
        arm.close()


def test_an_observe_only_profile_observes_and_refuses_to_write(ros, tmp_path, monkeypatch):
    import json
    from importlib import resources

    from strands_robots import ros as sros

    prof = json.loads(resources.files("strands_robots.ros.profiles").joinpath("ros2_control.json").read_text())
    del prof["command"]  # watch /joint_states before writing anything
    (tmp_path / "watch_arm.json").write_text(json.dumps(prof))
    monkeypatch.setenv("STRANDS_ROBOTS_ROS_PROFILES", str(tmp_path))
    arm = FakeArm()
    try:
        robot = sros.connect("watch_arm", timeout=1.0)
        obs = robot.observe(images=False)
        np.testing.assert_allclose(obs.joints.positions, [0.1, -0.2, 0.3])
        with pytest.raises(Exception, match="observe-only"):
            robot.act({"elbow": 0.5})
        assert not arm.commands
    finally:
        arm.close()


def test_ros2_control_arm_without_a_controller_joint_order_streams_nothing(ros):
    from strands_robots import ros as sros

    arm = FakeArm(controller_joints=None)
    try:
        with pytest.raises(ConnectionError, match="command.joints"):
            sros.connect("ros2_control", timeout=1.0)
        assert not arm.commands
    finally:
        arm.close()
    arm = FakeArm(controller_joints=["shoulder", "elbow"])  # a controller over other joints than the model's
    try:
        with pytest.raises(ValueError, match="not exactly"):
            sros.connect("ros2_control", timeout=1.0)
        assert not arm.commands
    finally:
        arm.close()


def test_g1_lowcmd_needs_release_then_streams_with_unitree_crc(ros, tmp_path):
    from strands_robots import ros as sros
    from strands_robots.robot.drivers.ros.crc import hg_lowcmd_crc

    path = tmp_path / "g1.urdf"
    path.write_text(urdf({f"{n}_joint": (-2.5, 2.5) for n in G1_NAMES}, "g1"))
    g1 = FakeG1()
    try:
        robot = sros.connect("unitree_g1", model=path, timeout=1.0)
        obs = robot.observe(images=False)
        assert obs.joints.positions[3] == pytest.approx(0.03) and obs.extras["mode_machine"] == 5
        with pytest.raises(PermissionError, match="released=True"):
            robot.body.write(np.array(obs.joints.positions))
        robot.close()
        from strands_robots.ros.bus import RosBus

        RosBus.shared().owners.clear()
        robot = sros.connect("unitree_g1", model=path, timeout=1.0, released=True)
        robot.body.write(np.array(obs.joints.positions))
        assert wait_until(lambda: len(g1.cmds) > 50, timeout=2.0)
        c = g1.cmds[-1]
        rows = [[m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve] for m in c.motor_cmd]
        assert c.crc == hg_lowcmd_crc(c.mode_pr, c.mode_machine, rows, c.reserve)
        assert c.mode_machine == 5 and c.motor_cmd[3].kp == 100 and c.motor_cmd[3].mode == 1
        assert c.motor_cmd[30].mode == 0 and c.motor_cmd[30].kp == 0  # unused slots disabled
        robot.close()
    finally:
        g1.close()


def test_an_unknown_profile_and_a_partial_joint_map_are_refused(ros, tmp_path):
    from strands_robots.description import Description
    from strands_robots.robot.drivers.ros import RosDriver

    path = tmp_path / "x.urdf"
    path.write_text(urdf({"a": (-1, 1)}, "x"))
    with pytest.raises(KeyError, match="no ROS profile"):
        RosDriver(Description.load(path), profile="nope", timeout=0.1)
    with pytest.raises(ValueError, match="no wire id"):
        RosDriver(Description.load(path), profile="unitree_g1", timeout=0.1)


@pytest.mark.network
def test_robot_facade_takes_driver_ros_and_host_for_a_registry_robot(ros):
    """The registry's G1 MJCF (robot_descriptions), Robot(...) unchanged but for host= meaning hardware."""
    import os

    from strands_robots import Robot

    g1 = FakeG1()
    try:
        robot = Robot("g1", driver="ros", host="192.168.123.164", timeout=1.0)
        assert robot.is_real and len(robot.action_keys) == 29
        assert os.environ["ROS_STATIC_PEERS"] == "192.168.123.164" and os.environ["ROS_DOMAIN_ID"] == "0"
        assert robot.observe(images=False).joints.positions[3] == pytest.approx(0.03)
        robot.close()
    finally:
        g1.close()


def test_a_ros2_control_arm_takes_the_profiles_on_disconnect_so_it_can_carry_a_watchdog(ros):
    """A model read from /robot_description has no registry hardware block. Without the profile's declared
    on_disconnect ("hold" in ros2_control.json) a real robot cannot carry a watchdog, so the mesh serves it
    observe-only and a generic ros2_control arm can never be commanded over the mesh (measured on a mock-hardware
    arm: "no proven safe on_disconnect for this body")."""
    from strands_robots import ros as sros
    from strands_robots.robot.safety import Watchdog

    arm = FakeArm()
    try:
        robot = sros.connect("ros2_control", timeout=1.0)
        assert robot.description.hardware.on_disconnect == "hold"
        assert {c.on_disconnect for c in robot.manifest.controls} == {"hold"}
        Watchdog(robot, 500.0)  # refused with RequestRefused when the body has no proven-safe on_disconnect
    finally:
        arm.close()


def test_connect_with_mesh_mounts_after_the_profiles_on_disconnect_is_applied(ros):
    """ros.connect(..., mesh=) used to forward mesh into Robot(), which mounts inside the constructor, before the
    profile's on_disconnect was applied, so the watchdog refused the body: "ros2_control: no proven safe
    on_disconnect for this body" (review on #4935). The one-call form must mount the patched robot."""
    import uuid

    from strands_robots import ros as sros
    from strands_robots.mesh import Mesh

    arm, m = FakeArm(), Mesh(f"mem://ros-connect-{uuid.uuid4().hex[:6]}", tenant="t")
    try:
        robot = sros.connect("ros2_control", timeout=1.0, mesh=m)
        assert robot.description.hardware.on_disconnect == "hold"
        assert {c.on_disconnect for c in robot.manifest.controls} == {"hold"}
        robot.close()
    finally:
        m.close()
        arm.close()
