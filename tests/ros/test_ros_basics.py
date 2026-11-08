"""Names, QoS, discovery and the bus, on the fake graph."""

from __future__ import annotations

import os

import pytest

from .conftest import wait_until
from .robots import BE, LATCH, REL


def test_names_are_checked_before_anything_touches_ros():
    from strands_robots.ros.names import canonical, domain_id, interface_type

    assert canonical("cmd_vel") == "/cmd_vel"
    for bad in ("", "/a//b", "/a/", "/1abc", "/a b", "/ü", "/a__b"):
        with pytest.raises(ValueError):
            canonical(bad)
    assert interface_type("std_msgs/String") == "std_msgs/msg/String"
    with pytest.raises(ValueError):
        interface_type("std_msgs/srv/String", kind="msg")
    assert domain_id(232) == 232
    for bad in (233, -1, True, 1.0, "3"):
        with pytest.raises(ValueError):
            domain_id(bad)


def test_host_becomes_a_zenoh_connect_or_a_dds_peer():
    from strands_robots.ros.discovery import Discovery

    z = Discovery(host="robot.local", rmw="rmw_zenoh_cpp").env()
    assert z["ZENOH_CONFIG_OVERRIDE"] == 'connect/endpoints=["tcp/robot.local:7447"]'
    assert z["RMW_IMPLEMENTATION"] == "rmw_zenoh_cpp"
    d = Discovery(host="10.0.0.5", rmw="rmw_cyclonedds_cpp", domain_id=0, iface="eth0").env()
    assert d["ROS_STATIC_PEERS"] == "10.0.0.5" and d["ROS_DOMAIN_ID"] == "0" and "eth0" in d["CYCLONEDDS_URI"]
    assert Discovery().env() == {}
    with pytest.raises(ValueError):
        Discovery(host="x; rm -rf /").env()


def test_a_second_different_graph_in_one_process_is_refused(ros):
    from strands_robots.ros.bus import RosBus
    from strands_robots.ros.discovery import Discovery

    RosBus.shared(Discovery(domain_id=3)).start()
    assert os.environ["ROS_DOMAIN_ID"] == "3"
    RosBus.shared(Discovery(domain_id=3))  # the same one is fine
    with pytest.raises(RuntimeError, match="already up"):
        RosBus.shared(Discovery(domain_id=4))


def test_auto_qos_hears_a_best_effort_writer_that_reliable_would_miss(ros):
    from strands_robots.ros.bus import RosBus

    bus = RosBus().start()
    n = ros.node("cam")
    pub = n.create_publisher(ros.msg("std_msgs/String"), "/imu", BE)
    reliable = bus.subscribe("/imu_rel", "std_msgs/msg/String", qos="reliable")
    pub2 = n.create_publisher(ros.msg("std_msgs/String"), "/imu_rel", BE)
    auto = bus.subscribe("/imu", "std_msgs/msg/String", qos="auto")
    assert auto.qos == "sensor"
    pub.publish(ros.msg("std_msgs/String")(data="hi"))
    pub2.publish(ros.msg("std_msgs/String")(data="lost"))
    assert wait_until(lambda: bus.latest("/imu"))[0].data == "hi"
    assert bus.latest("/imu_rel") is None and reliable.count == 0  # the old RELIABLE-only bug, reproduced
    bus.close()


def test_latched_topic_reaches_a_late_reader_and_one_type_per_topic(ros):
    from strands_robots.ros.bus import RosBus

    n = ros.node("mapper")
    n.create_publisher(ros.msg("std_msgs/String"), "/map_state", LATCH).publish(
        ros.msg("std_msgs/String")(data="Mapping")
    )
    bus = RosBus().start()
    assert bus.wait_first("/map_state", "std_msgs/msg/String", 1.0).data == "Mapping"
    with pytest.raises(ValueError, match="one type per topic"):
        bus.subscribe("/map_state", "std_msgs/msg/Bool")
    bus.close()


def test_publishers_are_cached_and_command_topics_have_one_owner(ros):
    from strands_robots.ros.bus import RosBus

    bus = RosBus().start()
    a = bus.publisher("/x", "std_msgs/msg/String")
    assert bus.publisher("/x", "std_msgs/msg/String") is a
    bus.claim("/cmd", "robot-a")
    bus.claim("/cmd", "robot-a")
    with pytest.raises(RuntimeError, match="one writer"):
        bus.claim("/cmd", "robot-b")
    bus.release("/cmd", "robot-a")
    bus.claim("/cmd", "robot-b")
    bus.close()


def test_graph_stats_and_heal(ros):
    from strands_robots.ros.bus import RosBus

    bus = RosBus().start()
    n = ros.node("talker")
    p = n.create_publisher(ros.msg("std_msgs/String"), "/chatter", REL)
    bus.subscribe("/chatter", "std_msgs/msg/String", qos="reliable")
    for _ in range(3):
        p.publish(ros.msg("std_msgs/String")(data="x"))
    wait_until(lambda: bus.stats()["/chatter"]["count"] == 3)
    g = bus.graph()
    assert "/chatter" in g["topics"] and "/talker" in g["nodes"]
    assert bus.heal(silent_s=0.0) == ["/chatter"]  # a quiet sub is re-created
    p.publish(ros.msg("std_msgs/String")(data="again"))
    assert wait_until(lambda: bus.latest("/chatter")[0].data == "again")
    bus.close()
