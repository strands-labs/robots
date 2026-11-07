from __future__ import annotations

import time

import pytest

from . import fake_rclpy


@pytest.fixture
def ros(monkeypatch, tmp_path):
    """Fake rclpy installed, a clean graph, a clean shared RosBus and discovery state."""
    import sys

    for k in [k for k in sys.modules if k == "rclpy" or k.startswith(("rclpy.", "rosidl_runtime_py"))]:
        monkeypatch.delitem(sys.modules, k, raising=False)
    handles = fake_rclpy.install()
    fake_rclpy.reset()
    for k in (
        "ROS_DOMAIN_ID",
        "RMW_IMPLEMENTATION",
        "ZENOH_CONFIG_OVERRIDE",
        "ROS_STATIC_PEERS",
        "CYCLONEDDS_URI",
        "STRANDS_ROS_ALLOW",
        "STRANDS_ROS_OPEN_COMMANDS",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "ZENOH_ROUTER_CHECK_ATTEMPTS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("STRANDS_ROBOTS_ROS_PROFILES", raising=False)
    from strands_robots.ros import describe, discovery, profile
    from strands_robots.ros.bus import RosBus

    monkeypatch.setattr(describe, "CACHE", tmp_path / "cache" / "ros")
    monkeypatch.setattr(profile, "_OVERLAY", tmp_path / "overlay")
    discovery.reset_for_tests()
    RosBus._shared = None
    yield handles
    if RosBus._shared is not None:
        RosBus._shared.close()
    RosBus._shared = None
    discovery.reset_for_tests()
    for k in list(__import__("os").environ):
        if k in (
            "ROS_DOMAIN_ID",
            "RMW_IMPLEMENTATION",
            "ZENOH_CONFIG_OVERRIDE",
            "ROS_STATIC_PEERS",
            "CYCLONEDDS_URI",
            "ROS_AUTOMATIC_DISCOVERY_RANGE",
            "ZENOH_ROUTER_CHECK_ATTEMPTS",
        ):
            __import__("os").environ.pop(k)


def wait_until(pred, timeout: float = 2.0, every: float = 0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(every)
    return pred()
