"""A real robot takes a remote LeRobot policy from off its own LAN only once the operator vouches for the link."""

from __future__ import annotations

import pytest

from strands_robots import Policy, Robot
from strands_robots.policy.lerobot import ZenohPolicy


class Connected(Exception):
    """The policy went on to connect: the router check let it through."""


@pytest.fixture
def connect(monkeypatch):
    def refuse(self):
        raise Connected

    monkeypatch.setattr(ZenohPolicy, "_connect", refuse)
    monkeypatch.delenv("STRANDS_ZENOH_TRUSTED", raising=False)


@pytest.mark.parametrize("link", ["tls", "quic", "ws", "wss"])
def test_a_real_robot_refuses_a_server_behind_a_router_nobody_vouched_for(connect, link):
    real = Robot("so101", driver="memory")
    try:
        with pytest.raises(PermissionError, match="vouch for every client"):
            Policy(f"zenoh://{link}/gpu.example:7447/smolvla").attach(real)
    finally:
        real.close()


@pytest.mark.parametrize("spec", ["zenoh://gpu.example:7447/smolvla", "zenoh://54.200.10.20:7447/smolvla"])
def test_a_real_robot_refuses_a_plain_internet_server_nobody_vouched_for(connect, spec):
    real = Robot("so101", driver="memory")
    try:
        with pytest.raises(PermissionError, match="across the network"):
            Policy(spec).attach(real)
    finally:
        real.close()


@pytest.mark.parametrize("how", ["kwarg", "env"])
def test_a_vouched_for_router_is_connected_to(connect, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("STRANDS_ZENOH_TRUSTED", "1")
    real = Robot("so101", driver="memory")
    try:
        with pytest.raises(Connected):
            Policy("zenoh://tls/gpu.example:7447/smolvla", **({"trusted": True} if how == "kwarg" else {})).attach(real)
    finally:
        real.close()


@pytest.mark.parametrize(
    "spec, real", [("zenoh://gpu-box:7447/smolvla", True), ("zenoh://tls/gpu.example:7447/s", False)]
)
def test_a_direct_link_or_a_simulated_robot_is_unchanged(connect, spec, real):
    robot = Robot("so101", driver="memory") if real else Robot("so101")
    try:
        with pytest.raises(Connected):
            Policy(spec).attach(robot)
    finally:
        robot.close()
