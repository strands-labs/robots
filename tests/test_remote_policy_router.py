"""A real robot takes a remote LeRobot policy's actions only over an authenticated channel, or from this machine."""

from __future__ import annotations

import json

import pytest

from strands_robots import Policy, Robot
from strands_robots.policy.lerobot import ZenohPolicy
from strands_robots.policy.lerobot.link import address, is_local


class Connected(Exception):
    """The policy went on to connect: the channel check let it through, to ``wire``."""

    def __init__(self, wire: str):
        super().__init__(wire)
        self.wire = wire


@pytest.fixture
def connect(monkeypatch):
    def record(self):
        raise Connected(self._wire)

    monkeypatch.setattr(ZenohPolicy, "_connect", record)
    for var in ("STRANDS_ZENOH_TRUSTED", "STRANDS_ZENOH_CONFIG"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def real():
    robot = Robot("so101", driver="memory")
    yield robot
    robot.close()


def zenoh_file(tmp_path, *, mtls: bool) -> str:
    pytest.importorskip("zenoh")
    tls = {"root_ca_certificate": str(tmp_path / "ca.pem"), "enable_mtls": mtls}
    if mtls:
        tls |= {"connect_certificate": str(tmp_path / "cert.pem"), "connect_private_key": str(tmp_path / "key.pem")}
    path = tmp_path / "zenoh.json5"
    path.write_text(json.dumps({"transport": {"link": {"tls": tls}}}))
    return str(path)


@pytest.mark.parametrize(
    "spec",
    [
        "zenoh://gpu-box:7447/smolvla",  # a LAN name: whoever answers there, or a zenohd shared by the LAN
        "zenoh://192.168.1.20:7447/smolvla",
        "zenoh://gpu.local:7447/smolvla",
        "zenoh://gpu.example:7447/smolvla",
        "zenoh://54.200.10.20:7447/smolvla",
        "zenoh://919079444:7447/smolvla",  # 54.200.10.20 as one number
        "zenoh://[2002:808:808::1]:7447/smolvla",  # 6to4 for 8.8.8.8
        "zenoh://0.0.0.0:7447/smolvla",
        "zenoh://tls/gpu.example:7447/smolvla",  # a router with no client identity
        "zenoh://quic/gpu.example:7447/smolvla",
        "zenoh://ws/gpu.example:7447/smolvla",
        "zenoh://wss/gpu.example:443/smolvla",
    ],
)
def test_a_real_robot_refuses_a_link_with_no_identity(connect, real, spec):
    with pytest.raises(PermissionError, match="only over an authenticated channel; the link carries no identity"):
        Policy(spec).attach(real)


def test_a_real_robot_refuses_tls_that_does_not_authenticate_the_client(connect, real, tmp_path):
    one_way = zenoh_file(tmp_path, mtls=False)
    with pytest.raises(PermissionError, match="does not set transport.link.tls.enable_mtls"):
        Policy("zenoh://tls/gpu.example:7447/smolvla", zenoh_config=one_way).attach(real)


@pytest.mark.parametrize("link", ["tls", "quic"])
@pytest.mark.parametrize("how", ["kwarg", "env", "lerobot-name"])
def test_a_mutual_tls_link_is_connected_to_as_given(connect, real, tmp_path, monkeypatch, link, how):
    path = zenoh_file(tmp_path, mtls=True)
    kw = {"kwarg": {"zenoh_config": path}, "lerobot-name": {"zenoh_config_path": path}, "env": {}}[how]
    if how == "env":
        monkeypatch.setenv("STRANDS_ZENOH_CONFIG", path)
    with pytest.raises(Connected) as got:
        Policy(f"zenoh://{link}/gpu.example:7447/smolvla", **kw).attach(real)
    assert got.value.wire == f"{link}/gpu.example:7447"


def test_mutual_tls_does_not_cover_a_websocket_link(connect, real, tmp_path):
    with pytest.raises(PermissionError, match="authenticated channel"):
        Policy("zenoh://ws/gpu.example:7447/smolvla", zenoh_config=zenoh_file(tmp_path, mtls=True)).attach(real)


def test_an_unreadable_zenoh_config_is_an_error_not_a_fallback(connect, real, tmp_path):
    pytest.importorskip("zenoh")
    bad = tmp_path / "zenoh.json5"
    bad.write_text("{ transport: ")
    with pytest.raises(ValueError, match="zenoh config"):
        Policy("zenoh://tls/gpu.example:7447/smolvla", zenoh_config=str(bad)).attach(real)


@pytest.mark.parametrize(
    ("host", "wire"),
    [
        ("127.0.0.1", "tcp/127.0.0.1:7447"),
        ("localhost", "tcp/127.0.0.1:7447"),
        ("LOCALHOST.", "tcp/127.0.0.1:7447"),
        ("127.1", "tcp/127.0.0.1:7447"),
        ("2130706433", "tcp/127.0.0.1:7447"),
        ("[::ffff:127.0.0.1]", "tcp/127.0.0.1:7447"),
        ("[::1]", "tcp/[::1]:7447"),
    ],
)
def test_this_machine_is_connected_to_at_its_literal_address(connect, real, host, wire):
    with pytest.raises(Connected) as got:
        Policy(f"zenoh://{host}:7447/smolvla").attach(real)
    assert got.value.wire == wire  # zenoh never resolves a name the check did not judge


@pytest.mark.parametrize("how", ["kwarg", "env"])
def test_a_vouched_for_link_is_connected_to_as_given(connect, real, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("STRANDS_ZENOH_TRUSTED", "1")
    with pytest.raises(Connected) as got:
        Policy("zenoh://gpu-box:7447/smolvla", **({"trusted": True} if how == "kwarg" else {})).attach(real)
    assert got.value.wire == "tcp/gpu-box:7447"


@pytest.mark.parametrize("value", ["0", "false", "on", "1 "])
def test_only_an_explicit_yes_vouches(connect, real, monkeypatch, value):
    monkeypatch.setenv("STRANDS_ZENOH_TRUSTED", value)
    with pytest.raises(PermissionError):
        Policy("zenoh://gpu-box:7447/smolvla").attach(real)


def test_a_simulated_robot_is_not_gated(connect):
    sim = Robot("so101")
    try:
        with pytest.raises(Connected):
            Policy("zenoh://tls/gpu.example:7447/smolvla").attach(sim)
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("host", "addr"),
    [
        ("919079444", "54.200.10.20"),
        ("0x36c80a14", "54.200.10.20"),
        ("54.200.2580", "54.200.10.20"),
        ("010.010.010.010", "8.8.8.8"),
        ("::ffff:8.8.8.8", "8.8.8.8"),
        ("2002:808:808::1", "8.8.8.8"),
        ("fe80::1%en0", "fe80::1"),
        ("8.8.8.8.", "8.8.8.8"),
        ("gpu-box", None),
    ],
)
def test_a_host_is_read_as_an_address_whatever_its_spelling(host, addr):
    assert (None if address(host) is None else str(address(host))) == addr


@pytest.mark.parametrize(
    ("host", "local"),
    [("919079444", False), ("[2002:808:808::1]", False), ("3232235796", True), ("gpu-box", True), ("x.LOCAL", True)],
)
def test_the_wan_settings_guess_follows_the_same_reading(host, local):
    assert is_local(f"tcp/{host}:7447") is local
