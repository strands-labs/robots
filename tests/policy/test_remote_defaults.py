"""What a zenoh link implies for LeRobot's client settings: the LAN keeps LeRobot's defaults, the internet does not."""

import pytest

from strands_robots.policy.lerobot.link import WAN_TIMEOUT_S, is_local, wan_defaults
from strands_robots.policy.lerobot.remote import ZenohPolicy


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("tcp/127.0.0.1:7447", True),
        ("tcp/localhost:7447", True),
        ("tcp/10.0.0.5:7447", True),
        ("tcp/192.168.1.20:7447", True),
        ("tcp/gpu-box.local:7447", True),
        ("tcp/gpu-box:7447", True),  # the documented LAN form: a single-label hostname has no TLD
        ("tcp/[::1]:7447", True),
        ("tcp/169.254.7.7:7447", True),
        ("tls/zenoh.api.agi.diy:7447", False),
        ("tls/h.example:7447", False),  # dotted = a domain = the internet, even a short one
        ("tcp/32.192.2.218:7449", False),
        ("quic/[2606:4700::1111]:7447", False),
    ],
)
def test_the_lan_is_known_by_its_address_or_its_bare_hostname(endpoint, local):
    assert is_local(endpoint) is local


def test_a_lan_endpoint_keeps_lerobots_defaults():
    assert wan_defaults("tcp/127.0.0.1:7447", {}) == {}
    assert wan_defaults("tcp/10.0.0.5:7447", {"encoding": "jpeg"}) == {"encoding": "jpeg"}
    assert wan_defaults("tcp/gpu-box:7447", {}) == {}  # docs/policies.md's own example stays raw + 5 s


def test_an_internet_endpoint_gets_wide_timeouts_jpeg_frames_and_a_router_mode():
    got = wan_defaults("tls/zenoh.api.agi.diy:7447", {})
    assert got == {
        "zenoh_mode": "client",
        "handshake_timeout_s": WAN_TIMEOUT_S,
        "action_timeout_s": WAN_TIMEOUT_S,
        "startup_timeout_s": WAN_TIMEOUT_S,
        "encoding": "jpeg",
    }
    assert wan_defaults("tcp/32.192.2.218:7449", {})["encoding"] == "jpeg"
    assert "zenoh_mode" not in wan_defaults("tcp/32.192.2.218:7449", {})  # a plain tcp link is a peer


def test_the_callers_word_wins_over_every_default():
    got = wan_defaults("tls/h.example:7447", {"action_timeout_s": 2.0, "encoding": "raw", "zenoh_mode": "peer"})
    assert (got["action_timeout_s"], got["encoding"], got["zenoh_mode"]) == (2.0, "raw", "peer")


def test_the_hello_widens_the_action_timeout_to_the_servers_own_deadline():
    hello = {"limits": {"action_deadline_s": 12.0}}
    assert wan_defaults("tcp/127.0.0.1:7447", {}, hello)["action_timeout_s"] == 12.0  # LAN: the deadline itself
    assert wan_defaults("tls/h.example:7447", {}, hello)["action_timeout_s"] == WAN_TIMEOUT_S  # 12 + 5 < 30
    assert wan_defaults("tls/h.example:7447", {}, {"limits": {"action_deadline_s": 40.0}})["action_timeout_s"] == 45.0
    assert wan_defaults("tcp/127.0.0.1:7447", {}, {"limits": {"action_deadline_s": 5.0}}) == {"action_timeout_s": 5.0}
    assert wan_defaults("tcp/127.0.0.1:7447", {"action_timeout_s": 1.0}, hello)["action_timeout_s"] == 1.0


def test_the_config_a_far_policy_builds_carries_the_defaults():
    pytest.importorskip("lerobot.remote_inference")
    cfg = ZenohPolicy(endpoint="tls/zenoh.api.agi.diy:7447", deployment="d")._config()
    assert (cfg.endpoint, cfg.zenoh_mode, cfg.encoding) == ("tls/zenoh.api.agi.diy:7447", "client", "jpeg")
    assert cfg.handshake_timeout_s == cfg.action_timeout_s == cfg.startup_timeout_s == WAN_TIMEOUT_S
    near = ZenohPolicy(endpoint="tcp/127.0.0.1:7447", deployment="d")._config()
    assert (near.zenoh_mode, near.encoding, near.action_timeout_s) == ("peer", "raw", 5.0)
