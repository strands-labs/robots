"""A server reached at 127.0.0.1 through a tunnel to a far machine gets the internet's timeouts and JPEG frames:
the link is judged by its measured round trip (or the caller's ``network=``), not by the address alone."""

import pytest
from delay_link import tunnel
from test_lerobot import _arm

from strands_robots import Policy
from strands_robots.policy.lerobot.link import WAN_TIMEOUT_S
from strands_robots.policy.lerobot.network import FAR_RTT_S, is_far


def _session_config(p):
    inner = getattr(p, "inner", p)
    return inner, inner._s.client.config


@pytest.fixture
def far(deployment):
    """(direct url, tunnelled url): the same server, straight and through a 127.0.0.1 tunnel of ~60 ms round trip."""
    cfg, server = deployment(name="far")
    with server as url:
        port = int(cfg.zenoh.listen_endpoints[0].rsplit(":", 1)[1])
        yield url, url.replace(f":{port}/", f":{tunnel(port, 0.030)}/")


def _attached(url, **kw):
    arm, p = _arm(), Policy(url, **kw)
    p.attach(arm)
    return p


def test_a_tunnel_at_127_0_0_1_gets_the_internets_settings(far):
    _, tunnelled = far
    p = _attached(tunnelled)
    try:
        inner, cfg = _session_config(p)
        assert inner.network == "wan" and inner.rtt_s > FAR_RTT_S
        assert (cfg.encoding, cfg.action_timeout_s, cfg.startup_timeout_s) == ("jpeg", WAN_TIMEOUT_S, WAN_TIMEOUT_S)
    finally:
        p.detach()


def test_the_same_server_on_this_machine_keeps_lerobots_defaults(far):
    direct, _ = far
    p = _attached(direct)
    try:
        inner, cfg = _session_config(p)
        assert inner.network == "lan" and inner.rtt_s < FAR_RTT_S
        assert (cfg.encoding, cfg.action_timeout_s) == ("raw", 5.0)
    finally:
        p.detach()


@pytest.mark.parametrize(("asked", "which", "encoding"), [("wan", 0, "jpeg"), ("lan", 1, "raw")])
def test_the_callers_word_wins_without_a_probe(far, asked, which, encoding):
    p = _attached(far[which], network=asked)
    try:
        inner, cfg = _session_config(p)
        assert (inner.network, inner.rtt_s, cfg.encoding) == (asked, None, encoding)
    finally:
        p.detach()


def test_a_tunnelled_run_drives_the_arm(far):
    r = _arm().run(Policy(far[1]), seconds=1)
    assert r.stopped_by == "seconds" and r.fault is None, r


def test_is_far_reads_the_word_then_the_address_then_the_round_trip():
    never = lambda: pytest.fail("probed")  # noqa: E731
    assert is_far("tcp/127.0.0.1:7447", "wan", never) == (True, None)
    assert is_far("tls/h.example:7447", "lan", never) == (False, None)
    assert is_far("tls/h.example:7447", "auto", never) == (True, None)  # written off the LAN: no probe needed
    assert is_far("tcp/127.0.0.1:7447", "auto", lambda: 0.002) == (False, 0.002)
    assert is_far("tcp/127.0.0.1:7447", "auto", lambda: 0.080) == (True, 0.080)
    assert is_far("tcp/127.0.0.1:7447", "auto", lambda: None) == (True, None)  # no answer: the safe side


def test_network_must_be_one_it_knows():
    with pytest.raises(ValueError, match="network must be one of 'auto', 'lan', 'wan'"):
        Policy("zenoh://127.0.0.1:7447/d", network="internet")
