"""The zenoh mesh over mutual TLS: built from a mesh identity, it admits only certificate holders of its CA."""

import json
import socket
import time

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("cryptography")

from strands_robots.mesh import pki  # noqa: E402
from strands_robots.mesh.transports import open_transport, zenoh_tls  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("mtls")
    ca = pki.make_ca(root / "ca")
    dirs = {n: pki.issue(ca, n, root / n, serves=["*"] if n == "device" else ()) for n in ("device", "operator")}
    dirs["rogue"] = pki.issue(pki.make_ca(root / "rogue-ca"), "rogue", root / "rogue")
    return dirs


def _ask(listener, peer, key="strands/t/arm/rpc/observe"):
    heard = []
    undo = listener.queryable(key, lambda k, p: heard.append(p) or b"ok")
    try:
        time.sleep(0.5)
        return peer.query(key, b"hi", 1.0), heard
    finally:
        undo()


def test_two_identities_of_one_ca_talk_over_mutual_tls(lab):
    port = free_port()
    dev = open_transport(f"zenoh://?identity={lab['device']}&listen=tls/127.0.0.1:{port}")
    op = open_transport(f"zenoh://?identity={lab['operator']}&connect=tls/127.0.0.1:{port}")
    try:
        assert (dev.identity.name, op.identity.name) == ("device", "operator")
        assert not dev.open_peers and not op.open_peers
        assert _ask(dev, op) == ([b"ok"], [b"hi"])
    finally:
        op.close(), dev.close()


@pytest.mark.parametrize("who", ["plain-tcp", "tls-without-cert", "other-ca"])
def test_a_peer_without_a_certificate_of_the_mesh_ca_never_reaches_it(lab, who, tmp_path):
    zenoh = pytest.importorskip("zenoh")
    port = free_port()
    dev = open_transport(f"zenoh://?identity={lab['device']}&listen=tls/127.0.0.1:{port}")
    if who == "plain-tcp":
        cfg = {"connect": {"endpoints": [f"tcp/127.0.0.1:{port}"]}}
    elif who == "tls-without-cert":
        link = {"tls": {"root_ca_certificate": str(lab["device"] / "ca.pem"), "verify_name_on_connect": False}}
        cfg = {"connect": {"endpoints": [f"tls/127.0.0.1:{port}"]}, "transport": {"link": link}}
    else:
        cfg = zenoh_tls.tls_config(lab["rogue"], connect=[f"tls/127.0.0.1:{port}"])
        cfg["transport"]["link"]["tls"]["root_ca_certificate"] = str(lab["device"] / "ca.pem")
    cfg.update(mode="peer", scouting={"multicast": {"enabled": False}})
    cfg.setdefault("listen", {"endpoints": []})
    cfg["connect"].update(timeout_ms=1000, exit_on_failure=False)
    stranger = zenoh.open(zenoh.Config.from_json5(json.dumps(cfg)))
    heard = []
    undo = dev.queryable("strands/t/arm/rpc/act", lambda k, p: heard.append(p) or b"moved")
    try:
        time.sleep(0.5)
        replies = [r for r in stranger.get("strands/t/arm/rpc/act", payload=b"x", timeout=1.0) if r.ok is not None]
        stranger.put("strands/t/@estop", b"x")
        assert (replies, heard) == ([], [])
    finally:
        undo(), stranger.close(), dev.close()


def test_an_identity_speaks_only_tls_endpoints(lab):
    with pytest.raises(ValueError, match="only tls/"):
        open_transport(f"zenoh://?identity={lab['device']}&listen=tcp/127.0.0.1:{free_port()}")


def test_an_identity_and_a_config_file_together_are_refused(lab, tmp_path):
    path = tmp_path / "z.json5"
    path.write_text("{}")
    with pytest.raises(ValueError, match="not both"):
        open_transport(f"zenoh://?identity={lab['device']}&config={path}")


def test_a_config_file_with_mutual_tls_carries_its_identity(lab, tmp_path):
    port = free_port()
    path = tmp_path / "dev.json5"
    path.write_text(json.dumps(zenoh_tls.tls_config(lab["device"], listen=[f"tls/127.0.0.1:{port}"])))
    dev = open_transport(f"zenoh://?config={path}")
    try:
        assert dev.identity is not None and dev.identity.name == "device" and not dev.open_peers
    finally:
        dev.close()


@pytest.mark.parametrize(
    "link",
    [
        {},
        {"tls": {"enable_mtls": True, "root_ca_certificate": "ca.pem"}},  # no protocols: tcp/ still admitted
        {"protocols": ["tls", "tcp"], "tls": {"enable_mtls": True, "root_ca_certificate": "ca.pem"}},
        {"protocols": ["tls"], "tls": {"enable_mtls": False, "root_ca_certificate": "ca.pem"}},
        {"protocols": ["tls"], "tls": {"enable_mtls": True}},
    ],
)
def test_a_config_that_admits_peers_without_a_certificate_is_not_authenticated(link):
    assert not zenoh_tls.authenticated(link)
    assert zenoh_tls.identity_of(link) is None


def test_a_config_file_without_mutual_tls_leaves_the_mesh_open(tmp_path):
    path = tmp_path / "z.json5"
    path.write_text('{mode: "peer"}')
    t = open_transport(f"zenoh://?config={path}&listen=tcp/127.0.0.1:{free_port()}")
    try:
        assert t.identity is None and t.open_peers
    finally:
        t.close()


def test_the_default_identity_directory_is_used_only_when_it_holds_one(monkeypatch, tmp_path):
    monkeypatch.delenv("STRANDS_MESH_IDENTITY", raising=False)
    monkeypatch.delenv("STRANDS_ZENOH_CONFIG", raising=False)
    monkeypatch.setattr(zenoh_tls, "DEFAULT_IDENTITY", tmp_path / "identity")
    assert zenoh_tls.identity_dir("zenoh://") is None
    pki.issue(pki.make_ca(tmp_path / "ca"), "me", tmp_path / "identity")
    assert zenoh_tls.identity_dir("zenoh://") == str(tmp_path / "identity")
    assert zenoh_tls.identity_dir("zenoh://?identity=none") is None
    monkeypatch.setenv("STRANDS_MESH_IDENTITY", "none")
    assert zenoh_tls.identity_dir("zenoh://") is None
