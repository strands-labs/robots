"""``python -m strands_robots.mesh``: CA, certificates and a zenoh config file a session accepts."""

import json

import pytest

pytest.importorskip("cryptography")

from strands_robots.mesh.__main__ import main  # noqa: E402
from strands_robots.mesh.identity import Identity  # noqa: E402


def test_ca_cert_and_config_make_an_identity_and_a_mutual_tls_config(tmp_path, capsys):
    ca, dev = tmp_path / "ca", tmp_path / "dev"
    assert main(["ca", "--ca", str(ca)]) == 0
    assert main(["cert", "lab-pc", "--ca", str(ca), "--out", str(dev), "--serves", "so101, arm-*"]) == 0
    assert Identity.load(dev).signer.serves == ("so101", "arm-*")
    out = tmp_path / "router.json5"
    assert (
        main(["config", "--identity", str(dev), "--mode", "router", "--listen", "tls/0.0.0.0:7447", "-o", str(out)])
        == 0
    )
    cfg = json.loads(out.read_text())
    assert cfg["mode"] == "router" and cfg["listen"]["endpoints"] == ["tls/0.0.0.0:7447"]
    assert cfg["transport"]["link"]["protocols"] == ["tls"] and cfg["transport"]["link"]["tls"]["enable_mtls"] is True
    assert "lab-pc" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["cert", "a.b", "--ca", "{ca}", "--out", "{tmp}/x"],
        ["cert", "x", "--ca", "{tmp}/nowhere", "--out", "{tmp}/x"],
        ["config", "--identity", "{tmp}/nowhere"],
        ["config", "--identity", "{tmp}/id", "--listen", "tcp/0.0.0.0:7447"],
        ["ca", "--ca", "{ca}"],
    ],
)
def test_a_bad_request_says_what_is_wrong_and_exits_2(tmp_path, capsys, argv):
    ca = tmp_path / "ca"
    main(["ca", "--ca", str(ca)])
    main(["cert", "id", "--ca", str(ca), "--out", str(tmp_path / "id")])
    capsys.readouterr()
    assert main([a.format(ca=ca, tmp=tmp_path) for a in argv]) == 2
    assert capsys.readouterr().err.startswith("error: ")
