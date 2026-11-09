"""LeRobot's role-based router setup: the generated configs, and a real robot that joins only in the robot role."""

from __future__ import annotations

import json
import stat

import pytest

from strands_robots import Policy, Robot
from strands_robots.policy.lerobot import ZenohPolicy, make
from strands_robots.policy.lerobot.__main__ import main
from strands_robots.policy.lerobot.roles import ROBOT, SERVER, common_name, issue, prefix, role_config

pytest.importorskip("zenoh")
pytest.importorskip("cryptography")


class Connected(Exception):
    def __init__(self, wire: str) -> None:
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


def test_make_writes_the_three_role_configs_and_their_certificates(tmp_path):
    roles = make("arm-1", tmp_path, hosts=("gpu-box", "10.0.0.5"), port=7450)
    assert roles.endpoint == "tls/gpu-box:7450"
    assert common_name(str(roles.robot)) == ROBOT and common_name(str(roles.server)) == SERVER
    for name in ("ca", "router", SERVER, ROBOT):
        assert stat.S_IMODE((tmp_path / "certs" / f"{name}-key.pem").stat().st_mode) == 0o600
    for f in (roles.server, roles.robot):
        cfg = json.loads(f.read_text())
        assert cfg["mode"] == "client" and cfg["connect"]["endpoints"] == ["tls/gpu-box:7450"]
        assert cfg["transport"]["link"]["tls"]["enable_mtls"] is True
        assert cfg["scouting"] == {"multicast": {"enabled": False}, "gossip": {"enabled": False}}


def test_the_router_denies_by_default_and_scopes_every_rule_to_the_deployment(tmp_path):
    acl = json.loads(make("arm-1", tmp_path).router.read_text())["access_control"]
    assert acl["enabled"] is True and acl["default_permission"] == "deny"
    assert {s["id"]: s["cert_common_names"] for s in acl["subjects"]} == {"server": [SERVER], "robot": [ROBOT]}
    rules = {r["id"]: r for r in acl["rules"]}
    assert all(k.startswith(prefix("arm-1") + "/") for r in rules.values() for k in r["key_exprs"])
    assert [k.rsplit("/", 1)[1] for k in rules["server-output"]["key_exprs"]] == ["act", "result"]
    assert [k.rsplit("/", 1)[1] for k in rules["robot-output"]["key_exprs"]] == ["obs", "request"]
    granted = {p["subjects"][0]: p["rules"] for p in acl["policies"]}
    assert "server-output" not in granted["robot"] and "robot-output" not in granted["server"]


def test_rerunning_keeps_the_ca_and_the_certificates(tmp_path):
    make("arm-1", tmp_path)
    before = {p.name: p.read_bytes() for p in (tmp_path / "certs").iterdir()}
    make("arm-2", tmp_path)
    assert {p.name: p.read_bytes() for p in (tmp_path / "certs").iterdir()} == before


@pytest.mark.parametrize("name", ["", "a/b", "*", "-x", "a" * 129])
def test_a_deployment_name_lerobot_would_refuse_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="not a LeRobot name"):
        make(name, tmp_path)


def test_the_cli_writes_the_configs_and_says_how_to_run_them(tmp_path, capsys):
    assert main(["zenoh-config", "--deployment", "arm-1", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"zenohd -c {tmp_path.resolve() / 'router.json5'}" in out and "zenoh://tls/localhost:7447/arm-1" in out


def test_a_real_robot_connects_with_the_generated_robot_config(connect, real, tmp_path):
    roles = make("arm-1", tmp_path, hosts=("gpu-box",))
    with pytest.raises(Connected) as got:
        Policy(f"zenoh://{roles.endpoint}/arm-1", zenoh_config=str(roles.robot)).attach(real)
    assert got.value.wire == roles.endpoint


@pytest.mark.parametrize("cn", [SERVER, "outsider", "router"])
def test_a_real_robot_refuses_a_certificate_that_is_not_the_robot_role(connect, real, tmp_path, cn):
    issue(tmp_path / "certs", cn, "other")
    cfg = role_config(ROBOT, "tls/gpu-box:7447", tmp_path / "certs")
    cfg["transport"]["link"]["tls"] |= {
        "connect_certificate": str(tmp_path / "certs" / "other.pem"),
        "connect_private_key": str(tmp_path / "certs" / "other-key.pem"),
    }
    (tmp_path / "other.json5").write_text(json.dumps(cfg))
    with pytest.raises(PermissionError, match=f"presents the certificate '{cn}'"):
        Policy("zenoh://tls/gpu-box:7447/arm-1", zenoh_config=str(tmp_path / "other.json5")).attach(real)


def test_a_real_robot_refuses_to_join_as_a_peer(connect, real, tmp_path):
    roles = make("arm-1", tmp_path)
    with pytest.raises(PermissionError, match="zenoh_mode='peer'"):
        Policy(f"zenoh://{roles.endpoint}/arm-1", zenoh_config=str(roles.robot), zenoh_mode="peer").attach(real)


def test_a_server_in_the_policy_server_role_joins_the_router_as_a_client(tmp_path):
    pytest.importorskip("lerobot")
    from strands_robots.policy.lerobot.link import server_zenoh

    roles = make("arm-1", tmp_path)
    z = server_zenoh(roles.endpoint, str(roles.server))
    assert (z.mode, z.connect_endpoints, z.listen_endpoints) == ("client", [roles.endpoint], [])
    z = server_zenoh("tcp/127.0.0.1:7447", None)
    assert (z.mode, z.listen_endpoints) == ("peer", ["tcp/127.0.0.1:7447"])
