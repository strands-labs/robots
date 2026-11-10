"""The generated role configs on a real ``zenohd`` (``ZENOHD=/path`` or on PATH; skipped without one).

Mirrors LeRobot's own ``test_documented_mtls_router_acl``: the server and the robot exchange frames and actions,
while a second robot certificate and a same-CA certificate with another name cannot answer for the server.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import time

import pytest

from strands_robots.policy.lerobot import make
from strands_robots.policy.lerobot.roles import ROBOT, issue, prefix, role_config

zenoh = pytest.importorskip("zenoh")
pytest.importorskip("cryptography")
ZENOHD = os.environ.get("ZENOHD") or shutil.which("zenohd")
pytestmark = pytest.mark.skipif(ZENOHD is None, reason="needs a zenohd binary")
SESSION = prefix("arm-1") + "/instances/i/sessions/s"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def router(tmp_path):
    port = free_port()
    roles = make("arm-1", tmp_path, port=port)
    with (tmp_path / "router.log").open("w") as log:
        proc = subprocess.Popen([str(ZENOHD), "-c", str(roles.router)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    socket.create_connection(("localhost", port), timeout=0.05).close()
                    break
                except OSError:
                    time.sleep(0.05)
            yield roles, tmp_path
        finally:
            proc.terminate()
            proc.wait(timeout=5)


def as_cn(roles, tmp_path, cn: str, name: str) -> str:
    """A robot-style config presenting a fresh certificate named ``cn`` from the same CA."""
    certs = tmp_path / "certs"
    issue(certs, cn, name)
    cfg = role_config(ROBOT, roles.endpoint, certs)
    tls = cfg["transport"]["link"]["tls"]
    tls |= {"connect_certificate": str(certs / f"{name}.pem"), "connect_private_key": str(certs / f"{name}-key.pem")}
    path = tmp_path / f"{name}.json5"
    path.write_text(json.dumps(cfg))
    return str(path)


def drain(q: queue.Queue, wait: float) -> list[bytes]:
    time.sleep(wait)
    return [q.get_nowait() for _ in range(q.qsize())]


def test_only_the_policy_server_role_answers_and_only_the_roles_see_frames(router):
    roles, tmp_path = router
    with (
        zenoh.open(zenoh.Config.from_file(str(roles.server))) as server,
        zenoh.open(zenoh.Config.from_file(str(roles.robot))) as robot,
        zenoh.open(zenoh.Config.from_file(as_cn(roles, tmp_path, ROBOT, "robot-2"))) as robot2,
        zenoh.open(zenoh.Config.from_file(as_cn(roles, tmp_path, "outsider", "outsider"))) as outsider,
    ):
        acts, frames, stolen = queue.Queue(), queue.Queue(), queue.Queue()
        robot.declare_subscriber(SESSION + "/act", lambda s: acts.put(bytes(s.payload)))
        server.declare_subscriber(SESSION + "/obs", lambda s: frames.put(bytes(s.payload)))
        outsider.declare_subscriber(SESSION + "/obs", lambda s: stolen.put(bytes(s.payload)))
        server.declare_queryable(prefix("arm-1") + "/describe", lambda q: q.reply(q.key_expr, b"descriptor"))
        time.sleep(0.5)
        robot.put(SESSION + "/obs", b"frame")
        assert drain(frames, 0.5) == [b"frame"] and drain(stolen, 0.0) == []
        robot2.put(SESSION + "/act", b"forged by a second robot")
        outsider.put(SESSION + "/act", b"forged by another name")
        assert drain(acts, 0.5) == []
        server.put(SESSION + "/act", b"action")
        assert drain(acts, 0.5) == [b"action"]
        describe = prefix("arm-1") + "/describe"
        assert [bytes(r.ok.payload) for r in robot.get(describe, timeout=2) if r.ok] == [b"descriptor"]
        assert [r for r in outsider.get(describe, timeout=0.5) if r.ok] == []


def test_a_client_without_a_certificate_cannot_connect(router):
    roles, tmp_path = router
    cfg = {"mode": "client", "connect": {"endpoints": [roles.endpoint], "timeout_ms": 2000, "exit_on_failure": True}}
    cfg["transport"] = {"link": {"tls": {"root_ca_certificate": str(tmp_path / "certs" / "ca.pem")}}}
    with pytest.raises(Exception, match="Unable to connect"):
        zenoh.open(zenoh.Config.from_json5(json.dumps(cfg)))
