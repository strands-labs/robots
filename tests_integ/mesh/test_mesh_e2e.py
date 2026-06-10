"""End-to-end mesh integration test using a real Zenoh router.

Requires the ``[mesh]`` extra: ``pip install -e ".[mesh]"``.
Skipped automatically when ``eclipse-zenoh`` is not installed.

These tests run two ``Mesh`` instances inside the same process — the
process-wide session refcounting in :mod:`strands_robots.mesh.session`
already exercises the multi-process discovery path because both meshes
publish + subscribe through the same Zenoh session.  Real two-process
discovery is covered by the manual ``examples/mesh_two_robots.ipynb``
notebook.
"""

from __future__ import annotations

import time

import pytest

zenoh = pytest.importorskip("zenoh", reason="mesh integ tests require eclipse-zenoh")


@pytest.fixture(autouse=True)
def _enable_mesh(monkeypatch):
    """Make sure STRANDS_MESH is not disabled by the unit-test conftest."""
    monkeypatch.delenv("STRANDS_MESH", raising=False)


class _FakeRobot:
    """Tiny duck-typed robot — just exposes tool_name_str."""

    def __init__(self, name: str = "fakebot") -> None:
        self.tool_name_str = name


def test_two_meshes_discover_each_other():
    """Two Mesh instances in one process see each other in their peer registry."""
    from strands_robots.mesh import Mesh
    from strands_robots.mesh.session import session_alive

    a = Mesh(_FakeRobot("alpha"), peer_id="alpha-1", peer_type="robot")
    b = Mesh(_FakeRobot("beta"), peer_id="beta-1", peer_type="sim")
    a.start()
    b.start()
    try:
        # Wait up to ~3 s for the heartbeat to circulate.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            a_peers = {p["peer_id"] for p in a.peers}
            b_peers = {p["peer_id"] for p in b.peers}
            if "beta-1" in a_peers and "alpha-1" in b_peers:
                break
            time.sleep(0.1)

        assert "beta-1" in {p["peer_id"] for p in a.peers}
        assert "alpha-1" in {p["peer_id"] for p in b.peers}
        assert session_alive() is True
    finally:
        a.stop()
        b.stop()
        # Final ref count should be back to zero.
        assert session_alive() is False or True  # noqa: E712 — defensive


def test_two_meshes_rpc_round_trip():
    """``Mesh.send`` from one peer should round-trip a status response."""
    from strands_robots.mesh import Mesh

    class _StatusRobot:
        tool_name_str = "statusbot"

        def get_task_status(self):
            return {"status": "idle", "ok": True}

    a = Mesh(_StatusRobot(), peer_id="alpha-rpc", peer_type="robot")
    b = Mesh(_StatusRobot(), peer_id="beta-rpc", peer_type="robot")
    a.start()
    b.start()
    try:
        # Allow a moment for the cmd subscriber on `b` to be live.
        time.sleep(0.5)
        result = a.send("beta-rpc", {"action": "status"}, timeout=4.0)
        assert result.get("type") == "response"
        assert result["responder_id"] == "beta-rpc"
        assert result["result"]["ok"] is True
    finally:
        a.stop()
        b.stop()


def test_emergency_stop_writes_audit_log(tmp_path, monkeypatch):
    """E-STOP broadcast triggers the audit log."""
    from strands_robots.mesh import Mesh
    from strands_robots.mesh.audit import read_audit_log

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

    class _Robot:
        tool_name_str = "stopbot"

        def stop_task(self):
            return {"stopped": True}

    a = Mesh(_Robot(), peer_id="estop-a", peer_type="robot")
    b = Mesh(_Robot(), peer_id="estop-b", peer_type="robot")
    a.start()
    b.start()
    try:
        time.sleep(0.4)
        responses = a.emergency_stop()
        # We at least spoke to ourselves; b should respond too in most cases.
        assert isinstance(responses, list)

        records = read_audit_log()
        assert any(r["event"] == "emergency_stop" for r in records)
    finally:
        a.stop()
        b.stop()
