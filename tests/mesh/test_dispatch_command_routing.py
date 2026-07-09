"""Behavior tests for the ``Mesh._dispatch`` command-router contract.

``Mesh._dispatch`` is the synchronous table that maps a validated wire command
(``{"action": ...}``) onto the local robot's control surface. Two contracts are
pinned here that the existing suite left uncovered:

1. Teleop actions (``teleop_status`` / ``teleop_receive`` / ``teleop_stop``) are
   forwarded to the robot when it implements the corresponding method, and the
   robot's own return dict is passed straight back to the caller. (The
   robot-lacks-the-method fallbacks are covered elsewhere; this pins the
   supported path, where the robot's result must be returned verbatim.)
2. The emergency-stop lockout gate: while ``_estop_lockout`` is engaged only
   ``status`` and ``resume`` are permitted; every other action is rejected with
   ``LockoutError`` (so ``_exec_cmd`` emits a generic wire error and an audit
   entry), and ``resume`` is routed to the lockout-release path instead of being
   blocked.

``_dispatch`` is called directly (not through the wire/validate path) on a
``Mesh`` built without ``start()`` -- the same construction the existing
``test_dispatch_policy_host_guard`` tests use, so no Zenoh session is created.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh import security as _security


class _TeleopRobot:
    """Robot that implements the teleop control surface.

    Each method records its arguments and returns a distinctive dict so the
    test can assert ``_dispatch`` forwards the call and returns the result
    verbatim.
    """

    def __init__(self) -> None:
        self.received: tuple[str, str] | None = None
        self.stopped: str | None = None

    def get_teleop_status(self) -> dict[str, Any]:
        return {"inputs": ["leader"], "publishers": {"leader": 1}, "receivers": {}}

    def start_teleop_receive(self, source: str, device: str) -> dict[str, Any]:
        self.received = (source, device)
        return {"receiving_from": source, "device": device}

    def stop_teleop(self, device: str | None) -> dict[str, Any]:
        self.stopped = device
        return {"stopped": device or "all"}


class _StatusRobot:
    """Minimal robot exposing only ``get_task_status`` (no teleop surface)."""

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}


def test_dispatch_teleop_status_forwarded_to_robot() -> None:
    m = Mesh(_TeleopRobot(), peer_id="p")

    out = m._dispatch({"action": "teleop_status"})

    assert out == {"inputs": ["leader"], "publishers": {"leader": 1}, "receivers": {}}


def test_dispatch_teleop_receive_forwards_source_and_device() -> None:
    robot = _TeleopRobot()
    m = Mesh(robot, peer_id="p")

    out = m._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1", "device_name": "leader"})

    assert out == {"receiving_from": "leader-1", "device": "leader"}
    assert robot.received == ("leader-1", "leader")


def test_dispatch_teleop_receive_device_defaults_to_leader() -> None:
    robot = _TeleopRobot()
    m = Mesh(robot, peer_id="p")

    # ``device_name`` omitted -> the dispatch table defaults it to "leader".
    out = m._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1"})

    assert out == {"receiving_from": "leader-1", "device": "leader"}
    assert robot.received == ("leader-1", "leader")


def test_dispatch_teleop_stop_forwarded_to_robot() -> None:
    robot = _TeleopRobot()
    m = Mesh(robot, peer_id="p")

    out = m._dispatch({"action": "teleop_stop", "device_name": "leader"})

    assert out == {"stopped": "leader"}
    assert robot.stopped == "leader"


def test_dispatch_rejects_actuation_while_estop_lockout_engaged() -> None:
    m = Mesh(_StatusRobot(), peer_id="p")
    m._estop_lockout.set()

    # Any action other than status/resume must be rejected loudly so
    # ``_exec_cmd`` records the audit entry and emits a generic wire error.
    with pytest.raises(_security.LockoutError):
        m._dispatch({"action": "execute", "instruction": "go"})


def test_dispatch_allows_status_while_estop_lockout_engaged() -> None:
    m = Mesh(_StatusRobot(), peer_id="p")
    m._estop_lockout.set()

    # ``status`` is one of the two always-permitted actions under lockout.
    out = m._dispatch({"action": "status"})

    assert out == {"status": "idle"}


def test_dispatch_routes_resume_to_lockout_release_even_under_lockout(monkeypatch) -> None:
    # No override code configured -> the release is rejected deterministically.
    monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
    m = Mesh(_StatusRobot(), peer_id="p")
    m._estop_lockout.set()

    # ``resume`` is not blocked by the lockout gate; it is routed to
    # ``_resume_lockout``. With no override code configured the release is
    # rejected, and the response is the generic (oracle-free) shape.
    out = m._dispatch({"action": "resume", "override_code": "wrong-code"})

    assert out.get("status") == "error"
    assert out.get("error") == "resume rejected"
    # The lockout stays engaged after a rejected resume.
    assert m._estop_lockout.is_set()


def test_dispatch_stop_falls_back_when_robot_lacks_stop_task() -> None:
    # A robot with no ``stop_task`` (e.g. a bare status-only peer) still gets a
    # well-formed acknowledgement rather than an error or an exception.
    m = Mesh(_StatusRobot(), peer_id="p")

    out = m._dispatch({"action": "stop"})

    assert out == {"ok": True}
