"""Pin: Mesh._dispatch re-checks policy_host even when validate_command is skipped.

The wire path (``_on_cmd`` -> ``_exec_cmd`` -> ``validate_command`` ->
``_dispatch``) already allowlist-checks ``policy_host``. This test pins a
defence-in-depth re-check INSIDE ``_dispatch`` so that any caller reaching
``_dispatch`` directly -- bypassing ``validate_command`` -- still cannot
drive the robot to connect to an un-allowlisted VLA policy server.

These tests call ``_dispatch`` directly (NOT through ``validate_command``)
with an off-allowlist ``policy_host`` and assert:

* a structured ``{"error": ...}`` dict is returned, and
* the robot's ``_execute_task_sync`` / ``start_task`` is never invoked.

This test fails on pre-fix code (where ``_dispatch`` forwarded any
``policy_host`` straight to ``_execute_task_sync``).
"""

from __future__ import annotations

from typing import Any

from strands_robots.mesh import Mesh


class _RecordingRobot:
    """Records whether the actuation entry points were reached."""

    def __init__(self) -> None:
        self.executed = False
        self.started = False

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}

    def _execute_task_sync(
        self, instruction: str, provider: str, port: Any, host: str, duration: float, **kw: Any
    ) -> dict[str, Any]:
        self.executed = True
        return {"executed": instruction, "host": host}

    def start_task(
        self, instruction: str, provider: str, port: Any, host: str, duration: float, **kw: Any
    ) -> dict[str, Any]:
        self.started = True
        return {"started": instruction, "host": host}


def test_dispatch_rejects_off_allowlist_policy_host_execute() -> None:
    r = _RecordingRobot()
    m = Mesh(r, peer_id="p")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "go",
            "policy_provider": "mock",
            "policy_host": "evil.example.com",
        }
    )
    assert "error" in out
    assert "allowlist" in out["error"]
    assert r.executed is False, "robot must not actuate against an un-allowlisted host"


def test_dispatch_rejects_off_allowlist_policy_host_start() -> None:
    r = _RecordingRobot()
    m = Mesh(r, peer_id="p")
    out = m._dispatch(
        {
            "action": "start",
            "instruction": "go",
            "policy_provider": "mock",
            "policy_host": "10.13.37.7",
        }
    )
    assert "error" in out
    assert r.started is False


def test_dispatch_allows_loopback_policy_host() -> None:
    """Default allowlist (loopback) still works -- the guard is not a blanket block."""
    r = _RecordingRobot()
    m = Mesh(r, peer_id="p")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "go",
            "policy_provider": "mock",
            "policy_host": "localhost",
        }
    )
    assert r.executed is True
    assert out.get("executed") == "go"


def test_dispatch_default_policy_host_is_loopback() -> None:
    """Omitting policy_host defaults to localhost and is allowed."""
    r = _RecordingRobot()
    m = Mesh(r, peer_id="p")
    out = m._dispatch({"action": "execute", "instruction": "go", "policy_provider": "mock"})
    assert r.executed is True
    assert out.get("executed") == "go"


def test_dispatch_honors_policy_host_allow_env(monkeypatch) -> None:
    """Operator-extended allowlist via env lets a custom host through _dispatch."""
    # Clear the cached allowlist parse so the new env value is picked up.
    from strands_robots.mesh import security as _security

    monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "vla.internal")
    _security._policy_host_allowlist_cached.cache_clear()
    try:
        r = _RecordingRobot()
        m = Mesh(r, peer_id="p")
        out = m._dispatch(
            {
                "action": "execute",
                "instruction": "go",
                "policy_provider": "mock",
                "policy_host": "vla.internal",
            }
        )
        assert r.executed is True
        assert out.get("executed") == "go"
    finally:
        _security._policy_host_allowlist_cached.cache_clear()
