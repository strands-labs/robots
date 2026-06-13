"""Tests for strands_robots.tools.robot_mesh — agent-facing dispatcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.robot_mesh import robot_mesh


def _make_tool_context(*, interrupt_response: str = "y", interrupt_raises: bool = False) -> MagicMock:
    """Build a stand-in ToolContext whose interrupt() returns *interrupt_response*.

    Tests that DON'T hit the interrupt path can still call this -- interrupt()
    will simply never be invoked. Tests that DO hit it (emergency_stop /
    broadcast) can vary `interrupt_response` to simulate operator approval /
    denial. Set `interrupt_raises=True` to model environments where interrupts
    aren't available (the tool should fail-closed).
    """
    ctx = MagicMock(name="ToolContext")
    if interrupt_raises:
        ctx.interrupt.side_effect = RuntimeError("interrupts not supported here")
    else:
        ctx.interrupt.return_value = interrupt_response
    return ctx


def _strands_call(*, _ctx: MagicMock | None = None, **kwargs):
    """Strands @tool wraps the function -- invoke via .original."""
    fn = getattr(robot_mesh, "original", None)
    ctx = _ctx if _ctx is not None else _make_tool_context()
    if fn is None:
        return robot_mesh(tool_context=ctx, **kwargs)
    return fn(tool_context=ctx, **kwargs)


@pytest.fixture
def fake_local_mesh():
    """Patch get_local_robots() to return a single fake mesh keyed by peer."""
    fake = MagicMock(name="LocalMesh")
    fake.peer_id = "local-a"
    fake.peer_type = "sim"
    fake.inbox = {}
    with (
        patch(
            "strands_robots.mesh.get_local_robots",
            return_value={"local-a": fake},
        ),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        yield fake


@pytest.fixture
def fake_no_local():
    """Patch get_local_robots()/get_peers() to return empty."""
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        yield


def test_peers_lists_local_and_remote(fake_local_mesh):
    with patch(
        "strands_robots.mesh.session.get_peers",
        return_value=[{"peer_id": "remote-1", "type": "robot", "hostname": "host1", "age": 3}],
    ):
        out = _strands_call(action="peers")
    assert out["status"] == "success"
    text = out["content"][0]["text"]
    assert "local-a" in text
    assert "remote-1" in text


def test_peers_no_local_no_remote(fake_no_local):
    out = _strands_call(action="peers")
    assert out["status"] == "success"
    assert "No peers" in out["content"][0]["text"]


def test_status_returns_counts(fake_local_mesh):
    out = _strands_call(action="status")
    assert out["status"] == "success"
    assert "local=1" in out["content"][0]["text"]


def test_tell_requires_target_and_instruction(fake_local_mesh):
    out = _strands_call(action="tell")
    assert out["status"] == "error"


def test_tell_invokes_mesh_tell(fake_local_mesh):
    fake_local_mesh.tell.return_value = {"executed": "go"}
    out = _strands_call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "success"
    fake_local_mesh.tell.assert_called_once()
    args = fake_local_mesh.tell.call_args
    assert args.args == ("peer-b", "go")


def test_send_requires_command(fake_local_mesh):
    out = _strands_call(action="send", target="peer-b")
    assert out["status"] == "error"
    assert "command" in out["content"][0]["text"].lower()


def test_send_rejects_invalid_json(fake_local_mesh):
    out = _strands_call(action="send", target="peer-b", command="not json")
    assert out["status"] == "error"
    assert "JSON" in out["content"][0]["text"]


def test_send_invokes_mesh_send(fake_local_mesh):
    fake_local_mesh.send.return_value = {"ok": 1}
    out = _strands_call(
        action="send",
        target="peer-b",
        command='{"action": "status"}',
        timeout=5.0,
    )
    assert out["status"] == "success"
    args = fake_local_mesh.send.call_args
    assert args.args[0] == "peer-b"
    assert args.args[1] == {"action": "status"}
    assert args.kwargs["timeout"] == 5.0


def test_broadcast_invokes_mesh_broadcast(fake_local_mesh):
    fake_local_mesh.broadcast.return_value = [{"a": 1}, {"b": 2}]
    out = _strands_call(action="broadcast", command='{"action":"status"}')
    assert out["status"] == "success"
    assert "2 responses" in out["content"][0]["text"]


def test_stop_requires_target(fake_local_mesh):
    out = _strands_call(action="stop")
    assert out["status"] == "error"


def test_stop_sends_stop_action(fake_local_mesh):
    fake_local_mesh.send.return_value = {"stopped": True}
    _strands_call(action="stop", target="peer-b")
    args = fake_local_mesh.send.call_args
    assert args.args[1] == {"action": "stop"}


def test_emergency_stop_invokes_mesh_emergency_stop(fake_local_mesh):
    fake_local_mesh.emergency_stop.return_value = [{"a": 1}, {"b": 2}]
    out = _strands_call(action="emergency_stop")
    assert out["status"] == "success"
    fake_local_mesh.emergency_stop.assert_called_once()
    assert "2 responses" in out["content"][0]["text"]


def test_subscribe_requires_target(fake_local_mesh):
    out = _strands_call(action="subscribe")
    assert out["status"] == "error"


def test_subscribe_calls_mesh_subscribe(fake_local_mesh):
    # Use an allowlisted topic class -- subscribing to a low-impact shared
    # class (presence) is permitted by the tool-layer allowlist.
    fake_local_mesh.subscribe.return_value = "topic-name"
    out = _strands_call(action="subscribe", target="**/presence", name="presence")
    assert out["status"] == "success"
    fake_local_mesh.subscribe.assert_called_once()


def test_subscribe_rejects_off_allowlist_target(fake_local_mesh):
    # Subscribing to another peer's cmd stream is not in the default
    # allowlist and subscribe is not gated by default -> rejected.
    out = _strands_call(action="subscribe", target="reachy/cmd", name="reachy")
    assert out["status"] == "error"
    assert "allowed topic set" in out["content"][0]["text"]
    fake_local_mesh.subscribe.assert_not_called()


def test_watch_requires_target(fake_local_mesh):
    out = _strands_call(action="watch")
    assert out["status"] == "error"


def test_watch_calls_on_stream(fake_local_mesh, monkeypatch):
    # Extend the subscribe allowlist so the watch target passes the
    # telemetry-leak defence-in-depth gate (watch validates against the
    # equivalent Zenoh key strands/<target>/stream).
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    from strands_robots.tools.robot_mesh import _reset_subscribe_allowlist_cache

    _reset_subscribe_allowlist_cache()
    fake_local_mesh.on_stream.return_value = "stream:peer-b"
    out = _strands_call(action="watch", target="peer-b")
    assert out["status"] == "success"
    fake_local_mesh.on_stream.assert_called_once_with("peer-b")
    _reset_subscribe_allowlist_cache()


def test_inbox_returns_buffered_messages(fake_local_mesh):
    fake_local_mesh.inbox = {"sub-a": [("topic", {"x": 1}), ("topic", {"x": 2})]}
    out = _strands_call(action="inbox", name="sub-a")
    assert out["status"] == "success"
    text = out["content"][0]["text"]
    assert "2 total" in text


def test_inbox_with_no_messages(fake_local_mesh):
    out = _strands_call(action="inbox", name="empty")
    assert out["status"] == "success"
    assert "no messages" in out["content"][0]["text"]


def test_unknown_action_returns_error(fake_local_mesh):
    out = _strands_call(action="warp")
    assert out["status"] == "error"
    assert "unknown action" in out["content"][0]["text"]


def test_actions_without_local_mesh_fail(fake_no_local):
    out = _strands_call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    assert "no local mesh" in out["content"][0]["text"]


# ---------------------------------------------------------------------------
# Regression: _resolve_mesh self-loop fix
#
# Before this fix, when the agent issued ``send/tell/stop`` to a target that
# matched a *local* peer_id, ``_resolve_mesh`` would return the target's own
# Mesh as the gateway.  ``Mesh.send`` then published on
# ``strands/{target}/cmd`` with ``sender_id == target`` — the receiving
# subscriber drops self-loops, so the call silently timed out.  The fix:
# pick a *different* local mesh as the gateway whenever one exists.
# ---------------------------------------------------------------------------


def test_resolve_mesh_avoids_self_loop_when_alternative_exists():
    """When target matches a local peer_id, pick a different local mesh."""
    from strands_robots.tools.robot_mesh import _resolve_mesh

    mesh_a = MagicMock(name="mesh_a")
    mesh_a.peer_id = "robot-a"
    mesh_b = MagicMock(name="mesh_b")
    mesh_b.peer_id = "robot-b"

    locals_ = {"robot-a": mesh_a, "robot-b": mesh_b}

    with patch("strands_robots.mesh.get_local_robots", return_value=locals_):
        gateway = _resolve_mesh("robot-b")
        # MUST be mesh_a (the OTHER local mesh) — never mesh_b itself,
        # which would self-loop.
        assert gateway is mesh_a, (
            f"_resolve_mesh returned {gateway.peer_id!r} but should have "
            "returned 'robot-a' to avoid the send-to-self self-loop."
        )


def test_resolve_mesh_fallback_when_target_is_only_local():
    """When the target IS the only local mesh, fall back to it.

    The caller will get a timeout (since the message self-drops) — that's
    the expected behaviour for "send to yourself" with no other local
    gateway available.
    """
    from strands_robots.tools.robot_mesh import _resolve_mesh

    only = MagicMock(name="only")
    only.peer_id = "robot-x"

    with patch("strands_robots.mesh.get_local_robots", return_value={"robot-x": only}):
        gateway = _resolve_mesh("robot-x")
        assert gateway is only


def test_resolve_mesh_returns_first_when_target_is_remote():
    """When target doesn't match any local peer, any local mesh is a fine gateway."""
    from strands_robots.tools.robot_mesh import _resolve_mesh

    mesh_a = MagicMock(name="mesh_a")
    mesh_a.peer_id = "robot-a"
    mesh_b = MagicMock(name="mesh_b")
    mesh_b.peer_id = "robot-b"

    with patch(
        "strands_robots.mesh.get_local_robots",
        return_value={"robot-a": mesh_a, "robot-b": mesh_b},
    ):
        gateway = _resolve_mesh("remote-c")
        assert gateway in (mesh_a, mesh_b)


def test_send_to_local_peer_does_not_use_target_as_gateway(fake_no_local):
    """End-to-end: robot_mesh(action='send', target=local_peer) must not
    route the call through the target's own Mesh (would self-loop)."""

    mesh_a = MagicMock(name="mesh_a")
    mesh_a.peer_id = "alpha"
    mesh_a.send.return_value = {"ok": "from-a"}

    mesh_b = MagicMock(name="mesh_b")
    mesh_b.peer_id = "beta"
    mesh_b.send.return_value = {"should-not-be-called": True}

    locals_ = {"alpha": mesh_a, "beta": mesh_b}
    with patch("strands_robots.mesh.get_local_robots", return_value=locals_):
        out = _strands_call(
            action="send",
            target="beta",
            command='{"action": "status"}',
            timeout=2.0,
        )

    assert out["status"] == "success"
    # mesh_a must be the gateway because target == "beta" must NOT route via
    # mesh_b (would self-loop).
    mesh_a.send.assert_called_once()
    mesh_b.send.assert_not_called()
    args = mesh_a.send.call_args
    assert args.args[0] == "beta"  # outbound target unchanged
