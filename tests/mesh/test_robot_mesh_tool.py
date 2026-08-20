"""Tests for strands_robots.tools.robot_mesh - agent-facing dispatcher."""

from __future__ import annotations

from typing import Any
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


# ---------------------------------------------------------------------------
# Observation / read-only action set: teardown + mesh-not-running paths.
#
# The read-only set (peers, status, inbox, unsubscribe) plus the
# subscribe/watch setup actions must each surface a clear, audited result -
# including the "no subscription named" guard and the "mesh not running"
# failure where the underlying transport returns None instead of a handle.
# ---------------------------------------------------------------------------


def test_unsubscribe_calls_mesh_unsubscribe(fake_local_mesh):
    out = _strands_call(action="unsubscribe", name="sub-a")
    assert out["status"] == "success"
    assert "unsubscribed from 'sub-a'" in out["content"][0]["text"]
    fake_local_mesh.unsubscribe.assert_called_once_with("sub-a")


def test_unsubscribe_falls_back_to_target_when_no_name(fake_local_mesh):
    # name is optional; the subscription handle defaults to target.
    out = _strands_call(action="unsubscribe", target="sub-b")
    assert out["status"] == "success"
    assert "unsubscribed from 'sub-b'" in out["content"][0]["text"]
    fake_local_mesh.unsubscribe.assert_called_once_with("sub-b")


def test_unsubscribe_requires_name_or_target(fake_local_mesh):
    out = _strands_call(action="unsubscribe")
    assert out["status"] == "error"
    assert "requires name" in out["content"][0]["text"]
    fake_local_mesh.unsubscribe.assert_not_called()


def test_inbox_requires_name_or_target(fake_local_mesh):
    out = _strands_call(action="inbox")
    assert out["status"] == "error"
    assert "requires name" in out["content"][0]["text"]


def test_subscribe_reports_mesh_not_running_when_handle_is_none(fake_local_mesh):
    # An allowlisted target that resolves but whose mesh.subscribe() returns
    # None (transport not running) must fail with an actionable message, not
    # claim success on a missing subscription handle.
    fake_local_mesh.subscribe.return_value = None
    out = _strands_call(action="subscribe", target="**/presence", name="presence")
    assert out["status"] == "error"
    assert "mesh not running" in out["content"][0]["text"]


def test_watch_reports_mesh_not_running_when_stream_is_none(fake_local_mesh, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    from strands_robots.tools.robot_mesh import _reset_subscribe_allowlist_cache

    _reset_subscribe_allowlist_cache()
    fake_local_mesh.on_stream.return_value = None
    out = _strands_call(action="watch", target="peer-b")
    _reset_subscribe_allowlist_cache()
    assert out["status"] == "error"
    assert "mesh not running" in out["content"][0]["text"]


def test_unknown_action_returns_error(fake_local_mesh):
    out = _strands_call(action="warp")
    assert out["status"] == "error"
    assert "unknown action" in out["content"][0]["text"]


def test_actions_without_local_mesh_fail(fake_no_local, monkeypatch):
    # Since the #10 gateway fallback, a robot-less process CAN reach the mesh
    # by lazily starting a gateway Mesh - so this test must forbid that path
    # (real zenoh session + discovery sleep + 30s RPC timeout in a unit test)
    # and pin the surviving contract: when NO gateway can be built (zenoh
    # unavailable), actions still fail fast with a clear, structured error.
    #
    # The premise is "zenoh unavailable", so state it: the suite-wide
    # STRANDS_MESH=false from tests/conftest.py is a *different* reason to have
    # no mesh, and it now has its own message naming that variable. Without this
    # delenv the assertion below would be graded against the kill-switch answer
    # and this test would stop covering the case it names.
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    with patch("strands_robots.tools.robot_mesh._gateway_mesh", return_value=None):
        out = _strands_call(action="tell", target="peer-b", instruction="go")
    assert out["status"] == "error"
    assert "no local mesh" in out["content"][0]["text"]


# ---------------------------------------------------------------------------
# Regression: _resolve_mesh self-loop fix
#
# Before this fix, when the agent issued ``send/tell/stop`` to a target that
# matched a *local* peer_id, ``_resolve_mesh`` would return the target's own
# Mesh as the gateway.  ``Mesh.send`` then published on
# ``strands/{target}/cmd`` with ``sender_id == target`` - the receiving
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
        # MUST be mesh_a (the OTHER local mesh) - never mesh_b itself,
        # which would self-loop.
        assert gateway is mesh_a, (
            f"_resolve_mesh returned {gateway.peer_id!r} but should have "
            "returned 'robot-a' to avoid the send-to-self self-loop."
        )


def test_resolve_mesh_fallback_when_target_is_only_local():
    """When the target IS the only local mesh, fall back to it.

    The caller will get a timeout (since the message self-drops) - that's
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


# --- mesh-path dispatch-error contract -----------------------------------
# AGENTS.md: an agent tool must convert a backend failure into an error dict
# (status="error") and must never let the exception propagate past dispatch.
# It must also audit the failure with success=False so the forensic trail is
# complete. These pin the Zenoh mesh-path actuation actions (tell/send/
# broadcast/stop/emergency_stop), distinct from the Device Connect dispatcher.


def _audit_capture(monkeypatch):
    """Patch the tool's audit hook and return the captured call list.

    Each entry is the (action, target, success, detail) tuple the tool logs.
    """
    calls: list[tuple[str, str, bool, str]] = []

    def _spy(action, target, success, detail):
        calls.append((action, target, success, detail))

    monkeypatch.setattr(
        "strands_robots.tools.robot_mesh._audit_tool_action",
        _spy,
    )
    return calls


def test_tell_dispatch_error_returns_error_dict_and_audits(fake_local_mesh, monkeypatch):
    calls = _audit_capture(monkeypatch)
    fake_local_mesh.tell.side_effect = RuntimeError("transport down")

    out = _strands_call(action="tell", target="peer-b", instruction="go")

    assert out["status"] == "error"
    assert "dispatch error" in out["content"][0]["text"]
    assert "RuntimeError" in out["content"][0]["text"]
    # failure path audited with success=False
    assert calls and calls[-1][0] == "tell"
    assert calls[-1][2] is False


def test_send_dispatch_error_returns_error_dict_and_audits(fake_local_mesh, monkeypatch):
    calls = _audit_capture(monkeypatch)
    fake_local_mesh.send.side_effect = RuntimeError("link reset")

    out = _strands_call(action="send", target="peer-b", command='{"action": "status"}')

    assert out["status"] == "error"
    assert "dispatch error" in out["content"][0]["text"]
    assert calls and calls[-1][0] == "send"
    assert calls[-1][2] is False


def test_broadcast_dispatch_error_returns_error_dict_and_audits(fake_local_mesh, monkeypatch):
    calls = _audit_capture(monkeypatch)
    fake_local_mesh.broadcast.side_effect = RuntimeError("no peers reachable")

    out = _strands_call(action="broadcast", command='{"action": "status"}')

    assert out["status"] == "error"
    assert "dispatch error" in out["content"][0]["text"]
    # broadcast audits against the wildcard target
    assert calls and calls[-1][0] == "broadcast"
    assert calls[-1][1] == "*"
    assert calls[-1][2] is False


def test_stop_dispatch_error_returns_error_dict_and_audits(fake_local_mesh, monkeypatch):
    calls = _audit_capture(monkeypatch)
    fake_local_mesh.send.side_effect = RuntimeError("stop unack")

    out = _strands_call(action="stop", target="peer-b")

    assert out["status"] == "error"
    assert "dispatch error" in out["content"][0]["text"]
    assert calls and calls[-1][0] == "stop"
    assert calls[-1][2] is False


def test_emergency_stop_dispatch_error_returns_error_dict_and_audits(fake_local_mesh, monkeypatch):
    calls = _audit_capture(monkeypatch)
    fake_local_mesh.emergency_stop.side_effect = RuntimeError("e-stop bus fault")

    out = _strands_call(action="emergency_stop")

    assert out["status"] == "error"
    assert "dispatch error" in out["content"][0]["text"]
    assert calls and calls[-1][0] == "emergency_stop"
    assert calls[-1][1] == "*"
    assert calls[-1][2] is False


def test_tell_forwards_nonzero_policy_port_and_omits_default(fake_local_mesh):
    """``policy_port`` reaches ``mesh.tell`` only when non-zero.

    ``tell`` lets a caller pin the port the remote peer serves its policy on.
    The default (``0``) means "let the mesh pick", so it must NOT be forwarded
    as an explicit kwarg (which would override the peer's own default), while a
    non-zero value must be passed through verbatim.
    """
    fake_local_mesh.tell.return_value = {"executed": "go"}

    _strands_call(action="tell", target="peer-b", instruction="go")
    assert "policy_port" not in fake_local_mesh.tell.call_args.kwargs

    fake_local_mesh.tell.reset_mock()
    out = _strands_call(action="tell", target="peer-b", instruction="go", policy_port=5556)
    assert out["status"] == "success"
    assert fake_local_mesh.tell.call_args.kwargs["policy_port"] == 5556


def test_broadcast_summary_truncates_to_ten_responses(fake_local_mesh):
    """A large fan-out is summarised with at most ten itemised responses.

    ``broadcast`` can return one response per peer; itemising all of them would
    bury the caller. The summary lists the first ten and reports the remainder
    as ``... and N more`` while still stating the true total.
    """
    fake_local_mesh.broadcast.return_value = [{"i": n} for n in range(13)]

    out = _strands_call(action="broadcast", command='{"action": "status"}')

    assert out["status"] == "success"
    text = out["content"][0]["text"]
    assert "13 responses" in text
    assert "... and 3 more" in text
    assert text.count("  - ") == 10


# --- Device Connect dispatch path: validation gate + fail-soft E-STOP -----
# The Device Connect dispatcher (``_device_connect_dispatch``) is a distinct
# code path from the Zenoh mesh, but it must uphold the same two contracts:
#   1. ``tell`` inherits the mesh command-validation gate - an instruction that
#      fails ``validate_command`` is rejected + audited and never reaches the
#      device (dropping this gate would let unvalidated instructions through).
#   2. fleet ``emergency_stop`` is best-effort - one unreachable device must
#      not abort the fan-out; the tool reports the partial count and audits it.


@pytest.fixture
def fake_dc_connection(monkeypatch):
    """Install a fake ``device_connect_agent_tools.connection`` module.

    The connection records every ``invoke()`` and can be told which device ids
    should raise, so a partially-unreachable fleet can be modelled. State is
    returned as a dict: set ``devices`` / ``raise_on`` before dispatching and
    read ``invoked`` afterwards.
    """
    import sys
    import types

    state: dict[str, Any] = {"devices": [], "raise_on": set(), "invoked": []}

    def _get_connection():
        conn = types.SimpleNamespace()
        conn.list_devices = lambda: state["devices"]

        def _invoke(device_id, function, params, timeout=None):
            state["invoked"].append((device_id, function))
            if device_id in state["raise_on"]:
                raise RuntimeError(f"{device_id} unreachable")
            return {"result": {"ok": True}}

        conn.invoke = _invoke
        return conn

    pkg = types.ModuleType("device_connect_agent_tools")
    conn_mod = types.ModuleType("device_connect_agent_tools.connection")
    conn_mod.get_connection = _get_connection  # type: ignore[attr-defined]
    conn_mod.connect = lambda: None  # type: ignore[attr-defined]
    pkg.connection = conn_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", conn_mod)
    return state


def _dc_dispatch(**kwargs):
    """Invoke ``_device_connect_dispatch`` with test-friendly defaults."""
    from strands_robots.tools.robot_mesh import _device_connect_dispatch

    params = {
        "action": "",
        "target": "",
        "instruction": "",
        "command": "",
        "policy_provider": "mock",
        "policy_port": 0,
        "duration": 30.0,
        "timeout": 5.0,
    }
    params.update(kwargs)
    return _device_connect_dispatch(**params)


def test_dc_tell_rejects_and_audits_overlong_instruction(fake_dc_connection, monkeypatch):
    from strands_robots.mesh.security import MAX_INSTRUCTION_LEN

    fake_dc_connection["devices"] = [{"device_id": "peer-b"}]
    calls = _audit_capture(monkeypatch)

    out = _dc_dispatch(action="tell", target="peer-b", instruction="x" * (MAX_INSTRUCTION_LEN + 1))

    assert out["status"] == "error"
    assert "tell rejected" in out["content"][0]["text"]
    # The validation gate fires before dispatch: the device is never invoked.
    assert fake_dc_connection["invoked"] == []
    # Rejection is audited with success=False for the forensic trail.
    assert calls and calls[-1][0] == "tell"
    assert calls[-1][2] is False


def test_dc_emergency_stop_is_best_effort_across_unreachable_devices(fake_dc_connection, monkeypatch):
    fake_dc_connection["devices"] = [
        {"device_id": "arm-1"},
        {"device_id": "arm-2"},
        {"device_id": "arm-3"},
    ]
    fake_dc_connection["raise_on"] = {"arm-2"}
    calls = _audit_capture(monkeypatch)

    out = _dc_dispatch(action="emergency_stop")

    assert out["status"] == "success"
    # Partial count reported: arm-2 failed, arms 1 and 3 stopped.
    assert "2/3" in out["content"][0]["text"]
    # Every device was attempted despite arm-2 raising mid fan-out.
    assert [d for d, _ in fake_dc_connection["invoked"]] == ["arm-1", "arm-2", "arm-3"]
    assert calls and calls[-1][0] == "emergency_stop"
    assert calls[-1][1] == "*"
    assert calls[-1][2] is True


# --- built-in Zenoh mesh fallback: observability + no-Device-Connect contract
# When Device Connect is absent, robot_mesh falls through to the built-in Zenoh
# mesh. Two user-facing contracts on that path had no coverage: the peers
# listing must surface a peer's in-flight task, and rpc must return an
# actionable error (the Zenoh mesh has no device-native call) rather than crash.


def test_peers_listing_surfaces_remote_task_status(fake_local_mesh):
    """A discovered peer that reports a running task renders that task and its
    instruction in the peers listing, so an agent can see what the fleet is
    doing before issuing new commands."""
    with patch(
        "strands_robots.mesh.session.get_peers",
        return_value=[
            {
                "peer_id": "remote-1",
                "type": "robot",
                "hostname": "host1",
                "age": 3,
                "task_status": "running",
                "instruction": "pick up the red cube",
            }
        ],
    ):
        out = _strands_call(action="peers")

    assert out["status"] == "success"
    text = out["content"][0]["text"]
    assert "task: running - pick up the red cube" in text


def test_rpc_without_device_connect_returns_actionable_error(fake_local_mesh):
    """rpc is a device-native function call with no Zenoh-mesh equivalent. With
    a local mesh present but Device Connect unavailable, the tool returns an
    actionable error dict (never raises) that names Device Connect as the
    requirement instead of silently timing out."""
    out = _strands_call(action="rpc", target="dev-1", function="nod")

    assert out["status"] == "error"
    text = out["content"][0]["text"]
    assert "rpc" in text
    assert "Device Connect" in text
