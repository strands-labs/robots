"""Tests for strands_robots.mesh — RPC (PR3), streams (PR4), safety (PR5).

All tests are 100% mocked.  No ``eclipse-zenoh`` install required: a
``MagicMock`` session is injected in place of the real Zenoh session by
patching :func:`strands_robots.mesh_session.get_session`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots import mesh as mesh_mod
from strands_robots.mesh import Mesh
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeRobot:
    """Minimal duck-typed robot for dispatch tests."""

    def __init__(self) -> None:
        self.tool_name_str = "fakebot"
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_task_status(self) -> dict[str, Any]:
        self.calls.append(("get_task_status", {}))
        return {"status": "idle"}

    def stop_task(self) -> dict[str, Any]:
        self.calls.append(("stop_task", {}))
        return {"stopped": True}

    def get_features(self) -> dict[str, Any]:
        self.calls.append(("get_features", {}))
        return {"foo": "bar"}

    def step(self, n: int) -> dict[str, Any]:
        self.calls.append(("step", {"n": n}))
        return {"stepped": n}

    def reset(self) -> dict[str, Any]:
        self.calls.append(("reset", {}))
        return {"ok": True}

    def _execute_task_sync(
        self, instruction: str, provider: str, port, host: str, duration: float, **kw: Any
    ) -> dict[str, Any]:
        self.calls.append(("execute", {"instruction": instruction, "provider": provider, "duration": duration}))
        return {"executed": instruction}

    def start_task(
        self, instruction: str, provider: str, port, host: str, duration: float, **kw: Any
    ) -> dict[str, Any]:
        self.calls.append(("start", {"instruction": instruction}))
        return {"started": instruction}


@pytest.fixture
def fake_session() -> Iterator[MagicMock]:
    """Patch get_session() / current_session() to return a MagicMock and
    capture put() calls.

    Both names are patched because :meth:`Mesh.start` uses ``get_session``
    while :meth:`Mesh.subscribe` uses ``current_session`` (no refcount
    bump).  Patching both keeps the in-test session identity stable.
    """
    sess = MagicMock()
    sess.declare_subscriber = MagicMock()
    with (
        patch.object(mesh_session, "get_session", return_value=sess),
        patch.object(mesh_session, "current_session", return_value=sess),
        patch.object(mesh_core, "get_session", return_value=sess),
        patch.object(mesh_core, "current_session", return_value=sess),
        patch.object(mesh_core, "release_session"),
    ):
        yield sess


@pytest.fixture
def captured_puts() -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Capture every mesh.put() call as (key, payload) tuples.

    Wire-level authentication is owned by the Zenoh transport in
    production; ``put()`` itself just hands plain JSON to the
    transport. The fixture records the payload as-is.
    """
    seen: list[tuple[str, dict[str, Any]]] = []

    def _spy(key: str, data: dict[str, Any]) -> None:
        seen.append((key, data))

    with patch.object(mesh_core, "put", side_effect=_spy):
        yield seen


@pytest.fixture
def started_mesh(fake_session: MagicMock) -> Iterator[Mesh]:
    """A Mesh that has start()-ed against a mocked session."""
    m = Mesh(_FakeRobot(), peer_id="peer-a", peer_type="robot")
    m.start()
    try:
        yield m
    finally:
        m.stop()


def _make_sample(payload: dict[str, Any], key: str = "strands/x/cmd") -> Any:
    """Build a fake zenoh sample object compatible with the mesh callbacks."""
    sample = MagicMock()
    sample.key_expr = key
    sample.payload.to_bytes.return_value = json.dumps(payload).encode()
    return sample


# ---------------------------------------------------------------------------
# Dispatch — _dispatch routes correctly for every action
# ---------------------------------------------------------------------------


def test_dispatch_status() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "status"})
    assert out == {"status": "idle"}


def test_dispatch_stop() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "stop"})
    assert out == {"stopped": True}


def test_dispatch_features() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "features"})
    assert out == {"foo": "bar"}


def test_dispatch_step() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "step", "steps": 5})
    assert out == {"stepped": 5}


def test_dispatch_reset() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "reset"})
    assert out == {"ok": True}


def test_dispatch_execute_calls_execute_task_sync() -> None:
    r = _FakeRobot()
    m = Mesh(r, peer_id="p")
    out = m._dispatch({"action": "execute", "instruction": "go", "duration": 5.0, "policy_provider": "mock"})
    assert out == {"executed": "go"}
    assert ("execute", {"instruction": "go", "provider": "mock", "duration": 5.0}) in r.calls


def test_dispatch_start_calls_start_task() -> None:
    r = _FakeRobot()
    m = Mesh(r, peer_id="p")
    out = m._dispatch({"action": "start", "instruction": "go", "policy_provider": "mock"})
    assert out == {"started": "go"}


def test_dispatch_execute_requires_instruction() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "execute"})
    assert "error" in out


def test_dispatch_unknown_action() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    out = m._dispatch({"action": "warp"})
    assert out == {"error": "unknown action: warp"}


def test_dispatch_state_falls_back_to_read_state() -> None:
    m = Mesh(_FakeRobot(), peer_id="p")
    with patch.object(m, "_read_state", return_value={"x": 1}):
        out = m._dispatch({"action": "state"})
    assert out == {"x": 1}


# ---------------------------------------------------------------------------
# Incoming command flow — _on_cmd → _exec_cmd → put(response)
# ---------------------------------------------------------------------------


def test_on_cmd_ignores_self_loop(started_mesh: Mesh, captured_puts: list[tuple[str, dict[str, Any]]]) -> None:
    sample = _make_sample({"sender_id": "peer-a", "command": {"action": "status"}})
    started_mesh._on_cmd(sample)
    # Give the maybe-spawned thread a moment; nothing should happen.
    time.sleep(0.05)
    response_keys = [k for k, _ in captured_puts if "response" in k]
    assert response_keys == []


def test_on_cmd_drops_undecodable_payload(started_mesh: Mesh) -> None:
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"not json"
    # Should not raise.
    started_mesh._on_cmd(sample)


def test_exec_cmd_publishes_response(captured_puts: list[tuple[str, dict[str, Any]]]) -> None:
    m = Mesh(_FakeRobot(), peer_id="me")
    m._exec_cmd(
        {
            "sender_id": "alice",
            "turn_id": "t1",
            "command": {"action": "status"},
        }
    )
    assert any(k == "strands/alice/response/me/t1" for k, _ in captured_puts)
    response = next(d for k, d in captured_puts if k == "strands/alice/response/me/t1")
    assert response["type"] == "response"
    assert response["responder_id"] == "me"
    assert response["turn_id"] == "t1"
    assert response["result"] == {"status": "idle"}


def test_exec_cmd_publishes_error_on_dispatch_exception(
    captured_puts: list[tuple[str, dict[str, Any]]],
) -> None:
    m = Mesh(_FakeRobot(), peer_id="me")
    # Use a valid action (status) so we get past command validation and exercise
    # the original dispatch-exception path that this test was written for.
    with patch.object(m, "_dispatch", side_effect=RuntimeError("boom-with-internal-detail")):
        m._exec_cmd({"sender_id": "alice", "turn_id": "t2", "command": {"action": "status"}})
    payload = next(d for k, d in captured_puts if k == "strands/alice/response/me/t2")
    assert payload["type"] == "error"
    # internal exception detail MUST NOT leak onto the wire. The
    # error string is sanitised to a static "dispatch error" so a remote
    # caller cannot pivot on path / attribute / library-trace fragments.
    assert payload["error"] == "dispatch error"
    assert "boom-with-internal-detail" not in payload["error"]


def test_exec_cmd_rejects_unknown_action_with_validation_error(
    captured_puts: list[tuple[str, dict[str, Any]]],
) -> None:
    """An unknown action is rejected at the validation gate, well before any
    robot side-effect runs. The reply is a structured error envelope that
    includes the offending action name."""
    m = Mesh(_FakeRobot(), peer_id="me")
    m._exec_cmd({"sender_id": "alice", "turn_id": "t3", "command": {"action": "rm_rf"}})
    payload = next(d for k, d in captured_puts if k == "strands/alice/response/me/t3")
    assert payload["type"] == "error"
    assert "validation" in payload["error"]
    assert "rm_rf" in payload["error"]


def test_exec_cmd_string_command_rejected() -> None:
    """bare-string commands on the wire bypass validate_command's dict-shape
    contract. _exec_cmd must reject them, not silently coerce to {action:execute}.

    Pre-the prior fix contract was auto-wrap (a peer publishing "hello" got a mock-policy
    execute), which let any mTLS+ACL-authorised peer drive the robot at the mock
    provider with arbitrary text.
    """
    r = _FakeRobot()
    m = Mesh(r, peer_id="me")
    with patch.object(mesh_mod, "put"):
        m._exec_cmd({"sender_id": "alice", "turn_id": "t", "command": "do thing"})
    # No execute call; bare string was rejected at the wire boundary.
    assert not any(name == "execute" for name, _ in r.calls), (
        "_exec_cmd must NOT coerce bare-string commands into mock-policy executes"
    )


# ---------------------------------------------------------------------------
# Outgoing RPC — send / broadcast / tell
# ---------------------------------------------------------------------------


def test_send_returns_first_response(started_mesh: Mesh, captured_puts) -> None:
    """Simulate the response subscriber firing during send.wait()."""

    def fake_responder() -> None:
        # Wait for send() to register a pending turn, then fake a response.
        for _ in range(50):
            with started_mesh._rpc_lock:
                if started_mesh._pending:
                    turn = next(iter(started_mesh._pending.keys()))
                    break
            time.sleep(0.01)
        else:  # pragma: no cover — defensive
            return
        # Phase-4 / D1: _on_response now requires responder_id to match the
        # target the sender originally addressed (otherwise an
        # authenticated peer that observes the turn_id can hijack the
        # response). Include responder_id="peer-b" to match send target.
        sample = _make_sample({"turn_id": turn, "responder_id": "peer-b", "result": {"ok": 1}})
        started_mesh._on_response(sample)

    threading.Thread(target=fake_responder, daemon=True).start()
    out = started_mesh.send("peer-b", {"action": "status"}, timeout=2.0)
    assert out["result"] == {"ok": 1}
    assert any(k == "strands/peer-b/cmd" for k, _ in captured_puts)


def test_send_timeout_returns_status_timeout(started_mesh: Mesh) -> None:
    out = started_mesh.send("peer-b", {"action": "status"}, timeout=0.05)
    assert out == {"status": "timeout"}


def test_send_when_not_running_returns_error() -> None:
    m = Mesh(_FakeRobot(), peer_id="x")
    out = m.send("peer-b", {"action": "status"})
    assert out["status"] == "error"


def test_broadcast_collects_multiple_responses(started_mesh: Mesh, captured_puts) -> None:
    def fake_responders() -> None:
        for _ in range(50):
            with started_mesh._rpc_lock:
                if started_mesh._pending:
                    turn = next(iter(started_mesh._pending.keys()))
                    break
            time.sleep(0.01)
        else:  # pragma: no cover
            return
        for i in range(3):
            sample = _make_sample({"turn_id": turn, "responder_id": f"p{i}", "result": {"i": i}})
            started_mesh._on_response(sample)

    threading.Thread(target=fake_responders, daemon=True).start()
    out = started_mesh.broadcast({"action": "status"}, timeout=1.0)
    assert len(out) == 3
    assert any(k == "strands/broadcast" for k, _ in captured_puts)


def test_broadcast_when_not_running_returns_empty() -> None:
    m = Mesh(_FakeRobot(), peer_id="x")
    assert m.broadcast({"action": "status"}) == []


def test_tell_wraps_send(started_mesh: Mesh) -> None:
    with patch.object(started_mesh, "send", return_value={"status": "ok"}) as mock_send:
        result = started_mesh.tell("peer-b", "pick the cube", duration=10.0)
    assert result == {"status": "ok"}
    mock_send.assert_called_once()
    args = mock_send.call_args
    assert args.args[0] == "peer-b"
    assert args.args[1]["action"] == "execute"
    assert args.args[1]["instruction"] == "pick the cube"
    assert args.args[1]["duration"] == 10.0


def test_on_response_drops_unknown_turn(started_mesh: Mesh) -> None:
    sample = _make_sample({"turn_id": "unknown-turn", "result": {"x": 1}})
    # Should not raise; with no pending entry the response is silently dropped.
    started_mesh._on_response(sample)
    with started_mesh._rpc_lock:
        assert "unknown-turn" not in started_mesh._responses


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe / inbox
# ---------------------------------------------------------------------------


def test_subscribe_with_callback_invokes_handler(started_mesh: Mesh, fake_session: MagicMock) -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    def cb(topic: str, data: dict[str, Any]) -> None:
        received.append((topic, data))

    name = started_mesh.subscribe("reachy_mini/joints", callback=cb, name="reachy")
    assert name == "reachy"

    # Find the subscription handler that mesh registered.
    last_call = fake_session.declare_subscriber.call_args_list[-1]
    handler = last_call.args[1] if len(last_call.args) >= 2 else last_call.kwargs["handler"]

    sample = _make_sample({"q": [1.0, 2.0]}, key="reachy_mini/joints")
    handler(sample)
    assert received == [("reachy_mini/joints", {"q": [1.0, 2.0]})]


def test_subscribe_without_callback_buffers_inbox(started_mesh: Mesh, fake_session: MagicMock) -> None:
    name = started_mesh.subscribe("sensor/data", name="sensor")
    assert name == "sensor"
    handler = fake_session.declare_subscriber.call_args_list[-1].args[1]
    handler(_make_sample({"x": 1}, "sensor/data"))
    handler(_make_sample({"x": 2}, "sensor/data"))
    assert started_mesh.inbox["sensor"] == [
        ("sensor/data", {"x": 1}),
        ("sensor/data", {"x": 2}),
    ]


def test_subscribe_inbox_caps_at_1000(started_mesh: Mesh, fake_session: MagicMock) -> None:
    started_mesh.subscribe("tick", name="tick")
    handler = fake_session.declare_subscriber.call_args_list[-1].args[1]
    for i in range(1100):
        handler(_make_sample({"i": i}, "tick"))
    # The cap (1000) triggers at the 1001st insert, slicing to keep last 500.
    # After 1100 total inserts the buffer contains entries 501..1099 (599 items).
    buf = started_mesh.inbox["tick"]
    assert len(buf) <= 1000  # never exceeds the cap
    assert len(buf) >= 500  # the slice retains at least 500 items
    # The most recent items are always present.
    assert buf[-1][1]["i"] == 1099
    # The oldest items have been trimmed.
    assert buf[0][1]["i"] >= 500


def test_subscribe_when_not_running_returns_none() -> None:
    m = Mesh(_FakeRobot(), peer_id="x")
    assert m.subscribe("anything") is None


def test_unsubscribe_unknown_name_is_noop(started_mesh: Mesh) -> None:
    # Should not raise.
    started_mesh.unsubscribe("nope")


def test_unsubscribe_drops_inbox(started_mesh: Mesh, fake_session: MagicMock) -> None:
    started_mesh.subscribe("a", name="a")
    assert "a" in started_mesh.inbox
    started_mesh.unsubscribe("a")
    assert "a" not in started_mesh.inbox
    assert "a" not in started_mesh._user_subs


def test_subscribe_handles_non_json_payload(started_mesh: Mesh, fake_session: MagicMock) -> None:
    received: list[tuple[str, dict[str, Any]]] = []
    started_mesh.subscribe("x", callback=lambda t, d: received.append((t, d)))
    handler = fake_session.declare_subscriber.call_args_list[-1].args[1]
    bad = MagicMock()
    bad.key_expr = "x"
    bad.payload.to_bytes.return_value = b"raw text not json"
    handler(bad)
    assert received[0][1] == {"raw": "raw text not json"}


# ---------------------------------------------------------------------------
# publish_step / on_stream
# ---------------------------------------------------------------------------


def test_publish_step_filters_camera_frames(started_mesh: Mesh, captured_puts) -> None:
    cam_frame = MagicMock()
    cam_frame.shape = (3, 240, 320)
    joints = MagicMock()
    joints.shape = (6,)
    joints.tolist = lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    started_mesh.publish_step(
        step=42,
        observation={
            "wrist_cam": cam_frame,
            "joint_positions": joints,
            "battery": 0.95,
        },
        action={"target_q": joints},
        instruction="grab",
        policy="mock",
    )

    keys = [k for k, _ in captured_puts]
    assert any(k.endswith("/stream") for k in keys)
    payload = next(d for k, d in captured_puts if k.endswith("/stream"))
    assert "wrist_cam" not in payload["observation"]
    assert payload["observation"]["joint_positions"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert payload["observation"]["battery"] == 0.95
    assert payload["step"] == 42


def test_publish_step_when_not_running_does_nothing(captured_puts) -> None:
    m = Mesh(_FakeRobot(), peer_id="x")
    m.publish_step(0, {}, {})
    assert captured_puts == []


def test_on_stream_subscribes_to_stream_key(started_mesh: Mesh) -> None:
    with patch.object(started_mesh, "subscribe", return_value="stream:peer-b") as mock_sub:
        out = started_mesh.on_stream("peer-b")
    assert out == "stream:peer-b"
    mock_sub.assert_called_once()
    assert mock_sub.call_args.args[0] == "strands/peer-b/stream"
    assert mock_sub.call_args.kwargs.get("name") == "stream:peer-b"


# ---------------------------------------------------------------------------
# emergency_stop + audit log
# ---------------------------------------------------------------------------


def test_emergency_stop_broadcasts_stop_action(started_mesh: Mesh) -> None:
    with (
        patch.object(started_mesh, "broadcast", return_value=[{"ok": 1}]) as mock_bc,
        patch.object(mesh_mod, "log_safety_event"),
    ):
        out = started_mesh.emergency_stop()
    assert out == [{"ok": 1}]
    args = mock_bc.call_args
    assert args.args[0] == {"action": "stop"}
    assert args.kwargs.get("timeout") == 3.0


def test_emergency_stop_writes_audit_log(started_mesh: Mesh, tmp_path) -> None:
    audit_dir = tmp_path / "audit"
    with (
        patch.object(started_mesh, "broadcast", return_value=[{"a": 1}, {"b": 2}]),
        patch.dict("os.environ", {"STRANDS_MESH_AUDIT_DIR": str(audit_dir)}),
    ):
        started_mesh.emergency_stop()

    log_file = audit_dir / "mesh_audit.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "emergency_stop"
    assert record["peer_id"] == "peer-a"
    assert record["payload"]["responses_received"] == 2


def test_emergency_stop_audit_log_failure_does_not_raise(started_mesh: Mesh) -> None:
    with (
        patch.object(started_mesh, "broadcast", return_value=[]),
        patch.object(mesh_mod, "log_safety_event", side_effect=OSError("disk full")),
    ):
        # Must not raise — audit log failure is non-fatal.
        out = started_mesh.emergency_stop()
    assert out == []


# ---------------------------------------------------------------------------
# Stop semantics — wakes blocked send/broadcast calls
# ---------------------------------------------------------------------------


def test_stop_wakes_blocked_send(fake_session: MagicMock) -> None:
    m = Mesh(_FakeRobot(), peer_id="me")
    m.start()

    out: list[dict[str, Any]] = []

    def caller() -> None:
        out.append(m.send("peer-b", {"action": "status"}, timeout=10.0))

    t = threading.Thread(target=caller, daemon=True)
    t.start()

    # Give send() a moment to register pending state.
    for _ in range(50):
        with m._rpc_lock:
            if m._pending:
                break
        time.sleep(0.01)

    m.stop()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert out == [{"status": "timeout"}]


class TestCommandDispatchErrorAudit:
    """Pin the dispatch-error audit posture for issue #257.

    The _exec_cmd dispatch-exception branch must emit a log_safety_event with
    event_type "command_rejected" and reason "dispatch error" so a remote
    prober cannot fish for adapter exceptions without leaving a forensic
    trail. The audit payload must never carry the internal exception detail,
    and an audit-sink failure on this path must not crash the handler.
    """

    def test_command_dispatch_error_emits_audit_event(self, captured_puts: list[tuple[str, dict[str, Any]]]) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        with (
            patch.object(mesh_core, "log_safety_event") as mock_audit,
            patch.object(m, "_dispatch", side_effect=RuntimeError("boom-with-internal-detail")),
        ):
            m._exec_cmd({"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}})
        assert mock_audit.called
        event_type = mock_audit.call_args.args[0]
        payload = mock_audit.call_args.args[2]
        assert event_type == "command_rejected"
        assert payload["reason"] == "dispatch error"

    def test_command_dispatch_error_audit_payload_excludes_exception_message(
        self, captured_puts: list[tuple[str, dict[str, Any]]]
    ) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        with (
            patch.object(mesh_core, "log_safety_event") as mock_audit,
            patch.object(m, "_dispatch", side_effect=RuntimeError("boom-with-internal-detail")),
        ):
            m._exec_cmd({"sender_id": "alice", "turn_id": "t2", "command": {"action": "status"}})
        payload = mock_audit.call_args.args[2]
        assert "boom-with-internal-detail" not in str(payload)

    def test_command_dispatch_error_audit_includes_sender_and_action(
        self, captured_puts: list[tuple[str, dict[str, Any]]]
    ) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        with (
            patch.object(mesh_core, "log_safety_event") as mock_audit,
            patch.object(m, "_dispatch", side_effect=RuntimeError("boom")),
        ):
            m._exec_cmd({"sender_id": "alice", "turn_id": "t3", "command": {"action": "status"}})
        payload = mock_audit.call_args.args[2]
        assert payload["sender"] == "alice"
        assert payload["action"] == "status"

    def test_command_dispatch_error_survives_audit_sink_failure(
        self, captured_puts: list[tuple[str, dict[str, Any]]]
    ) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        with (
            patch.object(mesh_core, "log_safety_event", side_effect=OSError("audit disk full")),
            patch.object(m, "_dispatch", side_effect=RuntimeError("boom")),
        ):
            # Must not raise despite the audit sink failing.
            m._exec_cmd({"sender_id": "alice", "turn_id": "t4", "command": {"action": "status"}})
        payload = next(d for k, d in captured_puts if k == "strands/alice/response/me/t4")
        assert payload["type"] == "error"
        assert payload["error"] == "dispatch error"
