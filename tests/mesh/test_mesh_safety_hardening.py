"""Mesh safety-hardening regression tests.

Covers the Zenoh/SDK in this change:
- E-stop lockout bypass via input path
- Permanent-lockout startup warning (override code unset)
- Teleop value bound tightened + apply-rate cap
- CMD replay dedup
- Resume override-code brute-force throttle
- Peer registry cap
- Presence freshness check
"""

from __future__ import annotations

import threading
import time

import pytest

from strands_robots.mesh import security as _sec
from strands_robots.mesh import session as _ses
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.input import InputReceiver, _input_max_hz


class _FakeMeshForInput:
    peer_id = "victim"

    def __init__(self) -> None:
        self._estop_lockout = threading.Event()

    def subscribe(self, *a, **k):
        return "sub"

    def unsubscribe(self, *a, **k):
        pass


class _RecordingRobot:
    def __init__(self) -> None:
        self.actions: list = []

    def send_action(self, action) -> None:
        self.actions.append(action)


# --------------------------------------------------------------------------- C-1
class TestC1InputLockout:
    def test_input_rejected_during_lockout(self):
        mesh = _FakeMeshForInput()
        robot = _RecordingRobot()
        rx = InputReceiver(mesh, robot, source_peer_id="leader")
        rx._running = True
        mesh._estop_lockout.set()
        rx._on_input(rx.topic, {"action": {"j0": 0.5}, "seq": 0})
        assert robot.actions == []
        assert rx.stats["rejected"] >= 1

    def test_input_applied_when_unlocked(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # disable rate cap
        mesh = _FakeMeshForInput()
        robot = _RecordingRobot()
        rx = InputReceiver(mesh, robot, source_peer_id="leader")
        rx._running = True
        rx._on_input(rx.topic, {"action": {"j0": 0.5}, "seq": 0})
        assert len(robot.actions) == 1


# --------------------------------------------------------------------------- H-2
class TestH2TeleopBoundAndRate:
    def test_default_bound_rejects_large_slew(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_INPUT_VALUE_ABS", raising=False)
        # 4*pi default ~= 12.57; 31 rad must reject
        with pytest.raises(_sec.ValidationError):
            _sec.validate_input_frame({"j0": 31.0})

    def test_default_bound_accepts_normal_radian(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_INPUT_VALUE_ABS", raising=False)
        out = _sec.validate_input_frame({"shoulder.pos": 1.5})
        assert out["shoulder.pos"] == 1.5

    def test_bound_is_env_tunable(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "100")
        out = _sec.validate_input_frame({"j0": 50.0})
        assert out["j0"] == 50.0

    def test_apply_rate_cap_drops_burst(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "50")
        mesh = _FakeMeshForInput()
        robot = _RecordingRobot()
        rx = InputReceiver(mesh, robot, source_peer_id="leader")
        rx._running = True
        # Two synchronous frames microseconds apart: 2nd exceeds 50Hz.
        rx._on_input(rx.topic, {"action": {"j0": 0.1}, "seq": 0})
        rx._on_input(rx.topic, {"action": {"j0": 0.2}, "seq": 1})
        assert len(robot.actions) == 1
        assert rx.stats["rate_dropped"] >= 1

    def test_rate_cap_disabled_with_zero(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        assert _input_max_hz() == 0.0


# --------------------------------------------------------------------------- M-2
class TestM2PeerCap:
    def test_registry_bounded(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "25")
        _ses.clear_peers()
        for i in range(500):
            _ses.update_peer(f"phantom-{i}", "robot", "h", {})
        assert _ses.peer_count() == 25

    def test_existing_peer_update_no_eviction(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "10")
        _ses.clear_peers()
        for i in range(10):
            _ses.update_peer(f"p{i}", "robot", "h", {})
        # Updating an existing peer should not evict anyone.
        _ses.update_peer("p0", "robot", "h2", {})
        assert _ses.peer_count() == 10


# --------------------------------------------------------------------------- M-3
class TestM3PresenceFreshness:
    def _sample(self, payload):
        import json
        from unittest.mock import MagicMock

        s = MagicMock()
        s.payload.to_bytes.return_value = json.dumps(payload).encode()
        return s

    def test_fresh_presence_accepted(self):
        _ses.clear_peers()
        m = Mesh.__new__(Mesh)
        m.peer_id = "self"
        m._on_presence(self._sample({"robot_id": "other", "timestamp": time.time()}))
        assert any(p["peer_id"] == "other" for p in _ses.get_peers())

    def test_stale_presence_rejected(self):
        _ses.clear_peers()
        m = Mesh.__new__(Mesh)
        m.peer_id = "self"
        m._on_presence(self._sample({"robot_id": "stale", "timestamp": time.time() - 300}))
        assert not any(p["peer_id"] == "stale" for p in _ses.get_peers())

    def test_missing_timestamp_rejected(self):
        _ses.clear_peers()
        m = Mesh.__new__(Mesh)
        m.peer_id = "self"
        m._on_presence(self._sample({"robot_id": "nots"}))
        assert not any(p["peer_id"] == "nots" for p in _ses.get_peers())


# --------------------------------------------------------------------------- M-1
class TestM1ResumeBruteForce:
    def _stub(self):
        m = Mesh.__new__(Mesh)
        m.peer_id = "p"
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m.publish_safety_event = lambda **kw: None
        return m

    def test_throttle_engages_after_threshold(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "the-correct-code-1234567890abcd")
        monkeypatch.setenv("STRANDS_MESH_RESUME_MAX_FAILS", "3")
        monkeypatch.setenv("STRANDS_MESH_RESUME_BACKOFF_S", "60")
        m = self._stub()
        m._estop_lockout.set()
        # 3 bad attempts arm the throttle.
        for _ in range(3):
            assert m._resume_lockout("wrong")["status"] == "error"
        # Now even the CORRECT code is refused (throttled) and lockout stays.
        assert m._resume_lockout("the-correct-code-1234567890abcd")["status"] == "error"
        assert m._estop_lockout.is_set()

    def test_correct_code_before_threshold_succeeds(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "code-xyz-1234567890abcdef0000")
        monkeypatch.setenv("STRANDS_MESH_RESUME_MAX_FAILS", "5")
        m = self._stub()
        m._estop_lockout.set()
        m._resume_lockout("wrong")  # 1 fail, under threshold
        assert m._resume_lockout("code-xyz-1234567890abcdef0000")["status"] == "ok"
        assert not m._estop_lockout.is_set()


# --------------------------------------------------------------------------- H-3
class TestH3CmdReplay:
    def _exec_stub(self):
        """A Mesh built via __new__ with just enough state for _exec_cmd's
        dedup path, with _dispatch stubbed to count actuations."""
        m = Mesh.__new__(Mesh)
        m.peer_id = "robot-1"
        m._cmd_replay_cache = {}
        m._cmd_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m.dispatched = []

        def _fake_dispatch(cmd):
            m.dispatched.append(cmd)
            return {"ok": True}

        m._dispatch = _fake_dispatch
        m.publish = lambda *a, **k: None
        return m

    def test_replayed_command_dispatched_once(self):
        m = self._exec_stub()
        env = {
            "sender_id": "operator",
            "turn_id": "turn-abc",
            "command": {"action": "execute", "instruction": "pick", "policy_provider": "mock"},
        }
        # Same envelope delivered 5x -> dispatch must fire exactly once.
        for _ in range(5):
            m._exec_cmd(dict(env))
        assert len(m.dispatched) == 1

    def test_distinct_turn_ids_each_dispatch(self):
        m = self._exec_stub()
        for i in range(3):
            m._exec_cmd(
                {
                    "sender_id": "operator",
                    "turn_id": f"turn-{i}",
                    "command": {"action": "execute", "instruction": "pick", "policy_provider": "mock"},
                }
            )
        assert len(m.dispatched) == 3

    def test_readonly_action_not_deduped(self):
        m = self._exec_stub()
        # status is idempotent -> repeats allowed (operator polling).
        for _ in range(4):
            m._exec_cmd(
                {
                    "sender_id": "operator",
                    "turn_id": "same-turn",
                    "command": {"action": "status"},
                }
            )
        assert len(m.dispatched) == 4


# --------------------------------------------------------------------------- M-5
class TestM5SuccessAudit:
    """Finding #9: successful actions must be audited, not only rejections."""

    def test_successful_command_is_audited(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "strands_robots.mesh.core.log_safety_event",
            lambda et, pid, payload: events.append((et, payload)),
        )
        m = Mesh.__new__(Mesh)
        m.peer_id = "robot-1"
        m._cmd_replay_cache = {}
        m._cmd_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._dispatch = lambda cmd: {"ok": True}
        m.publish = lambda *a, **k: None

        m._exec_cmd(
            {
                "sender_id": "op",
                "turn_id": "t1",
                "command": {"action": "execute", "instruction": "pick", "policy_provider": "mock"},
            }
        )
        assert any(et == "command_executed" for et, _ in events)

    def test_readonly_command_not_audited_as_executed(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "strands_robots.mesh.core.log_safety_event",
            lambda et, pid, payload: events.append((et, payload)),
        )
        m = Mesh.__new__(Mesh)
        m.peer_id = "robot-1"
        m._cmd_replay_cache = {}
        m._cmd_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._dispatch = lambda cmd: {"status": "idle"}
        m.publish = lambda *a, **k: None

        m._exec_cmd({"sender_id": "op", "turn_id": "t2", "command": {"action": "status"}})
        assert not any(et == "command_executed" for et, _ in events)

    def test_input_stream_sampled_audit(self, monkeypatch):
        from strands_robots.mesh import input as inp

        events = []
        monkeypatch.setattr(inp, "_log_safety_event", lambda et, pid, payload: events.append(et))
        monkeypatch.setenv("STRANDS_MESH_INPUT_AUDIT_EVERY", "5")
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # don't rate-limit the test

        mesh = _FakeMeshForInput()
        robot = _RecordingRobot()
        rx = inp.InputReceiver(mesh, robot, source_peer_id="leader")
        rx._running = True
        for i in range(5):
            rx._on_input(rx.topic, {"action": {"j0": 0.1 * i}, "seq": i})
        # 5th applied frame triggers one sampled audit.
        assert events.count("input_stream_applied") == 1
