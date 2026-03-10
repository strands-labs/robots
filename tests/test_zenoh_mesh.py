#!/usr/bin/env python3
"""Comprehensive tests for strands_robots/zenoh_mesh.py — Peer-to-peer mesh layer.

Coverage target: 17% → 60%+ of zenoh_mesh.py (378 statements).
All tests run without eclipse-zenoh installed — mock everything.
"""

import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 1. Module-level imports and dataclass tests ─────────────────────


class TestPeerInfo:
    """Test PeerInfo dataclass."""

    def test_peer_info_basic(self):
        from strands_robots.zenoh_mesh import PeerInfo

        p = PeerInfo(peer_id="robot-1", peer_type="robot", hostname="host1", last_seen=time.time())
        assert p.peer_id == "robot-1"
        assert p.peer_type == "robot"
        assert p.hostname == "host1"

    def test_peer_info_age(self):
        from strands_robots.zenoh_mesh import PeerInfo

        p = PeerInfo(peer_id="robot-1", peer_type="robot", last_seen=time.time() - 5.0)
        assert 4.5 < p.age < 6.0

    def test_peer_info_to_dict(self):
        from strands_robots.zenoh_mesh import PeerInfo

        now = time.time()
        p = PeerInfo(peer_id="sim-1", peer_type="sim", hostname="host2", last_seen=now, caps={"world": True})
        d = p.to_dict()
        assert d["peer_id"] == "sim-1"
        assert d["type"] == "sim"
        assert d["hostname"] == "host2"
        assert d["world"] is True
        assert "age" in d

    def test_peer_info_default_caps(self):
        from strands_robots.zenoh_mesh import PeerInfo

        p = PeerInfo(peer_id="r1", peer_type="agent")
        assert p.caps == {}
        assert p.hostname == ""


# ── 2. Module-level helper functions ────────────────────────────────


class TestModuleHelpers:
    """Test _update_peer, _prune_peers, get_peers, get_peer."""

    def setup_method(self):
        """Reset global state before each test."""
        import strands_robots.zenoh_mesh as zm

        zm._PEERS.clear()
        zm._PEERS_VERSION = 0

    def test_update_peer_new(self):
        from strands_robots.zenoh_mesh import _PEERS, _update_peer

        is_new = _update_peer("robot-a", "robot", "host1", {"arm": True})
        assert is_new is True
        assert "robot-a" in _PEERS

    def test_update_peer_existing(self):
        from strands_robots.zenoh_mesh import _update_peer

        _update_peer("robot-a", "robot", "host1", {})
        is_new = _update_peer("robot-a", "robot", "host1", {"updated": True})
        assert is_new is False

    def test_update_peer_version_increment(self):
        import strands_robots.zenoh_mesh as zm

        zm._PEERS_VERSION = 0
        zm._update_peer("peer-1", "robot", "h1", {})
        assert zm._PEERS_VERSION == 1
        zm._update_peer("peer-1", "robot", "h1", {})
        assert zm._PEERS_VERSION == 1  # Not incremented for existing
        zm._update_peer("peer-2", "robot", "h2", {})
        assert zm._PEERS_VERSION == 2

    def test_prune_peers_removes_stale(self):
        import strands_robots.zenoh_mesh as zm

        zm._PEERS["old-peer"] = zm.PeerInfo(
            peer_id="old-peer", peer_type="robot", last_seen=time.time() - 20.0  # Stale
        )
        zm._PEERS["fresh-peer"] = zm.PeerInfo(peer_id="fresh-peer", peer_type="robot", last_seen=time.time())
        zm._prune_peers()
        assert "old-peer" not in zm._PEERS
        assert "fresh-peer" in zm._PEERS

    def test_get_peers_empty(self):
        from strands_robots.zenoh_mesh import get_peers

        assert get_peers() == []

    def test_get_peers_with_data(self):
        from strands_robots.zenoh_mesh import _update_peer, get_peers

        _update_peer("r1", "robot", "h1", {})
        _update_peer("r2", "sim", "h2", {})
        peers = get_peers()
        assert len(peers) == 2
        ids = {p["peer_id"] for p in peers}
        assert "r1" in ids
        assert "r2" in ids

    def test_get_peer_found(self):
        from strands_robots.zenoh_mesh import _update_peer, get_peer

        _update_peer("r1", "robot", "h1", {"arm": True})
        p = get_peer("r1")
        assert p is not None
        assert p["peer_id"] == "r1"
        assert p["arm"] is True

    def test_get_peer_not_found(self):
        from strands_robots.zenoh_mesh import get_peer

        assert get_peer("nonexistent") is None


# ── 3. _put() function ─────────────────────────────────────────────


class TestPut:
    """Test _put() publishing function."""

    def test_put_with_no_session(self):
        import strands_robots.zenoh_mesh as zm

        old_session = zm._SESSION
        zm._SESSION = None
        # Should not raise
        zm._put("test/key", {"hello": "world"})
        zm._SESSION = old_session

    def test_put_with_session(self):
        import strands_robots.zenoh_mesh as zm

        mock_session = MagicMock()
        old_session = zm._SESSION
        zm._SESSION = mock_session
        try:
            zm._put("test/key", {"hello": "world"})
            mock_session.put.assert_called_once()
            args = mock_session.put.call_args
            assert args[0][0] == "test/key"
            assert b"hello" in args[0][1]
        finally:
            zm._SESSION = old_session

    def test_put_handles_exception(self):
        import strands_robots.zenoh_mesh as zm

        mock_session = MagicMock()
        mock_session.put.side_effect = RuntimeError("network fail")
        old_session = zm._SESSION
        zm._SESSION = mock_session
        try:
            # Should not raise
            zm._put("test/key", {"data": 1})
        finally:
            zm._SESSION = old_session


# ── 4. Session management ──────────────────────────────────────────


class TestSessionManagement:
    """Test _get_session and _release_session."""

    def test_get_session_zenoh_not_installed(self):
        import strands_robots.zenoh_mesh as zm

        old_session = zm._SESSION
        zm._SESSION = None
        zm._SESSION_REFS = 0
        try:
            with patch("importlib.import_module", side_effect=ImportError("no zenoh")):
                result = zm._get_session()
                assert result is None
        finally:
            zm._SESSION = old_session

    def test_get_session_returns_existing(self):
        import strands_robots.zenoh_mesh as zm

        mock_session = MagicMock()
        old_session = zm._SESSION
        old_refs = zm._SESSION_REFS
        zm._SESSION = mock_session
        zm._SESSION_REFS = 1
        try:
            result = zm._get_session()
            assert result is mock_session
            assert zm._SESSION_REFS == 2
        finally:
            zm._SESSION = old_session
            zm._SESSION_REFS = old_refs

    def test_get_session_opens_new(self):
        import strands_robots.zenoh_mesh as zm

        old_session = zm._SESSION
        old_refs = zm._SESSION_REFS
        zm._SESSION = None
        zm._SESSION_REFS = 0

        mock_zenoh = MagicMock()
        mock_config = MagicMock()
        mock_zenoh.Config.default.return_value = mock_config
        mock_opened = MagicMock()
        mock_zenoh.open.return_value = mock_opened

        try:
            with patch("importlib.import_module", return_value=mock_zenoh):
                result = zm._get_session()
                assert result is mock_opened
                assert zm._SESSION_REFS == 1
        finally:
            zm._SESSION = old_session
            zm._SESSION_REFS = old_refs

    def test_get_session_with_env_endpoints(self):
        import strands_robots.zenoh_mesh as zm

        old_session = zm._SESSION
        old_refs = zm._SESSION_REFS
        zm._SESSION = None
        zm._SESSION_REFS = 0

        mock_zenoh = MagicMock()
        mock_config = MagicMock()
        mock_zenoh.Config.return_value = mock_config
        mock_zenoh.open.return_value = MagicMock()

        try:
            with (
                patch("importlib.import_module", return_value=mock_zenoh),
                patch.dict(os.environ, {"ZENOH_CONNECT": "tcp/1.2.3.4:7447", "ZENOH_LISTEN": "tcp/0.0.0.0:7448"}),
            ):
                zm._get_session()
                # Should attempt to configure endpoints
                assert mock_config.insert_json5.call_count >= 1
        finally:
            zm._SESSION = old_session
            zm._SESSION_REFS = old_refs

    def test_release_session_decrements(self):
        import strands_robots.zenoh_mesh as zm

        mock_session = MagicMock()
        old_session = zm._SESSION
        old_refs = zm._SESSION_REFS
        zm._SESSION = mock_session
        zm._SESSION_REFS = 2
        try:
            zm._release_session()
            assert zm._SESSION_REFS == 1
            assert zm._SESSION is mock_session  # Not closed yet
        finally:
            zm._SESSION = old_session
            zm._SESSION_REFS = old_refs

    def test_release_session_closes_at_zero(self):
        import strands_robots.zenoh_mesh as zm

        mock_session = MagicMock()
        old_session = zm._SESSION
        old_refs = zm._SESSION_REFS
        zm._SESSION = mock_session
        zm._SESSION_REFS = 1
        try:
            zm._release_session()
            assert zm._SESSION_REFS == 0
            assert zm._SESSION is None
            mock_session.close.assert_called_once()
        finally:
            zm._SESSION = old_session
            zm._SESSION_REFS = old_refs


# ── 5. Mesh class ──────────────────────────────────────────────────


class TestMesh:
    """Test the Mesh class directly."""

    def setup_method(self):
        import strands_robots.zenoh_mesh as zm

        zm._PEERS.clear()
        zm._PEERS_VERSION = 0
        zm._LOCAL_ROBOTS.clear()

    def _make_mesh(self, peer_id="test-robot", peer_type="robot"):
        from strands_robots.zenoh_mesh import Mesh

        robot = MagicMock()
        robot.tool_name_str = "so100"
        return Mesh(robot, peer_id=peer_id, peer_type=peer_type)

    def test_mesh_init(self):
        m = self._make_mesh()
        assert m.peer_id == "test-robot"
        assert m.peer_type == "robot"
        assert m._running is False
        assert m._subs == []

    def test_mesh_alive_property(self):
        m = self._make_mesh()
        assert m.alive is False
        m._running = True
        assert m.alive is True

    def test_mesh_start_no_session(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        with patch.object(zm, "_get_session", return_value=None):
            m.start()
            assert m._running is False

    def test_mesh_start_with_session(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        mock_session = MagicMock()
        mock_sub = MagicMock()
        mock_session.declare_subscriber.return_value = mock_sub

        with patch.object(zm, "_get_session", return_value=mock_session), patch("threading.Thread") as mock_thread:
            mock_thread_inst = MagicMock()
            mock_thread.return_value = mock_thread_inst
            m.start()
            assert m._running is True
            assert m.peer_id in zm._LOCAL_ROBOTS
            assert mock_session.declare_subscriber.call_count == 4
            assert mock_thread.call_count == 2  # heartbeat + state

    def test_mesh_stop(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        m._running = True
        mock_sub = MagicMock()
        m._subs = [mock_sub]
        zm._LOCAL_ROBOTS[m.peer_id] = m

        with patch.object(zm, "_release_session"):
            m.stop()
            assert m._running is False
            assert m.peer_id not in zm._LOCAL_ROBOTS
            mock_sub.undeclare.assert_called_once()
            assert m._subs == []

    def test_mesh_stop_idempotent(self):
        m = self._make_mesh()
        m._running = False
        m.stop()  # Should be a no-op

    def test_mesh_start_idempotent(self):
        m = self._make_mesh()
        m._running = True
        m.start()  # Should be a no-op when already running

    def test_mesh_peers_property(self):
        import strands_robots.zenoh_mesh as zm

        zm._update_peer("r1", "robot", "h1", {})
        m = self._make_mesh()
        assert len(m.peers) == 1

    # ── presence ───────────────────────────────────────

    def test_build_presence_basic(self):
        m = self._make_mesh()
        p = m._build_presence()
        assert p["robot_id"] == "test-robot"
        assert p["robot_type"] == "robot"
        assert "hostname" in p
        assert "timestamp" in p

    def test_build_presence_with_attributes(self):
        from strands_robots.zenoh_mesh import Mesh

        robot = MagicMock()
        robot.tool_name_str = "so100"
        robot._task_state = MagicMock()
        robot._task_state.status.value = "running"
        robot._task_state.instruction = "pick up cube"
        robot.robot = MagicMock()
        robot.robot.is_connected = True
        robot.robot.name = "so100_follower"
        robot._action_features = {"action": MagicMock()}
        m = Mesh(robot, "test-1", "robot")
        p = m._build_presence()
        assert p["task_status"] == "running"
        assert p["instruction"] == "pick up cube"
        assert p["connected"] is True
        assert p["hw"] == "so100_follower"
        assert "action_keys" in p

    def test_build_presence_simulation(self):
        from strands_robots.zenoh_mesh import Mesh

        robot = MagicMock()
        robot.tool_name_str = "sim"
        robot._robots = {"arm1": MagicMock(), "arm2": MagicMock()}
        robot._world = MagicMock()
        m = Mesh(robot, "sim-1", "sim")
        p = m._build_presence()
        assert "sim_robots" in p
        assert p["world"] is True

    def test_on_presence_new_peer(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh(peer_id="local-robot")
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(
            {
                "robot_id": "remote-robot",
                "robot_type": "sim",
                "hostname": "remote-host",
            }
        ).encode()
        m._on_presence(sample)
        assert "remote-robot" in zm._PEERS

    def test_on_presence_ignores_self(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh(peer_id="local-robot")
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(
            {
                "robot_id": "local-robot",
            }
        ).encode()
        m._on_presence(sample)
        assert "local-robot" not in zm._PEERS

    def test_on_presence_bad_json(self):
        m = self._make_mesh()
        sample = MagicMock()
        sample.payload.to_bytes.return_value = b"not-json"
        # Should not raise
        m._on_presence(sample)

    # ── state ──────────────────────────────────────────

    def test_read_state_minimal(self):
        m = self._make_mesh()
        m.robot = MagicMock(spec=[])  # No robot/task attributes
        result = m._read_state()
        assert result is None  # Only peer_id and t — not enough

    def test_read_state_with_task(self):
        from strands_robots.zenoh_mesh import Mesh

        robot = MagicMock()
        robot._task_state = MagicMock()
        robot._task_state.status.value = "idle"
        robot._task_state.instruction = ""
        robot._task_state.step_count = 0
        robot._task_state.duration = 0.0
        # Remove robot attr so it doesn't trigger hardware path
        del robot.robot
        del robot._world
        del robot._data
        del robot._robots
        m = Mesh(robot, "r1", "robot")
        result = m._read_state()
        assert result is not None
        assert "task" in result

    def test_read_state_with_sim(self):
        from strands_robots.zenoh_mesh import Mesh

        robot = MagicMock()
        robot._world = MagicMock()
        robot._world._data = MagicMock()
        robot._world._data.time = 1.5
        robot._robots = {"arm1": MagicMock()}
        del robot.robot
        del robot._task_state
        m = Mesh(robot, "sim-1", "sim")
        result = m._read_state()
        assert result is not None
        assert result["sim_time"] == 1.5
        assert "arm1" in result.get("robots", {})

    # ── command dispatch ───────────────────────────────

    def test_on_cmd_ignores_self(self):
        m = self._make_mesh(peer_id="me")
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(
            {
                "sender_id": "me",
                "command": {"action": "status"},
            }
        ).encode()
        with patch("threading.Thread") as mock_t:
            m._on_cmd(sample)
            mock_t.assert_not_called()

    def test_on_cmd_dispatches(self):
        m = self._make_mesh(peer_id="me")
        m._running = True
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(
            {
                "sender_id": "them",
                "command": {"action": "status"},
            }
        ).encode()
        with patch("threading.Thread") as mock_t:
            m._on_cmd(sample)
            mock_t.assert_called_once()

    def test_dispatch_status(self):
        m = self._make_mesh()
        m.robot.get_task_status = MagicMock(return_value={"status": "idle"})
        result = m._dispatch({"action": "status"})
        assert result["status"] == "idle"

    def test_dispatch_status_no_method(self):
        from strands_robots.zenoh_mesh import Mesh

        robot = MagicMock(spec=[])
        m = Mesh(robot, "r1", "robot")
        result = m._dispatch({"action": "status"})
        assert "status" in result or "unknown" in str(result)

    def test_dispatch_stop(self):
        m = self._make_mesh()
        m.robot.stop_task = MagicMock(return_value={"stopped": True})
        result = m._dispatch({"action": "stop"})
        assert result["stopped"] is True

    def test_dispatch_features(self):
        m = self._make_mesh()
        m.robot.get_features = MagicMock(return_value={"arm": True})
        result = m._dispatch({"action": "features"})
        assert result["arm"] is True

    def test_dispatch_state(self):
        m = self._make_mesh()
        with patch.object(m, "_read_state", return_value={"peer_id": "r1"}):
            result = m._dispatch({"action": "state"})
            assert result["peer_id"] == "r1"

    def test_dispatch_execute_missing_instruction(self):
        m = self._make_mesh()
        result = m._dispatch({"action": "execute"})
        assert "error" in result

    def test_dispatch_execute_with_instruction(self):
        m = self._make_mesh()
        m.robot._execute_task_sync = MagicMock(return_value={"ok": True})
        m._dispatch(
            {
                "action": "execute",
                "instruction": "pick up cube",
                "policy_provider": "mock",
            }
        )
        m.robot._execute_task_sync.assert_called_once()

    def test_dispatch_start_action(self):
        m = self._make_mesh()
        m.robot.start_task = MagicMock(return_value={"started": True})
        m._dispatch(
            {
                "action": "start",
                "instruction": "wave",
            }
        )
        m.robot.start_task.assert_called_once()

    def test_dispatch_step_sim(self):
        m = self._make_mesh()
        m.robot.step = MagicMock(return_value={"stepped": True})
        m._dispatch({"action": "step", "steps": 5})
        m.robot.step.assert_called_once_with(5)

    def test_dispatch_reset_sim(self):
        m = self._make_mesh()
        m.robot.reset = MagicMock(return_value={"reset": True})
        m._dispatch({"action": "reset"})
        m.robot.reset.assert_called_once()

    def test_dispatch_unknown_action(self):
        m = self._make_mesh()
        result = m._dispatch({"action": "fly"})
        assert "error" in result

    # ── exec_cmd with response publishing ──────────────

    def test_exec_cmd_publishes_response(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        m.robot.get_task_status = MagicMock(return_value={"status": "ok"})

        with patch.object(zm, "_put") as mock_put:
            m._exec_cmd(
                {
                    "sender_id": "sender-1",
                    "turn_id": "turn-abc",
                    "command": {"action": "status"},
                }
            )
            mock_put.assert_called_once()
            key, data = mock_put.call_args[0]
            assert "sender-1" in key
            assert "turn-abc" in key
            assert data["type"] == "response"

    def test_exec_cmd_publishes_error(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        m.robot.get_task_status = MagicMock(side_effect=RuntimeError("boom"))

        with patch.object(zm, "_put") as mock_put:
            m._exec_cmd(
                {
                    "sender_id": "sender-1",
                    "turn_id": "turn-abc",
                    "command": {"action": "status"},
                }
            )
            mock_put.assert_called_once()
            _, data = mock_put.call_args[0]
            assert data["type"] == "error"
            assert "boom" in data["error"]

    # ── on_response ────────────────────────────────────

    def test_on_response_matching_turn(self):
        m = self._make_mesh()
        ev = threading.Event()
        m._pending["turn-123"] = ev
        m._responses["turn-123"] = []

        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(
            {
                "turn_id": "turn-123",
                "result": {"ok": True},
            }
        ).encode()
        m._on_response(sample)
        assert len(m._responses["turn-123"]) == 1
        assert ev.is_set()

    def test_on_response_no_matching_turn(self):
        m = self._make_mesh()
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(
            {
                "turn_id": "unknown-turn",
            }
        ).encode()
        m._on_response(sample)  # No error

    # ── subscribe / unsubscribe ────────────────────────

    def test_subscribe_not_running(self):
        m = self._make_mesh()
        m._running = False
        result = m.subscribe("topic/test")
        assert result is None

    def test_subscribe_with_callback(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        m._running = True
        mock_session = MagicMock()
        mock_sub = MagicMock()
        mock_session.declare_subscriber.return_value = mock_sub
        old_session = zm._SESSION
        zm._SESSION = mock_session

        try:
            cb = MagicMock()
            result = m.subscribe("sensors/*", callback=cb, name="sensors")
            assert result == "sensors"
            assert "sensors" in m.inbox
            mock_session.declare_subscriber.assert_called_once()
        finally:
            zm._SESSION = old_session

    def test_subscribe_buffer_mode(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        m._running = True
        mock_session = MagicMock()
        mock_session.declare_subscriber.return_value = MagicMock()
        old_session = zm._SESSION
        zm._SESSION = mock_session

        try:
            result = m.subscribe("data/topic")
            assert result == "data/topic"
            assert "data/topic" in m.inbox
        finally:
            zm._SESSION = old_session

    def test_unsubscribe(self):
        m = self._make_mesh()
        m._running = True
        mock_sub = MagicMock()
        m._user_subs = {"my_topic": mock_sub}
        m._subs = [mock_sub]
        m.inbox = {"my_topic": []}

        m.unsubscribe("my_topic")
        mock_sub.undeclare.assert_called_once()
        assert "my_topic" not in m._user_subs
        assert "my_topic" not in m.inbox

    def test_unsubscribe_no_subs(self):
        m = self._make_mesh()
        # No _user_subs attribute
        m.unsubscribe("anything")  # Should not raise

    # ── publish_step / on_stream ───────────────────────

    def test_publish_step(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh(peer_id="robot-1")
        import numpy as np

        with patch.object(zm, "_put") as mock_put:
            obs = {
                "joint.position": np.array([1.0, 2.0, 3.0]),
                "camera.top": np.zeros((480, 640, 3)),  # Should be filtered
                "status": "ok",
            }
            act = {
                "joint.target": np.array([1.1, 2.1, 3.1]),
                "gripper": 0.5,
            }
            m.publish_step(step=5, observation=obs, action=act, instruction="pick cube", policy="mock")
            mock_put.assert_called_once()
            key, data = mock_put.call_args[0]
            assert "robot-1/stream" in key
            assert data["step"] == 5
            assert "camera.top" not in data["observation"]
            assert "joint.position" in data["observation"]
            assert data["action"]["gripper"] == 0.5

    def test_on_stream(self):
        m = self._make_mesh()
        with patch.object(m, "subscribe", return_value="stream:other") as mock_sub:
            result = m.on_stream("other-robot", callback=lambda t, d: None)
            mock_sub.assert_called_once()
            assert result == "stream:other"

    # ── send / broadcast / tell / emergency_stop ───────

    def test_send_timeout(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        with patch.object(zm, "_put"):
            result = m.send("target-peer", {"action": "status"}, timeout=0.01)
            assert result["status"] == "timeout"

    def test_send_with_response(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()

        def fake_put(key, data):
            # Simulate immediate response
            for turn_id, ev in list(m._pending.items()):
                m._responses.setdefault(turn_id, []).append({"result": "ok", "turn_id": turn_id})
                ev.set()

        with patch.object(zm, "_put", side_effect=fake_put):
            result = m.send("target", {"action": "status"}, timeout=1.0)
            assert result.get("result") == "ok"

    def test_broadcast(self):
        import strands_robots.zenoh_mesh as zm

        m = self._make_mesh()
        with patch.object(zm, "_put"):
            result = m.broadcast({"action": "status"}, timeout=0.01)
            assert isinstance(result, list)

    def test_tell(self):
        m = self._make_mesh()
        with patch.object(m, "send", return_value={"ok": True}) as mock_send:
            m.tell("target", "pick up cube", policy_provider="mock")
            mock_send.assert_called_once()
            cmd = mock_send.call_args[0][1]
            assert cmd["action"] == "execute"
            assert cmd["instruction"] == "pick up cube"

    def test_emergency_stop(self):
        m = self._make_mesh()
        with patch.object(m, "broadcast", return_value=[{"ok": True}]) as mock_bc:
            m.emergency_stop()
            mock_bc.assert_called_once()
            cmd = mock_bc.call_args[0][0]
            assert cmd["action"] == "stop"


# ── 6. init_mesh() function ────────────────────────────────────────


class TestInitMesh:
    """Test the init_mesh() convenience function."""

    def test_init_mesh_disabled(self):
        from strands_robots.zenoh_mesh import init_mesh

        robot = MagicMock()
        result = init_mesh(robot, mesh=False)
        assert result is None

    def test_init_mesh_env_disabled(self):
        from strands_robots.zenoh_mesh import init_mesh

        robot = MagicMock()
        with patch.dict(os.environ, {"STRANDS_MESH": "false"}):
            result = init_mesh(robot)
            assert result is None

    def test_init_mesh_generates_peer_id(self):
        import strands_robots.zenoh_mesh as zm
        from strands_robots.zenoh_mesh import init_mesh

        robot = MagicMock()
        robot.tool_name_str = "so100"

        with patch.object(zm.Mesh, "start"):
            result = init_mesh(robot, peer_type="robot")
            assert result is not None
            assert result.peer_id.startswith("so100-")

    def test_init_mesh_explicit_peer_id(self):
        import strands_robots.zenoh_mesh as zm
        from strands_robots.zenoh_mesh import init_mesh

        robot = MagicMock()

        with patch.object(zm.Mesh, "start"):
            result = init_mesh(robot, peer_id="my-robot-1", peer_type="sim")
            assert result.peer_id == "my-robot-1"
            assert result.peer_type == "sim"
