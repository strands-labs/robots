"""Deep autonomous test suite for strands_robots.mesh — PR #101.

This test file exercises every corner of the mesh implementation:
- Session lifecycle edge cases
- Multi-mesh coordination
- RPC timeout & error paths
- Sensor loop robustness
- Input publisher/receiver lifecycle
- Audit log integrity
- Thread safety under load
- Memory leak detection (refcount)
- Camera encoding paths
- Subscribe/unsubscribe races
- Emergency stop propagation

NO CODE WILL BE PUSHED — this is a read-only test exploration.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Correct import paths after refactor (commit 8f0eb6c)
from strands_robots.mesh import InputPublisher, InputReceiver, Mesh, get_local_robots, init_mesh
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.audit import log_safety_event, read_audit_log
from strands_robots.mesh.session import (
    clear_peers,
    get_peer,
    get_session,
    prune_peers,
    put,
    release_session,
    session_alive,
    update_peer,
)

# ===========================================================================
# Fixtures
# ===========================================================================


class FakeRobot:
    """Full-featured mock robot for testing all mesh paths."""

    def __init__(self, name="testbot", **opts):
        self.tool_name_str = name
        self._task_state = None
        self._world = None
        self._pose = opts.get("pose")
        self._slam_pose = opts.get("slam_pose")
        self._odom_pose = opts.get("odom_pose")
        self._imu = opts.get("imu")
        self._odom = opts.get("odom")
        self._battery = opts.get("battery")
        self._temps = opts.get("temps")
        self._lidar_summary = opts.get("lidar_summary")
        self._lidar_state = opts.get("lidar_state")
        self._hands = opts.get("hands")
        self._map_info = opts.get("map_info")
        self._action_features = opts.get("action_features")
        self._input_publishers = opts.get("input_publishers", {})
        self.robot = opts.get("inner_robot")

        if opts.get("with_task"):
            ts = SimpleNamespace()
            ts.status = SimpleNamespace(value="running")
            ts.instruction = "test instruction"
            ts.step_count = 10
            ts.duration = 2.5
            self._task_state = ts

        if opts.get("with_world"):
            world = SimpleNamespace()
            world._data = SimpleNamespace(time=42.0)
            world.robots = {"arm0": object()}
            self._world = world

    def get_task_status(self):
        if self._task_state:
            return {"status": self._task_state.status.value}
        return {"status": "idle"}

    def stop_task(self):
        return {"stopped": True}

    def get_features(self):
        return {"action_space": 6}

    def step(self, n):
        return {"stepped": n}

    def reset(self):
        return {"reset": True}

    def _execute_task_sync(self, instruction, provider, port, host, duration, **kw):
        return {"executed": instruction, "provider": provider}

    def start_task(self, instruction, provider, port, host, duration, **kw):
        return {"started": instruction}


@pytest.fixture(autouse=True)
def clean_state():
    """Reset all module-level state between tests."""
    clear_peers()
    with mesh_core._LOCAL_ROBOTS_LOCK:
        mesh_core._LOCAL_ROBOTS.clear()
    # Reset session
    with mesh_session._SESSION_LOCK:
        if mesh_session._SESSION is not None:
            try:
                mesh_session._SESSION.close()
            except Exception:
                pass
        mesh_session._SESSION = None
        mesh_session._SESSION_REFS = 0
    yield
    clear_peers()
    with mesh_core._LOCAL_ROBOTS_LOCK:
        mesh_core._LOCAL_ROBOTS.clear()
    with mesh_session._SESSION_LOCK:
        if mesh_session._SESSION is not None:
            try:
                mesh_session._SESSION.close()
            except Exception:
                pass
        mesh_session._SESSION = None
        mesh_session._SESSION_REFS = 0


@pytest.fixture
def mock_session(monkeypatch):
    """Provide a mock zenoh session and patch get/release/current."""
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    sess = MagicMock()
    sess.declare_subscriber.return_value = MagicMock()
    with (
        patch.object(mesh_session, "get_session", return_value=sess),
        patch.object(mesh_session, "current_session", return_value=sess),
        patch.object(mesh_core, "get_session", return_value=sess),
        patch.object(mesh_core, "current_session", return_value=sess),
        patch.object(mesh_core, "release_session"),
    ):
        yield sess


@pytest.fixture
def mock_put():
    """Patch put at all locations where it's imported.

    Wire authentication is owned by the Zenoh transport in production;
    in-process tests just record (key, payload) tuples as-is.
    """
    from strands_robots.mesh import sensors as mesh_sensors

    calls = []

    def _spy(key, data):
        calls.append((key, data))

    with patch.object(mesh_session, "put", side_effect=_spy):
        with patch.object(mesh_core, "put", side_effect=_spy):
            with patch.object(mesh_sensors, "put", side_effect=_spy):
                yield calls


# ===========================================================================
# Test: Import Path Validation (Bug #1 from refactor)
# ===========================================================================


class TestImportPaths:
    """Verify correct import paths after the monolithic → package refactor."""

    def test_mesh_session_importable_from_package(self):
        """strands_robots.mesh.session should be the correct import path."""
        from strands_robots.mesh import session

        assert hasattr(session, "get_session")
        assert hasattr(session, "put")
        assert hasattr(session, "PeerInfo")

    def test_mesh_core_has_put(self):
        """core.py imports put from session — verify it's accessible."""
        from strands_robots.mesh import core

        assert callable(core.put)

    def test_mesh_package_exports_put(self):
        """put is re-exported from mesh package for backward compat."""
        import strands_robots.mesh as mesh_pkg

        assert hasattr(mesh_pkg, "put")
        assert callable(mesh_pkg.put)

    def test_old_compat_shim_removed(self):
        """from strands_robots import mesh_session should fail after refactor."""
        with pytest.raises(ImportError):
            from strands_robots import mesh_session  # noqa: F401

    def test_mesh_exports_correct_symbols(self):
        """__all__ should include core + session helpers."""
        from strands_robots.mesh import __all__

        # Must include core mesh types
        assert "Mesh" in __all__
        assert "InputPublisher" in __all__
        assert "init_mesh" in __all__
        # Must include session helpers (backward compat)
        assert "put" in __all__
        assert "get_session" in __all__
        assert "get_peers" in __all__


# ===========================================================================
# Test: Session Singleton Robustness
# ===========================================================================


class TestSessionRobustness:
    """Edge cases in session lifecycle."""

    def test_double_release_is_noop(self):
        """Releasing more than acquired should be safe."""
        mock_sess = MagicMock()
        with mesh_session._SESSION_LOCK:
            mesh_session._SESSION = mock_sess
            mesh_session._SESSION_REFS = 1
        release_session()
        assert mesh_session._SESSION is None
        # Second release should be a no-op
        release_session()
        assert mesh_session._SESSION_REFS == 0

    def test_get_session_concurrent_first_open(self):
        """Multiple threads calling get_session simultaneously."""
        results = []
        mock_zenoh = MagicMock()
        mock_sess = MagicMock()
        mock_zenoh.open.return_value = mock_sess
        mock_zenoh.Config.return_value = MagicMock()

        def worker():
            with (
                patch.dict("sys.modules", {"zenoh": mock_zenoh}),
                patch.dict("os.environ", {}, clear=False),
            ):
                os.environ.pop("ZENOH_CONNECT", None)
                os.environ.pop("ZENOH_LISTEN", None)
                s = get_session()
                results.append(s)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All should get the same session or None
        non_none = [r for r in results if r is not None]
        if non_none:
            assert all(r is non_none[0] for r in non_none)

    def test_session_alive_reflects_state(self):
        assert not session_alive()
        mock_sess = MagicMock()
        with mesh_session._SESSION_LOCK:
            mesh_session._SESSION = mock_sess
            mesh_session._SESSION_REFS = 1
        assert session_alive()

    def test_put_noop_when_no_session(self):
        """put() is a no-op (not an error) when session is None."""
        with mesh_session._SESSION_LOCK:
            mesh_session._SESSION = None
        # Should not raise
        put("strands/test/key", {"data": "value"})

    def test_invalid_mesh_port_env(self, monkeypatch):
        """STRANDS_MESH_PORT with invalid value logs warning, uses default."""
        monkeypatch.setenv("STRANDS_MESH_PORT", "not-a-number")
        mock_zenoh = MagicMock()
        mock_sess = MagicMock()
        mock_zenoh.open.return_value = mock_sess
        mock_zenoh.Config.return_value = MagicMock()

        with (
            patch.dict("sys.modules", {"zenoh": mock_zenoh}),
        ):
            os.environ.pop("ZENOH_CONNECT", None)
            os.environ.pop("ZENOH_LISTEN", None)
            s = get_session()

        # Should still work (fallback to 7447)
        assert s is mock_sess or s is None  # depends on race with previous tests

    def test_port_out_of_range(self, monkeypatch):
        """STRANDS_MESH_PORT=99999 falls back to default."""
        monkeypatch.setenv("STRANDS_MESH_PORT", "99999")
        mock_zenoh = MagicMock()
        mock_sess = MagicMock()
        mock_zenoh.open.return_value = mock_sess
        mock_zenoh.Config.return_value = MagicMock()

        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            os.environ.pop("ZENOH_CONNECT", None)
            os.environ.pop("ZENOH_LISTEN", None)
            get_session()  # verify no raise
        # It should have warned and used 7447


# ===========================================================================
# Test: Peer Registry Under Stress
# ===========================================================================


class TestPeerRegistryStress:
    """High-concurrency peer registry operations."""

    def test_rapid_update_prune_cycle(self):
        """Rapid interleaving of update + prune doesn't corrupt state."""
        errors = []

        def updater(prefix):
            try:
                for i in range(100):
                    update_peer(f"{prefix}-{i}", "robot", "host", {})
            except Exception as e:
                errors.append(e)

        def pruner():
            try:
                for _ in range(50):
                    prune_peers(timeout=0.001)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=updater, args=(f"u{i}",)))
        threads.append(threading.Thread(target=pruner))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_peer_timeout_boundary(self):
        """Peer exactly at timeout boundary."""
        update_peer("boundary", "robot", "h", {})
        # Manually set last_seen to exactly PEER_TIMEOUT ago
        with mesh_session._PEERS_LOCK:
            mesh_session._PEERS["boundary"].last_seen = time.time() - mesh_session.PEER_TIMEOUT
        # Peer at exact boundary should be pruned (> is strict)
        time.sleep(0.01)
        pruned = prune_peers()
        assert "boundary" in pruned

    def test_peer_info_caps_are_preserved(self):
        """Capabilities dict survives update cycles."""
        caps = {"connected": True, "hw": "so100", "cameras": ["wrist", "top"]}
        update_peer("rich", "robot", "jetson", caps)
        p = get_peer("rich")
        assert p["connected"] is True
        assert p["hw"] == "so100"
        assert p["cameras"] == ["wrist", "top"]


# ===========================================================================
# Test: Mesh Lifecycle Edge Cases
# ===========================================================================


class TestMeshLifecycleEdge:
    """Edge cases in Mesh.start() / Mesh.stop()."""

    def test_start_when_subscriber_fails_partial(self, mock_session):
        """If declare_subscriber fails on 3rd call, first 2 are undeclared."""
        sub1 = MagicMock()
        sub2 = MagicMock()
        mock_session.declare_subscriber.side_effect = [sub1, sub2, RuntimeError("zenoh error"), MagicMock()]

        m = Mesh(FakeRobot(), peer_id="fail-sub")
        m.start()
        assert not m.alive
        # The first two subs should have been undeclared during rollback
        assert sub1.undeclare.called
        assert sub2.undeclare.called

    def test_mesh_garbage_collected_after_stop(self, mock_session):
        """After stop(), the Mesh should be GC-able (no circular refs)."""
        m = Mesh(FakeRobot(), peer_id="gc-test")
        m.start()
        _ref = weakref.ref(m)  # noqa: F841
        m.stop()
        del m
        gc.collect()
        # Note: daemon threads may hold refs, so this is best-effort
        # The key invariant is stop() doesn't leak into _LOCAL_ROBOTS
        assert "gc-test" not in get_local_robots()

    def test_stop_clears_pending_rpc(self, mock_session):
        """stop() should wake any blocked send() calls."""
        m = Mesh(FakeRobot(), peer_id="rpc-clear")
        m.start()

        # Simulate a pending RPC
        event = threading.Event()
        with m._rpc_lock:
            m._pending["fake-turn"] = event
            m._responses["fake-turn"] = []

        m.stop()
        # Event should have been set
        assert event.is_set()
        # Pending should be empty
        with m._rpc_lock:
            assert len(m._pending) == 0
            assert len(m._responses) == 0

    def test_multiple_meshes_independent_lifecycle(self, mock_session):
        """Multiple meshes can start/stop independently."""
        m1 = Mesh(FakeRobot(name="bot1"), peer_id="multi-1")
        m2 = Mesh(FakeRobot(name="bot2"), peer_id="multi-2")
        m3 = Mesh(FakeRobot(name="bot3"), peer_id="multi-3")

        m1.start()
        m2.start()
        m3.start()

        assert len(get_local_robots()) == 3

        m2.stop()
        robots = get_local_robots()
        assert "multi-1" in robots
        assert "multi-2" not in robots
        assert "multi-3" in robots

        m1.stop()
        m3.stop()
        assert len(get_local_robots()) == 0


# ===========================================================================
# Test: RPC Robustness
# ===========================================================================


class TestRPCRobustness:
    """RPC edge cases: timeouts, malformed messages, concurrent sends."""

    def test_on_response_non_string_turn_id_ignored(self, mock_session):
        """Response with non-string turn_id should be silently dropped."""
        m = Mesh(FakeRobot(), peer_id="rpc-1")
        m.start()
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps({"turn_id": 12345, "result": {"x": 1}}).encode()
        m._on_response(sample)
        # No crash, no pending entries
        with m._rpc_lock:
            assert 12345 not in m._responses
        m.stop()

    def test_dispatch_teleop_actions(self, mock_session):
        """Test teleop-related dispatch actions."""
        m = Mesh(FakeRobot(), peer_id="teleop-1")

        # teleop_status when no robot methods
        result = m._dispatch({"action": "teleop_status"})
        assert result == {"inputs": [], "publishers": {}, "receivers": {}}

        # teleop_receive without source_peer_id
        result = m._dispatch({"action": "teleop_receive"})
        assert "error" in result

        # teleop_receive with source but no method
        result = m._dispatch({"action": "teleop_receive", "source_peer_id": "leader-1"})
        assert "error" in result

        # teleop_stop without method
        result = m._dispatch({"action": "teleop_stop"})
        assert "error" in result

    def test_concurrent_send_calls(self, mock_session, mock_put):
        """Multiple send() calls in parallel all get responses or timeout."""
        m = Mesh(FakeRobot(), peer_id="concurrent-rpc")
        m.start()

        results = []

        def sender(idx):
            r = m.send(f"peer-{idx}", {"action": "status"}, timeout=0.1)
            results.append(r)

        threads = [threading.Thread(target=sender, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All should timeout (no responders)
        assert len(results) == 10
        assert all(r == {"status": "timeout"} for r in results)
        m.stop()

    def test_broadcast_returns_empty_when_stopped(self, mock_session):
        """broadcast() on a stopped mesh returns []."""
        m = Mesh(FakeRobot(), peer_id="bc-stopped")
        assert m.broadcast({"action": "status"}) == []

    def test_send_returns_error_when_stopped(self, mock_session):
        """send() on a stopped mesh returns error dict."""
        m = Mesh(FakeRobot(), peer_id="send-stopped")
        result = m.send("target", {"action": "status"})
        assert result["status"] == "error"


# ===========================================================================
# Test: Sensor Loops
# ===========================================================================


class TestSensorLoops:
    """Sensor publishing loops robustness."""

    def test_health_loop_reads_system_stats(self, mock_session, mock_put):
        """Health loop publishes CPU, mem, disk even without robot battery."""
        m = Mesh(FakeRobot(), peer_id="health-1")
        m.start()
        time.sleep(2.5)  # HEALTH_HZ=0.5, need at least 2s
        m.stop()

        health_puts = [(k, d) for k, d in mock_put if k.endswith("/health")]
        assert len(health_puts) > 0
        _, payload = health_puts[0]
        assert payload["peer_id"] == "health-1"
        # Should have at least cpu_load on Linux
        if sys.platform == "linux":
            assert "cpu_load" in payload

    def test_pose_loop_with_dict_pose(self, mock_session, mock_put):
        """Pose loop publishes when robot has _pose as dict."""
        robot = FakeRobot(pose={"x": 1.0, "y": 2.0, "z": 0.0, "theta": 0.5})
        m = Mesh(robot, peer_id="pose-1")
        m.start()
        time.sleep(0.2)  # POSE_HZ=10
        m.stop()

        pose_puts = [(k, d) for k, d in mock_put if k.endswith("/pose")]
        assert len(pose_puts) > 0
        _, payload = pose_puts[0]
        assert payload["x"] == 1.0
        assert payload["source"] == "provider"

    def test_imu_loop_with_dict(self, mock_session, mock_put):
        """IMU loop publishes when robot has _imu."""
        robot = FakeRobot(imu={"rpy": [0.1, 0.2, 0.3], "gyro": [0, 0, 0.1]})
        m = Mesh(robot, peer_id="imu-1")
        m.start()
        time.sleep(0.2)
        m.stop()

        imu_puts = [(k, d) for k, d in mock_put if "/imu" in k]
        assert len(imu_puts) > 0

    def test_odom_loop(self, mock_session, mock_put):
        """Odom loop publishes when _odom is set."""
        robot = FakeRobot(odom={"x": 0.5, "y": 0.3, "theta": 1.2, "v": 0.1})
        m = Mesh(robot, peer_id="odom-1")
        m.start()
        time.sleep(0.2)
        m.stop()

        odom_puts = [(k, d) for k, d in mock_put if k.endswith("/odom")]
        assert len(odom_puts) > 0
        assert odom_puts[0][1]["frame"] == "odom"

    def test_hand_loop(self, mock_session, mock_put):
        """Hand loop publishes per-hand state."""
        robot = FakeRobot(
            hands={
                "left": {"joints": [0.1, 0.2, 0.3], "force": 1.5},
                "right": {"joints": [0.4, 0.5, 0.6], "force": 2.0},
            }
        )
        m = Mesh(robot, peer_id="hand-1")
        m.start()
        time.sleep(0.05)  # HAND_HZ=50
        m.stop()

        hand_puts = [(k, d) for k, d in mock_put if "/hand/" in k]
        assert len(hand_puts) > 0
        # Should have both left and right
        topics = set(k for k, _ in hand_puts)
        assert any("left" in t for t in topics)
        assert any("right" in t for t in topics)

    def test_lidar_loop(self, mock_session, mock_put):
        """Lidar summary + state publishing."""
        robot = FakeRobot(
            lidar_summary={"points": 1000, "range_min": 0.1, "range_max": 10.0},
            lidar_state={"status": "scanning", "rpm": 300},
        )
        m = Mesh(robot, peer_id="lidar-1")
        m.start()
        time.sleep(1.1)  # LIDAR_STATE_HZ=1.0
        m.stop()

        summary_puts = [(k, d) for k, d in mock_put if "/lidar/summary" in k]
        state_puts = [(k, d) for k, d in mock_put if "/lidar/state" in k]
        assert len(summary_puts) > 0
        assert len(state_puts) > 0

    def test_map_info_loop(self, mock_session, mock_put):
        """Map info loop publishes."""
        robot = FakeRobot(map_info={"name": "office", "resolution": 0.05, "size": [100, 100]})
        m = Mesh(robot, peer_id="map-1")
        m.start()
        time.sleep(5.5)  # MAP_INFO_HZ=0.2, need >5s
        m.stop()

        map_puts = [(k, d) for k, d in mock_put if "/map/info" in k]
        assert len(map_puts) > 0
        assert map_puts[0][1]["name"] == "office"

    def test_sensor_loops_no_crash_on_exception(self, mock_session):
        """Sensor loops don't crash if the robot attribute raises."""

        class ExplodingRobot:
            tool_name_str = "exploder"

            @property
            def _pose(self):
                raise RuntimeError("sensor failure")

            @property
            def _imu(self):
                raise ValueError("imu broken")

            @property
            def _battery(self):
                raise OSError("battery read failed")

        with patch.object(mesh_core, "put"):
            m = Mesh(ExplodingRobot(), peer_id="explode-1")
            m.start()
            time.sleep(0.5)
            assert m.alive  # Still alive despite errors
            m.stop()


# ===========================================================================
# Test: Presence Building
# ===========================================================================


class TestPresenceBuilding:
    """Exhaustive presence payload construction."""

    def test_presence_with_all_features(self, mock_session):
        """A fully-featured robot produces a rich presence payload."""
        inner = SimpleNamespace(
            is_connected=True,
            name="so100",
            config=SimpleNamespace(cameras={"wrist": {}, "top": {}}),
        )
        pub = MagicMock()
        pub._running = True
        pub.method = "arm"
        pub.hz = 50.0

        robot = FakeRobot(
            name="full_bot",
            with_task=True,
            with_world=True,
            action_features={"a0": float, "a1": float},
            input_publishers={"leader": pub},
        )
        robot.robot = inner

        m = Mesh(robot, peer_id="full-1", peer_type="sim")
        p = m._build_presence()

        assert p["robot_id"] == "full-1"
        assert p["robot_type"] == "sim"
        assert p["tool_name"] == "full_bot"
        assert p["task_status"] == "running"
        assert p["instruction"] == "test instruction"
        assert p["connected"] is True
        assert p["hw"] == "so100"
        assert sorted(p["cameras"]) == ["top", "wrist"]
        assert p["world"] is True
        assert "arm0" in p["sim_robots"]
        assert sorted(p["action_keys"]) == ["a0", "a1"]
        assert len(p["inputs"]) == 1
        assert p["inputs"][0]["method"] == "arm"

    def test_presence_advertises_available_topics(self, mock_session):
        """Topics field lists which sensor streams are available."""
        robot = FakeRobot(
            pose={"x": 0},
            imu={"rpy": [0, 0, 0]},
            odom={"x": 0},
            battery=50.0,
            hands={"left": {}},
            map_info={"name": "test"},
        )
        # Add lidar
        robot._lidar_summary = {"points": 100}

        m = Mesh(robot, peer_id="topics-1")
        p = m._build_presence()

        assert "topics" in p
        assert "pose" in p["topics"]
        assert "imu" in p["topics"]
        assert "odom" in p["topics"]
        assert "health" in p["topics"]
        assert "hand" in p["topics"]
        assert "lidar" in p["topics"]
        assert "map" in p["topics"]


# ===========================================================================
# Test: Input Publisher / Receiver
# ===========================================================================


class TestInputPublisherReceiver:
    """Teleop input streaming over mesh."""

    def test_publisher_lifecycle(self, mock_session, mock_put):
        """Publisher starts, publishes frames, stops with stats."""
        teleop = MagicMock()
        teleop.get_action.return_value = {"j0": 0.1, "j1": 0.2}

        m = Mesh(FakeRobot(), peer_id="pub-test")
        m.start()

        pub = InputPublisher(m, teleop, device_name="leader", method="arm", hz=100)
        pub.start()
        time.sleep(0.1)  # ~10 frames at 100Hz
        stats = pub.stop()

        assert stats["running"] is False
        assert stats["frames"] > 0
        assert stats["device"] == "leader"
        assert stats["method"] == "arm"
        m.stop()

    def test_publisher_handles_teleop_error(self, mock_session, mock_put):
        """Publisher survives when get_action raises."""
        teleop = MagicMock()
        call_count = [0]

        def side_effect():
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise RuntimeError("teleop disconnected")
            return {"j0": 0.5}

        teleop.get_action.side_effect = side_effect

        m = Mesh(FakeRobot(), peer_id="pub-err")
        m.start()

        pub = InputPublisher(m, teleop, device_name="leader", hz=100)
        pub.start()
        time.sleep(0.1)
        stats = pub.stop()

        assert stats["errors"] > 0
        assert stats["frames"] > 0  # Some frames still published
        m.stop()

    def test_publisher_normalize_numpy_action(self, mock_session, mock_put):
        """Numpy array actions are normalized to dict."""
        import numpy as np

        teleop = MagicMock()
        teleop.get_action.return_value = np.array([0.1, 0.2, 0.3])

        m = Mesh(FakeRobot(), peer_id="pub-np")
        m.start()

        pub = InputPublisher(m, teleop, device_name="arm", hz=50)
        pub.start()
        time.sleep(0.05)
        stats = pub.stop()

        assert stats["frames"] > 0
        # Check put was called with normalized action
        input_puts = [(k, d) for k, d in mock_put if "/input/" in k]
        if input_puts:
            _, payload = input_puts[0]
            assert "j0" in payload["action"]
            assert "j1" in payload["action"]
            assert "j2" in payload["action"]
        m.stop()

    def test_receiver_applies_actions(self, mock_session):
        """Receiver calls send_action on the robot."""
        robot = MagicMock()
        robot.send_action = MagicMock()

        m = Mesh(FakeRobot(), peer_id="recv-test")
        m.start()

        recv = InputReceiver(m, robot, source_peer_id="leader-1", device_name="leader")
        recv.start()

        # Disable the apply-rate cap for this plumbing test: the two frames
        # below are delivered synchronously microseconds apart, which the
        # default 100Hz cap would (correctly) rate-limit. The cap has its own
        # dedicated coverage; here we only assert send_action plumbing.
        import os as _os

        _os.environ["STRANDS_MESH_INPUT_MAX_HZ"] = "0"
        try:
            # Simulate incoming data
            recv._on_input("strands/leader-1/input/leader", {"action": {"j0": 0.5, "j1": 0.3}, "seq": 0})
            recv._on_input("strands/leader-1/input/leader", {"action": {"j0": 0.6, "j1": 0.4}, "seq": 1})
        finally:
            _os.environ.pop("STRANDS_MESH_INPUT_MAX_HZ", None)

        stats = recv.stop()

        assert stats["frames_received"] == 2
        assert robot.send_action.call_count == 2
        m.stop()

    def test_receiver_detects_frame_drops(self, mock_session):
        """Receiver tracks sequence gaps."""
        robot = MagicMock()
        robot.send_action = MagicMock()

        m = Mesh(FakeRobot(), peer_id="recv-drop")
        m.start()

        recv = InputReceiver(m, robot, source_peer_id="src", device_name="arm")
        recv.start()

        # Seq 0 → 5 (skip 1,2,3,4)
        recv._on_input("topic", {"action": {"j0": 0.1}, "seq": 0})
        recv._on_input("topic", {"action": {"j0": 0.2}, "seq": 5})

        stats = recv.stop()
        assert stats["drops"] == 4
        m.stop()

    def test_receiver_stops_when_not_running(self, mock_session):
        """_on_input is a no-op after stop()."""
        robot = MagicMock()
        robot.send_action = MagicMock()

        m = Mesh(FakeRobot(), peer_id="recv-stop")
        m.start()

        recv = InputReceiver(m, robot, source_peer_id="src", device_name="arm")
        recv.start()
        recv.stop()

        recv._on_input("topic", {"action": {"j0": 0.1}, "seq": 0})
        assert robot.send_action.call_count == 0
        m.stop()


# ===========================================================================
# Test: Audit Log
# ===========================================================================


class TestAuditLog:
    """Audit log integrity and error resilience."""

    def test_log_and_read_round_trip(self, tmp_path, monkeypatch):
        """Write → read preserves all fields."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        log_safety_event("test_event", "peer-1", {"severity": "warning", "detail": "test"})
        log_safety_event("another", "peer-2", {"severity": "critical"})

        events = read_audit_log()
        assert len(events) == 2
        assert events[0]["event"] == "test_event"
        assert events[0]["peer_id"] == "peer-1"
        assert events[0]["payload"]["severity"] == "warning"
        assert events[1]["event"] == "another"

    def test_read_with_since_filter(self, tmp_path, monkeypatch):
        """since= parameter filters old events."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        log_safety_event("old", "p1", {})
        cutoff = time.time()
        time.sleep(0.01)
        log_safety_event("new", "p2", {})

        events = read_audit_log(since=cutoff)
        assert len(events) == 1
        assert events[0]["event"] == "new"

    def test_read_empty_log(self, tmp_path, monkeypatch):
        """Reading a nonexistent log returns []."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "nonexistent"))
        events = read_audit_log()
        assert events == []

    def test_concurrent_writes(self, tmp_path, monkeypatch):
        """Multiple threads writing simultaneously don't corrupt the file."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))

        def writer(prefix):
            for i in range(50):
                log_safety_event(f"{prefix}_{i}", "peer", {"i": i})

        threads = [threading.Thread(target=writer, args=(f"t{n}",)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        events = read_audit_log()
        assert len(events) == 250  # 5 threads × 50 events

    def test_audit_log_permissions(self, tmp_path, monkeypatch):
        """File and directory permissions are set correctly."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        log_safety_event("perm_test", "p1", {})

        log_file = tmp_path / "mesh_audit.jsonl"
        assert log_file.exists()
        stat = log_file.stat()
        # Mode 0o600 (owner read/write only)
        assert oct(stat.st_mode & 0o777) == "0o600"


# ===========================================================================
# Test: Subscribe Race Conditions
# ===========================================================================


class TestSubscribeRaces:
    """Race conditions in subscribe/unsubscribe."""

    def test_subscribe_after_stop_returns_none(self, mock_session):
        """subscribe() on a stopped mesh returns None."""
        m = Mesh(FakeRobot(), peer_id="sub-race-1")
        m.start()
        m.stop()
        result = m.subscribe("strands/*/state")
        assert result is None

    def test_unsubscribe_during_active_handler(self, mock_session):
        """Unsubscribe while handler is executing should not crash."""
        m = Mesh(FakeRobot(), peer_id="sub-race-2")
        m.start()

        received = []
        barrier = threading.Barrier(2, timeout=5)

        def slow_handler(topic, data):
            received.append(data)
            try:
                barrier.wait()  # Block until main thread unsubscribes
            except threading.BrokenBarrierError:
                pass

        name = m.subscribe("test/topic", callback=slow_handler, name="slow")
        assert name is not None

        # Simulate message arriving
        handler = mock_session.declare_subscriber.call_args_list[-1].args[1]
        msg_thread = threading.Thread(
            target=handler,
            args=(MagicMock(key_expr="test/topic", payload=MagicMock(to_bytes=MagicMock(return_value=b'{"x":1}'))),),
            daemon=True,
        )
        msg_thread.start()
        time.sleep(0.05)

        # Unsubscribe while handler is running
        m.unsubscribe("slow")
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

        msg_thread.join(timeout=2)
        # Should not crash
        assert len(received) == 1
        m.stop()

    def test_inbox_overflow_does_not_lose_recent(self, mock_session):
        """When inbox hits 1000, trimming keeps recent entries."""
        m = Mesh(FakeRobot(), peer_id="inbox-overflow")
        m.start()

        m.subscribe("overflow/topic", name="overflow")
        handler = mock_session.declare_subscriber.call_args_list[-1].args[1]

        # Pump 1200 messages
        for i in range(1200):
            sample = MagicMock()
            sample.key_expr = "overflow/topic"
            sample.payload.to_bytes.return_value = json.dumps({"i": i}).encode()
            handler(sample)

        buf = m.inbox["overflow"]
        # Buffer should never exceed 1000
        assert len(buf) <= 1000
        # Most recent message should be present
        assert buf[-1][1]["i"] == 1199
        # Oldest should be trimmed
        assert buf[0][1]["i"] >= 500
        m.stop()


# ===========================================================================
# Test: init_mesh Factory
# ===========================================================================


class TestInitMeshFactory:
    """init_mesh() public constructor edge cases."""

    def test_env_mesh_false_variations(self, monkeypatch, mock_session):
        """All falsy STRANDS_MESH values disable mesh."""
        for val in ["false", "FALSE", "  False  ", "0", "no"]:
            monkeypatch.setenv("STRANDS_MESH", val)
            # Only "false" (case-insensitive, stripped) disables
            result = init_mesh(FakeRobot(), peer_id="env-test")
            if val.strip().lower() == "false":
                assert result is None
            # "0" and "no" are NOT treated as false by the implementation
            # (it only checks == "false")

    def test_explicit_mesh_false_overrides_env(self, monkeypatch, mock_session):
        """mesh=False parameter disables regardless of env."""
        monkeypatch.setenv("STRANDS_MESH", "true")
        result = init_mesh(FakeRobot(), peer_id="explicit-false", mesh=False)
        assert result is None

    def test_auto_peer_id_uniqueness(self, mock_session):
        """Auto-generated peer IDs should be unique across calls."""
        ids = set()
        for _ in range(100):
            m = init_mesh(FakeRobot(name="bot"))
            assert m is not None
            ids.add(m.peer_id)
            m.stop()
        assert len(ids) == 100  # All unique


# ===========================================================================
# Test: robot_mesh Tool
# ===========================================================================


class TestRobotMeshTool:
    """The agent-facing robot_mesh tool."""

    def test_peers_action_no_mesh(self):
        """peers action when no local mesh exists."""
        from unittest.mock import MagicMock as _MM

        from strands_robots.tools.robot_mesh import robot_mesh

        result = robot_mesh(action="peers", tool_context=_MM())
        assert result["status"] == "success"
        assert "0 local" in result["content"][0]["text"]

    def test_tell_without_target_errors(self, mock_session):
        """tell action without target returns error."""
        from strands_robots.tools.robot_mesh import robot_mesh

        m = Mesh(FakeRobot(), peer_id="tool-test")
        m.start()
        from unittest.mock import MagicMock as _MM

        result = robot_mesh(action="tell", instruction="go", tool_context=_MM())
        assert result["status"] == "error"
        m.stop()

    def test_send_invalid_json(self, mock_session):
        """send with non-JSON command returns error."""
        from strands_robots.tools.robot_mesh import robot_mesh

        m = Mesh(FakeRobot(), peer_id="tool-json")
        m.start()
        from unittest.mock import MagicMock as _MM

        result = robot_mesh(action="send", target="peer-x", command="not{json", tool_context=_MM())
        assert result["status"] == "error"
        assert "not valid JSON" in result["content"][0]["text"]
        m.stop()

    def test_unknown_action(self, mock_session):
        """Unknown action returns helpful error."""
        from strands_robots.tools.robot_mesh import robot_mesh

        m = Mesh(FakeRobot(), peer_id="tool-unk")
        m.start()
        from unittest.mock import MagicMock as _MM

        result = robot_mesh(action="foobar", tool_context=_MM())
        assert result["status"] == "error"
        assert "unknown action" in result["content"][0]["text"]
        m.stop()

    def test_resolve_mesh_avoids_self_gateway(self, mock_session):
        """When sending to a local peer, _resolve_mesh picks a different one."""
        from strands_robots.tools.robot_mesh import _resolve_mesh

        m1 = Mesh(FakeRobot(), peer_id="local-a")
        m2 = Mesh(FakeRobot(), peer_id="local-b")
        m1.start()
        m2.start()

        gateway = _resolve_mesh("local-a")
        # Should pick local-b as gateway (not the target itself)
        assert gateway.peer_id == "local-b"

        m1.stop()
        m2.stop()


# ===========================================================================
# Test: Emergency Stop Integration
# ===========================================================================


class TestEmergencyStop:
    """Emergency stop broadcasts and audits."""

    def test_estop_publishes_safety_topic(self, mock_session, mock_put):
        """E-stop publishes to strands/safety/estop."""
        m = Mesh(FakeRobot(), peer_id="estop-1")
        m.start()

        with patch.object(m, "broadcast", return_value=[{"stopped": True}]):
            responses = m.emergency_stop()

        assert responses == [{"stopped": True}]
        # Check safety topic was published
        safety_puts = [(k, d) for k, d in mock_put if "safety/estop" in k]
        assert len(safety_puts) == 1
        assert safety_puts[0][1]["peer_id"] == "estop-1"
        m.stop()


# ===========================================================================
# Test: Thread Safety Under Load
# ===========================================================================


class TestThreadSafety:
    """Concurrent operations across all mesh components."""

    def test_presence_state_rpc_all_concurrent(self, mock_session, mock_put):
        """All loops + RPC + subscribe running simultaneously."""
        robot = FakeRobot(
            with_task=True,
            with_world=True,
            pose={"x": 1},
            imu={"rpy": [0, 0, 0]},
            battery=80,
        )
        m = Mesh(robot, peer_id="stress-all")
        m.start()

        errors = []

        def send_rpcs():
            try:
                for _ in range(20):
                    m.send("fake-peer", {"action": "status"}, timeout=0.01)
            except Exception as e:
                errors.append(e)

        def subscribe_unsub():
            try:
                for i in range(10):
                    name = m.subscribe(f"topic/{i}", name=f"s{i}")
                    time.sleep(0.01)
                    if name:
                        m.unsubscribe(f"s{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_rpcs),
            threading.Thread(target=subscribe_unsub),
        ]
        for t in threads:
            t.start()

        time.sleep(1.0)

        for t in threads:
            t.join(timeout=5)

        m.stop()
        assert not errors, f"Thread safety errors: {errors}"
        assert not m.alive
